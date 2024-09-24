import functools
import os
import subprocess
import time
from dataclasses import field, dataclass
from typing import Optional, List, Union, Any, Dict

import torch
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from trl.core import flatten_dict

from peft import (
    LoraConfig,
    PeftConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    PromptTuningInit,
    PrefixTuningConfig
)


LORA_TARGET_MODULES = {
    "Phi-3-mini-128k-instruct": ["o_proj", "qkv_proj"],
    "deepseek-coder-6.7b-instruct": ["q_proj", "v_proj", "o_proj", "k_proj"],
    "CodeQwen1.5-7B-Chat": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    "Meta-Llama-3.1-8B-Instruct": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    "Qwen2.5-Coder-7B-Instruct": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
}

_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

INSTRUCTION_PREFIX = {
    "conala": (
        "Provide a self-contained Python script that solves the following problem in a markdown code block. "
        "Your solution should most likely contain a single line of code, or only a few ones."
    ),
    "mbpp": (
        "Provide a self-contained Python script that solves the following problem in a markdown code block. "
        "You are given example test cases from which you can infer the function signature."
    ),
    "apps": (
        "Provide a self-contained Python script that solves the following problem in a markdown code block. "
        "Make sure the solution obeys the constraints and passes the example test cases."
    )
}


def encode_chat_template(chat_template, tokenizer):
    prompt = tokenizer.apply_chat_template(chat_template, tokenize=False).split(_MAGIC_SPLITTER_)[0]
    return tokenizer(prompt, return_attention_mask=True, return_tensors="pt")


def make_chat_template_prompt(instruction, response, instruction_prefix):
    # https://github.com/evalplus/evalplus/blob/master/evalplus/provider/utility.py#L25
    user_content = f"{instruction_prefix}\n```\n{instruction.strip()}\n```"
    if response is None:
        assistant_content = f"```python\n{_MAGIC_SPLITTER_}\n```"
    else:
        assistant_content = f"```python\n{response.strip()}\n```"

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]


class CustomDataCollatorForCompletionOnlyLM(DataCollatorForCompletionOnlyLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        return batch


def get_gpu_memory_usage():
    try:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if visible_devices is None:
            # select all available GPUs
            gpu_ids = range(torch.cuda.device_count())
        else:
            # select active GPUs for the current script
            gpu_ids = list(map(int, visible_devices.split(',')))

        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )

        # sum GPU memory usage from active GPUs
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        total_memory_used = sum(gpu_memory[gpu_id] for gpu_id in gpu_ids)

        return total_memory_used
    except Exception:
        return -1


def track_gpu_usage(func):
    @functools.wraps(func)
    def wrapper_track_gpu_usage(*args, **kwargs):
        initial_gpu_memory = get_gpu_memory_usage()
        max_gpu_memory_usage = initial_gpu_memory
        start_time = time.time()
        result = []
        try:
            gen = func(*args, **kwargs)
            for sample_index, output in enumerate(gen):
                current_gpu_memory = get_gpu_memory_usage()
                max_gpu_memory_usage = max(max_gpu_memory_usage, current_gpu_memory)
                result.append(output)
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            end_time = time.time()

        return result, initial_gpu_memory, max_gpu_memory_usage, end_time - start_time

    return wrapper_track_gpu_usage


@dataclass
class SFTScriptArguments:
    dataset_name: str = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "the dataset name"},
    )
    dataset_train_split: str = field(default="train", metadata={"help": "The dataset split to train on"})
    dataset_test_split: str = field(default="validation", metadata={"help": "The dataset split to evaluate on"})
    config: str = field(default=None, metadata={"help": "Path to the optional config file"})
    gradient_checkpointing_use_reentrant: bool = field(
        default=False,
        metadata={"help": "Whether to apply `use_reentrant` for gradient_checkpointing"},
    )
    completion_only: bool = field(
        default=False,
        metadata={"help": "Whether to only consider the assistant's response in the loss calculation"}
    )
    response_template: str = field(
        default=None,
        metadata={"help": "Response template when setting `completion_only`"}
    )


@dataclass
class ModelConfig:
    """
    Arguments which define the model and tokenizer to load.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The model checkpoint for weights initialization.")},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: bool = field(default=True, metadata={"help": "Trust remote code when loading a model."})
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Which attention implementation to use; you can run --attn_implementation=flash_attention_2, in which case you must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": ("Whether to use PEFT or not for training.")},
    )
    # LoRA methods
    use_lora: bool = field(
        default=False,
        metadata={"help": ("Whether to use LoRA.")},
    )
    lora_r: Optional[int] = field(
        default=16,
        metadata={"help": ("LoRA R value.")},
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": ("LoRA alpha.")},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": ("LoRA dropout.")},
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("LoRA target modules.")},
    )
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Model layers to unfreeze & train")},
    )
    task_type: str = field(
        default="CAUSAL_LM", metadata={"help": "The task_type to pass for LoRA (use SEQ_CLS for reward modeling)"}
    )
    # QLoRA
    load_in_8bit: bool = field(
        default=False, metadata={"help": "use 8 bit precision for the base model - works only with LoRA"}
    )
    load_in_4bit: bool = field(
        default=False, metadata={"help": "use 4 bit precision for the base model - works only with LoRA"}
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"}
    )
    use_bnb_nested_quant: bool = field(default=False, metadata={"help": "use nested quantization"})
    # Prompt methods
    use_p_tuning: bool = field(
        default=False,
        metadata={"help": ("Whether to use p-tuning.")},
    )
    use_prefix_tuning: bool = field(
        default=False,
        metadata={"help": ("Whether to use prefix tuning.")},
    )
    use_prompt_tuning: bool = field(
        default=False,
        metadata={"help": ("Whether to use prompt tuning.")},
    )
    num_virtual_tokens: int = field(
        default=20,
        metadata={"help": ("Number of virtual tokens for p-tuning or prefix tuning.")},
    )
    encoder_hidden_size: int = field(
        default=128,
        metadata={"help": ("Encoder hidden size for p-tuning.")},
    )
    active_gpu: int = field(
        default=-1,
        metadata={"help": ("The index of the active GPU used for training.")}
    )

    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")

        if isinstance(self.lora_target_modules, list) and len(self.lora_target_modules) == 1:
            self.lora_target_modules = self.lora_target_modules[0]


def get_peft_config(model_config: ModelConfig, tokenizer: AutoTokenizer) -> "Optional[PeftConfig]":
    if model_config.use_peft is False:
        return None

    model_name = model_config.model_name_or_path.split("/")[-1]

    if model_config.use_lora:
        peft_config = LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            bias="none",
            task_type=model_config.task_type,
            target_modules=LORA_TARGET_MODULES[model_name],
            modules_to_save=model_config.lora_modules_to_save,
        )
    elif model_config.use_p_tuning:
        peft_config = PromptEncoderConfig(
            task_type=model_config.task_type,
            num_virtual_tokens=model_config.num_virtual_tokens,
            encoder_hidden_size=model_config.encoder_hidden_size,
        )
    elif model_config.use_prompt_tuning:
        prompt_tuning_init_text = "Generate a Python code that solves the given problem.\n"
        peft_config = PromptTuningConfig(
            task_type=model_config.task_type,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=len(tokenizer(prompt_tuning_init_text)["input_ids"]),
            prompt_tuning_init_text=prompt_tuning_init_text,
            tokenizer_name_or_path=model_config.model_name_or_path,
        )
    elif model_config.use_prefix_tuning:
        peft_config = PrefixTuningConfig(
            task_type=model_config.task_type,
            num_virtual_tokens=model_config.num_virtual_tokens
        )
    else:
        peft_config = None

    return peft_config
