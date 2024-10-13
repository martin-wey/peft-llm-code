import json
import warnings
from typing import List, Union, Dict, Any, Optional

import numpy as np
import torch

from datasets import load_dataset, Dataset, DatasetDict
from peft import (
    PeftModel,
    get_peft_model,
    TaskType,
    LoraConfig,
    IA3Config,
    PromptTuningConfig,
    PrefixTuningConfig
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)

LORA_IA3_TARGET_MODULES = {
    "codegen-350M-mono": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
    "codet5p-220m": {
        "target_modules_lora": ["q", "v", "k"],
        "target_modules_ia3": ["q", "v", "k", "wi", "wo"],
        "ff_modules": ["wi", "wo"]
    },
    "codet5p-770m": {
        "target_modules_lora": ["q", "v", "k"],
        "target_modules_ia3": ["q", "v", "k", "wi", "wo"],
        "ff_modules": ["wi", "wo"]
    },
    "codet5p-6b": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
    "codegen2-1B": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
    "codegen2-3_7B": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
    "codegen2-7B": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
    },
    "CodeLlama-7b-hf": {
        "target_modules_lora":["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "ff_modules": ["gate_proj", "up_proj", "down_proj"]
    },
    "CodeLlama-7b-Instruct-hf": {
        "target_modules_lora": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "ff_modules": ["gate_proj", "up_proj", "down_proj"]
    },
    "CodeLlama-7b-Python-hf": {
        "target_modules_lora": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "ff_modules": ["gate_proj", "up_proj", "down_proj"]
    },
    "CodeLlama-13b-Python-hf": {
        "target_modules_lora": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "ff_modules": ["gate_proj", "up_proj", "down_proj"]
    },
    "CodeLlama-34b-Python-hf": {
        "target_modules_lora": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "ff_modules": ["gate_proj", "up_proj", "down_proj"]
    }
}

#
# https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L86
#
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template: Union[str, List[int]],
        instruction_template: Optional[Union[str, List[int]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        padding_free: bool = False,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index
        self.padding_free = padding_free

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # find the indexes of the start of a response.
                    if (
                        self.response_token_ids
                        == batch["labels"][i][assistant_idx : assistant_idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))

                if len(response_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    # find the indexes of the start of a human answer.
                    if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find instruction key `{self.instruction_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        if self.padding_free:
            # remove padding, `attention_mask` and add `position_ids`
            attn_mask = batch.pop("attention_mask")
            batch["input_ids"] = batch["input_ids"][attn_mask.bool()].unsqueeze(0)
            batch["position_ids"] = attn_mask.cumsum(1)[attn_mask.bool()].unsqueeze(0) - 1
            batch["labels"] = batch["labels"][attn_mask.bool()].unsqueeze(0)
            batch["labels"][batch["position_ids"] == 0] = self.ignore_index

        return batch


def load_model_and_tokenizer(args, is_training=False):
    model_cls = AutoModelForCausalLM
    task_type = TaskType.CAUSAL_LM

    if "codet5" in args.model_name_or_path:
        model_cls = AutoModelForSeq2SeqLM
        task_type = TaskType.SEQ_2_SEQ_LM

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16
    }
    if not is_training:
        model_kwargs["device_map"] = "auto"

    # Apply quantization if necessary
    if args.tuning_method == "qlora-8bit":
        qconfig = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["quantization_config"] = qconfig
    elif args.tuning_method == "qlora-4bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_kwargs["quantization_config"] = qconfig

    model = model_cls.from_pretrained(args.model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    if is_training and args.adapter_path is None:
        if args.tuning_method in ["lora", "qlora-8bit", "qlora-4bit"]:
            peft_config = LoraConfig(
                task_type=task_type,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=LORA_IA3_TARGET_MODULES[args.model_name]["target_modules_lora"],
                lora_dropout=args.lora_dropout,
                bias="none"
            )
        elif args.tuning_method == "ia3":
            peft_config = IA3Config(
                task_type=task_type,
                target_modules=LORA_IA3_TARGET_MODULES[args.model_name]["target_modules_ia3"],
                feedforward_modules=LORA_IA3_TARGET_MODULES[args.model_name]["ff_modules"]
            )
        elif args.tuning_method == "prompt-tuning":
            peft_config = PromptTuningConfig(
                task_type=task_type,
                num_virtual_tokens=args.prompt_num_virtual_tokens,
                prompt_tuning_init="TEXT",
                prompt_tuning_init_text="Generate Python code given a natural language instruction.",
                tokenizer_name_or_path=args.model_name_or_path
            )
        elif args.tuning_method == "prefix-tuning":
            peft_config = PrefixTuningConfig(
                task_type=task_type,
                num_virtual_tokens=args.prefix_num_virtual_tokens
            )

        if args.tuning_method != "ft":
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
    else:
        if args.adapter_path is not None:
            if args.do_test:
                model = PeftModel.from_pretrained(model, args.adapter_path)
                model = model.merge_and_unload()
                model.config.use_cache = True
            else:
                # continue fine-tuning from PEFT adapter checkpoint
                model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
                model.print_trainable_parameters()

    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    # set padding side for non-seq2seq models
    if "codet5" not in args.model_name_or_path and args.adapter_path is None:
        tokenizer.padding_side = "left"

    return model, tokenizer
