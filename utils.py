from dataclasses import field, dataclass
from typing import Optional, List

from trl import is_peft_available
from trl.core import flatten_dict

if is_peft_available():
    from peft import LoraConfig, PeftConfig, PromptEncoderConfig, PrefixTuningConfig, PromptTuningConfig, \
        PromptTuningInit

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
    "codet5p-2b": {
        "target_modules_lora": ["qkv_proj"],
        "target_modules_ia3": ["qkv_proj", "fc_in", "fc_out"],
        "ff_modules": ["fc_in", "fc_out"]
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
        "target_modules_lora": ["q_proj", "k_proj", "v_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "ff_modules": ["gate_proj", "up_proj", "down_proj"]
    },
    "CodeLlama-7b-Instruct-hf": {
        "target_modules_lora": ["q_proj", "k_proj", "v_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "ff_modules": ["gate_proj", "up_proj", "down_proj"]
    },
    "CodeLlama-7b-Python-hf": {
        "target_modules_lora": ["q_proj", "k_proj", "v_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "ff_modules": ["gate_proj", "up_proj", "down_proj"]
    },
    "CodeLlama-13b-Python-hf": {
        "target_modules_lora": ["q_proj", "k_proj", "v_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "ff_modules": ["gate_proj", "up_proj", "down_proj"]
    },
    "CodeLlama-34b-Python-hf": {
        "target_modules_lora": ["q_proj", "k_proj", "v_proj"],
        "target_modules_ia3": ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "ff_modules": ["gate_proj", "up_proj", "down_proj"]
    }
}


@dataclass
class SFTScriptArguments:
    dataset_name: str = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "the dataset name"},
    )
    dataset_train_split: str = field(default="train", metadata={"help": "The dataset split to train on"})
    dataset_test_split: str = field(default="test", metadata={"help": "The dataset split to evaluate on"})
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
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code when loading a model."})
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
    use_dora: bool = field(
        default=False,
        metadata={"help": ("Whether to use DoRA.")},
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
    encoder_num_layers: int = field(
        default=2,
        metadata={"help": ("Encoder number of layers for p-tuning.")},
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


def get_peft_config(model_config: ModelConfig) -> "Optional[PeftConfig]":
    if model_config.use_peft is False:
        return None

    if not is_peft_available():
        raise ValueError(
            "You need to have PEFT library installed in your environment, make sure to install `peft`. "
            "Make sure to run `pip install -U peft`."
        )

    if model_config.use_lora:
        peft_config = LoraConfig(
            use_dora=model_config.use_dora,
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            bias="none",
            task_type=model_config.task_type,
            target_modules=model_config.lora_target_modules,
            modules_to_save=model_config.lora_modules_to_save,
        )
    elif model_config.use_p_tuning:
        peft_config = PromptEncoderConfig(
            task_type=model_config.task_type,
            num_virtual_tokens=model_config.num_virtual_tokens,
            encoder_hidden_size=model_config.encoder_hidden_size,
            encoder_num_layers=model_config.encoder_num_layers,
        )
    elif model_config.use_prefix_tuning:
        peft_config = PrefixTuningConfig(
            task_type=model_config.task_type,
            num_virtual_tokens=model_config.num_virtual_tokens,
        )
    elif model_config.use_prompt_tuning:
        prompt_tuning_init_text = "Generate a response given a natural language instruction."
        peft_config = PromptTuningConfig(
            task_type=model_config.task_type,
            prompt_tuning_init=PromptTuningInit.TEXT,
            prompt_tuning_init_text=prompt_tuning_init_text,
            tokenizer_name_or_path=model_config.model_name_or_path,
        )
    else:
        peft_config = None

    return peft_config
