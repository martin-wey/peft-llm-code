import json

from datasets import load_dataset

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


def load_conala_train_dataset():
    datasets = load_dataset("neulab/docprompting-conala")
    datasets = datasets.filter(lambda x: x["nl"] is not None)
    datasets = datasets.filter(lambda x: x["cmd"] is not None)
    del datasets["test"]
    return datasets


def load_conala_test_dataset():
    dataset = load_dataset("neulab/docprompting-conala")["test"]
    return dataset


def load_codealpaca_train_dataset():
    dataset = load_dataset("antolin/codealpaca-filtered")
    dataset["validation"] = dataset["valid"]
    del dataset["test"], dataset["valid"]
    return dataset


def load_codealpaca_test_dataset():
    return load_dataset("antolin/codealpaca-filtered")["test"]


def load_conala_icl_examples():
    with open("conala_icl_examples.json") as f:
        examples = json.load(f)
    return examples


def load_codealpaca_icl_examples():
    with open("codealpaca_icl_examples.json") as f:
        examples = json.load(f)
    return examples


def load_odex_test_dataset():
    dataset = load_dataset("neulab/odex")["test"]
    conala = load_dataset("neulab/docprompting-conala")["train"]

    # make sure we remove test samples that appear in the fine-tuning dataset to avoid data leakage
    dataset = dataset.filter(lambda example: example["intent"] not in conala["nl"])

    return dataset
