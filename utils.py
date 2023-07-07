import json

from datasets import load_dataset

LORA_TARGET_MODULES = {
    "PolyCoder-2.7B": ["query_key_value", "xxx"],
    "codegen-2B-mono": ["q_proj", "v_proj"],
    "codegen-6B-mono": ["q_proj", "v_proj"],
    "incoder-1B": ["q_proj", "v_proj"],
    "incoder-6B": ["q_proj", "v_proj"],
    "bloom-3b": ["query_key_value"],
    "bloom-7b1": ["query_key_value"]
}


def load_train_dataset():
    datasets = load_dataset("neulab/docprompting-conala")
    datasets = datasets.filter(lambda x: x["nl"] is not None)
    datasets = datasets.filter(lambda x: x["cmd"] is not None)

    return datasets


def read_icl_examples():
    with open("icl_examples.json") as f:
        examples = json.load(f)
    return examples


def load_odex_test_dataset():
    dataset = load_dataset("neulab/odex")["test"]
    conala = load_dataset("neulab/docprompting-conala")["train"]

    # make sure we remove test samples that appear in the fine-tuning dataset to avoid data leakage
    dataset = dataset.filter(lambda example: example["intent"] not in conala["nl"])

    return dataset


def load_conala_test_dataset():
    dataset = load_dataset("neulab/docprompting-conala")["test"]
    return dataset
