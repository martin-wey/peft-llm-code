import json

from datasets import load_dataset
from transformers import \
    AutoModelForSequenceClassification, \
    AutoModelForCausalLM, \
    AutoModelForSeq2SeqLM

LORA_TARGET_MODULES = {
    "PolyCoder-2.7B": ["query_key_value", "xxx"],
    "codegen-2B-mono": ["q_proj", "v_proj"],
    "codegen-6B-mono": ["q_proj", "v_proj"],
    "codegen-16B-mono": ["q_proj", "v_proj"],
    "codegen-6B-multi": ["q_proj", "v_proj"],
    "codet5p-2b": ["q_proj", "v_proj"],
    "codet5p-6b": ["q_proj", "v_proj"],
    "incoder-1B": ["q_proj", "v_proj"],
    "incoder-6B": ["q_proj", "v_proj"],
    "bloom-3b": ["query_key_value"],
    "bloom-7b1": ["query_key_value"]
}

DEFECT_MODEL_CLS = {
    "encoder": AutoModelForSequenceClassification,
    "decoder": AutoModelForCausalLM,
    "encoder-decoder": AutoModelForSeq2SeqLM
}

GENERATION_MODEL_CLS = {
    "encoder": AutoModelForCausalLM,
    "decoder": AutoModelForCausalLM,
    "encoder-decoder": AutoModelForSeq2SeqLM
}


def load_conala_dataset():
    datasets = load_dataset("neulab/docprompting-conala")
    datasets = datasets.filter(lambda x: x["nl"] is not None)
    datasets = datasets.filter(lambda x: x["cmd"] is not None)

    return datasets


def read_conala_few_shot_examples():
    with open("conala_few_shot_examples.json") as f:
        examples = json.load(f)
    return examples


def load_conala_unit_tests_dataset():
    dataset = load_dataset("neulab/odex")["test"]
    conala = load_dataset("neulab/docprompting-conala")["test"]

    # only select samples that appear in CoNaLa dataset
    dataset = dataset.filter(lambda example: example["intent"] in conala["nl"])

    return dataset
