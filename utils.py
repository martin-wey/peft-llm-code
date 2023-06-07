import os

from tqdm import tqdm

from datasets import Dataset, DatasetDict, concatenate_datasets

LANG_TO_EXT = {
    'C#': "cs",
    'C++': "cpp",
    'C': "c",
    'Java': "java",
    'Javascript': "js",
    'PHP': "php",
    'Python': "py",
    'comment': "txt",
    'desc': "txt"
}


def load_xlcost_code_translation_dataset(base_dir):
    dataset_name = "xlcost_code-translation"
    dataset_dir = os.path.join(base_dir, dataset_name)
    lang_pairs = os.listdir(dataset_dir)

    datasets = {"train": [], "val": [], "test": []}
    for lang_pair in tqdm(lang_pairs, desc="Loading XLCoST code translation dataset"):
        input_lang, target_lang = lang_pair.split("-")
        for split in ["train", "val", "test"]:
            input_file = os.path.join(dataset_dir, lang_pair,
                                      f"{split}-{input_lang}-{target_lang}-tok.{LANG_TO_EXT[input_lang]}")
            target_file = os.path.join(dataset_dir, lang_pair,
                                       f"{split}-{input_lang}-{target_lang}-tok.{LANG_TO_EXT[target_lang]}")
            input_data = [sample.strip() for sample in open(input_file, encoding="utf-8").readlines()]
            target_data = [sample.strip() for sample in open(target_file, encoding="utf-8").readlines()]

            ds = Dataset.from_dict({"input": input_data, "target": target_data})
            ds = ds.add_column(name="input_lang", column=[input_lang] * len(ds))
            ds = ds.add_column(name="target_lang", column=[target_lang] * len(ds))
            datasets[split].append(ds)
    for split in ["train", "val", "test"]:
        datasets[split] = concatenate_datasets(datasets[split])
    return DatasetDict(datasets)


def load_xlcost_code_generation_dataset(base_dir):
    dataset_name = "xlcost_code-generation"
    dataset_dir = os.path.join(base_dir, dataset_name)
    lang_pairs = os.listdir(dataset_dir)

    datasets = {"train": [], "val": [], "test": []}
    for lang_pair in tqdm(lang_pairs, desc="Loading XLCoST code generation dataset"):
        target_lang = lang_pair.split("-")[0]
        for split in ["train", "val", "test"]:
            input_file = os.path.join(dataset_dir, lang_pair,
                                      f"{split}-{target_lang}-desc-tok.txt")
            target_file = os.path.join(dataset_dir, lang_pair,
                                       f"{split}-{target_lang}-desc-tok.{LANG_TO_EXT[target_lang]}")
            input_data = [sample.strip() for sample in open(input_file, encoding="utf-8").readlines()]
            target_data = [sample.strip() for sample in open(target_file, encoding="utf-8").readlines()]
            ds = Dataset.from_dict({"input": input_data, "target": target_data})
            ds = ds.add_column(name="target_lang", column=[target_lang] * len(ds))
            datasets[split].append(ds)
    for split in ["train", "val", "test"]:
        datasets[split] = concatenate_datasets(datasets[split])
    return DatasetDict(datasets)


def load_concode_code_generation_dataset(base_dir):
    dataset_name = "concode_code-generation"
    dataset_dir = os.path.join(base_dir, dataset_name)

    datasets = {}
    for split in tqdm(["train", "valid", "test"], desc="Loading CONCODE code generation dataset"):
        ds = Dataset.from_json(f"{dataset_dir}/{split}.json")
        datasets[split] = ds
    return DatasetDict(datasets)


def load_devign_defect_detection_dataset(base_dir):
    dataset_name = "devign_defect-detection"
    dataset_dir = os.path.join(base_dir, dataset_name)

    datasets = {}
    for split in tqdm(["train", "valid", "test"], desc="Loading Devign defect detection dataset"):
        ds = Dataset.from_json(f"{dataset_dir}/{split}.jsonl")
        datasets[split] = ds
    return DatasetDict(datasets)
