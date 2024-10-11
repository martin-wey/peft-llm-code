import json

from datasets import load_dataset, DatasetDict, concatenate_datasets


def preprocess_apps(sample):
    starter_code = None if len(sample["starter_code"]) == 0 else sample["starter_code"]
    try:
        input_outpout = json.loads(sample["input_output"])
        fn_name = None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
    except ValueError:
        fn_name = None

    if fn_name:
        # see CodeLlama paper - Appendix F.
        guide = "use the provided function signature"
    else:
        guide = "read from and write to standard IO"

    return {"guide": guide, "starter_code": starter_code}


def load_apps_train_dataset():
    dataset = load_dataset("codeparrot/apps")["train"].shuffle(42)
    validation_set = dataset.select(range(0, 500))
    training_set = dataset.select(range(500, len(dataset)))

    dataset = DatasetDict({"train": training_set, "validation": validation_set})
    dataset = dataset.map(preprocess_apps, num_proc=16, load_from_cache_file=False)
    return dataset


def load_apps_test_dataset():
    dataset = load_dataset("codeparrot/apps", split="test")
    introductory_set = dataset.filter(lambda sample: sample["difficulty"] == "introductory").select(range(100))
    interview_set = dataset.filter(lambda sample: sample["difficulty"] == "interview").select(range(100))
    competition_set = dataset.filter(lambda sample: sample["difficulty"] == "competition").select(range(100))

    dataset = concatenate_datasets([introductory_set, interview_set, competition_set])
    dataset = dataset.map(preprocess_apps, num_proc=16, load_from_cache_file=False)

    return dataset


def load_conala_train_dataset():
    dataset = load_dataset("neulab/docprompting-conala")
    dataset = dataset.filter(lambda x: x["nl"] is not None)
    dataset = dataset.filter(lambda x: x["cmd"] is not None)
    del dataset["test"]
    return dataset


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


if __name__ == "__main__":
    _ = load_apps_test_dataset()