import json

from datasets import load_dataset, DatasetDict


def transform_conala(output_dir="datasets"):
    dataset = load_dataset("neulab/docprompting-conala")

    def process_example(e):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": e["nl"]
            },
            {
                "role": "assistant",
                "content": e["cmd"]
            }
        ]
        return {"messages": messages}

    dataset = dataset.map(process_example, num_proc=8)
    dataset.save_to_disk(f"{output_dir}/conala")


def transform_code_alpaca(output_dir="datasets"):
    dataset = load_dataset("HuggingFaceH4/CodeAlpaca_20K")

    def process_example(e):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": e["prompt"]
            },
            {
                "role": "assistant",
                "content": e["completion"]
            }
        ]
        return {"messages": messages}

    # create validation set
    train_set = dataset["train"].shuffle(42)
    validation_set = train_set.select(range(500))
    train_set = train_set.select(range(500, len(train_set)))
    dataset = DatasetDict({
        "train": train_set,
        "validation": validation_set,
        "test": dataset["test"]
    })

    dataset = dataset.map(process_example, num_proc=8)
    dataset.save_to_disk(f"{output_dir}/codealpaca")


def transform_apps(output_dir="datasets"):
    # this preprocessing follows the same format used in the original APPs paper:
    # https://github.com/hendrycks/apps/blob/main/train/dataset_apps/APPSBaseDataset.py

    dataset = load_dataset("codeparrot/apps", trust_remote_code=True)

    def process_example(e):
        try:
            solutions = json.loads(e["solutions"])
        except:
            solutions = [""]
        question = e["question"]

        if e["starter_code"] != "":
            question += "\n" + e["starter_code"] + "\n" + "\nUse Call-Based format\n"
        else:
            question += "\n" + "\nUse Standard Input format\n"

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": solutions[0]
            }
        ]
        return {"messages": messages}

    # create validation set
    train_set = dataset["train"].shuffle(42)
    validation_set = train_set.select(range(500))
    train_set = train_set.select(range(500, len(train_set)))
    dataset = DatasetDict({
        "train": train_set,
        "validation": validation_set,
        "test": dataset["test"]
    })

    dataset = dataset.map(process_example, num_proc=8)
    dataset.save_to_disk(f"{output_dir}/apps")


if __name__ == "__main__":
    # transform_conala()
    transform_code_alpaca()
    # transform_apps()
