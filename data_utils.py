import json

from datasets import load_dataset, DatasetDict


def transform_mbpp(output_dir="datasets"):
    dataset = load_dataset("google-research-datasets/mbpp")

    def process_example(e):
        prompt = f"{e['text']} Your code should pass these tests:"
        for test in e["test_list"]:
            prompt += f"\n{test}"

        messages = [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": e["code"]
            }
        ]
        return {"messages": messages}

    dataset = dataset.map(process_example, num_proc=8)
    dataset.save_to_disk(f"{output_dir}/mbpp")


def transform_conala(output_dir="datasets"):
    dataset = load_dataset("neulab/docprompting-conala")

    def process_example(e):
        messages = [
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
    # https://huggingface.co/spaces/codeparrot/apps_metric/blob/main/example_script.py

    dataset = load_dataset("codeparrot/apps", trust_remote_code=True)

    def process_example(e):
        starter_code = None if len(e["starter_code"]) == 0 else e["starter_code"]
        try:
            input_outpout = json.loads(e["input_output"])
            fn_name = None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
        except ValueError:
            fn_name = None
        try:
            solutions = json.loads(e["solutions"])
        except ValueError:
            solutions = [""]

        _input = e["question"]
        if starter_code:
            _input += starter_code
        if fn_name:
            _input += "\nUse Standard Input format\n"
        else:
            _input += "\nUse Call-Based format\n"

        messages = [
            {
                "role": "user",
                "content": _input
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
    # transform_code_alpaca()
    # transform_apps()
    transform_mbpp()