import json

from datasets import load_dataset, DatasetDict

from utils import make_chat_template_prompt, INSTRUCTION_PREFIX


def transform_magicoder(output_dir="datasets"):
    dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K")

    def process_example(e):
        messages = [
            {"role": "user", "content": e["instruction"]},
            {"role": "assistant", "content": e["response"]}
        ]
        return {"messages": messages}

    train_set = dataset["train"].shuffle(42)
    validation_set = train_set.select(range(1000))
    train_set = train_set.select(range(1000, len(train_set)))
    dataset = DatasetDict({
        "train": train_set,
        "validation": validation_set,
    })

    for split in dataset.keys():
        dataset[split] = dataset[split].map(lambda e: process_example(e), num_proc=8)
    dataset.save_to_disk(f"{output_dir}/magicoder")


def transform_conala(output_dir="datasets"):
    dataset = load_dataset("neulab/docprompting-conala", trust_remote_code=True)
    instruction_prefix = INSTRUCTION_PREFIX["conala"]

    def process_example(e, split):
        user_content = e["nl"]
        assistant_content = None if split == "test" else e["cmd"]
        messages = make_chat_template_prompt(user_content, assistant_content, instruction_prefix)
        return {"messages": messages}

    for split in dataset.keys():
        dataset[split] = dataset[split].map(lambda e: process_example(e, split), num_proc=8)
    dataset.save_to_disk(f"{output_dir}/conala")


def transform_mbpp(output_dir="datasets"):
    dataset = load_dataset("google-research-datasets/mbpp", trust_remote_code=True)
    instruction_prefix = INSTRUCTION_PREFIX["mbpp"]

    def process_example(e, split):
        user_content = f"{e['text']} Your code should pass these tests:"
        for test in e["test_list"]:
            user_content += f"\n{test}"
        assistant_content = None if split == "test" else e["code"]
        messages = make_chat_template_prompt(user_content, assistant_content, instruction_prefix)
        return {"messages": messages}

    for split in dataset.keys():
        dataset[split] = dataset[split].map(lambda e: process_example(e, split), num_proc=8)
    dataset.save_to_disk(f"{output_dir}/mbpp")


def transform_apps(output_dir="datasets"):
    # this preprocessing follows the same format used in the original APPs paper:
    # https://github.com/hendrycks/apps/blob/main/train/dataset_apps/APPSBaseDataset.py
    # https://huggingface.co/spaces/codeparrot/apps_metric/blob/main/example_script.py

    dataset = load_dataset("codeparrot/apps", trust_remote_code=True)
    instruction_prefix = INSTRUCTION_PREFIX["apps"]

    def process_example(e, split):
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

        user_content = e["question"]
        if starter_code:
            user_content += starter_code
        if fn_name:
            user_content += "\nUse Standard Input format\n"
        else:
            user_content += "\nUse Call-Based format\n"
        assistant_content = None if split == "test" else solutions[0]
        messages = make_chat_template_prompt(user_content, assistant_content, instruction_prefix)
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

    for split in dataset.keys():
        dataset[split] = dataset[split].map(lambda e: process_example(e, split), num_proc=8)
    dataset.save_to_disk(f"{output_dir}/apps")


if __name__ == "__main__":
    transform_magicoder()
    # transform_conala()
    # transform_mbpp()
    # transform_apps()
