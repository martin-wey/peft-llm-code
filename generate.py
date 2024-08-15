import argparse
import json
import os
import pickle

import evaluate
import torch
from peft import PeftModel
from rich.progress import MofNCompleteColumn, BarColumn, Progress, TextColumn, TimeElapsedColumn
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from datasets import load_from_disk

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--peft_checkpoint_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--use_icl", action="store_true", default=False)
    parser.add_argument("--icl_seed", type=int, default=42)
    parser.add_argument("--icl_n_examples", type=int, default=3)
    args = parser.parse_args()

    set_seed(42)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    if args.peft_checkpoint_path is not None:
        model = PeftModel.from_pretrained(model, args.peft_checkpoint_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_from_disk(args.dataset_name)["test"]
    # remove response from the messages
    dataset = dataset.map(lambda e: {"messages": e["messages"][:-1]}, num_proc=8)
    dataset_name = args.dataset_name.split("/")[-1]

    if dataset_name == "conala":
        max_new_tokens = 128
        instruction_field = "nl"
        reference_field = "cmd"
    elif dataset_name == "codealpaca":
        max_new_tokens = 512
        instruction_field = "prompt"
        reference_field = "completion"
    else:
        max_new_tokens = 1024
        instruction_field = "question"
        reference_field = "solution"

    if args.use_icl:
        train_set = load_from_disk(args.dataset_name)["train"].shuffle(args.icl_seed).select(range(args.icl_n_examples))
        chat_icl = []
        for example in train_set:
            chat_exemple = [
                {"role": "user", "content": example[instruction_field]},
                {"role": "assistant", "content": example[reference_field]},
            ]
            chat_icl += chat_exemple
        """
        icl_prompt = ""
        for icl_example in train_set:
            # @todo: add to argparse
            # add the <|end|> token to avoid the model generating more than the solution
            icl_prompt += f"\n### Instruction:\n{icl_example[instruction_field]}" \
                          f"\n### Response:\n{icl_example[reference_field]}\n"
        """
        def add_icl_prompt(example):
            # example["messages"][1]["content"] = icl_prompt + "\n" + example["messages"][1]["content"] + "\n### Response:\n"
            example["messages"] = [example["messages"][0]] + chat_icl + example["messages"][1:]
            return example

        dataset = dataset.map(add_icl_prompt, num_proc=8)

    responses = []
    references = []
    with (Progress(
            TextColumn(
                f"Generating responses •" + "[progress.percentage]{task.percentage:>3.0f}%"
            ),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
    ) as p):
        for sample in p.track(dataset):
            tokenized_sample = tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt").to("cuda")
            outputs = model.generate(tokenized_sample, max_new_tokens=args.max_new_tokens)
            response_ids = outputs[0][tokenized_sample.shape[1]:-1]
            response = tokenizer.decode(response_ids)
            responses.append(response)
            if dataset_name != "apps":
                references.append(sample[reference_field])

            print(tokenizer.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=True))
            print(response)
            print()

    if args.peft_checkpoint_path is not None:
        output_dir = f"{args.peft_checkpoint_path}/results"
    else:
        model_name = args.model_name_or_path.split("/")[-1]
        output_dir = f"runs/{model_name}/results"
    os.makedirs(output_dir, exist_ok=True)

    if dataset_name == "apps":
        # @todo: fix pyext and Python 3.12 --> try downgrading
        
        responses = [[resp] for resp in responses]
        apps_metric = evaluate.load('codeparrot/apps_metric')
        metrics = apps_metric.compute(predictions=responses, level="all")
        print(f"APPs: {metrics}")
        
        metrics = None
    else:
        chrf = evaluate.load("chrf")
        em = evaluate.load("exact_match")
        results_em = em.compute(predictions=responses, references=references)
        references_chrf = [[ref] for ref in references]
        results_chrf = chrf.compute(predictions=responses, references=references_chrf)
        results_chrf2 = chrf.compute(predictions=responses, references=references_chrf, word_order=2)

        print(f"EM: {results_em}")
        print(f"chrF: {results_chrf}")
        print(f"chrF++: {results_chrf2}")

        metrics = {
            "em": results_em,
            "chrf": results_chrf,
            "chrf2": results_chrf2
        }

    file_suffix = dataset_name
    if args.use_icl:
        file_suffix = f"{dataset_name}_icl_n{args.icl_n_examples}_s{args.icl_seed}"

    pickle.dump(metrics, open(f"{output_dir}/metrics_{file_suffix}.pkl", "wb"))
    with open(f"{output_dir}/responses_{file_suffix}.jsonl", "w") as fout:
        for response in responses:
            json.dump({"response": response}, fout)
            fout.write("\n")
