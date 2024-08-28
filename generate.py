import argparse
import json
import os

import evaluate
import torch
from peft import PeftModel
from rich.progress import MofNCompleteColumn, BarColumn, Progress, TextColumn, TimeElapsedColumn
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from datasets import load_from_disk
from utils import track_gpu_usage, MODELS_CHAT_USER


@track_gpu_usage
def generate(args, dataset, model, tokenizer):
    gen_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k
    }

    with (Progress(
            TextColumn(f"Generating responses •" + "[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
    ) as p):
        for sample in p.track(dataset):
            tokenized_sample = tokenizer.apply_chat_template(
                sample["messages"],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            ).to(model.device)

            outputs = model.generate(
                input_ids=tokenized_sample["input_ids"],
                attention_mask=tokenized_sample["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                **gen_kwargs
            )
            response_ids = outputs[0][tokenized_sample["input_ids"].shape[1]:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            print(response)

            # postprocess in case the model does not generate EOS token
            if args.model_name in MODELS_CHAT_USER:
                response_blocks = response.split(MODELS_CHAT_USER[args.model_name])
                response = response_blocks[0].strip()
            yield response


def compute_metrics(args, responses, dataset):
    if args.dataset_name == "apps":
        """
        @todo: fix pyext and Python 3.12 --> try downgrading to 3.11

        responses = [[resp] for resp in responses]
        apps_metric = evaluate.load('codeparrot/apps_metric')
        metrics = apps_metric.compute(predictions=responses, level="all")
        print(f"APPs: {metrics}")
        """
        return {}
    else:
        references = dataset[args.reference_field]
        chrf = evaluate.load("chrf")
        em = evaluate.load("exact_match")
        results_em = em.compute(predictions=responses, references=references)
        references_chrf = [[ref] for ref in references]
        results_chrf = chrf.compute(predictions=responses, references=references_chrf)
        results_chrf2 = chrf.compute(predictions=responses, references=references_chrf, word_order=2)

        print(f"EM: {results_em}")
        print(f"chrF: {results_chrf}")
        print(f"chrF++: {results_chrf2}")

        return {
            "em": results_em,
            "chrf": results_chrf,
            "chrf2": results_chrf2
        }


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    if args.peft_checkpoint_path is not None:
        model = PeftModel.from_pretrained(model, args.peft_checkpoint_path)
    args.model_name = args.model_name_or_path.split("/")[-1]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    dataset = load_from_disk(args.dataset_name_or_path)["test"]

    # remove ground truth from the prompt
    dataset = dataset.map(lambda e: {"messages": e["messages"][:-1]}, num_proc=8)
    args.dataset_name = args.dataset_name_or_path.split("/")[-1]

    if args.dataset_name == "conala":
        args.max_new_tokens = 128
        args.instruction_field = "nl"
        args.reference_field = "cmd"
    elif args.dataset_name == "codealpaca":
        args.max_new_tokens = 512
        args.instruction_field = "prompt"
        args.reference_field = "completion"
    else:
        args.max_new_tokens = 1024
        args.instruction_field = "question"
        args.reference_field = "solutions"

    if args.use_icl:
        train_set = load_from_disk(args.dataset_name_or_path)["train"].shuffle(args.icl_seed).select(
            range(args.num_icl_examples))
        chat_icl = []
        for example in train_set:
            if args.dataset_name == "apps":
                reference = json.loads(example[args.reference_field])[0]
            else:
                reference = example[args.reference_field]
            chat_exemple = [
                {"role": "user", "content": example[args.instruction_field]},
                {"role": "assistant", "content": reference},
            ]
            chat_icl += chat_exemple

        def add_icl_prompt(example):
            if "gemma" in args.model_name:
                example["messages"] = chat_icl + example["messages"][1:]
            else:
                example["messages"] = [example["messages"][0]] + chat_icl + example["messages"][1:]
            return example

        dataset = dataset.map(add_icl_prompt, num_proc=16)

    responses, init_gpu_memory, peak_gpu_memory, total_execution_time = generate(args, dataset, model, tokenizer)

    if args.peft_checkpoint_path is not None:
        output_dir = f"{args.peft_checkpoint_path}/results"
    else:
        output_dir = f"runs/{args.model_name}/results"
    os.makedirs(output_dir, exist_ok=True)

    metrics = compute_metrics(args, responses, dataset)
    metrics = {
        **metrics,
        "init_gpu_memory": f"{init_gpu_memory} MB",
        "peak_gpu_memory": f"{peak_gpu_memory} MB",
        "total_execution_time": f"{total_execution_time} seconds"
    }

    file_suffix = f"{args.dataset_name}_t{args.temperature}"
    if args.use_icl:
        file_suffix += f"_icl_n{args.num_icl_examples}_s{args.icl_seed}"

    with open(f"{output_dir}/metrics_{file_suffix}.jsonl", "w") as fout:
        json.dump(metrics, fout)

    with open(f"{output_dir}/responses_{file_suffix}.jsonl", "w") as fout:
        for response in responses:
            json.dump({"response": response}, fout)
            fout.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--peft_checkpoint_path", type=str, default=None)
    parser.add_argument("--dataset_name_or_path", type=str, default=None)

    parser.add_argument("--do_sample", default=True, type=bool, help="do sampling in generation")
    parser.add_argument("--temperature", default=0.2, type=float, help="temperature for sampling")
    parser.add_argument("--top_p", default=0.95, type=float, help="top p for sampling")
    parser.add_argument("--top_k", default=0, type=float, help="top k for sampling")

    parser.add_argument("--use_icl", action="store_true", default=False)
    parser.add_argument("--icl_seed", type=int, default=42)
    parser.add_argument("--num_icl_examples", type=int, default=3)

    args = parser.parse_args()
    set_seed(42)
    main(args)
