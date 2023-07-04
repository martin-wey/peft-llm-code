import json
import logging
import os
import re
from collections import defaultdict

import evaluate
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import \
    AutoModelForCausalLM, \
    AutoTokenizer, \
    default_data_collator, \
    StoppingCriteriaList, \
    StoppingCriteria

from utils import *

logger = logging.getLogger(__name__)

EOF_STRINGS = ["<|endoftext|>", "</s>", "\n"]
HUMAN_EVAL_EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n#"]


def load_model_and_tokenizer(args):
    if args.training_method == "ft":
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    else:
        inference_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                               torch_dtype=torch.float16,
                                                               trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = PeftModel.from_pretrained(inference_model, args.adapter_path).to(args.device)
        model.print_trainable_parameters()

    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    if "incoder" in args.model_name:
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 1

    tokenizer.padding_side = "left"

    return model, tokenizer


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length:])
        done = []
        for decoded_generation in decoded_generations:
            done.append(any([stop_string in decoded_generation for stop_string in self.eof_strings]))
        return all(done)


def test_code_generation(args):
    test_dataset = load_test_dataset()

    model, tokenizer = load_model_and_tokenizer(args)

    if args.num_few_shot_examples > -1:
        # zero-shot learning
        few_shot_prompt = "Generate one line of Python code given an instruction."
        if args.num_few_shot_examples > 0:
            # in-context learning
            examples = read_icl_examples()
            for n in range(1, args.num_few_shot_examples + 1):
                few_shot_prompt += f"\n### Instruction:\n{examples[f'instruction{n}']}\
                                     \n### Answer:\n{examples[f'solution{n}']}\n"

    def preprocess_function(example):
        prompt = "### Instruction:\n" + example["intent"] + "\n### Answer:\n"
        if args.num_few_shot_examples >= 0:
            prompt = few_shot_prompt + prompt
        # no need to pad/truncate, we do not do batched generation
        model_inputs = tokenizer(prompt)
        # we also tokenize the solution as each LLM has a specific tokenization process
        labels = tokenizer(example["canonical_solution"])["input_ids"]
        model_inputs["labels"] = labels

        return model_inputs

    test_dataset = test_dataset.map(preprocess_function,
                                    num_proc=args.num_workers,
                                    remove_columns=[cname for cname in test_dataset.column_names if
                                                    cname not in ["input_ids", "labels"]],
                                    desc="Generating samples features.")
    dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=default_data_collator, pin_memory=True)

    gen_kwargs = {
        "do_sample": True,
        "temperature": 0.8,
        "max_new_tokens": args.max_target_length,
        "num_return_sequences": args.num_return_sequences,
        "top_p": 0.95,
        "top_k": 0
    }

    predictions = [[] for _ in range(len(test_dataset))]
    references = []
    for step, sample in tqdm(enumerate(dataloader), total=len(test_dataset)):
        with torch.no_grad():
            generated_sequences = model.generate(
                input_ids=sample["input_ids"].to(args.device),
                stopping_criteria=StoppingCriteriaList(
                    [EndOfFunctionCriteria(sample["input_ids"].shape[1], EOF_STRINGS, tokenizer)]
                ),
                **gen_kwargs
            )
            generated_sequences = generated_sequences.cpu().numpy()
            generated_new_tokens = generated_sequences[:, sample["input_ids"].shape[1]:]

            for task, new_tokens in zip([step] * args.num_return_sequences, generated_new_tokens):
                new_tokens_decoded = tokenizer.decode(new_tokens, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=True)
                new_tokens_decoded = re.split("(%s)" % "|".join(EOF_STRINGS), new_tokens_decoded.strip())[0]
                predictions[task].append(new_tokens_decoded)

            reference_decoded = tokenizer.decode(sample["labels"][0], skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=True)
            references.append(reference_decoded)

    jsonl_data = []
    for preds, refs in zip(predictions, references):
        jsonl_data.append({
            "predictions": preds,
            "references": refs
        })

    logger.info(f"Exporting test predictions in directory {args.output_dir}.")
    fname = "output.jsonl"
    if args.num_few_shot_examples > -1:
        fname = f"output_{args.num_few_shot_examples}shot.jsonl"
    with open(os.path.join(args.output_dir, fname), "w", encoding="utf-8") as fout:
        for entry in jsonl_data:
            json.dump(entry, fout)
            fout.write("\n")


def test_pass_at_k(args):
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    dataset = load_test_dataset()
    code_eval_metric = evaluate.load("code_eval")

    model, tokenizer = load_model_and_tokenizer(args)

    if args.num_few_shot_examples > -1:
        # zero-shot learning
        few_shot_prompt = "Generate one line of Python code given an instruction."
        if args.num_few_shot_examples > 0:
            # few-shot learning
            examples = read_icl_examples()
            for n in range(1, args.num_few_shot_examples + 1):
                few_shot_prompt += f"\n### Instruction:\n{examples[f'instruction{n}']}\
                                     \n### Answer:\n{examples[f'solution{n}']}\n"

    def preprocess_function(example):
        prompt = "\n### Instruction:\n" + example["intent"] + "\n### Answer:\n"
        if args.num_few_shot_examples >= 0:
            prompt = few_shot_prompt + prompt
        model_inputs = tokenizer(prompt)
        return model_inputs

    test_dataset = dataset.map(preprocess_function,
                               num_proc=args.num_workers,
                               remove_columns=dataset.column_names,
                               desc="Generating samples features.")
    dataloader = DataLoader(test_dataset,
                            batch_size=1,
                            collate_fn=default_data_collator,
                            pin_memory=True)

    gen_token_dict = defaultdict(list)
    for step, sample in tqdm(enumerate(dataloader), total=len(test_dataset)):
        with torch.no_grad():
            generated_sequences = model.generate(
                input_ids=sample["input_ids"].to(args.device),
                num_beams=args.num_beams,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.conala_max_target_length,
                num_return_sequences=args.num_return_sequences,
                stopping_criteria=StoppingCriteriaList(
                    [EndOfFunctionCriteria(sample["input_ids"].shape[1], EOF_STRINGS, tokenizer)]
                )
            )
            generated_sequences = generated_sequences.cpu().numpy()
            new_tokens = generated_sequences[:, sample["input_ids"].shape[1]:]

            for task, new_tokens in zip([step] * args.num_return_sequences, new_tokens):
                gen_token_dict[task].append(new_tokens)

    code_gens = [[] for _ in range(len(test_dataset))]
    for (task, generations), sample in zip(gen_token_dict.items(), dataset):
        for gen_tokens in generations:
            gen_code = tokenizer.decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            gen_code = re.split("(%s)" % "|".join(EOF_STRINGS), gen_code.strip())[0]
            func_def = sample["prompt"].split("\n")[0] + "\n\treturn "
            solution = f"{func_def}{gen_code}".replace("\t", " " * 4)
            code_gens[task].append(solution)

    references = []
    for task in tqdm(range(len(test_dataset))):
        test_case = dataset[task]["test"][0]
        entry_point = dataset[task]["entry_point"]
        check_function = '\n'.join([
            dataset[task]['test_start'],
            ''.join(test_case),
            '',
            f"check({entry_point})",
        ])
        references.append(check_function)

    pass_at_k, results = code_eval_metric.compute(
        references=references, predictions=code_gens, num_workers=args.num_workers, k=[1, 2, 5, 10]
    )
    print(f"Results: {pass_at_k}")

    fname = "odex_results.json"
    if args.num_few_shot_examples > -1:
        fname = f"odex_results_{args.num_few_shot_examples}shot.json"
    with open(f"{args.output_dir}/{fname}", "w") as fp:
        json.dump(pass_at_k, fp)


def test_human_eval(args):
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    human_eval = load_dataset("openai_humaneval")
    code_eval_metric = evaluate.load("code_eval")

    model, tokenizer = load_model_and_tokenizer(args)

    test_dataset = human_eval.map(lambda example: tokenizer(example["prompt"].strip()),
                                  num_proc=args.num_workers,
                                  desc="Generating samples features.")["test"]

    dataloader = DataLoader(test_dataset,
                            batch_size=1,
                            collate_fn=default_data_collator,
                            pin_memory=True)

    gen_token_dict = defaultdict(list)
    for step, sample in tqdm(enumerate(dataloader), total=len(test_dataset)):
        with torch.no_grad():
            generated_sequences = model.generate(
                input_ids=sample["input_ids"].to(args.device),
                num_beams=args.num_beams,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.human_eval_max_new_tokens,
                num_return_sequences=args.num_return_sequences,
                stopping_criteria=StoppingCriteriaList(
                    [EndOfFunctionCriteria(sample["input_ids"].shape[1], HUMAN_EVAL_EOF_STRINGS, tokenizer)]
                )
            )
            generated_sequences = generated_sequences.cpu().numpy()
            func_signatures = generated_sequences[:, :sample["input_ids"].shape[1]]
            new_tokens = generated_sequences[:, sample["input_ids"].shape[1]:]

            for task, func_signatures, new_tokens in zip([step] * args.num_return_sequences,
                                                         func_signatures,
                                                         new_tokens):
                gen_token_dict[task].append((func_signatures, new_tokens))

    code_gens = [[] for _ in range(len(test_dataset))]
    for task, generations in gen_token_dict.items():
        for (signature, gen_tokens) in generations:
            func_sign = tokenizer.decode(signature, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            gen_code = tokenizer.decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            gen_code = re.split("(%s)" % "|".join(HUMAN_EVAL_EOF_STRINGS), gen_code)[0]
            code_gens[task].append(func_sign + gen_code)

    references = []
    for task in tqdm(range(len(test_dataset))):
        test_func = human_eval["test"][task]["test"]
        entry_point = f"check({human_eval['test'][task]['entry_point']})"
        references.append("\n" + test_func + "\n" + entry_point)

    pass_at_k, _ = code_eval_metric.compute(
        references=references, predictions=code_gens, num_workers=args.num_workers, k=[1, 2, 5, 10]
    )
    print(f"Results: {pass_at_k}")

    with open(f"{args.output_dir}/human_eval_results.json", "w") as fp:
        json.dump(pass_at_k, fp)
