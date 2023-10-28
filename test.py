import logging
import os
import random
import re
from collections import defaultdict

import evaluate
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import \
    AutoModelForCausalLM, \
    T5ForConditionalGeneration, \
    AutoTokenizer, \
    default_data_collator, \
    StoppingCriteriaList, \
    StoppingCriteria

from utils import *

logger = logging.getLogger(__name__)
EOF_STRINGS_CONALA = ["<|endoftext|>", "</s>", "\n"]
EOF_STRINGS_CODEALPACA = ["<|endoftext|>", "</s>"]


def load_model_and_tokenizer(args):
    model_cls = T5ForConditionalGeneration if "codet5" in args.model_name_or_path else AutoModelForCausalLM
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": True
    }
    if args.training_method != "ft":
        model_kwargs["torch_dtype"] = torch.float16

    model = model_cls.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.config.use_cache = True
    if args.training_method != "ft":
        model = PeftModel.from_pretrained(model, args.adapter_path).to(args.device)
        model.print_trainable_parameters()
    else:
        model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

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


def run_test(args):
    dataset_loading_func = globals().get(f"load_{args.dataset}_test_dataset")
    test_dataset = dataset_loading_func()
    intent_column = "nl"
    code_column = "cmd"

    model, tokenizer = load_model_and_tokenizer(args)

    if args.num_icl_examples >= 0:
        # zero-shot learning
        icl_prompt = "Generate Python code given a natural language instruction."
        if args.num_icl_examples > 0:
            train_loading_func = globals().get(f"load_{args.dataset}_train_dataset")
            train_dataset = train_loading_func()["train"]
            random_indices = random.sample(range(len(train_dataset)), args.num_icl_examples)
            icl_examples = train_dataset.select(random_indices)
            for n in icl_examples:
                icl_prompt += f"\n### Instruction:\n{n[intent_column]}\
                                \n### Response:\n{n[code_column]}\n"
        print(icl_prompt)

    def preprocess_function(example):
        prompt = "\n### Instruction:\n" + example[intent_column] + "\n### Response:\n"
        if args.num_icl_examples >= 0:
            prompt = icl_prompt + prompt
        # no need to pad/truncate, we do not do batched generation
        if "codet5" in args.model_name_or_path:
            prompt += "<extra_id_0>"
        model_inputs = tokenizer(prompt)

        labels = tokenizer(example[code_column])["input_ids"]
        model_inputs["labels"] = labels

        return model_inputs

    test_dataset = test_dataset.map(preprocess_function,
                                    num_proc=args.num_workers,
                                    remove_columns=[cname for cname in test_dataset.column_names if
                                                    cname not in ["input_ids", "labels"]],
                                    desc="Generating samples features.")
    dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=default_data_collator, pin_memory=True)

    eof_string = EOF_STRINGS_CONALA if args.dataset == "conala" else EOF_STRINGS_CODEALPACA
    predictions = [[] for _ in range(len(test_dataset))]
    references = []
    for step, sample in tqdm(enumerate(dataloader), total=len(test_dataset)):
        with torch.no_grad():
            generated_sequences = model.generate(
                input_ids=sample["input_ids"].to(args.device),
                num_beams=10,
                num_return_sequences=10,
                max_new_tokens=args.max_target_length,
                stopping_criteria=StoppingCriteriaList(
                    [EndOfFunctionCriteria(sample["input_ids"].shape[1], eof_string, tokenizer)]
                )
            )
            generated_sequences = generated_sequences.cpu().numpy()
            if "codet5" not in args.model_name_or_path:
                generated_sequences = generated_sequences[:, sample["input_ids"].shape[1]:]

            for task, new_tokens in zip([step] * args.num_return_sequences, generated_sequences):
                new_tokens_decoded = tokenizer.decode(new_tokens,
                                                      skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=True)
                new_tokens_decoded = re.split("(%s)" % "|".join(eof_string), new_tokens_decoded.strip())[0]
                new_tokens_decoded = new_tokens_decoded.replace("\n", " ").replace("\t", " ")
                predictions[task].append(new_tokens_decoded)

            reference_decoded = tokenizer.decode(sample["labels"][0],
                                                 skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=True)
            reference_decoded = reference_decoded.replace("\n", " ").replace("\t", " ")
            references.append(reference_decoded)

    # export top-10 predictions
    jsonl_data = []
    for preds, refs in zip(predictions, references):
        jsonl_data.append({
            "predictions": preds,
            "references": refs
        })
    logger.info(f"Exporting test predictions in directory {args.run_dir}.")
    base_fname = f"output_{args.dataset}"
    if args.num_icl_examples > -1:
        base_fname += f"_{args.num_icl_examples}shot"
    with open(os.path.join(args.run_dir, f"{base_fname}.jsonl"), "w", encoding="utf-8") as fout:
        for entry in jsonl_data:
            json.dump(entry, fout)
            fout.write("\n")

    # export top-1 predictions for CodeBLEU
    predictions = [p[0] for p in predictions]
    base_pred_fname = f"predictions_{args.dataset}"
    base_ref_fname = f"references_{args.dataset}"
    if args.num_icl_examples > -1:
        base_pred_fname += f"_{args.num_icl_examples}shot"
        base_ref_fname += f"_{args.num_icl_examples}shot"
    with open(os.path.join(args.run_dir, f"{base_pred_fname}.txt"), "w", encoding="utf-8") as fpred, \
            open(os.path.join(args.run_dir, f"{base_ref_fname}.txt"), "w", encoding="utf-8") as fref:
        for prediction, reference in zip(predictions, references):
            fpred.write(prediction + "\n")
            fref.write(reference + "\n")

'''
def test_pass_at_k(args):
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    dataset = load_odex_test_dataset()
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
        prompt = "\n### Instruction:\n" + example["intent"] + "\n### Response:\n"
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
'''