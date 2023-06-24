import json
import logging
import re
from collections import defaultdict

import evaluate
import torch

from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import \
    AutoTokenizer, \
    default_data_collator, \
    StoppingCriteriaList, \
    StoppingCriteria

from utils import *

logger = logging.getLogger(__name__)

EOF_STRINGS = ["<|endoftext|>", "</s>", "\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n"]
HUMAN_EVAL_EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]


def load_model_and_tokenizer(args):
    if args.training_method == "ft":
        kwargs = {}
        if args.fp16:
            kwargs["torch_dtype"] = torch.float16
        model = GENERATION_MODEL_CLS[args.model_type].from_pretrained(args.model_name_or_path, **kwargs).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    else:
        inference_model = GENERATION_MODEL_CLS[args.model_type].from_pretrained(args.model_name_or_path,
                                                                                torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = PeftModel.from_pretrained(inference_model, args.lora_adapter_path).to(args.device)
        model.print_trainable_parameters()

    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    if "incoder" in args.model_name:
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 1

    if args.model_type == "decoder":
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


def test_conala_code_generation(args):
    dataset = load_conala_dataset()
    test_dataset = dataset["test"]

    model, tokenizer = load_model_and_tokenizer(args)

    if args.num_few_shot_examples > 0:
        examples = read_conala_few_shot_examples(args)
        few_shot_prompt = "Answer the following instruction in one line of Python code:"
        for n in range(1, args.num_few_shot_examples + 1):
            few_shot_prompt += f"\n### Instruction:\n{examples[f'instruction{n}']}\
                                 \n### Answer:\n{examples[f'solution{n}']}\n"

    def preprocess_function_dec(example):
        prompt = "### Instruction:\n" + example["nl"] + "\n### Answer:\n"
        if args.num_few_shot_examples > 0:
            prompt = few_shot_prompt + prompt
        model_inputs = tokenizer(prompt,
                                 truncation=True,
                                 padding="max_length",
                                 max_length=args.conala_max_input_length)
        labels = tokenizer(example["cmd"],
                           truncation=True,
                           padding="max_length",
                           max_length=args.conala_max_target_length)["input_ids"]
        model_inputs["labels"] = labels

        return model_inputs

    def preprocess_function_encdec(example):
        prompt = "### Instruction:\n" + example["nl"] + "\n### Answer:\n"
        model_inputs = tokenizer(prompt,
                                 truncation=True,
                                 padding="max_length",
                                 max_length=args.conala_max_input_length,
                                 add_special_tokens=True)
        labels = tokenizer(example["cmd"],
                           truncation=True,
                           padding="max_length",
                           max_length=args.conala_max_target_length,
                           add_special_tokens=True)["input_ids"]
        model_inputs["labels"] = labels

        return model_inputs

    preprocess_function = preprocess_function_dec if args.model_type == "decoder" else preprocess_function_encdec
    test_dataset = test_dataset.map(preprocess_function,
                                    num_proc=args.num_workers,
                                    remove_columns=[cname for cname in test_dataset.column_names if
                                                    cname not in ["input_ids", "attention_mask", "labels"]],
                                    desc="Generating samples features.")

    print(tokenizer.decode(test_dataset[0]["input_ids"], skip_special_tokens=True))

    dataloader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            collate_fn=default_data_collator,
                            pin_memory=True)

    predictions = []
    references = []
    for batch in tqdm(dataloader, total=len(test_dataset) // args.batch_size):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            batch_generation = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                temperature=args.temperature,
                num_beams=args.beam_size,
                max_new_tokens=args.conala_max_target_length,
                stopping_criteria=StoppingCriteriaList(
                    [EndOfFunctionCriteria(batch["input_ids"].shape[1], EOF_STRINGS, tokenizer)]
                )
            )
            if args.model_type == "decoder":
                batch_generated_tokens = tokenizer.batch_decode(batch_generation[:, batch["input_ids"].shape[1]:],
                                                                skip_special_tokens=True)
            else:
                batch_generated_tokens = tokenizer.batch_decode(batch_generation, skip_special_tokens=True)
            batch_references = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

            print("*" * 100)
            print(batch_generated_tokens[0])
            print("-" * 100)
            print(batch_references[0])
            print("*" * 100)

            if "incoder" in args.model_name:
                # somehow the pad tokens do not get filtered when decoding with InCoder
                batch_references = [ref.replace("<pad>", "") for ref in batch_references]
            predictions += [generated_tokens for generated_tokens in batch_generated_tokens]
            references += [tokens for tokens in batch_references]

    logger.info(f"Exporting test predictions in directory {args.output_dir}.")
    pred_fname = "predictions.txt"
    ref_fname = "references.txt"
    if args.num_few_shot_examples > 0:
        pred_fname = f"predictions_{args.num_few_shot_examples}shot.txt"
        ref_fname = f"references_{args.num_few_shot_examples}shot.txt"
    with open(os.path.join(args.output_dir, pred_fname), "w", encoding="utf-8") as fpred, \
            open(os.path.join(args.output_dir, ref_fname), "w", encoding="utf-8") as fref:
        for prediction, reference, dataset in zip(predictions, references, test_dataset):
            fpred.write(prediction.replace("\n", " ") + "\n")
            fref.write(reference.replace("\n", " ") + "\n")


def test_human_eval(args):
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    human_eval = load_dataset("openai_humaneval")
    code_eval_metric = evaluate.load("code_eval")

    model, tokenizer = load_model_and_tokenizer(args)
    if args.model_type == "decoder":
        tokenizer.padding_side = "left"

    test_dataset = human_eval.map(lambda e: tokenizer(e["prompt"].strip()),
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
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.human_eval_max_new_tokens,
                num_return_sequences=args.human_eval_num_sequences,
                stopping_criteria=StoppingCriteriaList(
                    [EndOfFunctionCriteria(sample["input_ids"].shape[1], HUMAN_EVAL_EOF_STRINGS, tokenizer)]
                )
            )
            generated_sequences = generated_sequences.cpu().numpy()
            func_signatures = generated_sequences[:, :sample["input_ids"].shape[1]]
            new_tokens = generated_sequences[:, sample["input_ids"].shape[1]:]

            for task, func_signatures, new_tokens in zip([step] * args.human_eval_num_sequences,
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
        references=references, predictions=code_gens, num_workers=args.num_workers, k=[1, 10, 100]
    )
    print(f"Results: {pass_at_k}")

    with open(f"{args.output_dir}/human_eval.json", "w") as fp:
        json.dump(pass_at_k, fp)
