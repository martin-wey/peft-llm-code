import logging

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

EOF_STRINGS = ["<|endoftext|>", "</s>"]


def load_model_and_tokenizer(args):
    if args.training_method == "ft":
        model = GENERATION_MODEL_CLS[args.model_type].from_pretrained(args.model_name_or_path).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    elif args.training_method == "lora":
        inference_model = GENERATION_MODEL_CLS[args.model_type].from_pretrained(args.model_name_or_path,
                                                                                torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = PeftModel.from_pretrained(inference_model, args.lora_adapter_path).to(args.device)
        model.eval()
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


def test_devign_defect_detection(args):
    dataset = load_devign_defect_detection_dataset(args.dataset_dir)
    test_dataset = dataset["test"]

    model, tokenizer = load_model_and_tokenizer(args)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    if args.model_type == "encoder":
        def preprocess_function(examples):
            model_inputs = tokenizer(examples["func"],
                                     truncation=True,
                                     padding="max_length",
                                     max_length=args.defect_max_seq_length)
            model_inputs["labels"] = examples["target"]
            return model_inputs

        test_dataset = test_dataset.map(preprocess_function,
                                        batched=True,
                                        num_proc=args.num_workers,
                                        remove_columns=test_dataset.column_names,
                                        desc="Generating samples features.")
        dataloader = DataLoader(test_dataset,
                                collate_fn=default_data_collator,
                                batch_size=args.val_batch_size,
                                pin_memory=True)
        total_correct = 0
        for batch in tqdm(dataloader, total=len(test_dataset) // args.val_batch_size):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch).logits
            predicted_labels = torch.argmax(outputs, dim=1)
            total_correct += (predicted_labels == batch["labels"]).sum().item()
        accuracy = total_correct / len(test_dataset)
        logger.info(f"Accuracy: {round(accuracy * 100, 2)}%")
    else:
        def preprocess_function_dec(example):
            suffix = tokenizer(" Label : ")
            label = tokenizer(example["text_label"])
            max_input_len = args.defect_max_seq_length - len(suffix.input_ids)
            # perform truncation only on the code to avoid truncating the suffix and label
            model_inputs = tokenizer(example["func"],
                                     truncation=True,
                                     max_length=max_input_len)
            model_inputs["input_ids"] = model_inputs["input_ids"] + suffix.input_ids
            model_inputs["labels"] = label.input_ids

            # left-padding
            padding_len = args.defect_max_seq_length - len(model_inputs["input_ids"])
            model_inputs["attention_mask"] = [1] * len(model_inputs["input_ids"])
            model_inputs["attention_mask"] = [0] * padding_len + model_inputs["attention_mask"]
            model_inputs["input_ids"] = [tokenizer.pad_token_id] * padding_len + model_inputs["input_ids"]

            return model_inputs

        def preprocess_function_encdec(examples):
            model_inputs = tokenizer(examples["func"],
                                     truncation=True,
                                     padding="max_length",
                                     max_length=args.defect_max_seq_length)
            labels = tokenizer(examples["text_label"]).input_ids
            model_inputs["labels"] = labels
            return model_inputs

        # transform the labels into textual labels for generation
        id2label = {0: "No", 1: "Yes"}
        target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in id2label.values()])
        test_dataset = test_dataset.map(lambda x: {"text_label": [id2label[label] for label in x["target"]]},
                                        batched=True,
                                        num_proc=args.num_workers)
        preprocess_function = preprocess_function_dec if args.model_type == "decoder" else preprocess_function_encdec
        test_dataset = test_dataset.map(preprocess_function,
                                        num_proc=args.num_workers,
                                        remove_columns=test_dataset.column_names,
                                        desc="Generating samples features.")
        dataloader = DataLoader(test_dataset,
                                collate_fn=default_data_collator,
                                batch_size=args.val_batch_size,
                                pin_memory=True)
        total_correct = 0
        for batch in tqdm(dataloader, total=len(test_dataset) // args.val_batch_size):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            with torch.no_grad():
                batch_generation = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=target_max_length,
                )
                if args.model_type == "decoder":
                    batch_predictions = tokenizer.batch_decode(batch_generation[:, batch["input_ids"].shape[1]:],
                                                               skip_special_tokens=True)
                else:
                    batch_predictions = tokenizer.batch_decode(batch_generation, skip_special_tokens=True)
                batch_references = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
                for pred, ref in zip(batch_predictions, batch_references):
                    if pred == ref:
                        total_correct += 1
        accuracy = total_correct / len(test_dataset)
        logger.info(f"Accuracy: {round(accuracy * 100, 2)}%")


def test_xlcost_code_translation(args):
    dataset = load_xlcost_code_translation_dataset(args.dataset_dir)
    test_dataset = dataset["test"]

    model, tokenizer = load_model_and_tokenizer(args)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    def preprocess_function_dec(examples):
        examples_inputs = [f"{input_lang} -> {target_lang} : {input}"
                           for (input_lang, target_lang, input) in zip(examples["input_lang"],
                                                                       examples["target_lang"],
                                                                       examples["input"])]
        model_inputs = tokenizer(examples_inputs,
                                 truncation=True,
                                 max_length=args.translation_max_input_length - 1)
        model_inputs["input_ids"] = [input_ids + [tokenizer.eos_token_id] for input_ids in model_inputs["input_ids"]]
        model_inputs["attention_mask"] = [mask + [1] for mask in model_inputs["attention_mask"]]

        model_inputs["input_ids"] = [[tokenizer.pad_token_id] * (args.translation_max_target_length - len(input_ids))
                                     + input_ids for input_ids in model_inputs["input_ids"]]
        model_inputs["attention_mask"] = [[0] * (args.translation_max_target_length - len(mask)) + mask
                                          for mask in model_inputs["attention_mask"]]
        labels = tokenizer(examples["target"],
                           truncation=True,
                           padding="max_length",
                           max_length=args.translation_max_target_length)
        model_inputs["labels"] = labels.input_ids

        return model_inputs

    def preprocess_function_encdec(examples):
        examples_inputs = [f"{input_lang} -> {target_lang} : {input}"
                           for (input_lang, target_lang, input) in zip(examples["input_lang"],
                                                                       examples["target_lang"],
                                                                       examples["input"])]
        model_inputs = tokenizer(examples_inputs,
                                 truncation=True,
                                 padding="max_length",
                                 max_length=args.translation_max_input_length)
        labels = tokenizer(examples["target"],
                           truncation=True,
                           padding="max_length",
                           max_length=args.translation_max_target_length)
        model_inputs["labels"] = labels.input_ids
        return model_inputs

    preprocess_function = preprocess_function_dec if args.model_type == "decoder" else preprocess_function_encdec
    test_dataset = test_dataset.map(preprocess_function,
                                    batched=True,
                                    batch_size=args.val_batch_size,
                                    num_proc=args.num_workers,
                                    remove_columns=[cname for cname in test_dataset.column_names if
                                                    cname not in ["input_ids", "attention_mask", "labels",
                                                                  "input_lang", "target_lang"]],
                                    desc="Generating samples features.")
    dataloader = DataLoader(test_dataset,
                            batch_size=args.val_batch_size,
                            collate_fn=default_data_collator,
                            pin_memory=True)

    predictions = []
    references = []
    for batch in tqdm(dataloader, total=len(test_dataset) // args.val_batch_size):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            batch_generation = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=args.translation_max_target_length,
                use_cache=True,
                do_sample=args.do_sample,
                temperature=args.temperature,
                num_beams=args.beam_size,
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
            predictions += [generated_tokens for generated_tokens in batch_generated_tokens]
            references += [tokens for tokens in batch_references]

    logger.info(f"Exporting test predictions in directory {args.run_dir}.")
    with open(os.path.join(args.run_dir, f"predictions.txt"), "w", encoding="utf-8") as fpred, \
            open(os.path.join(args.run_dir, f"references.txt"), "w", encoding="utf-8") as fref:
        for prediction, reference, dataset in zip(predictions, references, test_dataset):
            fpred.write(
                dataset["input_lang"] + ";" + dataset["target_lang"] + " | " +
                prediction.replace("\n", "") + "\n")
            fref.write(
                dataset["input_lang"] + ";" + dataset["target_lang"] + " | " +
                reference.replace("\n", "") + "\n")


def test_code_generation(args):
    if args.task == "xlcost_code_generation":
        dataset = load_xlcost_code_generation_dataset(args.dataset_dir)
    else:
        dataset = load_concode_code_generation_dataset(args.dataset_dir)
    test_dataset = dataset["test"]

    model, tokenizer = load_model_and_tokenizer(args)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    def preprocess_function_dec(example):
        suffix = tokenizer(f" Code ({example['target_lang']}) : ")
        max_input_len = args.codegen_max_input_length - len(suffix.input_ids) - 1
        model_inputs = tokenizer(example["input"],
                                 truncation=True,
                                 max_length=max_input_len)
        model_inputs["input_ids"] = model_inputs["input_ids"] + suffix.input_ids + [tokenizer.eos_token_id]

        padding_len = args.codegen_max_input_length - len(model_inputs["input_ids"])
        model_inputs["input_ids"] = [tokenizer.pad_token_id] * padding_len + model_inputs["input_ids"]
        model_inputs["attention_mask"] = [0] * padding_len + model_inputs["attention_mask"] \
                                         + suffix.attention_mask + [1]

        labels = tokenizer(example["target"],
                           truncation=True,
                           padding="max_length",
                           max_length=args.codegen_max_target_length)
        model_inputs["labels"] = labels.input_ids

        return model_inputs

    def preprocess_function_encdec(example):
        suffix = tokenizer(f" Code ({example['target_lang']}) : ", add_special_tokens=False)
        model_inputs = tokenizer(example["input"],
                                 truncation=True,
                                 max_length=args.codegen_max_input_length - len(suffix.input_ids) - 2,
                                 add_special_tokens=False)
        model_inputs["input_ids"] = [tokenizer.cls_token_id] + \
                                    model_inputs["input_ids"] + suffix.input_ids + \
                                    [tokenizer.eos_token_id]

        padding_len = args.codegen_max_input_length - len(model_inputs["input_ids"])
        model_inputs["attention_mask"] = [1] * len(model_inputs["input_ids"]) + [tokenizer.pad_token_id] * padding_len
        model_inputs["input_ids"] = model_inputs["input_ids"] + [tokenizer.pad_token_id] * padding_len

        labels = tokenizer(example["target"],
                           truncation=True,
                           padding="max_length",
                           max_length=args.codegen_max_target_length)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    preprocess_function = preprocess_function_dec if args.model_type == "decoder" else preprocess_function_encdec
    test_dataset = test_dataset.map(preprocess_function,
                                    num_proc=args.num_workers,
                                    remove_columns=[cname for cname in test_dataset.column_names if
                                                    cname not in ["input_ids", "attention_mask", "labels",
                                                                  "target_lang"]],
                                    desc="Generating samples features.")
    dataloader = DataLoader(test_dataset,
                            batch_size=args.val_batch_size,
                            collate_fn=default_data_collator,
                            pin_memory=True)

    predictions = []
    references = []
    for batch in tqdm(dataloader, total=len(test_dataset) // args.val_batch_size):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            batch_generation = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=args.codegen_max_target_length,
                use_cache=True,
                do_sample=args.do_sample,
                temperature=args.temperature,
                num_beams=args.beam_size,
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
            predictions += [generated_tokens for generated_tokens in batch_generated_tokens]
            references += [tokens for tokens in batch_references]

    logger.info(f"Exporting test predictions in directory {args.run_dir}.")
    with open(os.path.join(args.run_dir, f"predictions.txt"), "w", encoding="utf-8") as fpred, \
            open(os.path.join(args.run_dir, f"references.txt"), "w", encoding="utf-8") as fref:
        for prediction, reference, dataset in zip(predictions, references, test_dataset):
            fpred.write("text;" + dataset["target_lang"] + " | " + prediction.replace("\n", "") + "\n")
            fref.write("text;" + dataset["target_lang"] + " | " + reference.replace("\n", "") + "\n")
