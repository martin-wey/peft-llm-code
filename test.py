import logging

import torch
from torch.utils.data import DataLoader
from transformers import \
    AutoModelForSequenceClassification, \
    AutoModelForCausalLM, \
    AutoTokenizer, \
    default_data_collator, \
    StoppingCriteriaList, \
    StoppingCriteria

from utils import *

logger = logging.getLogger(__name__)


EOF_STRINGS = ["<|endoftext|>", "</s>"]
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
    def convert_samples_to_features(examples):
        tokenized_inputs = tokenizer(examples["func"],
                                     truncation=True,
                                     padding="max_length",
                                     max_length=args.defect_max_seq_length)
        tokenized_inputs["labels"] = examples["target"]
        return tokenized_inputs

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    dataset = load_devign_defect_detection_dataset(args.dataset_dir)
    test_dataset = dataset["test"]
    test_dataset = test_dataset.map(convert_samples_to_features,
                                    batched=True,
                                    num_proc=args.num_workers,
                                    remove_columns=[cname for cname in test_dataset.column_names if
                                                    cname not in ["input_ids", "labels", "attention_mask"]],
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


def test_xlcost_code_translation(args):
    def convert_samples_to_features(examples):
        examples_inputs = [f"# {input_lang} -> {target_lang} : {input}"
                           for (input_lang, target_lang, input) in zip(examples["input_lang"],
                                                                       examples["target_lang"],
                                                                       examples["input"])]
        tokenized_inputs = tokenizer(examples_inputs,
                                     truncation=True,
                                     max_length=args.translation_max_input_length - 2,
                                     add_special_tokens=False)
        input_ids = [[tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
                     for input_ids in tokenized_inputs.input_ids]
        attention_mask = [[1] + mask + [1] for mask in tokenized_inputs.attention_mask]

        # pad the sequences dynamically to the length of the longest sequence in the batch
        max_len = max(len(ids) for ids in input_ids)
        padded_input_ids = torch.tensor([ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids in input_ids])
        padded_attention_mask = torch.tensor([mask + [0] * (max_len - len(mask)) for mask in attention_mask])

        return {"input_ids": padded_input_ids, "attention_mask": padded_attention_mask}

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    dataset = load_xlcost_code_translation_dataset(args.dataset_dir)
    test_dataset = dataset["test"]
    test_dataset = test_dataset.map(convert_samples_to_features,
                                    batched=True,
                                    batch_size=args.val_batch_size,
                                    num_proc=args.num_workers,
                                    remove_columns=[cname for cname in test_dataset.column_names if
                                                    cname not in ["input_ids", "attention_mask",
                                                                  "target", "input_lang", "target_lang"]],
                                    desc="Generating samples features.").select(range(32))
    dataloader = DataLoader(test_dataset,
                            batch_size=args.val_batch_size,
                            collate_fn=default_data_collator,
                            pin_memory=True)

    predictions = []
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
            batch_generated_tokens = tokenizer.batch_decode(batch_generation[:, batch["input_ids"].shape[1]:],
                                                            skip_special_tokens=True)
            predictions += [generated_tokens for generated_tokens in batch_generated_tokens]
    logger.info(f"Exporting test predictions in directory {args.run_dir}.")
    with open(os.path.join(args.run_dir, f"predictions.txt"), "w") as fpred, \
            open(os.path.join(args.run_dir, f"references.txt"), "w") as fref:
        for prediction, reference in zip(predictions, test_dataset):
            fpred.write(
                reference["input_lang"] + ";" + reference["target_lang"] + " | " +
                prediction.replace("\n", "") + "\n")
            fref.write(
                reference["input_lang"] + ";" + reference["target_lang"] + " | " +
                reference["target"].replace("\n", "") + "\n")
