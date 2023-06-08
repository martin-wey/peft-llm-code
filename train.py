import evaluate
import numpy as np
from transformers import \
    AutoModelForSequenceClassification, \
    AutoModelForCausalLM, \
    AutoModelForSeq2SeqLM, \
    AutoTokenizer, \
    default_data_collator, \
    TrainingArguments, \
    Trainer

from utils import *

DEFECT_MODEL_CLS = {
    "encoder": AutoModelForSequenceClassification,
    "decoder": AutoModelForCausalLM,
    "encoder-decoder": AutoModelForSeq2SeqLM
}

GENERATION_MODEL_CLS = {
    "encoder": AutoModelForSeq2SeqLM,
    "decoder": AutoModelForCausalLM,
    "encoder-decoder": AutoModelForSeq2SeqLM
}


def train_devign_defect_detection(args):
    dataset = load_devign_defect_detection_dataset(args.dataset_dir)
    del dataset["test"]

    model = DEFECT_MODEL_CLS[args.model_type].from_pretrained(args.model_name_or_path, return_dict=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    if args.model_type == "encoder":
        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return metric.compute(predictions=predictions, references=labels)

        def preprocess_function(examples):
            tokenized_inputs = tokenizer(examples["func"],
                                         truncation=True,
                                         padding="max_length",
                                         max_length=args.defect_max_seq_length)
            tokenized_inputs["labels"] = examples["target"]
            return tokenized_inputs

        dataset = dataset.map(preprocess_function,
                              num_proc=args.num_workers,
                              remove_columns=dataset["train"].column_names,
                              desc="Generating samples features.")
    elif args.model_type == "decoder":
        metric = evaluate.load("seqeval")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions[0], axis=2)

            true_predictions = [
                [tokenizer.decode(p).strip() for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [tokenizer.decode(l) for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = metric.compute(predictions=true_predictions, references=true_labels)
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

        def preprocess_function(examples):
            suffix = tokenizer(" Labels : ")
            labels = tokenizer(examples["text_label"])
            max_input_len = args.defect_max_seq_length - len(suffix.input_ids) - len(labels.input_ids)
            # perform truncation only on the code to avoid truncating the suffix and labels
            inputs = tokenizer(examples["func"],
                               truncation=True,
                               max_length=args.defect_max_seq_length - max_input_len)
            input_ids = inputs.input_ids + suffix.input_ids + labels.input_ids + [tokenizer.pad_token_id]
            attention_mask = [1] * len(input_ids)
            labels = [-100] * (len(inputs.input_ids) + len(suffix.input_ids)) + labels.input_ids + [-100]

            # padding
            attention_mask = [0] * (args.defect_max_seq_length - len(input_ids)) + attention_mask
            labels = [-100] * (args.defect_max_seq_length - len(input_ids)) + labels
            input_ids = [tokenizer.pad_token_id] * (args.defect_max_seq_length - len(input_ids)) + input_ids

            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        # transform the labels into textual labels
        id2label = {
            0: "No",
            1: "Yes"
        }
        dataset = dataset.map(lambda x: {"text_label": [id2label[label] for label in x["target"]]},
                              batched=True,
                              num_proc=args.num_workers)
        dataset = dataset.map(preprocess_function,
                              num_proc=args.num_workers,
                              remove_columns=dataset["train"].column_names,
                              desc="Generating samples features.")

    training_args = TrainingArguments(
        output_dir=args.run_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()


def train_xlcost_code_translation(args):
    def convert_samples_to_features(example):
        # This is the implementation for causal language modeling,
        # for conditional generation (i.e., using encoder-decoder), the feature extraction is different (@todo)

        # we inform the model what are the input and target languages
        #   by defining a prefix as follow: # {input_lang} -> {target_lang}
        example_input = f"# {example['input_lang']} -> {example['target_lang']} : {example['input']}"
        tokenized_input = tokenizer(example_input,
                                    truncation=True,
                                    max_length=args.translation_max_input_length - 2,
                                    add_special_tokens=False)
        tokenized_input_ids = [tokenizer.bos_token_id] + tokenized_input.input_ids + [tokenizer.eos_token_id]
        input_padding_len = (args.translation_max_input_length - len(tokenized_input_ids))
        input_attention_mask = [1] + tokenized_input.attention_mask + [1] + [0] * input_padding_len
        padded_input_ids = tokenized_input_ids + [tokenizer.pad_token_id] * input_padding_len

        tokenized_target = tokenizer(example["target"],
                                     truncation=True,
                                     max_length=args.translation_max_input_length - 1,
                                     add_special_tokens=False)
        tokenized_target_ids = tokenized_target.input_ids + [tokenizer.eos_token_id]
        target_padding_len = (args.translation_max_target_length - len(tokenized_target_ids))
        target_attention_mask = tokenized_target.attention_mask + [1] + [0] * target_padding_len
        padded_target_ids = tokenized_target_ids + [tokenizer.pad_token_id] * target_padding_len

        input_ids = padded_input_ids + padded_target_ids
        attention_mask = input_attention_mask + target_attention_mask
        labels = [-100] * len(padded_input_ids) + tokenized_target_ids + [-100] * target_padding_len

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    dataset = load_xlcost_code_translation_dataset(args.dataset_dir)
    del dataset["test"]
    dataset = dataset.map(convert_samples_to_features,
                          num_proc=args.num_workers,
                          remove_columns=[cname for cname in dataset["train"].column_names if
                                          cname not in ["input_ids", "labels", "attention_mask"]],
                          desc="Generating samples features.")

    training_args = TrainingArguments(
        output_dir=args.run_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=tokenizer,
        data_collator=default_data_collator
    )
    trainer.train()


def train_concode_code_generation(args):
    pass


def train_xlcost_code_generation(args):
    pass