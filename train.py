import evaluate
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model
from transformers import \
    AutoModelForSequenceClassification, \
    AutoModelForCausalLM, \
    AutoTokenizer, \
    default_data_collator, \
    TrainingArguments, \
    Trainer

from utils import *


def train_devign_defect_detection(args):
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    def convert_samples_to_features(examples):
        tokenized_inputs = tokenizer(examples["func"],
                                     truncation=True,
                                     padding="max_length",
                                     max_length=args.defect_max_seq_length)
        tokenized_inputs["labels"] = examples["target"]
        return tokenized_inputs

    dataset = load_devign_defect_detection_dataset(args.dataset_dir)
    del dataset["test"]

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, return_dict=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    dataset = dataset.map(convert_samples_to_features,
                          batched=True,
                          num_proc=args.num_workers,
                          remove_columns=[cname for cname in dataset["train"].column_names if
                                          cname not in ["input_ids", "labels", "attention_mask"]],
                          desc="Generating samples features.")

    if args.training_method == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

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


def train_concode_code_generation(args):
    pass


def train_xlcost_code_generation(args):
    pass


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
