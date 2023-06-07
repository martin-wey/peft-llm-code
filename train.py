import evaluate
import numpy as np
from transformers import \
    AutoModelForSequenceClassification, \
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = dataset.map(convert_samples_to_features,
                          batched=True,
                          num_proc=args.num_workers,
                          remove_columns=[cname for cname in dataset["train"].column_names if
                                          cname not in ["input_ids", "labels", "attention_mask"]],
                          desc="Generating samples features.")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, return_dict=True)
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
    pass
