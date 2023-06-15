import os.path

from peft import get_peft_model, TaskType, LoraConfig
from transformers import \
    AutoTokenizer, \
    default_data_collator, \
    TrainingArguments, \
    Trainer, \
    Seq2SeqTrainingArguments, \
    Seq2SeqTrainer, \
    TrainerCallback, \
    EarlyStoppingCallback

from utils import *


def load_model_and_tokenizer(args):
    peft_task_type = TaskType.SEQ_2_SEQ_LM if args.model_type == "encoder-decoder" else TaskType.CAUSAL_LM
    if args.training_method == "ft":
        model = GENERATION_MODEL_CLS[args.model_type].from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    elif args.training_method == "lora":
        model = GENERATION_MODEL_CLS[args.model_type].from_pretrained(args.model_name_or_path,
                                                                      trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        peft_config = LoraConfig(task_type=peft_task_type,
                                 r=args.lora_r,
                                 lora_alpha=args.lora_alpha,
                                 target_modules=LORA_TARGET_MODULES[args.model_name],
                                 lora_dropout=args.lora_dropout,
                                 bias=args.lora_bias)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    if getattr(tokenizer, "cls_token_id") is None:
        tokenizer.cls_token_id = tokenizer.bos_token_id
        model.config.cls_token_id = model.config.bos_token_id

    return model, tokenizer


class SaveModelCallback(TrainerCallback):
    def __init__(self, output_dir, tokenizer):
        self.output_dir = output_dir
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint_ep{int(state.epoch)}")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)


def train_devign_defect_detection(args):
    dataset = load_devign_defect_detection_dataset(args.dataset_dir)
    del dataset["test"]

    model, tokenizer = load_model_and_tokenizer(args)

    if args.model_type == "encoder":
        def preprocess_function(examples):
            model_inputs = tokenizer(examples["func"],
                                     truncation=True,
                                     padding="max_length",
                                     max_length=args.defect_max_seq_length)
            model_inputs["labels"] = examples["target"]
            return model_inputs

        dataset = dataset.map(preprocess_function,
                              batched=True,
                              num_proc=args.num_workers,
                              remove_columns=dataset["train"].column_names,
                              desc="Generating samples features.")
    else:
        def preprocess_function_dec(example):
            suffix = tokenizer(" Label : ")
            label = tokenizer(example["text_label"])
            max_input_len = args.defect_max_seq_length - len(suffix.input_ids) - len(label.input_ids) - 1
            # perform truncation only on the code to avoid truncating the suffix and label
            model_inputs = tokenizer(example["func"],
                                     truncation=True,
                                     max_length=max_input_len)
            model_inputs["labels"] = [-100] * (len(model_inputs["input_ids"]) + len(suffix.input_ids)) \
                                     + label.input_ids + [-100]
            model_inputs["input_ids"] = model_inputs["input_ids"] + suffix.input_ids + label.input_ids \
                                        + [tokenizer.eos_token_id]
            model_inputs["attention_mask"] = [1] * len(model_inputs["input_ids"])

            # left-padding
            padding_len = args.defect_max_seq_length - len(model_inputs["input_ids"])
            model_inputs["attention_mask"] = [0] * padding_len + model_inputs["attention_mask"]
            model_inputs["labels"] = [-100] * padding_len + model_inputs["labels"]
            model_inputs["input_ids"] = [tokenizer.pad_token_id] * padding_len + model_inputs["input_ids"]

            return model_inputs

        def preprocess_function_encdec(examples):
            model_inputs = tokenizer(examples["func"],
                                     truncation=True,
                                     padding="max_length",
                                     max_length=args.defect_max_seq_length)
            labels = tokenizer(examples["text_label"],
                               truncation=True,
                               padding="max_length",
                               max_length=target_max_length)
            labels = labels["input_ids"]
            labels[labels == tokenizer.pad_token_id] = -100
            model_inputs["labels"] = labels
            return model_inputs

        preprocess_function = preprocess_function_dec if args.model_type == "decoder" else preprocess_function_encdec
        # transform the labels into textual labels for generation
        id2label = {0: "No", 1: "Yes"}
        target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in id2label.values()])
        dataset = dataset.map(lambda x: {"text_label": [id2label[label] for label in x["target"]]},
                              batched=True,
                              num_proc=args.num_workers)
        dataset = dataset.map(preprocess_function,
                              num_proc=args.num_workers,
                              remove_columns=dataset["train"].column_names,
                              desc="Generating samples features.")

    training_args_cls = Seq2SeqTrainingArguments if args.model_type == "encoder-decoder" else TrainingArguments
    trainer_cls = Seq2SeqTrainer if args.model_type == "encoder-decoder" else Trainer
    training_args = training_args_cls(
        output_dir=args.run_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.max_num_epochs,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb" if args.use_wandb else "none"
    )
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[SaveModelCallback(args.run_dir, tokenizer),
                   EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )
    trainer.train()


def train_xlcost_code_translation(args):
    dataset = load_xlcost_code_translation_dataset(args.dataset_dir, train_samples_percentage=0.25)
    del dataset["test"]

    model, tokenizer = load_model_and_tokenizer(args)
    def preprocess_function_dec(example):
        # we inform the model what are the input and target languages
        #   by defining a prefix as follow: # {input_lang} -> {target_lang}
        example_input = f"{example['input_lang']} -> {example['target_lang']} : {example['input']}"
        model_inputs = tokenizer(example_input,
                                 truncation=True,
                                 max_length=args.translation_max_input_length - 1)
        tokenized_target = tokenizer(example["target"],
                                     truncation=True,
                                     max_length=args.translation_max_target_length - 1)
        model_inputs["input_ids"] = model_inputs["input_ids"] + [tokenizer.eos_token_id]
        tokenized_target["input_ids"] = tokenized_target["input_ids"] + [tokenizer.eos_token_id]

        padding_len = (args.translation_max_input_length + args.translation_max_target_length) - \
                      (len(model_inputs["input_ids"]) + len(tokenized_target["input_ids"]))

        model_inputs["attention_mask"] = [0] * padding_len + model_inputs["attention_mask"] + [1]
        model_inputs["input_ids"] = [tokenizer.pad_token_id] * padding_len + model_inputs["input_ids"]
        tokenized_target["attention_mask"] = tokenized_target["attention_mask"] + [1]

        model_inputs["labels"] = [-100] * len(model_inputs["input_ids"]) + tokenized_target["input_ids"]
        model_inputs["input_ids"] = model_inputs["input_ids"] + tokenized_target["input_ids"]
        model_inputs["attention_mask"] = model_inputs["attention_mask"] + tokenized_target["attention_mask"]

        return model_inputs

    def preprocess_function_encdec(example):
        example_input = f"{example['input_lang']} -> {example['target_lang']} : {example['input']}"
        model_inputs = tokenizer(example_input,
                                 truncation=True,
                                 padding="max_length",
                                 max_length=args.translation_max_input_length)
        labels = tokenizer(example["target"],
                           truncation=True,
                           padding="max_length",
                           max_length=args.translation_max_target_length)
        labels = labels.input_ids
        labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]
        model_inputs["labels"] = labels
        return model_inputs

    preprocess_function = preprocess_function_dec if args.model_type == "decoder" else preprocess_function_encdec
    dataset = dataset.map(preprocess_function,
                          num_proc=args.num_workers,
                          remove_columns=dataset["train"].column_names,
                          desc="Generating samples features.")

    training_args_cls = Seq2SeqTrainingArguments if args.model_type == "encoder-decoder" else TrainingArguments
    trainer_cls = Seq2SeqTrainer if args.model_type == "encoder-decoder" else Trainer
    training_args = training_args_cls(
        output_dir=args.run_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.max_num_epochs,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb" if args.use_wandb else "none"
    )
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[SaveModelCallback(args.run_dir, tokenizer),
                   EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )
    trainer.train()


def train_code_generation(args):
    if args.task == "xlcost_code_generation":
        dataset = load_xlcost_code_generation_dataset(args.dataset_dir, train_samples_percentage=0.50)
    else:
        dataset = load_concode_code_generation_dataset(args.dataset_dir, train_samples_percentage=0.25)
    del dataset["test"]

    model, tokenizer = load_model_and_tokenizer(args)
    def preprocess_function_dec(example):
        suffix = tokenizer(f" Code ({example['target_lang']}) : ")
        max_input_len = args.codegen_max_input_length - len(suffix.input_ids) - 1
        # perform truncation only on the code to avoid truncating the suffix
        model_inputs = tokenizer(example["input"],
                                 truncation=True,
                                 max_length=max_input_len)
        tokenized_target = tokenizer(example["target"],
                                     truncation=True,
                                     max_length=args.codegen_max_target_length - 1)
        model_inputs["input_ids"] = model_inputs["input_ids"] + suffix.input_ids + [tokenizer.eos_token_id]
        tokenized_target["input_ids"] = tokenized_target["input_ids"] + [tokenizer.eos_token_id]

        padding_len = (args.codegen_max_input_length + args.codegen_max_target_length) - \
                      (len(model_inputs["input_ids"]) + len(tokenized_target["input_ids"]))

        model_inputs["input_ids"] = [tokenizer.pad_token_id] * padding_len + model_inputs["input_ids"]
        model_inputs["attention_mask"] = [0] * padding_len + model_inputs["attention_mask"] \
                                         + suffix.attention_mask + [1]
        tokenized_target["attention_mask"] = tokenized_target["attention_mask"] + [1]

        model_inputs["labels"] = [-100] * len(model_inputs["input_ids"]) + tokenized_target["input_ids"]
        model_inputs["input_ids"] = model_inputs["input_ids"] + tokenized_target["input_ids"]
        model_inputs["attention_mask"] = model_inputs["attention_mask"] + tokenized_target["attention_mask"]

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
        model_inputs["attention_mask"] = [1] * len(model_inputs["input_ids"]) + [0] * padding_len
        model_inputs["input_ids"] = model_inputs["input_ids"] + [tokenizer.pad_token_id] * padding_len

        labels = tokenizer(example["target"],
                           truncation=True,
                           padding="max_length",
                           max_length=args.codegen_max_target_length,
                           add_special_tokens=True)

        labels = labels.input_ids
        labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]
        model_inputs["labels"] = labels
        return model_inputs

    preprocess_function = preprocess_function_dec if args.model_type == "decoder" else preprocess_function_encdec
    dataset = dataset.map(preprocess_function,
                          num_proc=args.num_workers,
                          remove_columns=dataset["train"].column_names,
                          desc="Generating samples features.")

    training_args_cls = Seq2SeqTrainingArguments if args.model_type == "encoder-decoder" else TrainingArguments
    trainer_cls = Seq2SeqTrainer if args.model_type == "encoder-decoder" else Trainer
    training_args = training_args_cls(
        output_dir=args.run_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.max_num_epochs,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb" if args.use_wandb else "none"
    )
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[SaveModelCallback(args.run_dir, tokenizer),
                   EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )
    trainer.train()
