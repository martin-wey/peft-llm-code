import os.path

from peft import get_peft_model, LoraConfig, TaskType
from transformers import \
    AutoTokenizer, \
    default_data_collator, \
    TrainingArguments, \
    Trainer, \
    Seq2SeqTrainingArguments, \
    Seq2SeqTrainer, \
    TrainerCallback

from utils import *


class SaveModelCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint_{state.global_step}")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)


def train_devign_defect_detection(args):
    dataset = load_devign_defect_detection_dataset(args.dataset_dir)
    del dataset["test"]

    model = DEFECT_MODEL_CLS[args.model_type].from_pretrained(args.model_name_or_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    if args.model_type == "encoder":
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
        def preprocess_function(example):
            suffix = tokenizer(" Labels : ")
            labels = tokenizer(example["text_label"])
            max_input_len = args.defect_max_seq_length - len(suffix.input_ids) - len(labels.input_ids) - 1
            # perform truncation only on the code to avoid truncating the suffix and labels
            input = tokenizer(example["func"],
                              truncation=True,
                              max_length=max_input_len)
            input_ids = input.input_ids + suffix.input_ids + labels.input_ids + [tokenizer.pad_token_id]
            attention_mask = [1] * len(input_ids)
            labels = [-100] * (len(input.input_ids) + len(suffix.input_ids)) + labels.input_ids + [-100]

            # padding
            attention_mask = [0] * (args.defect_max_seq_length - len(input_ids)) + attention_mask
            labels = [-100] * (args.defect_max_seq_length - len(input_ids)) + labels
            input_ids = [tokenizer.pad_token_id] * (args.defect_max_seq_length - len(input_ids)) + input_ids

            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        # transform the labels into textual labels for generation
        id2label = {0: "No", 1: "Yes"}
        dataset = dataset.map(lambda x: {"text_label": [id2label[label] for label in x["target"]]},
                              batched=True,
                              num_proc=args.num_workers)
        dataset = dataset.map(preprocess_function,
                              num_proc=args.num_workers,
                              remove_columns=dataset["train"].column_names,
                              desc="Generating samples features.")
    elif args.model_type == "encoder-decoder":
        def preprocess_function(examples):
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
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="no",
        report_to="wandb" if args.use_wandb else "none"
    )
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[SaveModelCallback(args.run_dir)]
    )
    trainer.train()


def train_xlcost_code_translation(args):
    dataset = load_xlcost_code_translation_dataset(args.dataset_dir, train_samples_percentage=0.25)
    del dataset["test"]

    model = GENERATION_MODEL_CLS[args.model_type].from_pretrained(args.model_name_or_path, device_map="auto")
    if args.training_method == "lora":
        task_type = TaskType.CAUSAL_LM if args.model_type == "decoder" else TaskType.SEQ_2_SEQ_LM
        peft_config = LoraConfig(task_type=task_type,
                                 inference_mode=False,
                                 r=16,
                                 lora_alpha=32,
                                 lora_dropout=0.0)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    def preprocess_function_dec(example):
        # we inform the model what are the input and target languages
        #   by defining a prefix as follow: # {input_lang} -> {target_lang}
        example_input = f"# {example['input_lang']} -> {example['target_lang']} : {example['input']}"
        tokenized_input = tokenizer(example_input,
                                    truncation=True,
                                    max_length=args.translation_max_input_length - 1,
                                    add_special_tokens=False)
        tokenized_target = tokenizer(example["target"],
                                     truncation=True,
                                     max_length=args.translation_max_target_length - 1,
                                     add_special_tokens=False)
        tokenized_input_ids = tokenized_input.input_ids + [tokenizer.eos_token_id]
        tokenized_target_ids = tokenized_target.input_ids + [tokenizer.eos_token_id]

        padding_len = (args.translation_max_input_length + args.translation_max_target_length) - \
                      (len(tokenized_input_ids) + len(tokenized_target_ids))

        input_attention_mask = [0] * padding_len + tokenized_input.attention_mask + [1]
        padded_input_ids = [tokenizer.pad_token_id] * padding_len + tokenized_input_ids
        target_attention_mask = tokenized_target.attention_mask + [1]

        input_ids = padded_input_ids + tokenized_target_ids
        attention_mask = input_attention_mask + target_attention_mask
        labels = [-100] * len(padded_input_ids) + tokenized_target_ids

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def preprocess_function_encdec(example):
        example_input = f"# {example['input_lang']} -> {example['target_lang']} : {example['input']}"
        model_inputs = tokenizer(example_input,
                                 truncation=True,
                                 padding="max_length",
                                 max_length=args.translation_max_input_length)
        labels = tokenizer(example["target"],
                           truncation=True,
                           padding="max_length",
                           max_length=args.translation_max_target_length)
        labels = labels["input_ids"]
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
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="no",
        report_to="wandb" if args.use_wandb else "none"
    )
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[SaveModelCallback(args.run_dir)]
    )
    trainer.train()


def train_xlcost_code_generation(args):
    dataset = load_xlcost_code_generation_dataset(args.dataset_dir, train_samples_percentage=0.50)
    del dataset["test"]

    model = GENERATION_MODEL_CLS[args.model_type].from_pretrained(args.model_name_or_path, device_map="auto")
    if args.training_method == "lora":
        task_type = TaskType.CAUSAL_LM if args.model_type == "decoder" else TaskType.SEQ_2_SEQ_LM
        peft_config = LoraConfig(task_type=task_type,
                                 inference_mode=False,
                                 r=8,
                                 lora_alpha=32,
                                 lora_dropout=0.1)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    def preprocess_function_dec(example):
        suffix = tokenizer(f" Code ({example['target_lang']}) : ")
        max_input_len = args.codegen_max_input_length - len(suffix.input_ids) - 1
        # perform truncation only on the code to avoid truncating the suffix
        tokenized_input = tokenizer(example["input"],
                                    truncation=True,
                                    max_length=max_input_len)
        tokenized_target = tokenizer(example["target"],
                                     truncation=True,
                                     max_length=args.codegen_max_target_length - 1)

        tokenized_input_ids = tokenized_input.input_ids + suffix.input_ids + [tokenizer.eos_token_id]
        tokenized_target_ids = tokenized_target.input_ids + [tokenizer.eos_token_id]

        padding_len = (args.codegen_max_input_length + args.codegen_max_target_length) - \
                      (len(tokenized_input_ids) + len(tokenized_target_ids))

        padded_input_ids = [tokenizer.pad_token_id] * padding_len + tokenized_input_ids
        input_attention_mask = [0] * padding_len + tokenized_input.attention_mask + suffix.attention_mask + [1]
        target_attention_mask = tokenized_target.attention_mask + [1]

        input_ids = padded_input_ids + tokenized_target_ids
        attention_mask = input_attention_mask + target_attention_mask
        labels = [-100] * len(padded_input_ids) + tokenized_target_ids

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def preprocess_function_encdec(example):
        examples_inputs = f"{example['input']} Code ({example['target_lang']}) : "
        model_inputs = tokenizer(examples_inputs,
                                 truncation=True,
                                 padding="max_length",
                                 max_length=args.codegen_max_input_length)
        labels = tokenizer(example["target"],
                           truncation=True,
                           padding="max_length",
                           max_length=args.codegen_max_target_length)
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
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
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="no",
        report_to="wandb" if args.use_wandb else "none"
    )
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[SaveModelCallback(args.run_dir)]
    )
    trainer.train()


def train_concode_code_generation(args):
    dataset = load_concode_code_generation_dataset(args.dataset_dir, train_samples_percentage=0.25)
    del dataset["test"]

    print(dataset["train"][1000])
