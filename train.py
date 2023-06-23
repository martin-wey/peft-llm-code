import torch
from peft import get_peft_model, TaskType, LoraConfig
from transformers import \
    AutoTokenizer, \
    default_data_collator, \
    TrainingArguments, \
    Trainer, \
    Seq2SeqTrainingArguments, \
    Seq2SeqTrainer, \
    EarlyStoppingCallback

from utils import *


def load_model_and_tokenizer(args):
    peft_task_type = TaskType.SEQ_2_SEQ_LM if args.model_type == "encoder-decoder" else TaskType.CAUSAL_LM
    if args.training_method == "ft":
        model = GENERATION_MODEL_CLS[args.model_type].from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    else:
        model = GENERATION_MODEL_CLS[args.model_type].from_pretrained(args.model_name_or_path,
                                                                      torch_dtype=torch.float16,
                                                                      trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if args.training_method == "lora":
            peft_config = LoraConfig(task_type=peft_task_type,
                                     r=args.lora_r,
                                     lora_alpha=args.lora_alpha,
                                     target_modules=LORA_TARGET_MODULES[args.model_name],
                                     lora_dropout=args.lora_dropout,
                                     bias="none")
        model = get_peft_model(model, peft_config)
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


def train_conala_code_generation(args):
    dataset = load_conala_dataset(args)
    del dataset["test"]

    model, tokenizer = load_model_and_tokenizer(args)

    def preprocess_function_dec(example):
        """
        # we tokenize, pad and truncate the samples in the following way:
        #   <pad><pad>...<intent + \n><snippet><eos>
        #
        #   - prompt tokens `<pad><pad>...<intent + \n>` are ignored in the computation of the loss (-100 labels)
        #   - `<eos>` delimits the snippet and allows the model to have more focused predictions at inference
        """
        tokenized_target = tokenizer(example["snippet"],
                                     truncation=True,
                                     max_length=args.conala_max_target_length - 1,
                                     # incoder adds eos token before the start of a sequence -> ignore
                                     add_special_tokens=False)
        tokenized_target["input_ids"] = tokenized_target["input_ids"] + [tokenizer.eos_token_id]
        tokenized_target["attention_mask"] = tokenized_target["attention_mask"] + [1]

        max_prompt_len = (args.conala_max_input_length + args.conala_max_target_length) - \
                         len(tokenized_target["input_ids"])
        model_inputs = tokenizer(example["rewritten_intent"] + "\n",
                                 truncation=True,
                                 padding="max_length",
                                 max_length=max_prompt_len)

        model_inputs["labels"] = [-100] * len(model_inputs["input_ids"]) + tokenized_target["input_ids"]
        model_inputs["input_ids"] = model_inputs["input_ids"] + tokenized_target["input_ids"]
        model_inputs["attention_mask"] = model_inputs["attention_mask"] + tokenized_target["attention_mask"]

        return model_inputs

    def preprocess_function_encdec(example):
        model_inputs = tokenizer(example["rewritten_intent"] + "\n",
                                 truncation=True,
                                 padding="max_length",
                                 max_length=args.conala_max_input_length,
                                 add_special_tokens=True)
        tokenized_target = tokenizer(example["snippet"],
                                     truncation=True,
                                     padding="max_length",
                                     max_length=args.conala_max_target_length,
                                     add_special_tokens=True)
        labels = tokenized_target["input_ids"]
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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.num_warmup_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_strategy="steps",
        logging_steps=20,
        save_total_limit=2,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )
    trainer.train()
    # save best model after training
    trainer.model.save_pretrained(f"{args.run_dir}/best_model_checkpoint")
    trainer.tokenizer.save_pretrained(f"{args.run_dir}/best_model_checkpoint")
