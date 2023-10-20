import logging

import torch
from peft import get_peft_model, TaskType, LoraConfig, PromptTuningConfig, PeftModel
from transformers import \
    AutoModelForCausalLM, \
    AutoTokenizer, \
    default_data_collator, \
    TrainingArguments, \
    Trainer, \
    EarlyStoppingCallback

from utils import *


logger = logging.getLogger(__name__)


def load_model_and_tokenizer(args):
    if args.training_method == "ft":
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                     torch_dtype=torch.float16,
                                                     trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.adapter_path is not None:
        # continue fine-tuning an existing PEFT checkpoint
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True).to(args.device)
        model.print_trainable_parameters()
    else:
        if args.training_method == "lora":
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                     r=args.lora_r,
                                     lora_alpha=args.lora_alpha,
                                     target_modules=LORA_TARGET_MODULES[args.model_name],
                                     lora_dropout=args.lora_dropout,
                                     bias="none")
        elif args.training_method == "prompt-tuning":
            peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM,
                                             prompt_tuning_init="TEXT",
                                             prompt_tuning_init_text="Generate one line of Python code given an "
                                                                     "instruction",
                                             num_virtual_tokens=args.num_virtual_tokens,
                                             tokenizer_name_or_path=args.model_name_or_path)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    if "incoder" in args.model_name:
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 1

    tokenizer.padding_side = "left"

    return model, tokenizer


def run_train(args):
    dataset_loading_func = globals().get(f"load_{args.dataset}_train_dataset")
    dataset = dataset_loading_func()
    code_column = "cmd"
    intent_column = "nl"
    print(dataset["train"], dataset["validation"])

    for example in dataset["train"].select(range(3)):
        print(f"### Instruction:\n{example[intent_column]}\n### Response:\n{example[code_column]}")
        print("=" * 100)

    model, tokenizer = load_model_and_tokenizer(args)

    def preprocess_function(example):
        """
        # we tokenize, pad and truncate the samples in the following way:
        #   <pad><pad>...### Instruction:\n<intent>\n### Answer:\n<snippet><eos>
        #
        #   - prompt tokens `<pad><pad>...<intent + \n>` are ignored in the computation of the loss (-100 labels)
        #   - `<eos>` delimits the snippet and allows the model to have more focused predictions at inference
        """
        tokenized_target = tokenizer(example[code_column],
                                     truncation=True,
                                     max_length=args.max_target_length - 1,
                                     # incoder adds eos token before the start of a sequence -> ignore
                                     add_special_tokens=False)
        tokenized_target["input_ids"] = tokenized_target["input_ids"] + [tokenizer.eos_token_id]
        tokenized_target["attention_mask"] = tokenized_target["attention_mask"] + [1]

        prompt = "### Instruction:\n" + example[intent_column] + "\n### Response:\n"
        max_prompt_len = (args.max_input_length + args.max_target_length) - \
                         len(tokenized_target["input_ids"])
        model_inputs = tokenizer(prompt,
                                 truncation=True,
                                 padding="max_length",
                                 max_length=max_prompt_len)

        model_inputs["labels"] = [-100] * len(model_inputs["input_ids"]) + tokenized_target["input_ids"]
        model_inputs["input_ids"] = model_inputs["input_ids"] + tokenized_target["input_ids"]
        model_inputs["attention_mask"] = model_inputs["attention_mask"] + tokenized_target["attention_mask"]

        return model_inputs

    dataset = dataset.map(preprocess_function,
                          num_proc=args.num_workers,
                          remove_columns=dataset["train"].column_names,
                          desc="Generating samples features.")

    training_args = TrainingArguments(
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
        report_to=["wandb"] if args.use_wandb else ["none"]
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation loss before training: {round(eval_results['eval_loss'], 4)}")
    trainer.train()
    # save best model after training
    trainer.model.save_pretrained(f"{args.run_dir}/best_model_checkpoint")
    trainer.tokenizer.save_pretrained(f"{args.run_dir}/best_model_checkpoint")
