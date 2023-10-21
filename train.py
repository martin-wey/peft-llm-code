import logging

import torch
from peft import (
    get_peft_model,
    TaskType,
    LoraConfig,
    IA3Config,
    PromptTuningConfig,
    PrefixTuningConfig,
    PeftModel
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    default_data_collator,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

from utils import *


logger = logging.getLogger(__name__)


def load_model_and_tokenizer(args):
    model_kwargs = {}
    if args.bit8_training:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif args.bit4_training:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16

    if args.training_method == "ft":
        del model_kwargs["torch_dtype"]

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
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
                                     target_modules=LORA_IA3_TARGET_MODULES[args.model_name]["target_modules"],
                                     lora_dropout=args.lora_dropout,
                                     bias="none",
                                     inference_mode=False)
        elif args.training_method == "ia3":
            peft_config = IA3Config(task_type=TaskType.CAUSAL_LM,
                                    target_modules=LORA_IA3_TARGET_MODULES[args.model_name]["target_modules"],
                                    feedforward_modules=LORA_IA3_TARGET_MODULES[args.model_name]["ff_modules"],
                                    inference_mode=False)
        elif args.training_method == "prompt-tuning":
            peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM,
                                             prompt_tuning_init="TEXT",
                                             prompt_tuning_init_text="Generate a Python code corresponding to the "
                                                                     "natural language instruction.",
                                             num_virtual_tokens=args.num_virtual_tokens,
                                             tokenizer_name_or_path=args.model_name_or_path,
                                             inference_mode=False)
        elif args.training_method == "prefix-tuning":
            peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM,
                                             num_virtual_tokens=args.num_virtual_tokens,
                                             inference_mode=False)
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
        optim="adamw_bnb_8bit" if args.bit8_training or args.bit4_training else "adamw_torch",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
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
