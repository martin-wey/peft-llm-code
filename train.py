import logging
import math
import random

import wandb
from transformers import (
    default_data_collator,
    TrainingArguments,
    Trainer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback
)

from data_utils import *
from utils import load_model_and_tokenizer, DataCollatorForCompletionOnlyLM

logger = logging.getLogger(__name__)


class SaveBestModelCallback(TrainerCallback):
    def __init__(self, trainer, eval_steps):
        self.trainer = trainer
        self.eval_steps = eval_steps
        self.best_loss = float("inf")
        self.saved_models_dir = []

    def on_step_end(self, args, state, control, model, tokenizer, **kwargs):
        if state.global_step % self.eval_steps == 0:
            evaluation_results = self.trainer.evaluate()
            eval_loss = evaluation_results["eval_loss"]

            if eval_loss < self.best_loss:
                model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                self.best_loss = eval_loss

                if "wandb" in args.report_to:
                    wandb.run.summary["best_evaluation_loss"] = eval_loss


def run_train(args):
    dataset_loading_func = globals().get(f"load_{args.dataset}_train_dataset")
    dataset = dataset_loading_func()

    instruction_column = "nl"
    response_column = "cmd"

    if args.dataset == "apps":
        instruction_column = "question"
        response_column = "solutions"

    model, tokenizer = load_model_and_tokenizer(args, is_training=True)

    def preprocess_function(examples):
        instructions = examples[instruction_column]
        responses = examples[response_column]

        if args.dataset == "apps":
            responses = [random.choice(json.loads(solutions)) for solutions in responses]
            guides = examples["guide"]
            prompt_template = (
                "### Instruction:\nWrite a python code to solve the following coding problem that obeys the constraints "
                "and passes the example test cases. The output code needs to {guide}:\n"
                "{instruction}\n### Response:\n```python\n{response}\n```"
            )
            prompts = [
                prompt_template.format(guide=guide, instruction=instruction, response=response.strip())
                for guide, instruction, response in zip(guides, instructions, responses)
            ]
        else:
            prompt_template = "### Instruction:\n{instruction}\n### Response:\n```python\n{response}\n```"

            prompts = [
                prompt_template.format(instruction=instruction, response=response.strip())
                for instruction, response in zip(instructions, responses)
            ]

        model_inputs = tokenizer(
            prompts,
            truncation=True,
            max_length=args.max_seq_length,
        )

        return model_inputs

    def preprocess_function_seq2seq(example):
        prompt = "Generate Python code: ### Instruction:\n" + example[instruction_column] + "\n### Response:\n"
        model_inputs = tokenizer(prompt, max_length=args.max_input_length, padding="max_length", truncation=True)
        labels = tokenizer(example[response_column], max_length=args.max_target_length, padding="max_length", truncation=True)

        labels["input_ids"] = [l if l != tokenizer.pad_token_id else -100 for l in labels["input_ids"]]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    if "codet5" in args.model_name_or_path:
        tokenize_fn = preprocess_function_seq2seq
        trainer_cls = Seq2SeqTrainer
        training_args_cls = Seq2SeqTrainingArguments
        collator = default_data_collator
    else:
        tokenize_fn = preprocess_function
        trainer_cls = Trainer
        training_args_cls = TrainingArguments
        # ignore instruction tokens
        response_template_with_context = "\n```python"
        response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
        collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=dataset["train"].column_names,
        desc="Generating samples features."
    )

    logger.info(f"Training/validation size before filtering lengthy samples: {len(dataset['train'])} / {len(dataset['validation'])}")
    dataset = dataset.filter(lambda ex: len(ex["input_ids"]) <= 1024, num_proc=args.num_workers)
    logger.info(f"Training/validation size after filtering: {len(dataset['train'])} / {len(dataset['validation'])}")

    n_samples = len(dataset["train"])
    n_samples_per_step = args.batch_size * args.num_gpus * args.gradient_accumulation_steps
    eval_steps = math.ceil((n_samples // n_samples_per_step) * args.ratio_samples_per_eval_step)

    training_args = training_args_cls(
        output_dir=args.run_dir,
        eval_strategy="no",
        save_strategy="no",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=0.05,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        optim="adafactor",
        logging_strategy="steps",
        logging_steps=10,
        bf16=True,
        report_to=["wandb"] if args.use_wandb else ["none"]
    )
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.add_callback(SaveBestModelCallback(trainer, eval_steps))

    eval_results = trainer.evaluate()
    logger.info(f"Evaluation loss before training: {round(eval_results['eval_loss'], 4)}")

    trainer.train()
