# base script: https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py

import logging

from trl.commands.cli_utils import init_zero_verbose, TrlParser

init_zero_verbose()
FORMAT = "%(message)s"

from rich.console import Console
from rich.logging import RichHandler

from datasets import load_dataset, load_from_disk

from tqdm.rich import tqdm
from transformers import AutoTokenizer

from trl import (
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    DataCollatorForCompletionOnlyLM,
    get_quantization_config,
    get_kbit_device_map,
)

from utils import SFTScriptArguments, ModelConfig, get_peft_config

tqdm.pandas()

logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)

if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    training_args.disable_tqdm = True
    console = Console()

    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    if "meta-llama" in model_config.model_name_or_path:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    raw_datasets = load_from_disk(args.dataset_name)
    train_dataset = raw_datasets[args.dataset_train_split]
    eval_dataset = raw_datasets[args.dataset_test_split]

    if args.completion_only:
        # ensures the instruction is ignored during loss computation
        response_template = args.response_template
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    console.print(model_config)

    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        peft_config=get_peft_config(model_config, tokenizer),
        callbacks=[RichProgressCallback]
    )

    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    console.print(f"Number of trainable parameters: {trainable_params:,}")

    """
    # check data preprocessing
    train_dataloader = trainer.get_train_dataloader()

    for i, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        print(input_ids)
        print(labels)
        print(tokenizer.decode(input_ids))
        break
    """

    trainer.train()
