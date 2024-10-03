import logging

from rich.console import Console
from rich.logging import RichHandler
from tqdm.rich import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
from trl import (
    SFTTrainer,
    SFTConfig,
    get_quantization_config,
    get_kbit_device_map,
    RichProgressCallback,
    DataCollatorForCompletionOnlyLM
)
from trl.commands.cli_utils import init_zero_verbose, TrlParser

from datasets import load_from_disk
from utils import (
    SFTScriptArguments,
    ModelConfig,
    get_peft_config
)

init_zero_verbose()
tqdm.pandas()
logging.basicConfig(format="%(message)s", datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    training_args.disable_tqdm = True
    console = Console()

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    dataset = load_from_disk(args.dataset_name)
    train_dataset = dataset[args.dataset_train_split]
    eval_dataset = dataset[args.dataset_test_split]

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    if args.completion_only:
        # ensures the instruction is ignored during loss computation
        collator = DataCollatorForCompletionOnlyLM(args.response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        peft_config=get_peft_config(model_config, tokenizer),
        callbacks=[RichProgressCallback()]
    )

    """
    train_dataloader = trainer.get_train_dataloader()

    for i, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"][3]
        labels = batch["labels"][3]
        print(input_ids)
        print(labels)
        print(batch["attention_mask"][3])
        print(tokenizer.decode(input_ids))
        break

    """
    trainer.train()

    console.log(model_config)
    trainable_params, all_param = trainer.model.get_nb_trainable_parameters()
    console.log(f"trainable params: {trainable_params:,d} || "
                f"all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}")
