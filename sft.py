import logging

from datasets import load_from_disk

from peft import get_peft_model

from rich.console import Console
from rich.logging import RichHandler

from tqdm.rich import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer
)

from transformers.trainer_callback import PrinterCallback

from trl import (
    SFTConfig,
    DataCollatorForCompletionOnlyLM,
    get_quantization_config,
    get_kbit_device_map, RichProgressCallback
)
from trl.commands.cli_utils import init_zero_verbose, TrlParser

from utils import SFTScriptArguments, ModelConfig, get_peft_config

init_zero_verbose()
tqdm.pandas()
logging.basicConfig(format="%(message)s", datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


def load_model_and_tokenizer(model_config):
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
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    model.enable_input_require_grads()
    peft_config = get_peft_config(model_config, tokenizer)
    model = get_peft_model(model, peft_config) if peft_config is not None else model

    return model, tokenizer


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    training_args.disable_tqdm = True
    console = Console()

    model, tokenizer = load_model_and_tokenizer(model_config)

    def tokenize(examples):
        outputs = tokenizer.apply_chat_template(
            examples["messages"],
            truncation=True,
            max_length=training_args.max_seq_length,
            return_dict=True
        )
        return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

    dataset = load_from_disk(args.dataset_name)
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=[cn for cn in dataset["train"].column_names if cn not in ["input_ids", "attention_mask"]],
        num_proc=training_args.dataset_num_proc,
        batch_size=training_args.dataset_batch_size
    )

    train_dataset = tokenized_dataset[args.dataset_train_split]
    eval_dataset = tokenized_dataset[args.dataset_test_split]

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    if args.completion_only:
        # ensures the instruction is ignored during loss computation
        response_template = args.response_template
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[RichProgressCallback]
    )

    console.print(model_config)
    trainer.model.print_trainable_parameters()

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
    trainer.remove_callback(PrinterCallback)
    trainer.train()
