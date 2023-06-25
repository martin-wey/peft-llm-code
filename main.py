import argparse
import logging
from pathlib import Path

import wandb
from transformers import set_seed

import train
import test

logger = logging.getLogger(__name__)


def main(args):
    if args.do_train:
        logger.info(f"[Fine-tuning] Model: {args.model_name_or_path} | Task: {args.task}.")
        train_func = getattr(train, f"train_{args.task}")
        train_func(args)

    if args.do_test:
        logger.info(f"[Test] Model: {args.model_name_or_path} | Task: {args.task}.")
        test_func = getattr(test, f"test_{args.task}")
        test_func(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="NinedayWang/PolyCoder-160M", type=str,
                        help="Name of the pretrained model on Huggingface Hub or in local storage.")
    parser.add_argument("--model_type", default="encoder", type=str, help="Model architecture type.")
    parser.add_argument("--dataset_dir", default="./datasets", type=str, help="Dataset base directory.")
    parser.add_argument("--output_dir", default="./runs", type=str, help="Output directory.")
    parser.add_argument("--run_name", default=None, type=str)

    parser.add_argument("--task", default="conala_code_generation", type=str,
                        help="Task on which to fine-tune the model.")
    parser.add_argument("--training_method", default="ft", type=str, help="Method used to fine-tuning the model.")

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--patience", type=int, default=2)

    parser.add_argument("--learning_rate", type=float,  default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--fp16", type=bool, default=False)

    parser.add_argument("--conala_max_input_length", default=64, type=int)
    parser.add_argument("--conala_max_target_length", default=64, type=int)

    parser.add_argument("--human_eval_max_new_tokens", default=256, type=int)
    parser.add_argument("--human_eval_num_sequences", default=10, type=int)

    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--beam_size", default=4, type=int)

    parser.add_argument("--lora_adapter_path", default=None, type=str)
    parser.add_argument("--lora_r", default=32, type=int)
    parser.add_argument("--lora_alpha", default=64, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)

    parser.add_argument("--num_few_shot_examples", default=-1, type=int)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project_name", default="peft-code", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    # Setup logging and output directories
    args.model_name = args.model_name_or_path.split('/')[-1]
    if args.do_train:
        if args.run_name is None:
            args.run_name = f"{args.model_name_or_path.split('/')[-1]}_{args.training_method}"
        args.run_dir = Path(f"{args.output_dir}/{args.task}/{args.run_name}")
        args.run_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    if args.use_wandb:
        wandb.init(project=args.wandb_project_name, name=f"{args.task}/{args.run_name}", config=vars(args), mode="offline")

    main(args)
