import argparse
import logging
from pathlib import Path

import wandb
from transformers import set_seed

from test import *
from train import *

logger = logging.getLogger(__name__)


def main(args):
    if args.task == "xlcost_code_translation":
        if args.do_train:
            logger.info(f"Running fine-tuning of {args.model_name_or_path} for code translation.")
            train_xlcost_code_translation(args)
        if args.do_test:
            logger.info(f"Testing model {args.model_name_or_path} on code translation.")
            test_xlcost_code_translation(args)
    elif args.task == "xlcost_code_generation" or args.task == "concode_code_generation":
        if args.do_train:
            logger.info(f"Running fine-tuning of {args.model_name_or_path} for code generation ({args.task.split('_')[0]}).")
            train_code_generation(args)
        if args.do_test:
            logger.info(f"Testing model {args.model_name_or_path} on code generation ({args.task.split('_')[0]}).")
            test_code_generation(args)
    elif args.task == "devign_defect_detection":
        if args.do_train:
            logger.info(f"Running fine-tuning of {args.model_name_or_path} for defect detection.")
            train_devign_defect_detection(args)
        if args.do_test:
            logger.info(f"Testing model {args.model_name_or_path} on defect detection.")
            test_devign_defect_detection(args)
    else:
        raise ValueError("Wrong task argument name.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str,
                        help="Name of the pretrained model on Huggingface Hub or in local storage.")
    parser.add_argument("--model_type", default="encoder", type=str, help="Model architecture type.")
    parser.add_argument("--dataset_dir", default="./datasets", type=str, help="Dataset base directory.")
    parser.add_argument("--output_dir", default="./runs", type=str, help="Output directory.")

    parser.add_argument("--task", default="devign_defect_detection", type=str,
                        help="Task on which to fine-tune the model.")
    parser.add_argument("--training_method", default="ft", type=str, help="Method used to fine-tuning the model.")
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--val_batch_size", default=16, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--max_num_epochs", default=10, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=0, type=int)
    parser.add_argument("--fp16", default=False, type=bool)
    parser.add_argument("--patience", default=2, type=int)

    parser.add_argument("--defect_max_seq_length", default=400, type=int)

    parser.add_argument("--translation_max_input_length", default=256, type=int)
    parser.add_argument("--translation_max_target_length", default=256, type=int)

    parser.add_argument("--codegen_max_input_length", default=256, type=int)
    parser.add_argument("--codegen_max_target_length", default=256, type=int)

    parser.add_argument("--do_sample", default=False, type=bool)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--beam_size", default=5, type=int)

    parser.add_argument("--lora_adapter_path", default=None, type=str)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_bias", default="none", type=str)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project_name", default="peft-code", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    # Setup logging and output directories
    args.model_name = args.model_name_or_path.split('/')[-1]
    if args.do_train:
        args.run_name = f"{args.model_name_or_path.split('/')[-1]}_{args.training_method}"
        args.run_dir = Path(f"{args.output_dir}/{args.task}/{args.run_name}")
        args.run_dir.mkdir(exist_ok=True)
    else:
        args.run_dir = args.model_name_or_path
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    if args.use_wandb:
        wandb.init(project=args.wandb_project_name, name=args.run_name, group=args.task, mode="offline")

    main(args)
