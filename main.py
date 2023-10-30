import argparse
import logging
from pathlib import Path

import torch
import wandb
from transformers import set_seed

from train import run_train
from test import run_test

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="NinedayWang/PolyCoder-160M", type=str,
                        help="Name of the pretrained model on Huggingface Hub or in local storage.")
    parser.add_argument("--output_dir", default="./runs", type=str, help="Output directory.")
    parser.add_argument("--run_name", default=None, type=str)

    parser.add_argument("--dataset", default="conala", type=str,
                        help="Dataset on which to fine-tune the model.")
    parser.add_argument("--tuning_method", default="ft", type=str,
                        help="Method used to fine-tuning the model.")

    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--ratio_samples_per_eval_step", type=float, default=0.2,
                        help="The percentage of samples seen between each model evaluation step.")

    parser.add_argument("--learning_rate", type=float,  default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--weight_decay", type=float, default=0)

    parser.add_argument("--temperature", default=1, type=float)
    parser.add_argument("--num_beams", default=10, type=int)
    parser.add_argument("--num_return_sequences", type=int, default=10)
    parser.add_argument("--do_sample", action='store_true')

    parser.add_argument("--adapter_path", default=None, type=str)

    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)

    parser.add_argument("--prompt_num_virtual_tokens", default=20, type=int)
    parser.add_argument("--prefix_num_virtual_tokens", default=10, type=int)

    parser.add_argument("--num_icl_examples", default=-1, type=int)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project_name", default="peft-llm-code", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    args.num_gpus = torch.cuda.device_count()

    # Setup logging and output directories
    if args.adapter_path is not None:
        args.model_name = args.adapter_path.split('/')[-1]
    else:
        args.model_name = args.model_name_or_path.split('/')[-1]
    if args.run_name is None:
        if args.do_train:
            args.run_name = f"{args.dataset}/{args.model_name}_{args.tuning_method}"
        else:
            args.run_name = args.model_name
    run_intermediate_path = "checkpoints" if args.do_train else "test_results"
    args.run_dir = Path(f"{args.output_dir}/{run_intermediate_path}/{args.run_name}")
    args.run_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    if args.use_wandb:
        wandb.init(project=args.wandb_project_name,
                   name=args.run_name,
                   config=vars(args))

    if args.dataset == "conala":
        args.max_input_length = 64
        args.max_target_length = 64
    else:
        args.max_input_length = 64
        args.max_target_length = 128

    if args.do_train:
        logger.info(f"[Fine-tuning] Model: {args.model_name_or_path} | Dataset: {args.dataset}.")
        run_train(args)

    if args.do_test:
        logger.info(f"[Test] Model: {args.model_name_or_path} | Dataset: {args.dataset}.")
        run_test(args)
