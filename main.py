import argparse
import logging
from pathlib import Path

from transformers import set_seed

from train import *
from utils import *

logger = logging.getLogger(__name__)


def main(args):
    if args.task == "xlcost_code-translation":
        train_dataset, val_dataset, test_dataset = load_xlcost_code_translation_dataset(args.dataset_dir)
    elif args.task == "xlcost_code-generation":
        train_dataset, val_dataset, test_dataset = load_xlcost_code_generation_dataset(args.dataset_dir)
    elif args.task == "concode_code-generation":
        train_dataset, val_dataset, test_dataset = load_concode_code_generation_dataset(args.dataset_dir)
    elif args.task == "devign_defect-detection":
        if args.do_train:
            logger.info(f"Running fine-tuning of {args.model_name_or_path} for defect detection.")
            train_devign_defect_detection(args)
        if args.do_test:
            pass
    else:
        raise ValueError("Wrong task argument name.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str,
                        help="Name of the pretrained model on Huggingface Hub or in local storage.")
    parser.add_argument("--model_type", default="roberta", type=str, help="Model architecture type.")
    parser.add_argument("--dataset_dir", default="./datasets", type=str, help="Dataset base directory.")
    parser.add_argument("--output_dir", default="./runs", type=str, help="Output directory.")

    parser.add_argument("--task", default="devign_defect-detection", type=str, help="Task on which to fine-tune the model.")
    parser.add_argument("--training_method", default="ft", type=str, help="Method used to fine-tuning the model.")
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--val_batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--num_epochs", default=5, type=float)

    parser.add_argument("--defect_max_seq_length", default=400, type=int)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    # Setup logging and output directories
    args.run_name = f"{args.model_name_or_path.split('/')[-1]}_{args.training_method}"
    args.run_dir = Path(f"{args.output_dir}/{args.task}/{args.run_name}")
    args.run_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(args.run_dir / "info.log"), logging.StreamHandler()],
    )

    main(args)
