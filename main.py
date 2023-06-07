import argparse

from utils import *


def main(args):
    # train_dataset, val_dataset, test_dataset = load_xlcost_code_translation_dataset(args.dataset_dir)
    # train_dataset, val_dataset, test_dataset = load_xlcost_code_generation_dataset(args.dataset_dir)
    # train_dataset, val_dataset, test_dataset = load_concode_code_generation_dataset(args.dataset_dir)
    train_dataset, val_dataset, test_dataset = load_devign_defect_prediction_dataset(args.dataset_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="./datasets", type=str,
                        help="Dataset base directory.")

    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    main(args)
