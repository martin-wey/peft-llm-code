import argparse
import codebleu
import csv
import glob
import json
import os
import subprocess
from tqdm import tqdm


def compute_exact_match(predictions, references, k):
    num_correct = 0
    total = len(predictions)

    for i in range(total):
        prediction = predictions[i][:k]
        reference = references[i]

        if reference in prediction:
            num_correct += 1

    accuracy = num_correct / total
    return accuracy


def get_em_metrics(predictions_fp, references_fp):
    with open(predictions_fp, "r") as f:
        predictions = [json.loads(l) for l in f]

    with open(references_fp, "r") as f:
        references = [json.loads(l)[0] for l in f]

    em_1 = round(compute_exact_match(predictions, references, 1) * 100, 3)
    em_2 = round(compute_exact_match(predictions, references, 2) * 100, 3)
    em_5 = round(compute_exact_match(predictions, references, 5) * 100, 3)
    em_10 = round(compute_exact_match(predictions, references, 10) * 100, 3)

    return em_1, em_2, em_5, em_10


def get_codebleu(predictions_fp, references_fp):
    with open(predictions_fp, "r") as f:
        predictions = [json.loads(l)[0] for l in f]

    with open(references_fp, "r") as f:
        references = [json.loads(l)[0] for l in f]

    result = codebleu.calc_codebleu(references, predictions, lang="python")
    return result["codebleu"]


if __name__ == "__main__":
    dataset = "conala"
    results_dir = "runs/test_results"
    references_file = os.path.join(results_dir, "conala_references.jsonl")

    for run_dir in tqdm(os.listdir(results_dir)):
        print(run_dir)
        if not os.path.isdir(f"{results_dir}/{run_dir}"):
            continue

        model_dir = os.path.join(results_dir, run_dir)
        for file in glob.glob(f"{model_dir}/predictions_{dataset}_icl_n*.jsonl"):
            n = int(file.split('_n')[1].split('.jsonl')[0])
            print(f"ICL - n={n}")
            print(get_em_metrics(file, references_file))
            codebleu_result = get_codebleu(file, references_file)
            print(f"CodeBLEU: {codebleu_result}")

        print("-" * 25)



"""
if __name__ == "__main__":
    methods = ["joint", "qlora-8bit", "qlora-4bit", "ft", "lora", "ia3", "prompt-tuning", "prefix-tuning"]
    datasets = ["conala", "codealpaca"]
    max_num_icl_examples = 3
    results_dir = "runs/test_results"

    with open(f"{results_dir}/data_metrics.csv", "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["model", "dataset", "method", "em_1", "em_2", "em_5", "em_10", "codebleu", "seed"])
        for run_dir in tqdm(os.listdir(results_dir)):
            if not os.path.isdir(f"{results_dir}/{run_dir}"):
                continue
            if "icl" in run_dir:
                for n in range(1, max_num_icl_examples + 1):
                    for dataset in datasets:
                        # EM metrics
                        output_fp = f"{results_dir}/{run_dir}/output_{dataset}_{n}shot.jsonl"
                        if not os.path.exists(output_fp):
                            continue
                        em_1, em_2, em_5, em_10 = get_em_metrics(output_fp)

                        # CodeBLEU
                        predictions_fp = f"../../{results_dir}/{run_dir}/predictions_{dataset}_{n}shot.txt"
                        references_fp = f"../../{results_dir}/{run_dir}/references_{dataset}_{n}shot.txt"
                        codebleu = get_codebleu(predictions_fp, references_fp)

                        run_dir_splitted = run_dir.split("_icl")
                        model = run_dir_splitted[0]
                        method = f"icl_{n}"
                        seed = int(run_dir_splitted[1].split("_seed")[1])
                        writer.writerow([model, dataset, method, em_1, em_2, em_5, em_10, codebleu, seed])
            else:
                for dataset in datasets:
                    output_fp = f"{results_dir}/{run_dir}/output_{dataset}.jsonl"
                    if not os.path.exists(output_fp):
                        continue
                    em_1, em_2, em_5, em_10 = get_em_metrics(output_fp)

                    predictions_fp = f"../../{results_dir}/{run_dir}/predictions_{dataset}.txt"
                    references_fp = f"../../{results_dir}/{run_dir}/references_{dataset}.txt"
                    codebleu = get_codebleu(predictions_fp, references_fp)

                    method = next((m for m in methods if m in run_dir), None)
                    model = run_dir.split(f"_{method}")[0]
                    seed = 42
                    writer.writerow([model, dataset, method, em_1, em_2, em_5, em_10, codebleu, seed])
"""