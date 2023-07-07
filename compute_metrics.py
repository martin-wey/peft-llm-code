import argparse
import json

from utils import load_conala_test_dataset


def compute_exact_match(predictions, references, k):
    num_correct = 0
    total = len(predictions)

    for i in range(total):
        prediction = predictions[i][:k]  # Get the top-k predictions
        reference = references[i]

        if reference in prediction:
            num_correct += 1

    accuracy = num_correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", "-ref", help="Jsonl file containing the predictions and references.")
    args = parser.parse_args()

    test_dataset = load_conala_test_dataset()

    predictions = []
    references = []
    with open(args.output_file, "r", encoding="utf-8") as file:
        for line in file:
            json_data = json.loads(line)
            predictions.append(json_data["predictions"])
            references.append(json_data["references"])

    em_at_1 = round(compute_exact_match(predictions, references, 1) * 100, 2)
    em_at_2 = round(compute_exact_match(predictions, references, 2) * 100, 2)
    em_at_5 = round(compute_exact_match(predictions, references, 5) * 100, 2)
    em_at_10 = round(compute_exact_match(predictions, references, 10) * 100, 2)

    print(f"EM@1: {em_at_1} | EM@2: {em_at_2} | EM@5: {em_at_5} | EM@10: {em_at_10}")

    lib_samples = {}
    for preds, refs, sample in zip(predictions, references, test_dataset):
        if sample["oracle_man"] is []:
            lib_samples["None"] += 1
        else:
            for lib in sample["oracle_man"]:
                if "python" in lib:
                    lib_name = lib.split("#")[0]
                else:
                    lib_name = lib.split(".")[0]

                if lib_name not in lib_samples:
                    lib_samples[lib_name] = {"predictions": [], "references": [], "n": 0}
                lib_samples[lib_name]["predictions"].append(preds)
                lib_samples[lib_name]["references"].append(refs)
                lib_samples[lib_name]["n"] += 1

    # top-10 interfaces/libraries
    lib_samples_filtered = dict(sorted(lib_samples.items(), key=lambda x: x[1]["n"], reverse=True)[:10])
    for lib, lib_data in lib_samples_filtered.items():
        print(lib)
        em_at_1 = round(compute_exact_match(lib_data["predictions"], lib_data["references"], 1) * 100, 2)
        em_at_2 = round(compute_exact_match(lib_data["predictions"], lib_data["references"], 2) * 100, 2)
        em_at_5 = round(compute_exact_match(lib_data["predictions"], lib_data["references"], 5) * 100, 2)
        em_at_10 = round(compute_exact_match(lib_data["predictions"], lib_data["references"], 10) * 100, 2)
        print(f"EM@1: {em_at_1} | EM@2: {em_at_2} | EM@5: {em_at_5} | EM@10: {em_at_10}")


if __name__ == "__main__":
    main()
