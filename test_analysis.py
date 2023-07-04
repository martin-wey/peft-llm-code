import argparse
import re
from statistics import mean

from utils import load_test_dataset


def retrieve_hints_from_intent(text):
    pattern = r'`([^`]+)`|\'([^\']+)\'|"([^"]+)"'
    matches = re.findall(pattern, text)
    words = [match[0] or match[1] or match[2] for match in matches]
    return words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds')
    parser.add_argument('--refs')
    args = parser.parse_args()

    dataset = load_test_dataset()
    predictions = [x.strip() for x in open(args.preds, 'r', encoding='utf-8').readlines()]
    references = [x.strip() for x in open(args.refs, 'r', encoding='utf-8').readlines()]

    domain_scores_dict = {"None": []}
    for (sample, pred, ref) in zip(dataset, predictions, references):
        score = 0
        if pred == ref:
            score = 1
        if len(sample["library"]) == 0:
            domain_scores_dict["None"].append(score)
        for lib in sample["library"]:
            if lib not in domain_scores_dict:
                domain_scores_dict[lib] = []
            domain_scores_dict[lib].append(score)

    for domain, score_list in domain_scores_dict.items():
        print(f"{domain} - EM: {mean(score_list)}% ({len(score_list)})")

    print("=" * 100)

    length = len(references)
    count = 0
    for i in range(length):
        r = references[i]
        p = predictions[i]
        if r == p:
            count += 1
    acc = round(count / length * 100, 2)
    print(f"EM: {acc}")

    print("=" * 100)

    n_hints = 0
    acc = 0
    for (ref, pred) in zip(dataset, predictions):
        hints = retrieve_hints_from_intent(ref["intent"])
        for hint in hints:
            n_hints += 1
            if hint in pred:
                acc += 1
    print(round((acc / n_hints) * 100, 3))


if __name__ == "__main__":
    main()
