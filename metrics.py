import argparse
import os
from collections import Counter

from crystalbleu import corpus_bleu, SmoothingFunction
from nltk.util import ngrams

from utils import LANG_TO_EXT


def extract_trivially_shared_ngrams(dataset_dir):
    trivially_shared_ngrams = {}
    max_ngram_order = 4
    task_lang_pairs = os.listdir(dataset_dir)
    for task_lang_pair in task_lang_pairs:
        all_ngrams = []
        lang, _ = task_lang_pair.split("-")
        data_fp = os.path.join(dataset_dir, task_lang_pair,
                               f"train-{lang}-desc-tok.{LANG_TO_EXT[lang]}")
        data = [line.strip() for line in open(data_fp, encoding="utf-8").readlines()]
        for sample in data:
            sample_tokenized = sample.split()
            for j in range(1, max_ngram_order + 1):
                n_grams = list(ngrams(sample_tokenized, j))
                all_ngrams.extend(n_grams)
        freq = Counter(all_ngrams)
        most_common_dict = dict(freq.most_common(500))
        trivially_shared_ngrams[lang] = most_common_dict
    return trivially_shared_ngrams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refs", default=None, type=str, help="References file.")
    parser.add_argument("--preds", default=None, type=str, help="References file.")
    args = parser.parse_args()

    # extract trivially shared n-grams to compute CrystalBLEU
    trivially_shared_ngrams = extract_trivially_shared_ngrams("datasets/pair_data_tok_full_desc_comment")
    # smoothing function for BLEU and CrystalBLEU
    sm_func = SmoothingFunction(epsilon=0.0001).method1

    references = [x.strip() for x in open(args.refs, encoding="utf-8").readlines()]
    predictions = [x.strip() for x in open(args.preds, encoding="utf-8").readlines()]
    data = {}
    total_samples = 0
    for ref, pred in zip(references, predictions):
        ref_splitted = ref.split("|")
        pred_splitted = pred.split("|")
        target_lang = ref_splitted[0].strip().split(";")[1].strip()
        if target_lang not in data:
            data[target_lang] = {
                "references": [ref_splitted[1].strip()],
                "predictions": [pred_splitted[1].strip()],
            }
        else:
            data[target_lang]["references"].append(ref_splitted[1].strip())
            data[target_lang]["predictions"].append(pred_splitted[1].strip())
        total_samples += 1

    weighted_bleu = 0
    weighted_crystalbleu = 0
    for lang, lang_data in data.items():
        references = [[ref.split()] for ref in lang_data["references"]]
        predictions = [pred.split() for pred in lang_data["predictions"]]

        bleu_score = corpus_bleu(references, predictions, smoothing_function=sm_func, ignoring=None)
        crystalbleu_score = corpus_bleu(references, predictions, smoothing_function=sm_func,
                                        ignoring=trivially_shared_ngrams[lang])
        print(f"[{lang}] CrystalBLEU: {(crystalbleu_score * 100)}")
        weighted_bleu += (len(predictions) / total_samples) * (bleu_score * 100)
        weighted_crystalbleu += (len(predictions) / total_samples) * (crystalbleu_score * 100)
    print(f"Weighted avg BLEU: {round(weighted_bleu, 2)}, Weighted avg CrystalBLEU: {round(weighted_crystalbleu, 2)}")


if __name__ == "__main__":
    main()
