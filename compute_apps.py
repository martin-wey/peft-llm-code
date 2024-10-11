import json

from evaluate import load


if __name__ == "__main__":
    results_dir = "runs/test_results"
    models = "CodeLlama-7b-Python-hf_lora"

    apps_metric = load('codeparrot/apps_metric')

    predictions_fp = f"{results_dir}/{models}/predictions_apps.jsonl"

    with open(predictions_fp, "r") as fpred:
        predictions = [json.loads(l) for l in fpred]

    introductory_predictions = predictions[:100]
    interview_predictions = predictions[100:200]
    competition_predictions = predictions[200:]

    results = apps_metric.compute(predictions=introductory_predictions, k_list=[1, 2, 5], count_errors=True,
                                  level="introductory")
    output_file = f"{results_dir}/{models}/apps_introductory_metrics.jsonl"
    with open(output_file, "w") as f:
        json.dump(results, f)

    results = apps_metric.compute(predictions=introductory_predictions, k_list=[1, 2, 5], count_errors=True,
                                  level="interview")
    output_file = f"{results_dir}/{models}/apps_interview_metrics.jsonl"
    with open(output_file, "w") as f:
        json.dump(results, f)

    results = apps_metric.compute(predictions=introductory_predictions, k_list=[1, 2, 5], count_errors=True,
                                  level="competition")
    output_file = f"{results_dir}/{models}/apps_competition_metrics.jsonl"
    with open(output_file, "w") as f:
        json.dump(results, f)
