import json

from evaluate import load
from datasets import load_dataset


if __name__ == "__main__":
    results_dir = "runs/test_results"
    models = [
        f"{results_dir}/CodeLlama-7b-Instruct-hf",
        f"{results_dir}/CodeLlama-7b-Instruct-hf_lora",
        f"{results_dir}/CodeLlama-7b-Instruct-hf_qlora-4b",
        f"{results_dir}/CodeLlama-7b-Instruct-hf_qlora-8b",
    ]

    apps_metric = load("apps_metric")

    for model in models:
        predictions_fp = f"{model}/predictions_apps_t0.6.jsonl"

        with open(predictions_fp, "r") as fpred:
            predictions = [json.loads(l) for l in fpred]

        introductory_predictions = predictions[:250]
        interview_predictions = predictions[250:500]
        competition_predictions = predictions[500:]

        results = apps_metric.compute(
            predictions=introductory_predictions,
            k_list=[1, 2, 5],
            count_errors=True,
            level="introductory"
        )
        output_file = f"{model}/apps_introductory_metrics_t0.6.jsonl"
        with open(output_file, "w") as f:
            json.dump(results, f)

        results = apps_metric.compute(
            predictions=interview_predictions,
            k_list=[1, 2, 5],
            count_errors=True,
            level="interview"
        )
        output_file = f"{model}/apps_interview_metrics_t0.6.jsonl"
        with open(output_file, "w") as f:
            json.dump(results, f)
        results = apps_metric.compute(
            predictions=competition_predictions,
            k_list=[1, 2, 5],
            count_errors=True,
            level="competition"
        )
        output_file = f"{model}/apps_competition_metrics_t0.6.jsonl"
        with open(output_file, "w") as f:
            json.dump(results, f)
