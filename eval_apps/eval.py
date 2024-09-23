import evaluate
import json
import os
import sys

input_fp = sys.argv[1]
base_path = os.path.dirname(input_fp)

with open(input_fp, "r") as f:
    responses = json.load(f)

apps_metric = evaluate.load('codeparrot/apps_metric', keep_in_memory=True)

interview_responses = responses[:3000]
competition_responses = responses[3000:4000]
intro_responses = responses[4000:5000]

for difficulty_responses, difficulty in [(interview_responses, "interview"),
                                         (competition_responses, "competition"),
                                         (intro_responses, "introductory")]:
    print(f"Evaluating {difficulty} -- {input_fp}")
    results = apps_metric.compute(predictions=difficulty_responses, level=difficulty, debug=False, count_errors=True)
    print(results)

    output_fp = os.path.join(base_path, f"apps_metrics_{difficulty}.json")
    with open(output_fp, "w") as fout:
        json.dump(results, fout)
