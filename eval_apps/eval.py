import evaluate
import json
import os
import sys

input_fp = sys.argv[1]
output_file_name = sys.argv[2]

base_path = os.path.dirname(input_fp)
output_fp = os.path.join(base_path, output_file_name)

with open(input_fp, "r") as f:
    responses = [[json.loads(l)["response"]] for l in f]

apps_metric = evaluate.load('codeparrot/apps_metric', keep_in_memory=True)
results = apps_metric.compute(predictions=responses, k_list=[1], level="all", debug=False)
print(results)

with open(output_fp, "w") as fout:
    json.dump(results, fout)
