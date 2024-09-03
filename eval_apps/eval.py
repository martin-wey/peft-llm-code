import evaluate
import json
import time

with open("../runs/deepseek-coder-6.7b-instruct_apps_lora/checkpoint-281/results/responses_apps_t0.2.jsonl", "r") as f:
    responses = [[json.loads(l)["response"]] for l in f]

start = time.time()
print(start)

apps_metric = evaluate.load('codeparrot/apps_metric', keep_in_memory=True)
results = apps_metric.compute(predictions=responses, level="all")
print(results)

end = time.time()
print(end)

print(f"Total time: {end - start}")
