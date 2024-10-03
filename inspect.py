import json

with open("responses_apps_t0.2.json") as f:
    responses = json.load(f)

for response in responses[:10]:
    print(response[0])
    print()
    print("-" * 30)
    print()
