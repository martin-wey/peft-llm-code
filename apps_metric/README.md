---
title: APPS Metric
emoji: 📊
colorFrom: blue
colorTo: pink
tags:
- evaluate
- metric
description: "Evaluation metric for the APPS benchmark"
sdk: gradio
sdk_version: 3.0.2
app_file: app.py
pinned: false
---

# Metric Card for apps_metric 

## Metric Description
This metric is used to evaluate code generation on the [APPS benchmark](https://huggingface.co/datasets/codeparrot/apps).

## How to Use
You can load the metric and use it with the following commands:

```python
from evaluate import load
apps_metric = load('codeparrot/apps_metric')
# to evaluate generations made for all levels for example
results = apps_metric.compute(predictions=generations, level="all")
```

### Inputs
**generations** list(list(str)): List of code generations, each sub-list corresponds to the generations for a problem in APPS dataset, **the order of the samples in the dataset must be kept (with respect to the difficulty level)**.

### Output Values

**average accuracy**: when a single solution is generated, average accuracy computes the average of test cases that are passed.

**strict accuracy**: when a single solution is generated, strict accuracy computes the average number of problems that pass all their test cases.

**pass@k**: when multiple solutions are generated per problem, pass@k is the metric originally used for the [HumanEval](https://huggingface.co/datasets/openai_humaneval) benchmark. For more details please refer to the [metric space](https://huggingface.co/spaces/evaluate-metric/code_eval) and [Codex paper](https://arxiv.org/pdf/2107.03374v2.pdf).

## Citation
```
@article{hendrycksapps2021,
  title={Measuring Coding Challenge Competence With APPS},
  author={Dan Hendrycks and Steven Basart and Saurav Kadavath and Mantas Mazeika and Akul Arora and Ethan Guo and Collin Burns and Samir Puranik and Horace He and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
``` 
