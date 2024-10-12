import json
from evaluate import load

solution_sample1 = json.load(open("test_examples/solutions_problem_1.json", "r"))
solution_sample2 = json.load(open("test_examples/solutions_problem_2.json", "r"))
single_solutions = [solution_sample1[:1], solution_sample2[:1]]
multiple_solutions = [solution_sample1[:3], solution_sample2[:3]]

metric = load("codeparrot/apps_metric")
result_1 = metric.compute(predictions=single_solutions, level="all")
result_2 = metric.compute(predictions=multiple_solutions, level="all", k_list=[1, 2, 3])

assert result_1 == {'avg_accuracy': 1.0, 'strict_accuracy': 1.0, 'pass_at_k': None}
assert result_2 == {'avg_accuracy': None, 'strict_accuracy': None, 'pass_at_k': {'pass@1': 1.0, 'pass@2': 1.0, 'pass@3': 1.0}}