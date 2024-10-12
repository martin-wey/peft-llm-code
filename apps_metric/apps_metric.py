# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluation of code generation on the APPS benchmark"""

import evaluate
import datasets
from .utils import compute_metrics
from .testing_util import run_test


_CITATION = """\
@article{hendrycksapps2021,
  title={Measuring Coding Challenge Competence With APPS},
  author={Dan Hendrycks and Steven Basart and Saurav Kadavath and Mantas Mazeika and Akul Arora and Ethan Guo and Collin Burns and Samir Puranik and Horace He and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
"""


_DESCRIPTION = """\
This is a metric to evaluate code generation using the APPS benchmark "Measuring Coding Challenge Competence With
APPS" (https://arxiv.org/pdf/2105.09938.pdf).
"""


# TODO: Add description of the arguments of the module here
_KWARGS_DESCRIPTION = """
Computes Average accuracy and strict accuracy for single generations, and pass@k for multiple generations.
Args:
    predictions: list of code generations to score. It's a list of list(s), each corresponding to a problem from APPS dataset.

Returns:
    metrics: dict of three metrics: average accuracy, stric accuracy, and pass@k.
Examples:
    >>> my_new_module = evaluate.load("loubnabnl/apps_metric")
    >>> results = my_new_module.compute(predictions=[["s=input()\nprint(s)"]])
    >>> print(results)
    {'avg_accuracy': 0, 'strict_accuracy': 0, 'pass_at_k': None}
"""




@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class apps_metric(evaluate.EvaluationModule):
    """Evaluate code generation on APPS benchmark. 
    The generations are compiled and their corresponding unit tests are run"""

    def _info(self):

        return evaluate.EvaluationModuleInfo(

            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,

            features=datasets.Features({
                'predictions': datasets.Sequence(datasets.Value("string")),
            }),
            homepage="https://github.com/hendrycks/apps",
            reference_urls=["https://huggingface.co/datasets/codeparrot/apps"]
        )



    def _compute(self, predictions, k_list=[1, 10, 100], count_errors=True, level="all", debug=False):
        """Returns the scores"""
        metrics = compute_metrics(predictions, k_list=k_list, count_errors=count_errors, level=level, debug=debug)
        return metrics