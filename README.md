
# No More In-Context Learning? Exploring Parameter-Efficient Fine-Tuning Techniques for Code Generation with Large Language Models
This repository is the official replication package for the paper "*No More In-Context Learning? Exploring Parameter-Efficient Fine-Tuning Techniques for Code Generation with Large Language Models*" submitted to ICSE 24.

In this readme, we provide detailed instructions on how to setup the repository, run the paper experiments, and compute the evaluation metrics. Our code can easily be adapted to further investigate the usage of parameter-efficient fine-tuning techniques for large language models for other generation tasks.

## Installation
1. Clone this repository using `git`.
2. Setup a  `Python 3`  virtual environment and install the requirements.
```sh
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
mkdir runs
```
We used Python 3.9 to run all our experiments, and a single NVIDIA GeForce RTX 3090 GPU. 
We used CUDA release 11.8, V11.8.89. Please, make sure the PyTorch version match your hardware requirements.

## Dataset and models checkpoints
We leveraged CoNaLa dataset for all our experiments. The dataset can be downloaded from HuggingFace at the following URL https://huggingface.co/datasets/neulab/docprompting-conala. To replicate our experiments or use the fine-tuning/inference scripts, you do not need to manually download the data. The `utils.py` file contains all the methods required to load the dataset.

We make all our models checkpoints available along with the prediction files generated when testing each model for code generation at the following URL https://zenodo.org/record/8191801. The directory `/checkpoints` contains all the models checkpoints, for all configuration of hyperparameters tested for the paper, and the `/test_code_generation_conala` contains the prediction file for each model checkpoint. Extract both folders into the `runs` folder for further usage.

The `/checkpoints` folder is organized as follows:
 - `{model}_ft` contains the checkpoint of a model fine-tuned using full fine-tuning (*e.g.,* PolyCoder-160M_ft). 
 - `{model}_lora_r{r}a{alpha}` contains the checkpoint of a model fine-tuned using LoRA (*e.g.,* codegen-2B-mono_lora_r8a16). 
 - `{model}_prompt{n}` contains the checkpoint of a model fine-tuned using Prompt tuning (*e.g.,* codegen-2B-mono_prompt20).
 
The `/test_code_generation_conala` folder is organized as follows:

- Each subfolder contains the predictions and references files for the corresponding model checkpoint. 
- For the in-context learning case, the subfolder is the name of the model (*e.g.,* bloom-7b1) and contains one the predictions and references file for each number of prompt examples.
- We also include `output.json`files which contains the top-*10* predictions for each test samples.

## Running the experiments

#### Fine-tune an LLM using LoRA or Prompt tuning
To fine-tune an LLM using LoRA or Prompt tuning, run the `train_peft.sh` script:

    ./train_peft.sh {model_name_or_path} {training_method} {batch_size} {gradient_accumulation_steps} {gpu_id}
    ./train_peft.sh Salesforce/CodeGen-2B-mono lora 4 4 0
    ./train_peft.sh Salesforce/CodeGen-2B-mono prompt-tuning 4 4 0

The script declares two variables `hps` and `n_tokens`, which contains the hyperparameters values for LoRA and Prompt tuning, respectively. Therefore, running the script will perform one fine-tuning per hyperparameter configuration. For instance, if you want to fine-tune an LLM using a single LoRA hyperparameter configuration, you can change the `hps` array to:

    declare -a hps=(
      8 16
    )
, where 8 is the *r* value and 16 is the *alpha* value.

#### Test a fine-tuned LLM on code generation
To test an LLM on code generation, run the `test_code_generation.sh` script:

    ./test_code_generation.sh {model_name_or_path} {training_method} {test_dataset} {gpu_id}
    ./test_code_generation.sh CodeGen-2B-mono lora conala 0
    ./test_code_generation.sh CodeGen-2B-mono prompt-tuning conala 0
    ./test_code_generation.sh CodeGen-2B-mono icl conala 0
Similarly to `train_peft.sh`, the test script also defines the hyperparameters values for each fine-tuning technique. The script fetches the corresponding checkpoint folder for each hyperparameter configuration. Therefore, make sure to change the hyperparameter values accordingly.

#### Computing the evaluation metrics
To compute the EM@*k*, run the `compute_metrics.sh` script:

    ./compute_metrics.sh {model_name_or_path} {training_method} {test_dataset}
    ./compute_metrics.sh CodeGen-2B-mono lora conala
    ./compute_metrics.sh CodeGen-2B-mono prompt-tuning conala
    ./compute_metrics.sh CodeGen-2B-mono icl conala
Again, make sure to change the hyperparameter values in the `compute_metrics.sh` file according to your existing model checkpoints.

To compute the Bleu and CodeBLEU scores: 
```sh
cd evaluator
./bleu.sh CodeGen-2B-mono lora conala
./bleu.sh CodeGen-2B-mono prompt-tuning conala
./bleu.sh CodeGen-2B-mono icl conala

cd CodeBLEU
./codebleu.sh CodeGen-2B-mono lora conala
./codebleu.sh CodeGen-2B-mono prompt-tuning conala
./codebleu.sh CodeGen-2B-mono icl conala
```
