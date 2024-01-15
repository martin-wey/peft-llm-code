
# Exploring Parameter-Efficient Fine-Tuning Techniques for Code Generation with Large Language Models
Official replication package for our submission to TOSEM.

In this readme, we provide detailed instructions on how to setup the repository and run the paper experiments. 
Our code can easily be adapted to further investigate the usage of parameter-efficient fine-tuning techniques for large language models for other generation tasks.

## Installation
1. Clone this repository using `git`.
2. Setup a  `Python 3`  virtual environment and install the requirements.
```sh
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
mkdir runs
```
We used Python 3.11.5 to run all our experiments, and a single NVIDIA RTX A5000. 
We used CUDA release 12.3, V12.3.52. Please, make sure the PyTorch version match your hardware requirements.

## Running the experiments

### Fine-tune an LLM using PEFT
```shell
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model_name_or_path codellama/CodeLlama-7b-hf \
  --dataset codealpaca \
  --tuning_method lora \
  --num_epochs 5 \
  --batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 3e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --do_train \
  --use_wandb
```

- You can also decide to not use WanDB by removing the `use_wandb` argument.
- For `QLoRA`, set `--tuning_method qlora-8bit` or `--tuning_method qlora-4bit`.
- For joint training, set `--dataset joint`. 
- The script automatically saves the best model checkpoint in the directory: `/runs/checkpoints/{dataset}/{model_name}_{tuning_method}/`
In our example: `/runs/checkpoints/codealpaca/CodeLlama-7b-hf_lora`

### Evaluating fine-tuned LLMs
```shell
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model_name_or_path codellama/CodeLlama-7b-hf \
  --adapter_path runs/checkpoints/conala/CodeLlama-7b-hf_lora \
  --tuning_method lora \
  --dataset conala \
  --tuning_method lora \
  --do_test
```
- `--adapter_path` corresponds to the local path of the best model checkpoint.
- The script saves files in the directory: `/runs/test_results/{model_name}_{tuning_method}`:
  - `output_{dataset}.jsonl`: top-10 predictions to compute EM@*k*.
  - `predictions_{dataset}.txt` and `references_{dataset}.txt`: top-1 predictions to compute CodeBLEU.

### Evaluating LLMs with ICL
Use the `scripts/inference_icl_seeds.sh` bash script to replicate the results from the paper:
```shell
bash scripts/inference_icl_seeds.sh codellama/CodeLlama-7b-hf conala 0
```
The script is going to run inference on the input model with the following parameters (that you can adjust to your needs):
- `n_examples=(1 2 3)`: 1 to 3 few-shot examples 
- `seeds=(42 777 5432 55555 97)`: run inference using 5 seeds
- Note that it results in running inference 15 times, which can be time consuming.

### Computing EM@k and CodeBLEU
Use `compute_metrics.py`, which computes the EM@*k* and CodeBLEU on all the test results stored in the `runs/test_results` directory.
