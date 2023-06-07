# Study on parameter-efficient fine-tuning techniques for code

## Tasks and datasets.

1. Vulnerability detection - Devign dataset
2. Program translation - XLCoST dataset
3. Code generation - Subset of CONCODE dataset
4. Code generation - XLCoST dataset

## Studied PLMs and LLMs

1. Medium-sized PLMs
   1. CodeBERT
   2. GraphCodeBERT
   3. CodeT5
2. LLMs (from 1B to 12B)
   1. Bloom
   2. InCoder
   3. CodeGen
   4. CodeT5+
   5. PolyCoder

## Setup

```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

mkdir runs
cd runs
mkdir devign_defect-detection
mkdir xlcost_code-translation
mkdir xlcost_code-generation
mkdir concode_code-generation
```

## Examples of running arguments

Running arguments using a single Nvidia V100L 32Gi GPU.

### Defect detection

- CodeBERT fine-tuning:
```shell
python main.py \
  --model_name_or_path microsoft/codebert-base \
  --model_type roberta \
  --task devign_defect-detection \
  --training_method ft \
  --train_batch_size 32 \
  --val_batch_size 32 \
  --learning_rate 5e-5 \
  --weight_decay 0 \
  --num_epochs 5 \
  --defect_max_seq_length 400 
```
Training time ~42min, peak GPU memory usage 17Gi.