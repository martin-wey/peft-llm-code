#!/bin/bash

export HF_HOME="/Tmp/weyssowm/.cache/huggingface"
export WANDB_PROJECT="PEFT-Code-LLM"

model=$1
model_name=$(echo "$model" | rev | cut -d'/' -f1 | rev)
gpu_id=$2

datasets=(
  # "conala 256 8 3e-4"
  "mbpp 512 4 3e-6"
)

for dataset in "${datasets[@]}"; do
  arr=($dataset)
  ds=${arr[0]}
  max_len=${arr[1]}
  bsz=${arr[2]}
  lr=${arr[3]}

  # LoRA
  run_name="${model_name}_${ds}_lora"
  CUDA_VISIBLE_DEVICES=$gpu_id python sft.py \
    --output_dir "runs/${run_name}" \
    --model_name_or_path "$model" \
    --dataset_name "datasets/${ds}" \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size $bsz \
    --per_device_eval_batch_size $bsz \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --learning_rate "$lr" \
    --max_seq_length "$max_len" \
    --gradient_checkpointing \
    --optim adafactor \
    --use_peft \
    --use_lora \
    --completion_only \
    --response_template '```python' \
    --eval_strategy "epoch" \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --report_to "wandb" \
    --run_name "${run_name}" \
    --active_gpu "$gpu_id"

  # DoRA
  run_name="${model_name}_${ds}_dora"
  CUDA_VISIBLE_DEVICES=$gpu_id python sft.py \
    --output_dir "runs/${run_name}" \
    --model_name_or_path "$model" \
    --dataset_name datasets/"${ds}" \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size $bsz \
    --per_device_eval_batch_size $bsz \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --learning_rate "$lr" \
    --max_seq_length "$max_len" \
    --gradient_checkpointing \
    --optim adafactor \
    --use_peft \
    --use_lora \
    --use_dora \
    --completion_only \
    --response_template '```python' \
    --eval_strategy "epoch" \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --report_to "wandb" \
    --run_name "${run_name}" \
    --active_gpu "$gpu_id"

  # QLoRA-8b
  run_name="${model_name}_${ds}_qlora-8b"
  CUDA_VISIBLE_DEVICES=$gpu_id python sft.py \
    --output_dir "runs/${run_name}" \
    --model_name_or_path "$model" \
    --dataset_name datasets/"${ds}" \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size $bsz \
    --per_device_eval_batch_size $bsz \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --learning_rate "$lr" \
    --max_seq_length "$max_len" \
    --gradient_checkpointing \
    --optim adafactor \
    --use_peft \
    --use_lora \
    --load_in_8bit \
    --completion_only \
    --response_template '```python' \
    --eval_strategy "epoch" \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --report_to "wandb" \
    --run_name "${run_name}" \
    --active_gpu "$gpu_id"

  # QLoRA-4b
  run_name="${model_name}_${ds}_qlora-4b"
  CUDA_VISIBLE_DEVICES=$gpu_id python sft.py \
    --output_dir "runs/${run_name}" \
    --model_name_or_path "$model" \
    --dataset_name datasets/"${ds}" \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size $bsz \
    --per_device_eval_batch_size $bsz \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --learning_rate "$lr" \
    --max_seq_length "$max_len" \
    --gradient_checkpointing \
    --optim adafactor \
    --use_peft \
    --use_lora \
    --load_in_4bit \
    --completion_only \
    --response_template '```python' \
    --eval_strategy "epoch" \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --report_to "wandb" \
    --run_name "${run_name}" \
    --active_gpu "$gpu_id"
done