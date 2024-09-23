from datasets import load_from_disk

from peft import AutoPeftModelForCausalLM

import torch
from transformers import AutoTokenizer

_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

model = AutoPeftModelForCausalLM.from_pretrained("runs/Phi-3-mini-128k-instruct_conala_p-tuning/checkpoint-1335").to("cuda")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

dataset = load_from_disk("datasets/conala")["test"].select(range(1))

for example in dataset:
    prompt = tokenizer.apply_chat_template(example["messages"], tokenize=False).split(_MAGIC_SPLITTER_)[0]
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128
        )
        print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))