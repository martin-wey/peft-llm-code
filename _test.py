import torch
from transformers import AutoModelForCausalLM
from peft import (
    get_peft_model,
    TaskType,
    LoraConfig,
    IA3Config,
    PromptTuningConfig,
    PrefixTuningConfig,
    PeftModel
)

model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf",
                                             torch_dtype=torch.float16,
                                             low_cpu_mem_usage=True,
                                             device_map="auto",
                                             trust_remote_code=True)
print(model)
