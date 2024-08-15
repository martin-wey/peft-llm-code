from datasets import load_from_disk

dataset = load_from_disk("datasets/apps")
print(dataset)
"""
tokenizer = AutoTokenizer.from_pretrained("Microsoft/Phi-3-mini-128k-instruct")

dataset = dataset.map(lambda e: {
    "sample_len": len(tokenizer.apply_chat_template(e["messages"]))
})
dataset = dataset.filter(lambda e: e["sample_len"] > 2048)
print(len(dataset))
"""