import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, default_data_collator
from torch.utils.data import DataLoader

from utils import *


def test_devign_defect_detection(args):
    def convert_samples_to_features(examples):
        tokenized_inputs = tokenizer(examples["func"],
                                     truncation=True,
                                     padding="max_length",
                                     max_length=args.defect_max_seq_length)
        tokenized_inputs["labels"] = examples["target"]
        return tokenized_inputs

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = load_devign_defect_detection_dataset(args.dataset_dir)
    test_dataset = dataset["test"]
    test_dataset = test_dataset.map(convert_samples_to_features,
                                    batched=True,
                                    num_proc=args.num_workers,
                                    remove_columns=[cname for cname in test_dataset.column_names if
                                                    cname not in ["input_ids", "labels", "attention_mask"]],
                                    desc="Generating samples features.")
    test_dataloader = DataLoader(test_dataset, collate_fn=default_data_collator, batch_size=args.val_batch_size)

    total_correct = 0
    for batch in tqdm(test_dataloader, total=len(test_dataset) // args.val_batch_size):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch).logits
        predicted_labels = torch.argmax(outputs, dim=1)
        total_correct += (predicted_labels == batch["labels"]).sum().item()
    accuracy = total_correct / len(test_dataset)
    print(f"Accuracy: {round(accuracy * 100, 2)}%")
