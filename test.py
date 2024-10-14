import logging
import os
import random

import torch
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    default_data_collator,
    StoppingCriteriaList,
    StoppingCriteria
)

from data_utils import *
from utils import load_model_and_tokenizer

logger = logging.getLogger(__name__)
EOF_STRINGS = ["<|endoftext|>", "</s>", "```"]


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` that checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length:])
        done = []
        for decoded_generation in decoded_generations:
            done.append(any([stop_string in decoded_generation for stop_string in self.eof_strings]))
        return all(done)


def run_test(args):
    dataset_loading_func = globals().get(f"load_{args.dataset}_test_dataset")
    test_dataset = dataset_loading_func()

    instruction_column = "nl"
    response_column = "cmd"

    if args.dataset == "apps":
        instruction_column = "question"
        response_column = "solutions"

    model, tokenizer = load_model_and_tokenizer(args, is_training=False)

    # in-context learning with random examples
    if args.num_icl_examples > 0:
        train_loading_func = globals().get(f"load_{args.dataset}_train_dataset")
        train_dataset = train_loading_func()["train"]
        random_indices = random.sample(range(len(train_dataset)), args.num_icl_examples)
        icl_examples = train_dataset.select(random_indices)

        icl_prompt = ""
        for example in icl_examples:
            icl_prompt += (
                f"### Instruction:\n{example[instruction_column]}\n"
                f"### Response:\n```python\n{example[response_column]}\n```\n\n"
            )
    # retrieval-augmented generation, select closest samples from training set
    elif args.num_rag_examples > 0:
        train_loading_func = globals().get(f"load_{args.dataset}_train_dataset")
        train_dataset = train_loading_func()["train"]

        knowledge_base = [
            LangchainDocument(
                page_content=sample[instruction_column],
                metadata={"code": sample[response_column]}
            ) for sample in tqdm(train_dataset)
        ]
        embedding_model = HuggingFaceEmbeddings(
            model_name=args.rag_encoder_model,
            multi_process=False,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )
        knowledge_base_vectors = FAISS.from_documents(
            knowledge_base, embedding_model, distance_strategy=DistanceStrategy.COSINE
        )

    def preprocess_function(examples):
        prompts = []
        prompt_template = "### Instruction:\n{instruction}\n### Response:\n```python\n"
        # CodeLlama prompt for APPs, similar to what is reported in the paper
        prompt_template_apps = (
            "<s>[INST] Write a python code to solve the following coding problem that obeys the constraints "
            "and passes the example test cases. The output code needs to {guide}:\n{instruction}\n[/INST]\n"
            "```python\n{starter_code}"
        )

        for i, instruction in enumerate(examples[instruction_column]):
            if args.dataset == "apps":
                starter_code = "" if examples["starter_code"][i] is None else examples["starter_code"][i]
                prompt = prompt_template_apps.format(
                    guide=examples["guide"][i],
                    instruction=instruction,
                    starter_code=starter_code
                )
            else:
                prompt = prompt_template.format(instruction=instruction, add_special_tokens=False)

            # in-context learning
            if args.num_icl_examples >= 0:
                prompt = icl_prompt + prompt
            # RAG
            elif args.num_rag_examples > 0:
                retrieved_docs = knowledge_base_vectors.similarity_search(query=instruction, k=args.num_rag_examples)
                rag_prompt = ""
                for doc in retrieved_docs:
                    rag_prompt += (
                        f"### Instruction:\n{doc.page_content}\n"
                        f"### Response:\n```python\n{doc.metadata["code"]}\n```\n\n"
                    )
                prompt = rag_prompt + prompt

            print(prompt)

            if "codet5" in args.model_name_or_path:
                prompt += "<extra_id_0>"

            prompts.append(prompt)

        # pad to max sequence in batch
        model_inputs = tokenizer(prompts, padding=True)

        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
        }

    test_dataset_tokenized = test_dataset.map(
        preprocess_function,
        # num_proc=args.num_workers,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=[cname for cname in test_dataset.column_names if
                        cname not in ["input_ids", "attention_mask"]],
        desc="Generating samples features."
    )
    dataloader = DataLoader(
        test_dataset_tokenized,
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        pin_memory=True
    )


    if args.dataset in ["conala", "codealpaca"]:
        kwargs = {
            "num_beams": 10,
            "num_return_sequences": 10
        }
    else:
        kwargs = {
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": .95,
            "num_return_sequences": 5
        }

    predictions = [[] for _ in range(len(test_dataset_tokenized))]
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        with torch.no_grad():
            batch = {k: v.to(model.device) for k, v in batch.items()}
            generated_sequences = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=args.max_target_length,
                stopping_criteria=StoppingCriteriaList(
                    [EndOfFunctionCriteria(batch["input_ids"].shape[1], EOF_STRINGS, tokenizer)]
                ),
                **kwargs
            )
            # (batch_size, num_return_sequences, num_token_generated)
            generated_sequences = generated_sequences.view(args.batch_size, kwargs["num_return_sequences"], -1)
            generated_sequences = generated_sequences.detach().cpu().numpy()

            if "codet5" not in args.model_name_or_path:
                generated_sequences = generated_sequences[:, :, batch["input_ids"].shape[1]:]

            decoded_sequences = [
                [
                    tokenizer.decode(
                        output,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    ).split("```")[0].strip() for output in sample_seq
                ]
                for sample_seq in generated_sequences
            ]

            for i, outputs in enumerate(decoded_sequences):
                predictions[step * args.batch_size + i].extend(outputs)

    output_filename = f"predictions_{args.dataset}"
    if args.dataset == "apps":
        output_filename += f"_{args.temperature}"

    if args.num_icl_examples > -1:
        output_filename += f"_icl_n{args.num_icl_examples}"

    if args.num_rag_examples > 0:
        output_filename += f"_rag_n{args.num_rag_examples}"

    with open(os.path.join(args.run_dir, f"{output_filename}.jsonl"), "w") as f:
        for outputs in predictions:
            f.write(json.dumps(outputs) + "\n")
