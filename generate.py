import argparse
import json
import os

import anthropic
import evaluate
import torch
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from peft import PeftModelForCausalLM
from ragatouille import RAGPretrainedModel
from rich.progress import MofNCompleteColumn, BarColumn, Progress, TextColumn, TimeElapsedColumn
from tqdm import tqdm
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from datasets import load_from_disk
from utils import track_gpu_usage, make_chat_template_prompt, encode_chat_template, INSTRUCTION_PREFIX


def prepare_input(sample, knowledge_base_vectors, reranker, args):
    if args.use_rag:
        query = sample[args.instruction_field]
        retrieved_docs = knowledge_base_vectors.similarity_search(query=query, k=args.rag_n_retrieve)

        if args.use_reranking:
            docs = [doc.page_content for doc in retrieved_docs]
            docs_scores = reranker.rerank(query, docs, k=args.rag_n_retrieve)
            result_index = [doc["result_index"] for doc in docs_scores]
            retrieved_docs = [retrieved_docs[i] for i in result_index[:args.rag_n_final]]
        else:
            retrieved_docs = retrieved_docs[:args.rag_n_final]

        chat_docs = []
        for doc in retrieved_docs:
            chat_docs += make_chat_template_prompt(
                doc.page_content,
                doc.metadata["code"],
                instruction_prefix=INSTRUCTION_PREFIX[args.dataset_name],
            )
        return chat_docs + sample["messages"]
    return sample["messages"]


class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, start, eos_tokens, tokenizer):
        self.start = start
        self.eos_tokens = eos_tokens
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        tokens = self.tokenizer.decode(input_ids[0, self.start:])
        return any([eos_token in tokens for eos_token in self.eos_tokens])


@track_gpu_usage
def generate(args, dataset, model, tokenizer, knowledge_base_vectors=None, reranker=None):
    gen_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }

    with (Progress(
            TextColumn(f"Generating responses •" + "[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
    ) as p):
        for sample in p.track(dataset):
            messages = prepare_input(sample, knowledge_base_vectors, reranker, args)
            if not args.api_model:
                inputs = encode_chat_template(messages, tokenizer)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=args.max_new_tokens,
                    stopping_criteria=[CustomStoppingCriteria(inputs["input_ids"].shape[1], args.eos, tokenizer)],
                    **gen_kwargs
                )
                response_ids = outputs[0][inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(response_ids, skip_special_tokens=True)
                response = response.split("```")[0].strip()
            else:
                if isinstance(model, OpenAI):
                    response = model.chat.completions.create(
                        model=args.model_name_or_path,
                        messages=messages,
                        max_tokens=args.max_new_tokens,
                        temperature=args.temperature
                    )
                    response = response.choices[0].message.content
                elif isinstance(model, anthropic.Anthropic):
                    pass
            print(response)
            print("-" * 25)
            yield response


def compute_metrics(args, responses, dataset):
    chrf = evaluate.load("chrf")
    em = evaluate.load("exact_match")

    references = dataset[args.reference_field]
    results_em = em.compute(predictions=responses, references=references)

    references_chrf = [[ref] for ref in references]
    results_chrf = chrf.compute(predictions=responses, references=references_chrf)
    results_chrf2 = chrf.compute(predictions=responses, references=references_chrf, word_order=2)

    print(f"EM: {results_em}")
    print(f"chrF: {results_chrf}")
    print(f"chrF++: {results_chrf2}")

    return {
        "em": results_em,
        "chrf": results_chrf,
        "chrf2": results_chrf2
    }


def main(args):
    if not args.api_model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        if args.peft_checkpoint_path is not None:
            model = PeftModelForCausalLM.from_pretrained(model, args.peft_checkpoint_path)
        args.model_name = args.model_name_or_path.split("/")[-1]

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        args.eos = [tokenizer.eos_token, "\n```\n"]
    else:
        if "deepseek" in args.model_name_or_path:
            model = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        elif "claude" in args.model_name_or_path:
            model = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        else:
            model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        tokenizer = None
        args.model_name = args.model_name_or_path

    dataset = load_from_disk(args.dataset_name_or_path)["test"]
    args.dataset_name = args.dataset_name_or_path.split("/")[-1]

    if args.dataset_name == "conala":
        args.max_new_tokens = 128
        args.instruction_field = "nl"
        args.reference_field = "cmd"
    elif args.dataset_name == "mbpp":
        args.max_new_tokens = 512
        args.instruction_field = "text"
        args.reference_field = "code"
    else:
        args.max_new_tokens = 1024
        args.instruction_field = "question"
        args.reference_field = "solutions"

    knowledge_base_vectors = None
    reranker = None
    if args.use_icl:
        examples = (
            load_from_disk(args.dataset_name_or_path)["train"]
            .shuffle(args.icl_seed)
            .select(range(args.num_icl_examples))
        )
        chat_icl = []
        for example in examples:
            if args.dataset_name == "apps":
                reference = json.loads(example[args.reference_field])[0]
            else:
                reference = example[args.reference_field]
            chat_exemple = make_chat_template_prompt(
                example[args.instruction_field],
                reference,
                instruction_prefix=INSTRUCTION_PREFIX[args.dataset_name],
            )
            chat_icl += chat_exemple

        def add_icl_prompt(example):
            example["messages"] = chat_icl + example["messages"]
            return example

        dataset = dataset.map(add_icl_prompt, num_proc=16)
    elif args.use_rag:
        examples = load_from_disk(args.dataset_name_or_path)["train"]
        # @todo: special case for APPs
        knowledge_base = [
            LangchainDocument(
                page_content=sample[args.instruction_field],
                metadata={"code": sample[args.reference_field]}
            ) for sample in tqdm(examples)
        ]

        embedding_model = HuggingFaceEmbeddings(
            model_name=args.rag_encoder_model,
            multi_process=False,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )

        if args.use_reranking:
            reranker = RAGPretrainedModel.from_pretrained(args.reranking_model)

        knowledge_base_vectors = FAISS.from_documents(
            knowledge_base, embedding_model, distance_strategy=DistanceStrategy.COSINE
        )

    responses, init_gpu_memory, peak_gpu_memory, total_execution_time = (
        generate(args, dataset, model, tokenizer, knowledge_base_vectors, reranker)
    )

    metrics = {
        "init_gpu_memory": f"{init_gpu_memory} MB",
        "peak_gpu_memory": f"{peak_gpu_memory} MB",
        "total_execution_time": f"{total_execution_time} seconds"
    }

    if args.dataset_name == "conala":
        conala_metrics = compute_metrics(args, responses, dataset)
        metrics = {**metrics, **conala_metrics}

    output_dir = (
        f"{args.peft_checkpoint_path}/results" if args.peft_checkpoint_path else f"runs/{args.model_name}/results"
    )
    os.makedirs(output_dir, exist_ok=True)

    file_suffix = f"{args.dataset_name}_t{args.temperature}"
    if args.use_icl:
        file_suffix += f"_icl_n{args.num_icl_examples}_s{args.icl_seed}"
    elif args.use_rag:
        file_suffix += "_rag"
        if args.use_reranking:
            file_suffix += "_reranking"
        file_suffix += f"_n{args.rag_n_final}"

    with open(f"{output_dir}/metrics_{file_suffix}.jsonl", "w") as fout:
        json.dump(metrics, fout)

    data = [[response] for response in responses]
    with open(f"{output_dir}/responses_{file_suffix}.json", "w") as fout:
        json.dump(data, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--peft_checkpoint_path", type=str, default=None)
    parser.add_argument("--dataset_name_or_path", type=str, default=None)
    parser.add_argument("--api_model", action="store_true", default=False)

    parser.add_argument("--do_sample", default=True, type=bool, help="do sampling in generation")
    parser.add_argument("--temperature", default=0.2, type=float, help="temperature for sampling")
    parser.add_argument("--top_p", default=0.95, type=float, help="top p for sampling")
    parser.add_argument("--top_k", default=0, type=float, help="top k for sampling")

    parser.add_argument("--use_icl", action="store_true", default=False)
    parser.add_argument("--icl_seed", type=int, default=42)
    parser.add_argument("--num_icl_examples", type=int, default=3)

    parser.add_argument("--use_rag", action="store_true", default=False)
    parser.add_argument("--rag_encoder_model", default="thenlper/gte-small", type=str)
    parser.add_argument("--rag_n_retrieve", default=1, type=int)
    parser.add_argument("--rag_n_final", default=1, type=int)
    parser.add_argument("--use_reranking", action="store_true", default=False)
    parser.add_argument("--reranking_model", default="colbert-ir/colbertv2.0", type=str)

    args = parser.parse_args()
    set_seed(42)
    main(args)
