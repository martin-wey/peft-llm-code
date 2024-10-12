"""This is an example script to evaluate a code generation model on APPS, you can also use the APPS solutions as code generations
> python example_script.py --model_ckpt MODEL_NAME --num_tasks 10 --difficulty introductory --n_samples 1
> python example_script.py --use_solutions True --num_tasks 10 --difficulty introductory --n_samples 1"""

import json
import pprint
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
from evaluate import load

def generate_prompt(sample):
    starter_code = None if len(sample["starter_code"]) == 0 else sample["starter_code"] 
    try:
        input_outpout = json.loads(sample["input_output"])
        fn_name = None if not input_outpout.get("fn_name") else input_outpout["fn_name"] 
    except ValueError:
        fn_name = None 
    _input = "\nQUESTION:\n"
    _input += sample["question"]
    if starter_code:
        _input += starter_code
    if fn_name:
        _input += "\nUse Standard Input format"
    else:
        _input += "\nUse Call-Based format"
    
    _input += "\nANSWER:\n"
    return _input


def complete_code(pipe, prompt, num_completions=1, max_length=256, **gen_kwargs):
    """Complete prompt with text generation pipeline and return num_completions."""
    prompt = pipe.tokenizer.eos_token + prompt
    try:
        code_gens = pipe(prompt, num_return_sequences=num_completions, max_length=max_length, **gen_kwargs)
        return [code_gen["generated_text"][len(prompt):] for code_gen in code_gens]
    except IndexError:
        print("prompt is longer than the context size of the model, generation skipped")
        code_gens = ""
        return [""]


def make_generations(dataset, args, model, tokenizer):
    set_seed(args.seed)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=args.device_int)

    # Generation settings
    gen_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k
    }

    # Generate completions for evaluation set
    n_tasks = args.num_tasks if args.num_tasks is not None else len(dataset)
    print(f"ntasks is {n_tasks}")
    generations = []
    for task in tqdm(range(n_tasks)):
        task_generations = []
        prompt = generate_prompt(dataset[task]).strip()
        task_generations.extend(complete_code(pipe, prompt, num_completions=args.n_samples, max_length=args.max_length, **gen_kwargs))
        generations.append([gen.replace(args.eos, "") for gen in task_generations])
    return generations


def main(args):
    DATA_PATH = "codeparrot/apps"
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # setup
    print("Loading evaluation dataset...")
    dataset = load_dataset(DATA_PATH, split="test", difficulties=[args.difficulty])
    if args.use_solutions:
        print("Using data solutions as code generations")
        model = None
        tokenizer = None
        generations = []
        for index in range(args.num_tasks+1):
            try:
                sol = json.loads(dataset[index]["solutions"])
                generations.append(sol[:args.n_solutions])
            except ValueError:
                print(f"No solutions for task {index} or not enough to have {args.n_solutions} solutions")
                break
        
    else:
        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        model = AutoModelForCausalLM.from_pretrained(args.model_ckpt)
        generations = make_generations(dataset, args, model, tokenizer)
        
    metric = load("loubnabnl/apps_metric")
    results = metric.compute(predictions=generations, level=args.difficulty, k_list=args.k_list, count_errors=args.count_errors, debug=args.debug)
    print(results)
    with open(args.output_file, "w") as fp:
        json.dump(results, fp)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Testing a Language Model on APPS Python Code dataset")
    #model and tokenizer arguments
    parser.add_argument("--model_ckpt", default="loubnabnl/apps-1.5B-model", type=str, help="path to model checkpoint.")
    parser.add_argument("--tokenizer", default="gpt2", type=str, help="tokenizer to use.")
    parser.add_argument("--eos", default="<|endoftext|>", type=str, help="end of sentence token.")
    # generation arguments
    parser.add_argument("--do_sample", default=True, type=bool, help="do sampling in generation")
    parser.add_argument("--temperature", default=0.2, type=float, help="temperature for sampling")
    parser.add_argument("--top_p", default=0.95, type=float, help="top p for sampling")
    parser.add_argument("--top_k", default=0, type=float, help="top k for sampling")
    parser.add_argument("--max_length", default=1024, type=int, help="max length of generated code")
    # evaluation arguments
    parser.add_argument("--difficulty", default="all", type=str, help="difficulty level to select in the dataset from:\
     'all', 'introductory', 'interview'  and 'competition' ")
    parser.add_argument("--num_tasks", default=6, type=int, help="number of tasks to evaluate")
    parser.add_argument("--use_solutions", default=False, type=bool, help="use solutions instead of generating new code")
    parser.add_argument("--n_samples", default=1, type=int, help="number of samples to generate")
    parser.add_argument("--n_solutions", default=1, type=int, help="number of solutions to use")
    parser.add_argument("--k_list", default=[1, 2, 3], type=list, help="list of k values to evaluate pass@k")
    parser.add_argument("--count_errors", default=False, type=bool, help="count compilation and runtime errors for single generations")
    # configuration
    parser.add_argument("--seed", default=0, type=int, help="generation seed")
    parser.add_argument("--device_int", default=-1, type=int, help="device on which code generation is run, if positive use GPU")
    parser.add_argument("--debug", default=False, type=bool, help="debug mode")
    # save
    parser.add_argument("--output_file", default="apps_metrics.json", type=str, help="output file to save the results")

    args = parser.parse_args()
    main(args)