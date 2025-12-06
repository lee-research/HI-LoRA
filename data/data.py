from datasets import load_dataset, Dataset
import typing as tp
import functools
import os
import pickle
import logging
import hashlib

log = logging.getLogger(__name__)


def cache_to_disk(root_datadir="data_cache"):
    """
    Decorator to cache function results to disk.
    Uses MD5 hash of arguments to create unique cache filenames.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not os.path.exists(root_datadir):
                os.makedirs(root_datadir)

            func_name = func.__name__.replace("/", "")
            args_str = "_".join(map(str, args))
            kwargs_str = "_".join(f"{k}={v}" for k, v in kwargs.items())
            params_str = f"{args_str}_{kwargs_str}"

            # Hash parameters for unique filename
            params_hash = hashlib.md5(params_str.encode()).hexdigest()
            cache_filename = os.path.join(root_datadir, f"{func_name}_{params_hash}.pkl")

            if os.path.exists(cache_filename):
                with open(cache_filename, "rb") as f:
                    log.info(f"Loading cached data for {func.__name__}")
                    return pickle.load(f)

            result = func(*args, **kwargs)

            with open(cache_filename, "wb") as f:
                pickle.dump(result, f)
                log.info(f"Cached data for {func.__name__}")

            # Record hash mapping
            hash_table_filename = os.path.join(root_datadir, "hash_table.txt")
            with open(hash_table_filename, "a") as f:
                f.write(f"{cache_filename}: {params_str}\n")

            return result

        return wrapper

    return decorator


@cache_to_disk("data_cache")
def load_emo():
    """Load EMO emotion classification dataset."""
    dataset = load_dataset("emo")
    label_map = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
    instruction = "classify the emotion of the text: "
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["text"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    train_set = dataset["train"]
    test_set = dataset["test"]
    return train_set, test_set, test_set


@cache_to_disk("data_cache")
def load_sst2():
    """Load SST-2 sentiment classification dataset from GLUE."""
    dataset = load_dataset("glue", "sst2")
    instruction = "classify the sentiment of the text: "
    label_map = {0: "negative", 1: "positive", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["sentence"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set


@cache_to_disk("data_cache")
def load_cola():
    """Load CoLA grammaticality classification dataset from GLUE."""
    dataset = load_dataset("glue", "cola")
    instruction = "classify the grammaticality of the text: "
    label_map = {0: "unacceptable", 1: "acceptable", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["sentence"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set


@cache_to_disk("data_cache")
def load_qqp():
    """Load QQP (Quora Question Pairs) semantic similarity dataset from GLUE."""
    dataset = load_dataset("glue", "qqp")
    instruction = "classify the semantic similarity of the text: "
    label_map = {0: "different", 1: "duplicate", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["question1"]}\n{e["question2"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set


@cache_to_disk("data_cache")
def load_mrpc():
    """Load MRPC (Microsoft Research Paraphrase Corpus) dataset from GLUE."""
    dataset = load_dataset("glue", "mrpc")
    instruction = "classify the semantic similarity of the text: "
    label_map = {0: "different", 1: "equivalent", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["sentence1"]}\n{e["sentence2"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set


@cache_to_disk("data_cache")
def load_mnli():
    """Load MNLI (Multi-Genre Natural Language Inference) dataset from GLUE."""
    dataset = load_dataset("glue", "mnli")
    instruction = "classify the semantic similarity of the text: "
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["premise"]}\n{e["hypothesis"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation_matched"]
    return train_set, validation_set, validation_set


@cache_to_disk("data_cache")
def load_squad():
    """Load SQuAD question answering dataset."""
    dataset = load_dataset("rajpurkar/squad")
    instruction = "answer the question: "
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["question"]}\ncontext: {e["context"]}\nresult: ',
            "y": ", ".join(e["answers"]["text"]),
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    return train_set, validation_set, validation_set


@cache_to_disk("data_cache")
def load_qnli():
    """Load QNLI (Question Natural Language Inference) dataset from GLUE."""
    dataset = load_dataset("glue", "qnli")
    instruction = "classify the semantic similarity of the question and the sentence: "
    label_map = {0: "entailment", 1: "not_entailment", -1: "other"}
    dataset = dataset.map(
        lambda e: {
            "x": f'{instruction}{e["question"]}\n{e["sentence"]}\nresult: ',
            "y": label_map[e["label"]],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    test_set = dataset["test"]
    return train_set, validation_set, test_set


# Alpaca instruction templates
template_with_input = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

template_wo_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


@cache_to_disk("data_cache")
def load_alpaca():
    """Load Alpaca instruction-following dataset."""
    dataset = load_dataset("tatsu-lab/alpaca")

    def alpaca_preprocess(instruction, input, output):
        if input == "":
            x = template_wo_input.format(instruction=instruction)
        else:
            x = template_with_input.format(instruction=instruction, input=input)
        return {"x": x, "y": output}

    dataset = dataset.map(
        lambda e: alpaca_preprocess(e["instruction"], e["input"], e["output"])
    )
    # Sample 10% of training set as validation
    train_set = dataset["train"].train_test_split(test_size=0.1)["train"]
    validation_set = dataset["train"].train_test_split(test_size=0.1)["test"]
    return train_set, validation_set, validation_set


@cache_to_disk("data_cache")
def load_gsm8k():
    """Load GSM8K math word problem dataset."""
    dataset = load_dataset("gsm8k", "main")
    dataset = dataset.map(
        lambda e: {
            "x": f'Q: {e["question"]}\nA: ',
            "y": e["answer"],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["test"]
    return train_set, validation_set, validation_set


@cache_to_disk("data_cache")
def load_alpaca_gpt4():
    """Load Alpaca GPT-4 instruction-following dataset."""
    dataset = load_dataset("tatsu-lab/alpaca")

    def alpaca_preprocess(instruction, input, output):
        if input == "":
            x = template_wo_input.format(instruction=instruction)
        else:
            x = template_with_input.format(instruction=instruction, input=input)
        return {"x": x, "y": output}

    dataset = dataset.map(
        lambda e: alpaca_preprocess(e["instruction"], e["input"], e["output"])
    )
    train_set = dataset["train"].train_test_split(test_size=0.1)["train"]
    validation_set = dataset["train"].train_test_split(test_size=0.1)["test"]
    return train_set, validation_set, validation_set


@cache_to_disk("data_cache")
def load_flan():
    """Load FLAN instruction dataset (100k train, 10k eval)."""
    dataset = load_dataset("Muennighoff/flan", split="train", streaming=True)

    def preprocess(data):
        return {
            "x": template_wo_input.format(instruction=data["inputs"]),
            "y": data["targets"],
        }

    train_samples = []
    eval_samples = []
    count = 0
    dataset = dataset.shuffle(buffer_size=5000, seed=42)
    
    from tqdm import tqdm
    for sample in tqdm(dataset, total=110000, desc="Loading FLAN"):
        processed_sample = preprocess(sample)
        if count < 100000:
            train_samples.append(processed_sample)
        elif count < 110000:
            eval_samples.append(processed_sample)
        else:
            break
        count += 1
    
    train_set = Dataset.from_list(train_samples)
    eval_set = Dataset.from_list(eval_samples)
    return train_set, eval_set, eval_set


@cache_to_disk("data_cache")
def load_meta_math_5k(max_tokens=512):
    """Load MetaMathQA dataset (5k train, 500 eval) filtered for GSM problems."""
    dataset = load_dataset("meta-math/MetaMathQA", split="train")
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    def preprocess(data):
        return {
            "x": f'Q: {data["query"]}\nA: ',
            "y": data["response"].split("\nThe answer is:")[0],
        }

    train_samples = []
    eval_samples = []
    count = 0
    dataset = dataset.shuffle(seed=42)
    
    from tqdm import tqdm
    for sample in tqdm(dataset, desc="Loading MetaMath-5k"):
        temp = preprocess(sample)
        if (len(tokenizer(temp["x"] + " " + temp["y"])["input_ids"]) >= max_tokens
            or "GSM" not in sample["type"]):
            continue
        
        if count < 5000:
            train_samples.append(temp)
        elif count < 5500:
            eval_samples.append(temp)
        else:
            break
        count += 1
    
    train_set = Dataset.from_list(train_samples)
    eval_set = Dataset.from_list(eval_samples)
    return train_set, eval_set, eval_set


@cache_to_disk("data_cache")
def load_meta_math(max_tokens=512):
    """Load MetaMathQA dataset (100k train, 10k eval) filtered for GSM problems."""
    dataset = load_dataset("meta-math/MetaMathQA", split="train")
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    def preprocess(data):
        return {
            "x": f'Q: {data["query"]}\nA: ',
            "y": data["response"].split("\nThe answer is:")[0],
        }

    train_samples = []
    eval_samples = []
    count = 0
    dataset = dataset.shuffle(seed=42)
    
    from tqdm import tqdm
    for sample in tqdm(dataset, desc="Loading MetaMath"):
        temp = preprocess(sample)
        if (len(tokenizer(temp["x"] + " " + temp["y"])["input_ids"]) >= max_tokens
            or "GSM" not in sample["type"]):
            continue
        
        if count < 100000:
            train_samples.append(temp)
        elif count < 110000:
            eval_samples.append(temp)
        else:
            break
        count += 1
    
    train_set = Dataset.from_list(train_samples)
    eval_set = Dataset.from_list(eval_samples)
    return train_set, eval_set, eval_set


@cache_to_disk("data_cache")
def load_meta_math_full(max_tokens=512):
    """Load full MetaMathQA dataset with token filtering."""
    dataset = load_dataset("meta-math/MetaMathQA", split="train")
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    def preprocess(data):
        return {
            "x": f'Q: {data["query"]}\nA: ',
            "y": data["response"].split("\nThe answer is:")[0],
        }

    train_samples = []
    eval_samples = []
    count = 0
    dataset = dataset.shuffle(seed=42)
    
    from tqdm import tqdm
    for sample in tqdm(dataset, desc="Loading MetaMath-Full"):
        temp = preprocess(sample)
        if len(tokenizer(temp["x"] + " " + temp["y"])["input_ids"]) >= max_tokens:
            continue
        
        train_samples.append(temp)
        if count < 1000:
            eval_samples.append(temp)
        count += 1
    
    train_set = Dataset.from_list(train_samples)
    eval_set = Dataset.from_list(eval_samples)
    return train_set, eval_set, eval_set


@cache_to_disk("data_cache")
def load_flan_v2(max_tokens=512):
    """Load FLAN v2 dataset with token filtering."""
    dataset = load_dataset("SirNeural/flan_v2", split="train", streaming=True)
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    def preprocess(data):
        return {
            "x": data["inputs"],
            "y": data["targets"],
        }

    train_samples = []
    eval_samples = []
    count = 0
    dataset = dataset.shuffle(buffer_size=5000, seed=42)
    
    from tqdm import tqdm
    for sample in tqdm(dataset, total=110000, desc="Loading FLAN-v2"):
        temp = preprocess(sample)
        if len(tokenizer(temp["x"] + " " + temp["y"])["input_ids"]) >= max_tokens:
            continue
        
        if count < 100000:
            train_samples.append(temp)
        elif count < 110000:
            eval_samples.append(temp)
        else:
            break
        count += 1
    
    train_set = Dataset.from_list(train_samples)
    eval_set = Dataset.from_list(eval_samples)
    return train_set, eval_set, eval_set


@cache_to_disk("data_cache")
def load_codefeedback(max_tokens=1024):
    """Load CodeFeedback instruction dataset with token filtering."""
    dataset = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    def preprocess(data):
        y = data["answer"]
        y = "```".join(y.split("```")[:2]) + "```"  # Keep only first code block
        return {
            "x": template_wo_input.format(instruction=data["query"]),
            "y": y,
        }

    train_samples = []
    eval_samples = []
    count = 0
    dataset = dataset.shuffle(seed=42)
    
    from tqdm import tqdm
    for sample in tqdm(dataset, total=110000, desc="Loading CodeFeedback"):
        if "```" not in sample["answer"]:
            continue
        
        temp = preprocess(sample)
        if len(tokenizer(temp["x"] + " " + temp["y"])["input_ids"]) >= max_tokens:
            continue
        
        if count < 100000:
            train_samples.append(temp)
        elif count < 110000:
            eval_samples.append(temp)
        else:
            break
        count += 1
    
    train_set = Dataset.from_list(train_samples)
    eval_set = Dataset.from_list(eval_samples)
    return train_set, eval_set, eval_set


@cache_to_disk("data_cache")
def load_wizardlm(max_tokens=1024):
    """Load WizardLM Chinese instruction dataset."""
    dataset = load_dataset("silk-road/Wizard-LM-Chinese-instruct-evol", split="train")
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    def preprocess(data):
        return {
            "x": template_wo_input.format(instruction=data["instruction"]),
            "y": data["output"],
        }

    train_samples = []
    eval_samples = []
    count = 0
    dataset = dataset.shuffle(seed=42)
    
    from tqdm import tqdm
    for sample in tqdm(dataset, total=70000, desc="Loading WizardLM"):
        temp = preprocess(sample)
        if "sorry" in temp["y"].lower() or "as an ai" in temp["y"].lower():
            continue
        if len(tokenizer(temp["x"] + " " + temp["y"])["input_ids"]) >= max_tokens:
            continue
        
        if count < 52000:
            train_samples.append(temp)
        elif count < 70000:
            eval_samples.append(temp)
        else:
            break
        count += 1
    
    train_set = Dataset.from_list(train_samples)
    eval_set = Dataset.from_list(eval_samples)
    return train_set, eval_set, eval_set


# Dataset registry
DATASET_MAP = {
    "sst2": load_sst2,
    "cola": load_cola,
    "qqp": load_qqp,
    "mrpc": load_mrpc,
    "mnli": load_mnli,
    "emo": load_emo,
    "squad": load_squad,
    "alpaca": load_alpaca,
    "qnli": load_qnli,
    "gsm8k": load_gsm8k,
    "alpaca_gpt4": load_alpaca_gpt4,
    "flan": load_flan,
    "flan_v2": load_flan_v2,
    "meta_math": load_meta_math,
    "meta_math_full": load_meta_math_full,
    "meta_math_5k": load_meta_math_5k,
    "codefeedback": load_codefeedback,
    "wizard_lm": load_wizardlm,
}