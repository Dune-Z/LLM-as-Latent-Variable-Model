import os
import re
import json
import time
import torch
import openai
import random
import asyncio
import pathlib
import torch.utils
import transformers
import multiprocessing
from random import shuffle
from datasets import Dataset, load_dataset
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
MATH_FEWSHOT_PROMPT = r"""The following examples demonstrate how to solve various math problems step by step. For each problem, the solution should begin by identifying the key elements and then proceed with a logical sequence of steps to find the answer. The final answer should be clearly highlighted using $\\boxed{}$.
Question: A positive multiple of 45 less than 1000 is randomly selected. What is the probability that it is a two-digit integer? Express your answer as a common fraction.

Solution: The positive multiples of 45 are  \\[45,90,135,\\ldots,990=1\\cdot45,2\\cdot45,3\\cdot45,\\ldots,22\\cdot45.\\] There are 22 multiples on this list. Every positive multiple of 45 less than 1000 is either a two-digit integer or a three-digit integer. Out of the $99-10+1=90$ two-digit integers, $45$ and $90$ are multiples of 45. Therefore, the probability that the selected multiple of 45 has two digits is $2/22=\\boxed{\\frac{1}{11}}$.

Question: Factor $x^3 - 9x^2 + 27x - 35$.

Solution: We could check to see which divisors of $-35$ are roots of the cubic $x^3 - 9x^2 + 27x - 35 = 0$.\n\nHowever, notice that $x^3 - 9x^2 + 27x - 35 = (x - 3)^3 - 2^3$. As such, we can factor this as a difference of cubes: $(x-3)^3 - 2^3 = ((x-3)-2)((x-3)^2+2(x-3)+2^2) = (x-5)(x^2-4x+7)$.\n\nWe see that $x^2-4x+7$ cannot be factored any further, so our answer is $\\boxed{(x-5)(x^2-4x+7)}$.

"""

GSM8K_FEWSHOT_PROMPT = r"""The following examples demonstrate how to solve various math problems step by step. For each problem, the solution should begin by identifying the key elements and then proceed with a logical sequence of steps to find the answer. The final answer should be clearly highlighted after '#### ' in the solution.
Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?

Solution: How many bolts of white fiber does it take? ** It takes 2/2=<<2/2=1>>1 bolt of white fiber How many bolts in total does it take? ** So the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric. #### 3

Question: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?

Solution: The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000 He increased the value of the house by 80,000*1.5=<<80000*1.5=120000>>120,000 So the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000 So he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000. #### 70,000

"""

FEWSHOT_PROMPTS = {
    "MATH": MATH_FEWSHOT_PROMPT,
    "GSM8K": GSM8K_FEWSHOT_PROMPT,
}

GENERATION_PROMPT = """Question: {question}

Solution: """


def load_client():
    setattr(openai, "api_key", "2NLsETn3aHwZCumtAjIrB4E82erOvtGp")
    setattr(openai, "base_url", "https://azure-openai-api.shenmishajing.workers.dev/v1/")
    return openai


CLIENT = load_client()

    
class MathDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Dict], tokenizer: Optional[transformers.PreTrainedTokenizer]):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = MATH_FEWSHOT_PROMPT + GENERATION_PROMPT.format(question=item["problem"])
        inputs = self.tokenizer(prompt) if self.tokenizer is not None else None
        label = item["label"]
        label = self.tokenizer(label)['input_ids'] if self.tokenizer is not None else label
        return prompt, inputs, label

    def sample(self, size: int):
        sampled_indices = random.sample(range(len(self.data)), size)
        sampled_items = [self[i] for i in sampled_indices]
        return sampled_items

        
class GSM8KDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, name: str, split: str, tokenizer: Optional[transformers.PreTrainedTokenizer] = None):
        self.datasets = load_dataset(path, name)[split]
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.datasets)
    
    def __getitem__(self, idx):
        item = self.datasets[idx]
        prompt = GSM8K_FEWSHOT_PROMPT + GENERATION_PROMPT.format(question=item["question"])
        inputs = self.tokenizer(prompt) if self.tokenizer is not None else None
        label = item["answer"].split("#### ")[-1]
        label = self.tokenizer(label)['input_ids'] if self.tokenizer is not None else label
        return prompt, inputs, label
    
    def sample(self, size: int):
        self.datasets.shuffle()
        sampled_items = [self[i] for i in range(size)]
        return sampled_items


def index_processed_math_dataset(
    processed_ds: Dataset,
    splits: List[str] = ["train", "test"],
):
    index = {split: {} for split in splits}
    for split in splits:
        for d in processed_ds[split]:
            index[split][d["problem"]] = d["solution"]
    return index

    
def load_json_file(file: pathlib.Path) -> Dict:
    with open(file, "r") as f:
        return json.load(f)


def math_dataset_provider(
    path: str = "datasets/MATH",
    splits: List[str] = ["train", "test"],
    tokenizer: transformers.PreTrainedTokenizer = None,
) -> Dict[str, List]:
    ds = {s: list() for s in splits}
    processed_ds = load_dataset("gohsyi/math")
    processed_index = index_processed_math_dataset(processed_ds, splits)
    datasets = {split: list() for split in splits}
    for split in splits:
        split_path = os.path.join(path, split)
        fields = [f for f in pathlib.Path(split_path).iterdir() if f.is_dir()]

        # Iterate through each field and process the JSON files in parallel
        with ThreadPoolExecutor() as executor:
            for field in fields:
                json_files = [file for file in pathlib.Path(field).iterdir() if file.suffix == ".json"]
                
                # Load all JSON files concurrently
                json_data_list = list(executor.map(load_json_file, json_files))
                
                for data in json_data_list:
                    # Find the corresponding solution using the index
                    if data["problem"] in processed_index[split]:
                        data["label"] = processed_index[split][data["problem"]]
                        # remove "level", "type" keys from the dict
                        data.pop("level", None)
                        data.pop("type", None)
                        ds[split].append(data)
                    else:
                        raise ValueError(f"Could not find the corresponding data in the processed dataset for problem: {data['problem']}")
        
        datasets[split] = MathDataset(ds[split], tokenizer)

    return datasets


def gsm8k_dataset_provider(
    path: str = "openai/gsm8k",
    name: str = "main",
    splits: List[str] = ["train", "test"],
    tokenizer: transformers.PreTrainedTokenizer = None,
):
    datasets = {split: GSM8KDataset(path=path, name=name, split=split, tokenizer=tokenizer) for split in splits}
    return datasets

    
def filtered_math_dataset_provider(filename: str, tokenizer: transformers.PreTrainedTokenizer):
    dataset = load_dataset('json', data_files=filename)

    def _train_data_preprocess_fn(example):
        inputs, labels = list(), list()
        for question, answers in example.items():
            for answer in answers[0]:
                inputs.append(f"{MATH_FEWSHOT_PROMPT}{question}{answer}")
                labels.append(answer)

        model_inputs = tokenizer(inputs)
        labels_tokenized = tokenizer(labels)["input_ids"]
        model_inputs["labels"] = labels_tokenized
        return model_inputs
    
    tokenized_dataset = dataset.map(_train_data_preprocess_fn, batched=True, remove_columns=dataset['train'].column_names)
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
    # test_data = math_dataset_provider(splits=["test"], tokenizer=tokenizer)['test']
    # inputs = [d[1] for d in test_data][:10]
    # labels = [d[2] for d in test_data][:10]
    # input_keys = inputs[0].keys()
    # inputs = {k: [d[k] for d in inputs] for k in input_keys}
    # inputs["labels"] = labels
    # test_dataset = Dataset.from_dict(inputs)
    # tokenized_dataset['test'] = test_dataset

    return tokenized_dataset, data_collator


DATASET_PROVIDERS = {
    "MATH": math_dataset_provider,
    "GSM8K": gsm8k_dataset_provider,
}

    
def model_provider(cfg) -> Tuple[transformers.PreTrainedTokenizer, transformers.PreTrainedModel]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        _attn_implementation=cfg.attention_impl,
        torch_dtype=torch.bfloat16 if cfg.trainer.bf16 else torch.float32,
    )
    tokenizer.padding_side = "left"
    return tokenizer, model

    
def static_verification(
    text_outputs: List[str],
    label: str,
):
    results = list()
    for text_output in text_outputs:
        # boxed_text = text_output.split("\\boxed{")[-1].split("}")[0]
        boxed_text = text_output.split("#### ")[-1]
        results.append(boxed_text == label)

    return results


def verification(
    text_outputs: List[str],
    label: str,
    client: openai.Client,
    model_name: str = "gpt-4-turbo",
) -> List[bool]:
    results = list()
    contents = list()
    prompts = [
        {
            "role": "system",
            "content": f"""
            The following text contains a few-shot prompt followed by a math problem, reasoning process, and a conclusion. 
            Please verify if the final solution matches the true answer provided below.

            Text:
            {text_output}

            True Answer:
            {label}

            Does the solution in the text match the true answer? Answer with 'yes, the solution "..." matches the true answer "..."' or 'no, the solution ... does not match the true answer ...'.
            """
        } for text_output in text_outputs
    ]

    for prompt in prompts:
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=128,
            temperature=0.0,
            messages=[prompt]
        )
        results.append(response.choices[0].message.content.lower().startswith("yes"))
        contents.append(response.choices[0].message.content)

    if not any(results):
        print("ALL SOLUTIONS IN THIS PROBLEM ARE INCORRECT: ")
        for content in contents:
            print("\t- " + content)

    return results

    
def single_verification(
    text_output: str,
    label: str,
    client: openai.Client,
    model_name: str = "gpt-4-turbo",
    max_retries: int = 5,
    initial_delay: float = 32,
) -> bool:
    prompt = {
        "role": "system",
        "content": f"""
        The following text contains a few-shot prompt followed by a math problem, reasoning process, and a conclusion. 
        Please verify if the final solution matches the true answer provided below.

        Text:
        {text_output}

        True Answer:
        {label}

        Does the solution in the text match the true answer? Answer with 'yes, the solution "..." matches the true answer "..."' or 'no, the solution ... does not match the true answer ...'.
        """
    }
    # response = client.chat.completions.create(
    #     model=model_name,
    #     max_tokens=128,
    #     temperature=0.0,
    #     messages=[prompt]
    # )
    # result = response.choices[0].message.content.lower().startswith("yes")
    delay = initial_delay
    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                max_tokens=128,
                temperature=0.0,
                messages=[prompt]
            )
            result = response.choices[0].message.content.lower().startswith("yes")
            return result
        except openai.RateLimitError:
            print(f"Rate limit exceeded. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            raise e
    
    return result


def batch_verification(
    text_outputs: List[str],
    label: str,
    client: openai.Client,
    model_name: str = "gpt-4-turbo",
    num_workers: int = multiprocessing.cpu_count(),
):
    results = [None] * len(text_outputs)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_index = {
            executor.submit(
                single_verification,
                text_outputs[i],
                label,
                client,
                model_name
            ): i for i in range(len(text_outputs))
        }
        
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:
                print(f"An error occurred for index {index}: {e}")
                results[index] = False
        
    return results
