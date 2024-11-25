import os
import json
import hydra
import multiprocessing
from dataclasses import dataclass
from omegaconf import DictConfig
from typing import Any, Dict, List
from vllm import LLM, SamplingParams, EngineArgs
from torch.utils.data import Dataset 
from utils import DATASET_PROVIDERS, FEWSHOT_PROMPTS, static_verification


def vllm_sample_ddp(
    model_name_or_path: str,
    dtype: str,
    datasets: Dict[str, Dataset],
    sample_sizes: Dict[str, int],
    sample_params: Dict[str, Any],
    sample_batch_size: int = 1,
    cutoff: int = 10,
    rank: int = 0,
    world_size: int = 1,
    output_file: str = "sample_output.jsonl",
) -> Dict[str, List]:
    collections = {}

    for key, value in datasets.items():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        llm = LLM(model=model_name_or_path, dtype=dtype)
        total_size = len(value)
        per_worker_size = total_size // world_size
        start_idx = rank * per_worker_size
        end_idx = total_size if rank == world_size - 1 else (rank + 1) * per_worker_size
        distributed_dataset = value[start_idx:end_idx]
        sample_params["n"] = sample_sizes[key]
        sample_params = SamplingParams(**sample_params)
        batches = [(prompt, label) for prompt, _, label in distributed_dataset]
        batch_index = 0

        while batch_index < len(batches):
            sample_batch_size = min(sample_batch_size, len(batches) - batch_index)
            batch = batches[batch_index:batch_index+sample_batch_size]
            prompts, labels = zip(*batch)
            if rank == 0:
                print(f"SAMPLING THE ({batch_index+1}-{batch_index+sample_batch_size}) EXAMPLES FROM {key} DATASET...")
            batch_outputs = llm.generate(prompts, sample_params)

            for i, single_outputs in enumerate(batch_outputs):
                text_outputs = [output.text for output in single_outputs.outputs]
                reward = static_verification(text_outputs, labels[i])
                verified_outputs = [output.text for output, r in zip(single_outputs.outputs, reward) if r]
                stop_string = "I hope it is correct."
                for i, output in enumerate(verified_outputs):
                    if stop_string in output:
                        index = output.find(stop_string)
                        verified_outputs[i] = output[:index-1]

                if len(verified_outputs) > 0:
                    verified_outputs = verified_outputs[:min(cutoff, len(verified_outputs))]
                    collections[single_outputs.prompt[len(FEWSHOT_PROMPTS[key]):]] = verified_outputs

            batch_index += sample_batch_size
    
    filename = output_file.split(".")[:-1]
    filename = ".".join(filename)
    filetype = output_file.split(".")[-1]
    rank_output_file = f"{filename}_{rank}.{filetype}"
    with open(rank_output_file, "w") as f:
        json.dump(collections, f, indent=4)


def vllm_sample(
    llm: LLM,
    datasets: Dict[str, Dataset],
    sample_sizes: Dict[str, int],
    sample_params: Dict[str, Any],
    sample_batch_size: int = 1,
    cutoff: int = 10,
) -> Dict[str, List]:
    collections = {}

    for key, value in datasets.items():
        sample_params["n"] = sample_sizes[key]
        sample_params = SamplingParams(**sample_params)
        batches = [(prompt, label) for prompt, _, label in value]
        batch_index = 0

        while batch_index < len(batches):
            sample_batch_size = min(sample_batch_size, len(batches) - batch_index)
            batch = batches[batch_index:batch_index+sample_batch_size]
            prompts, labels = zip(*batch)
            print(f"SAMPLING THE ({batch_index+1}-{batch_index+sample_batch_size}) EXAMPLES FROM {key} DATASET...")
            batch_outputs = llm.generate(prompts, sample_params)

            for i, single_outputs in enumerate(batch_outputs):
                text_outputs = [output.text for output in single_outputs.outputs]
                reward = static_verification(text_outputs, labels[i])
                verified_outputs = [output.text for output, r in zip(single_outputs.outputs, reward) if r]
                stop_string = "I hope it is correct."
                for i, output in enumerate(verified_outputs):
                    if stop_string in output:
                        index = output.find(stop_string)
                        verified_outputs[i] = output[:index-1]

                if len(verified_outputs) > 0:
                    verified_outputs = verified_outputs[:min(cutoff, len(verified_outputs))]
                    collections[single_outputs.prompt[len(FEWSHOT_PROMPTS[key]):]] = verified_outputs

            batch_index += sample_batch_size

    return collections

    
@hydra.main(config_path="../configs", config_name="sample_config", version_base="1.2")
def main(cfg: DictConfig):
    sample_params = {
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "repetition_penalty": cfg.repetition_penalty,
    }
    datasets = {}
    sample_sizes = {}
    for dataset_name, dataset_configs in cfg.dataset_kwargs.items():
        dataset = DATASET_PROVIDERS[dataset_name](tokenizer=None)
        train_dataset = dataset["train"]
        if dataset_configs.problem_size > 0 and dataset_configs.problem_size < len(train_dataset):
            train_dataset = train_dataset.sample(dataset_configs.problem_size)
        if dataset_configs.num_partitions > 1:
            size_per_partition = len(train_dataset) // dataset_configs.num_partitions
            start_idx = dataset_configs.partition_id * size_per_partition
            end_idx = len(train_dataset) if dataset_configs.partition_id == dataset_configs.num_partitions - 1 else (dataset_configs.partition_id + 1) * size_per_partition
            train_dataset = train_dataset[start_idx:end_idx]
        print(f"DATASET: {dataset_name} | PROBLEM SIZE: {len(train_dataset)} | SAMPLE SIZE: {dataset_configs.sample_size} | BATCH SIZE: {cfg.sample_batch_size} | DP SIZE: {cfg.world_size}")
        print(f"PER WORKER SIZE: {len(train_dataset) // cfg.world_size}")
        datasets[dataset_name] = train_dataset
        sample_sizes[dataset_name] = dataset_configs.sample_size

    if cfg.world_size > 1:
        process = []
        for rank in range(cfg.world_size):
            p = multiprocessing.Process(target=vllm_sample_ddp, args=(cfg.model_name_or_path, cfg.model_dtype, datasets, sample_sizes, sample_params, cfg.sample_batch_size, cfg.cutoff, rank, cfg.world_size, cfg.output_file))
            p.start()
            process.append(p)
        
        for p in process:
            p.join()

        combined_collections = {}
        for rank in range(cfg.world_size):
            filename = cfg.output_file.split(".")[:-1]
            filename = ".".join(filename)
            filetype = cfg.output_file.split(".")[-1]
            rank_output_file = f"{filename}_{rank}.{filetype}"
            with open(rank_output_file, "r") as f:
                data = json.load(f)
                combined_collections.update(data)
        
        with open(cfg.output_file, "w") as f:
            json.dump(combined_collections, f, indent=4)

    else:
        collections = vllm_sample(
            model_name_or_path=cfg.model_name_or_path,
            dtype=cfg.model_dtype,
            datasets=datasets,
            sample_sizes=sample_sizes,
            sample_params=sample_params,
            sample_batch_size=cfg.sample_batch_size,
            cutoff=cfg.cutoff,
        )

        with open(cfg.output_file, "w") as f:
            json.dump(collections, f, indent=4)

    with open(cfg.output_file, "r") as f:
        data = json.load(f)

    transformed_data = list()
    for question_prompt, answers in data.items():
        # Separate question and solution part from the prompt
        question = question_prompt.split("Solution:")[0].split("Question:")[1].strip()
        transformed_data.append({
            "Question": question,
            "Answers": answers
        })

    with open(cfg.output_file, "w") as f:
        for item in transformed_data:
            f.write(json.dumps(item) + "\n")
        
            
if __name__ == '__main__':
    main()


