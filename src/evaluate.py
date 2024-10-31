import torch
import hydra
from datasets import Dataset
from omegaconf import DictConfig
from vllm import LLM, SamplingParams
from utils import DATASET_PROVIDERS
from typing import Dict


def vllm_generation(
    llm: LLM,
    max_tokens: int,
    eval_datasets: Dict[str, Dataset],
    generation_batch_size: int = 1,
):

    correct_count = 0
    total_count = 0
    for dataset_name, eval_dataset in eval_datasets.items():
        total_count += len(eval_dataset)
        batches = [(prompt, label) for prompt, _, label in eval_dataset]
        print(f"TOTAL EXAMPLES: {len(batches)}")
        sample_params = SamplingParams(n=1, max_tokens=max_tokens, temperature=1.0)
        batch_index = 0

        while batch_index < len(batches):
            generation_batch_size = min(generation_batch_size, len(batches) - batch_index)
            batch = batches[batch_index:batch_index+generation_batch_size] 
            prompts, labels = zip(*batch)
            print(f"GENERATING THE ({batch_index+1}-{batch_index+generation_batch_size}) EXAMPLES...")
            batch_outputs = llm.generate(prompts, sample_params)

            for i, single_outputs in enumerate(batch_outputs):
                prompt = single_outputs.prompt
                text_output = [prompt + output.text for output in single_outputs.outputs][0]
                if dataset_name == "GSM8K":
                    answer = text_output.split("#### ")[-1]
                elif dataset_name == "MATH":
                    answer = text_output.split("\\boxed{")[-1].split("}")[0]
                if answer == labels[i]:
                    correct_count += 1

            batch_index += generation_batch_size

    acc = correct_count / total_count
    return acc


@hydra.main(config_path="../configs", config_name="eval_config", version_base="1.2")
def main(cfg: DictConfig):
    llm = LLM(cfg.model_name_or_path, dtype=cfg.model_dtype)
    batch_size = cfg.batch_size
    test_datasets = {}
    for dataset in cfg.datasets:
        test_dataset = DATASET_PROVIDERS[dataset](splits=['test'], tokenizer=None)['test']
        test_size = len(test_dataset)
        if cfg.eval_ratio < 1.0:
            test_dataset = test_dataset.sample(size=int(test_size * cfg.eval_ratio))
            batch_size = min(cfg.batch_size, len(test_dataset))
        test_datasets[dataset] = test_dataset
    acc = vllm_generation(
        llm=llm,
        max_tokens=cfg.max_tokens,
        eval_datasets=test_datasets,
        generation_batch_size=batch_size
    )
    with open(cfg.output_file, "w") as f:
        f.write(f"{acc}")

    
if __name__ == "__main__":
    main()
