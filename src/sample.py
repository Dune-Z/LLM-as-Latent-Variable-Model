import json
import hydra
from openai import OpenAI
from omegaconf import DictConfig
from typing import Any, Dict, List
from torch.utils.data import Dataset
from vllm import LLM, SamplingParams
from utils import (
    CLIENT,
    DATASET_PROVIDERS,
    FEWSHOT_PROMPT,
    verification,
    batch_verification
)


def vllm_sample(
    llm: LLM,
    client: OpenAI,
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
            batch = batches[batch_index:batch_index+sample_batch_size]
            prompts, labels = zip(*batch)
            print(f"SAMPLING THE ({batch_index+1}-{batch_index+sample_batch_size}) EXAMPLES FROM {key} DATASET...")
            batch_outputs = llm.generate(prompts, sample_params)

            for i, single_outputs in enumerate(batch_outputs):
                prompt = single_outputs.prompt
                text_outputs = [prompt + output.text for output in single_outputs.outputs]
                reward = batch_verification(text_outputs, labels[i], client)
                # reward = verification(text_outputs, labels[i], client)
                verified_outputs = [output.text for output, r in zip(single_outputs.outputs, reward) if r]

                if len(verified_outputs) > 0:
                    verified_outputs = verified_outputs[:min(cutoff, len(verified_outputs))]
                    collections[single_outputs.prompt[len(FEWSHOT_PROMPT):]] = verified_outputs

            batch_index += sample_batch_size

    return collections

    
@hydra.main(config_path="../configs", config_name="sample_config", version_base="1.2")
def main(cfg: DictConfig):
    llm = LLM(model=cfg.model_name_or_path, dtype=cfg.model_dtype)
    sample_params = {
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
    }
    datasets = {}
    sample_sizes = {}
    for dataset_name, dataset_configs in cfg.dataset_kwargs.items():
        train_dataset, _ = DATASET_PROVIDERS[dataset_name](tokenizer=None)
        train_dataset = train_dataset.sample(dataset_configs.problem_size)
        datasets[dataset_name] = train_dataset
        sample_sizes[dataset_name] = dataset_configs.sample_size
        
    collections = vllm_sample(
        llm=llm,
        client=CLIENT,
        datasets=datasets,
        sample_sizes=sample_sizes,
        sample_params=sample_params,
        sample_batch_size=cfg.sample_batch_size,
    )

    with open(cfg.output_file, "w") as f:
        json.dump(collections, f, indent=4)

    from torch.distributed import destroy_process_group
    destroy_process_group()

            
if __name__ == '__main__':
    main()

