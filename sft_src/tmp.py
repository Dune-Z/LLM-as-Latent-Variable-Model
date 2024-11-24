import os
import multiprocessing
from vllm import LLM, SamplingParams


def worker(worker_idx):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_idx)
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    llm = LLM(model="google/gemma-2-2b-it")
    outputs = llm.generate(prompts, sampling_params)


if __name__ == "__main__":
    
    with multiprocessing.Pool(4) as pool:
        pool.map(worker, range(4))