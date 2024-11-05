import torch
import hydra
import transformers
from tqdm import tqdm
from datasets import Dataset
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments
from utils import model_provider, filtered_dataset_provider


@hydra.main(config_path="../configs", config_name="train_config", version_base="1.2")
def main(cfg: DictConfig):
    tokenizer, model = model_provider(cfg)
    model.config._attn_implementation = cfg.attention_impl
    dataset, data_collator = filtered_dataset_provider(cfg.dataset_path, tokenizer)
    training_args = TrainingArguments(**cfg.trainer)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        data_collator=data_collator,
        args=training_args,
    )
    trainer.train()


if __name__ == '__main__':
    main()
