import torch
import hydra
import transformers
from tqdm import tqdm
from datasets import Dataset
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments, GenerationConfig
from utils import model_provider, filtered_math_dataset_provider


class RestEMTrainer(Trainer):
    def compute_loss(self, model, input, return_outputs=False):
        labels = input.pop("labels")
        outputs = model(**input)
        logits = outputs.logits[..., -labels.shape[-1]:, :]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )
        return (loss, outputs) if return_outputs else loss


@hydra.main(config_path="../configs", config_name="train_config", version_base="1.2")
def main(cfg: DictConfig):
    tokenizer, model = model_provider(cfg)
    model.config._attn_implementation = cfg.attention_impl
    dataset, data_collator = filtered_math_dataset_provider(cfg.dataset_path, tokenizer)
    training_args = TrainingArguments(**cfg.trainer)
    trainer = RestEMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        data_collator=data_collator,
        args=training_args,
    )
    trainer.train()


if __name__ == '__main__':
    main()
