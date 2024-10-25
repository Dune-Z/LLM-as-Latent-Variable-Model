import torch
import hydra
import transformers
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments
from utils import model_provider, filtered_math_dataset_provider


class RestEMTrainer(Trainer):
    def compute_loss(self, model, input, return_outputs=False):
        labels = input.pop("labels")
        outputs = model(**input)
        logits = outputs.logits[..., -labels.shape[-1]:, :]
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )
        return (loss, outputs) if return_outputs else loss


@hydra.main(config_path="../configs", config_name="train_config", version_base="1.2")
def main(cfg: DictConfig):
    tokenizer, model = model_provider(cfg)
    model.config._attn_implementation = cfg.attention_impl
    train_dataset, data_collator = filtered_math_dataset_provider(cfg.dataset_path, tokenizer)
    training_args = TrainingArguments(**cfg.trainer)
    trainer = RestEMTrainer(
        model=model,
        data_collator=data_collator,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )
    trainer.train()


if __name__ == '__main__':
    main()
