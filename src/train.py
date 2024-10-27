import torch
import hydra
import transformers
from tqdm import tqdm
from datasets import Dataset
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments, GenerationConfig
from utils import model_provider, filtered_math_dataset_provider


def compute_metrics(pred, tokenizer):
    import numpy as np
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    text_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoder_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)


@torch.no_grad()
def evaluate(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    data_collator: transformers.DataCollatorForSeq2Seq,
    eval_dataset: Dataset,
    eval_batch_size: int = 1,
):
    model.eval()
    if not next(model.parameters()).is_cuda:
        model.to("cuda")
    model = model.half()
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=data_collator,
    )
    generation_config = GenerationConfig(do_sample=False, max_length=2048)
    correct_count = 0

    for batch in tqdm(eval_dataloader, "Evaluating"):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            synced_gpus=False
        )
        batch_predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        batch_references = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        for pred, ref in zip(batch_predictions, batch_references):
            answer = pred.split("\\boxed{")[-1].split("}")[0]
            if answer == ref:
                correct_count += 1
    
    acc = correct_count / len(eval_dataset)
    return acc
    

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
    acc = evaluate(
        model=trainer.model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        eval_dataset=dataset["test"],
        eval_batch_size=1,
    )
    print(f"Accuracy: {acc}")
    trainer.train()


if __name__ == '__main__':
    main()
