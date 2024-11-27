import os
import re
import torch
import transformers
from trl import SFTTrainer
from typing import Optional
from typing import Dict, Sequence
from datasets import load_dataset
from dataclasses import dataclass, field
from torch.nn.utils.rnn import pad_sequence
from transformers import HfArgumentParser, TrainingArguments, AutoTokenizer, AutoModelForCausalLM


@dataclass
class ScriptArguments:
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=2)
    per_device_eval_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=8)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[float] = field(default=0.0)
    warmup_ratio: Optional[float] = field(default=0.1)
    model_name: Optional[str] = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=3,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    max_seq_length: Optional[int] = field(default=2048)
    train_set_path: Optional[str] = field(
        default="meta-math/MetaMathQA",
        metadata={"help": "The dir of the subset of the training data to use"},
    )
    output_dir: Optional[str] = field(
        default="./baseline-Llama-3-8B-Instruct-sft",
        metadata={"help": "The dir for output model"},
    )
    save_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Save the model every x steps"},
    )
    push_to_hub: Optional[bool] = field(
        default=True,
        metadata={"help": "Push to hub."},
    )
    hub_model_id: Optional[str] = field(
        default="baseline-Llama-3-8B-Instruct-sft",
        metadata={"help": "Hub model id"},
    )
    attn_implementation: Optional[str] = field(
        default=None,#"flash_attention_2",
        metadata={"help": "Which attention implementation to use"},
    )
    train_ratio: Optional[float] = field(
        default=1.0,
        metadata={"help": "The ratio of the training data to use"},
    )


@dataclass
class DataCollatorForSFT:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        input_ids = [torch.tensor(x) for x in input_ids]
        labels = pad_sequence(input_ids, batch_first=True, padding_value=-100)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def filtered_dataset_provider(filename, tokenizer):
    dataset = load_dataset('json', data_files=filename)

    def _extract_answers(response):
        answer = re.search(r"The answer is: (.*)", response)
        if answer: return answer.group(1)
        elif answer := re.search(r"#### (\-?[0-9\.,]+)", response): return answer.group(1)
        else: return re.split(r'(####)', response, maxsplit=1)[0] + '####' + re.split(r'(####)', response, maxsplit=1)[2].split('####')[0] if '####' in response else response

    def _train_data_preprocess_fn(example):
        inputs = example['Question']
        answers = example['Answers']
        qa_pairs = list()
        labels = list()
        for q, answer in zip(inputs, answers):
            answer = [_extract_answers(a) for a in answer]
            qa_pair = [f"Question: {q}\nSolution: {a}" for a in answer]
            qa_pairs.extend(qa_pair)
            labels.extend([a for a in answer])
        # model_inputs = tokenizer(qa_pairs)
        model_inputs = dict()
        model_inputs["text"] = qa_pairs
        return model_inputs
    
    tokenized_dataset = dataset.map(_train_data_preprocess_fn, batched=True, remove_columns=dataset['train'].column_names)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data_collator = DataCollatorForSFT(tokenizer)
    return tokenized_dataset, data_collator 


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    weight_decay=script_args.weight_decay,
    warmup_ratio=script_args.warmup_ratio,
    do_eval=False,
    eval_strategy="no",
    save_strategy="epoch",
    save_steps=script_args.save_every_steps,
    overwrite_output_dir=True,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    gradient_checkpointing_kwargs={'use_reentrant':False},
    deepspeed=script_args.deepspeed,
    remove_unused_columns=True,
    bf16=script_args.bf16,
    log_level="info",
    logging_strategy="steps",
    logging_steps=1,
    push_to_hub=script_args.push_to_hub,
    hub_model_id=script_args.hub_model_id,
    report_to="none"
)

model_kwargs = dict(
    attn_implementation=script_args.attn_implementation,
    torch_dtype=torch.bfloat16,
    use_cache=False if script_args.gradient_checkpointing else True,
)

def cot_prefix(sample):
    sample["text"] = 'Question: ' + sample["question"] + ' Answer: ' + sample["answer"]
    return sample

# train_dataset = load_dataset(script_args.train_set_path, split="train").shuffle(seed=42)
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
train_dataset, data_collator = filtered_dataset_provider(script_args.train_set_path, tokenizer)
train_dataset = train_dataset["train"].shuffle(seed=42)
if script_args.train_ratio < 1.0:
    size = int(len(train_dataset) * script_args.train_ratio)
    train_dataset = train_dataset.select(range(size))
# column_names = list(train_dataset.features)
# train_dataset = train_dataset.map(cot_prefix, remove_columns=column_names, num_proc=16)
trainer = SFTTrainer(
    model=script_args.model_name,
    model_init_kwargs=model_kwargs,
    args=training_args,
    max_seq_length=script_args.max_seq_length,
    train_dataset=train_dataset,
    # data_collator=data_collator,
    dataset_text_field="text",
    packing=True,
)
# trainer.model.to(trainer.args.device)
train_result = trainer.train()
metrics = train_result.metrics
metrics["train_samples"] = len(train_dataset)
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
trainer.save_model(training_args.output_dir)
print(f"Model saved to {training_args.output_dir}")

if trainer.accelerator.is_main_process:
    trainer.model.config.use_cache = True
    trainer.model.config.save_pretrained(training_args.output_dir)
