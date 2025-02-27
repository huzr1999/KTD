import os
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import os 


import argparse
from lib.utils import get_input_col_name, get_original_checkpoint

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True, choices=["gpt2", "pythia", "gpt-neo", "pythia-410m", "pythia-1b", "pythia-2.8b", "pythia-6.9b", "mistralai"])
parser.add_argument("--uselora", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--bp16", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--batch_size", default=2, type=int)


args = parser.parse_args()

dataset_name = args.dataset
model_name = args.model
use_lora = args.uselora
num_train_epochs = args.epochs
batch_size = args.batch_size
bp16 = args.bp16

checkpoint = get_original_checkpoint(model_name)
input_column_name = get_input_col_name(dataset_name)

ds = load_from_disk(f"./datasets/{dataset_name}")
train_set = ds.filter(lambda example: example['label'] == 1)



tokenizer = AutoTokenizer.from_pretrained(checkpoint)
if bp16:
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16)
else:
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

def tokenizer_function(example):
    return tokenizer(example[input_column_name], truncation=True)

tokenized_set = train_set.map(tokenizer_function, batched=True, remove_columns=train_set.column_names)


peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
)

if use_lora:
    model = get_peft_model(model, peft_config)



data_collactor = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

finetuned_model_path=f"/path/to/finetuned_models/{model_name}-finetuned/{dataset_name}-finetuned"

training_args = TrainingArguments(
    output_dir=finetuned_model_path,
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    eval_strategy="no",
    save_strategy="no",
    eval_steps=100,
    save_steps=100,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    bf16=True,
    optim="adamw_hf", 
    gradient_checkpointing=True,
    logging_strategy="steps",        
    logging_steps=10,                
    report_to=["tensorboard"],  
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_set,
    eval_dataset=tokenized_set,
    data_collator=data_collactor,
)


trainer.train()

import json

eval_results = trainer.evaluate()

import os
import shutil

    
saved_dir_name = os.path.join(finetuned_model_path, "final_model")

if os.path.exists(saved_dir_name):
    shutil.rmtree(saved_dir_name)

os.makedirs(saved_dir_name)
        
model.save_pretrained(saved_dir_name, safe_serialization=True)
tokenizer.save_pretrained(saved_dir_name, safe_serialization=True)

with open(f'{model_name}-{dataset_name}-eval_results.json', 'w') as f:
    json.dump(eval_results, f, indent=4)
