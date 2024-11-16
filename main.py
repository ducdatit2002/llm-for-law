# -*- coding: utf-8 -*-
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import glob
import pandas as pd
import json

# Path Configuration
data_path = "/home/datpham/datpham/llm-for-law/datasets"
output_dir = "/home/datpham/datpham/llm-for-law/models/PhoGPT-4B-Chat"
log_dir = os.path.join(output_dir, "logs")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Load and Process Dataset
excel_files = glob.glob(os.path.join(data_path, "*.xlsx"))
if not excel_files:
    raise FileNotFoundError(f"No Excel files found in {data_path}")

all_dataframes = []
for file in excel_files:
    df = pd.read_excel(file)
    df["text"] = "<s>[INST] " + df["question"] + " [/INST] " + df["answer"] + " </s>"
    all_dataframes.append(df)

combined_df = pd.concat(all_dataframes, ignore_index=True)
dataset = Dataset.from_pandas(combined_df[["text"]])
print(f"Loaded {len(excel_files)} files from {data_path}. Total records: {len(combined_df)}")

# Model Configuration
model_id = "vinai/PhoGPT-4B-Chat"
use_4bit = True
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Load Model and Tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    use_auth_token=True,
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=True,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Tokenization Function
def tokenize_function(examples):
    """Tokenize the input data."""
    return tokenizer(
        examples["text"],  # Process batch of text
        truncation=True,
        padding="max_length",
        max_length=1024,
    )

# Tokenize Dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# LoRA Configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training Configuration
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=150,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    logging_dir=log_dir,
)

# SFT Trainer Configuration
sft_config = SFTConfig(
    peft_config=peft_config,
    max_seq_length=1024,
)

# Fine-Tuning Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    args=training_args,
    sft_config=sft_config,
)

# Training and Saving
trainer.train()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")

# Gather Metrics
hyperparams = {
    "learning_rate": training_args.learning_rate,
    "train_batch_size": training_args.per_device_train_batch_size,
    "num_train_epochs": training_args.num_train_epochs,
    "weight_decay": training_args.weight_decay,
}
training_metrics = trainer.state.log_history

# Save Training Report
report = {
    "hyperparameters": hyperparams,
    "training_metrics": training_metrics,
}
with open(os.path.join(output_dir, "training_report.json"), "w") as f:
    json.dump(report, f, indent=4)
print(f"Training report saved to {os.path.join(output_dir, 'training_report.json')}")

# TensorBoard Visualization
print(f"To visualize training logs, use: tensorboard --logdir {log_dir}")
