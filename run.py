# -*- coding: utf-8 -*-
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig
from trl import SFTTrainer
import pandas as pd
import glob

# Ensure the Hugging Face token is securely handled
HUGGINGFACE_TOKEN = os.getenv("HF_AUTH_TOKEN", "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")

# Path to dataset
data_path = os.getenv("DATA_PATH", "/home/datpham/datpham/llm-for-law/datasets")

# Correctly join the path
excel_files = glob.glob(os.path.join(data_path, "*.xlsx"))
print("Excel files found:", excel_files)  # Debugging line

if not excel_files:
    raise FileNotFoundError(f"No .xlsx files found in directory: {data_path}")

all_dataframes = []

for file in excel_files:
    print(f"Processing file: {file}")  # Debugging line
    df = pd.read_excel(file)
    # Fill NaN or None values in question and answer
    df['question'] = df['question'].fillna("")
    df['answer'] = df['answer'].fillna("")
    # Create the text field
    df['text'] = '<s>[INST] ' + df['question'] + ' [/INST] ' + df['answer'] + ' </s>'
    all_dataframes.append(df)

combined_df = pd.concat(all_dataframes, ignore_index=True)

# Debugging: Check the combined dataframe
print("Sample data from combined dataframe:")
print(combined_df[['text']].head())

# Convert to Dataset
dataset = Dataset.from_pandas(combined_df[['text']])

# Model and training configuration
model_id = "vinai/PhoGPT-4B-Chat"
output_dir = os.getenv("OUTPUT_DIR", "/home/datpham/datpham/llm-for-law/output")
fine_tuned_model_dir = os.path.join(output_dir, "SaveTrained")

lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

device_map = {"": 0}

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map=device_map,
    token=HUGGINGFACE_TOKEN
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=HUGGINGFACE_TOKEN,  # Replaced use_auth_token with token
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Configure LoRA
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)

# Define training arguments
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=8,
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
    report_to="tensorboard"
)

# Tokenization function
def tokenize(batch):
    texts = batch['text']
    texts = [str(text) for text in texts]  # Ensure all inputs are strings
    return tokenizer(texts, padding=True, truncation=True, max_length=1024)

# Apply tokenization
tokenized_dataset = dataset.map(tokenize, batched=True)

# Train model
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
    max_seq_length=1024  # Explicitly set max_seq_length
)

# Start training
trainer.train()
trainer.model.save_pretrained(fine_tuned_model_dir)
tokenizer.save_pretrained(fine_tuned_model_dir)

# Inference pipeline
prompt = "Lợi dụng lúc bà B đang ngủ, An đã lén vào nhà bà B để trộm cắp tài sản, giá trị tài sản là 1 chiếc xe máy. An sẽ bị kết tội gì?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=3000)

result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'].split("[/INST]")[1])
