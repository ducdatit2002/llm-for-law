# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset as HFDataset  # Ensure 'datasets' is correctly imported

# Define the data path (update this to your new local directory path)
data_path = '/home/datpham/datpham/llm-in-law/datasets'  # Changed from 'datasets' to 'data'

# Function to load text data from folder
def load_data_from_folder(data_folder):
    texts = []
    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)

        if file_name.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())

        elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            # Read Excel file and convert content to string
            df = pd.read_excel(file_path)
            excel_text = " ".join(df.astype(str).values.flatten())
            texts.append(excel_text)

    return texts

# Load dataset
texts = load_data_from_folder(data_path)
dataset = HFDataset.from_dict({"text": texts})
print(f"Loaded {len(texts)} files from {data_path}")

# Model and tokenizer setup
model_name = "vinai/PhoGPT-4B-Chat"  # Replace with your specific model path
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token="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")
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token="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")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",  # For TensorBoard
    logging_steps=10,      # Log every 10 steps
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train model and log metrics
trainer.train()

# Save final model and tokenizer
model_save_path = "/home/datpham/datpham/llm-in-law/models/PhoGPT-4B-Chat"  # Changed to 'models'
os.makedirs(model_save_path, exist_ok=True)  # Ensure the directory exists
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")

# Gather hyperparameters and training metrics
hyperparams = {
    "learning_rate": training_args.learning_rate,
    "train_batch_size": training_args.per_device_train_batch_size,
    "eval_batch_size": training_args.per_device_eval_batch_size,
    "num_train_epochs": training_args.num_train_epochs,
    "weight_decay": training_args.weight_decay
}

training_metrics = trainer.state.log_history
eval_metrics = trainer.evaluate()

# Save hyperparameters, training metrics, and evaluation metrics to JSON
report = {
    "hyperparameters": hyperparams,
    "training_metrics": training_metrics,
    "evaluation_metrics": eval_metrics
}

with open("training_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("Training report saved to training_report.json")

# Optional TensorBoard command
print("To visualize training with TensorBoard, use: tensorboard --logdir ./logs")
