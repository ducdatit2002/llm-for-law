# -*- coding: utf-8 -*-
!pip install -q accelerate==0.21.0
!pip install peft==0.4.0
!pip install bitsandbytes==0.40.2
!pip install transformers==4.31.0
!pip install trl==0.4.7
!pip install datasets
import os
import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
import glob
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
from google.colab import files
import pandas as pd
from torch import cuda, bfloat16
import json

import pandas as pd
# Define the data path (update this to your new local directory path)
data_path = '/home/datpham/datpham/llm-in-law/datasets'  # Changed from 'datasets' to 'data'
excel_files = glob.glob(data_path + "*.xlsx")
# Function to load text data from folder
all_dataframes = []

# Iterate through each Excel file and read it into a DataFrame
for file in excel_files:
    df = pd.read_excel(file)
    # Process the DataFrame as needed (e.g., clean data, add columns)
    df['text'] = '<s>[INST] ' + df['question'] + ' [/INST] ' + df['answer'] + ' </s>'
    all_dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(all_dataframes, ignore_index=True)
# Convert the combined DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(combined_df[['text']])
dataset
print(f"Loaded {len(excel_files)} files from {data_path}")

# Model and tokenizer setup
model_id = "vinai/PhoGPT-4B-Chat"  # Replace with your specific model path

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token="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")
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token="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")

#LoRA settings for modifying attention mechanisms
lora_r = 64  # Dimension of LoRA attention
lora_alpha = 16  # Scaling factor for LoRA
lora_dropout = 0.1  # Dropout rate in LoRA layers

#4-bit precision settings for model efficiency
use_4bit = True  # Enable 4-bit precision
bnb_4bit_compute_dtype = "float16"  # Data type for computations
bnb_4bit_quant_type = "nf4"  # Quantization method
use_nested_quant = False  # Enable double quantization for more compression

#Training settings
output_dir = "/home/datpham/datpham/llm-in-law/models/PhoGPT-4B-Chat"  # Where to save model and results
num_train_epochs = 5  # Total number of training epochs
fp16 = False  # Use mixed precision training
bf16 = False  # Use bfloat16 precision with A100 GPUs
per_device_train_batch_size = 4  # Training batch size per GPU
per_device_eval_batch_size = 4  # Evaluation batch size per GPU
gradient_accumulation_steps = 1  # Number of steps for gradient accumulation
gradient_checkpointing = True  # Save memory during training
max_grad_norm = 0.3  # Max norm for gradient clipping
learning_rate = 2e-4  # Initial learning rate
weight_decay = 0.001  # Weight decay for regularization
optim = "paged_adamw_32bit"  # Optimizer choice
lr_scheduler_type = "cosine"  # Learning rate scheduler
max_steps = -1  # Set total number of training steps
warmup_ratio = 0.03  # Warmup ratio for learning rate
group_by_length = True  # Group sequences by length for efficiency
save_steps = 0  # Checkpoint save frequency
logging_steps = 150  # Logging frequency

#Sequence-to-sequence (SFT) training settings
max_seq_length = None  # Max sequence length
packing = False  # Pack short sequences together
device_map = {"": 0}  # Load model on specific GPU

#Setting up the data type for computation based on the precision setting
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

#Configuring the 4-bit quantization and precision for the model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

#Verifying if the current GPU supports bfloat16 to suggest using it for better performance
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Accelerate training with bf16=True")
        print("=" * 80)

#Loading the specified model with the above quantization configuration
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map=device_map,
    use_auth_token="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"
)
model.config.use_cache = False  # Disable caching to save memory
model.config.pretraining_tp = 1  # Setting pre-training task parallelism

#Initializing the tokenizer for the model and setting padding configurations
tokenizer = AutoTokenizer.from_pretrained(model_id,use_auth_token="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", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Setting the pad token
tokenizer.padding_side = "right"  # Adjusting padding to the right to avoid issues during training

#Configuring LoRA parameters for the model to fine-tune its attention mechanisms
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",  # Setting the bias option for LoRA
    task_type="CAUSAL_LM",  # Defining the task type as causal language modeling
)

#Defining various training parameters such as directory, epochs, batch sizes, optimization settings, etc.
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,  # Grouping by length for efficient batching
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"  # Reporting to TensorBoard for monitoring
)

#Setting up the fine-tuning trainer with the specified model, dataset, tokenizer, and training arguments
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",  # Specifying which dataset field to use for text
    max_seq_length=max_seq_length,  # Setting the maximum sequence length
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,  # Enabling packing for efficiency
)

#Starting the training process
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
