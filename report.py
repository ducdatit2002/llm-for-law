# -*- coding: utf-8 -*-
"""llm_law_report_llama3

Server version of the code to load and run the model for law report generation.

# Install dependencies (on your server environment)
"""

# Ensure dependencies are installed:
# pip install -U bitsandbytes
# pip install -U sentence-transformers
# pip install transformers
# pip install peft
# pip install pandas
# pip install scikit-learn
# pip install matplotlib seaborn tqdm

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline  # <-- added pipeline here
from peft import PeftModel
import gc
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# Set up the server's paths and configurations
base_model_id = "NousResearch/Hermes-3-Llama-3.1-8B"
finetuned_path = "/home/datpham/datpham/llm-for-law/output/llama3/SaveTrained"  # Path to the fine-tuned model
output_path = "/home/datpham/datpham/llm-for-law/output/llama3/merged_model"

# Load and merge the base model with the adapter
def load_and_merge(base_model_id, finetuned_path):
    # Memory cleanup
    torch.cuda.empty_cache()
    gc.collect()

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="offload",
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True)
    )

    # Load and merge PEFT model
    model = PeftModel.from_pretrained(base_model, finetuned_path)
    merged_model = model.merge_and_unload()

    # Clean up
    del base_model, model
    torch.cuda.empty_cache()
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return merged_model, tokenizer

# Load the model and tokenizer
model, tokenizer = load_and_merge(base_model_id, finetuned_path)
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

# Format the prompt for inference
def format_prompt(system_prompt, user_prompt):
    return f"""<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"""

# Run inference with the model
def inference(model, tokenizer, prompt, max_gen_tokens=500):
    system_prompt = """Bạn là một chuyên gia pháp lý với kiến thức sâu rộng về luật hình sự Việt Nam. Nhiệm vụ của bạn là phân tích các tình huống pháp lý và xác định tội danh dựa trên Bộ luật Hình sự."""

    formatted_prompt = format_prompt(system_prompt, prompt)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_gen_tokens)
    result = pipe(formatted_prompt)[0]['generated_text']
    return result.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()

# Validate model predictions
def evaluate_predictions(predictions, ground_truth, embedding_model, threshold=0.8):
    embeddings_pred = embedding_model.encode(predictions, normalize_embeddings=True)
    embeddings_true = embedding_model.encode(ground_truth, normalize_embeddings=True)

    similarities = cosine_similarity(embeddings_pred, embeddings_true)
    binary_preds = similarities.diagonal() > threshold

    precision, recall, f1, _ = precision_recall_fscore_support(
        np.ones(len(ground_truth)), binary_preds, average='binary'
    )

    cm = confusion_matrix(np.ones(len(ground_truth)), binary_preds)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'similarities': similarities.diagonal(),
        'confusion_matrix': cm
    }

def run_validation(model, tokenizer, val_df, batch_size=4):
    embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert')
    predictions = batch_inference(model, tokenizer, val_df['question'].tolist(), batch_size)

    metrics = evaluate_predictions(predictions, val_df['answer'].tolist(), embedding_model)

    results_df = val_df.copy()
    results_df['prediction'] = predictions
    results_df['similarity'] = metrics['similarities']
    results_df['passed_threshold'] = metrics['similarities'] > 0.8

    plot_validation_results(metrics, results_df)

    print("\nValidation Results:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Average Similarity: {np.mean(metrics['similarities']):.4f}")

    return results_df, metrics

# Load your dataset
df = pd.read_excel("/home/datpham/datpham/llm-for-law/datasets/data_law_4.xlsx")  # Path to your dataset
df

# Load the embedding model for comparison
embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert')
answers = df['answer'].values.tolist()[:15000]
y = embedding_model.encode(answers)

# Perform predictions on the validation set
df['predict'] = df.apply(lambda row: inference(model, tokenizer, row['question']), axis=1)

# Visualize the results
def plot_validation_results(metrics, results_df, threshold=0.8):
    plt.figure(figsize=(20, 15))

    plt.subplot(2, 2, 1)
    metrics_vals = [metrics['precision'], metrics['recall'], metrics['f1']]
    plt.bar(['Precision', 'Recall', 'F1'], metrics_vals)
    plt.title('Metrics')
    plt.ylim(0, 1)

    plt.subplot(2, 2, 2)
    sns.histplot(metrics['similarities'], bins=20)
    plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
    plt.title('Similarity Distribution')
    plt.xlabel('Similarity')
    plt.ylabel('Count')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(range(len(metrics['similarities'])), metrics['similarities'], 'bo-')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.title('Similarity by Example')
    plt.xlabel('Example ID')
    plt.ylabel('Similarity')
    plt.legend()

    plt.subplot(2, 2, 4)
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')

    plt.tight_layout()
    plt.savefig('validation_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    detailed_results = pd.DataFrame({
        'question': results_df['question'],
        'answer': results_df['answer'],
        'prediction': results_df['prediction'],
        'similarity': results_df['similarity'],
        'passed': results_df['passed_threshold']
    })

# Run validation on your dataset
validation_results, validation_metrics = run_validation(model, tokenizer, df)

# Perform any additional processing or evaluation
# You can now use the results and metrics for further analysis or saving to a file
