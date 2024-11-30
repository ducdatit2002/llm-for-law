# -*- coding: utf-8 -*-
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Sử dụng mô hình gốc thay vì mô hình fine-tuned
base_model_id = "vinai/PhoGPT-4B-Chat"

try:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device == 0 else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Tắt cache để tránh bất ổn định
    model.config.use_cache = False
except Exception as e:
    print(f"Error loading base model or tokenizer: {e}")
    exit()

try:
    qa_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True
    )
except Exception as e:
    print(f"Error creating pipeline: {e}")
    exit()

while True:
    print("\nĐặt câu hỏi (hoặc nhập 'exit' để thoát):")
    question = input("> ").strip()
    if question.lower() == "exit":
        print("Thoát chương trình.")
        break

    prompt = f"<s>[INST] {question.strip()} [/INST]"
    print("\nPrompt đầu vào:", prompt)

    try:
        result = qa_pipeline(prompt, truncation=True)
        response = result[0]["generated_text"].split("[/INST]")[1].strip()
        print("\nCâu trả lời từ mô hình:")
        print(response)
    except Exception as e:
        print(f"Lỗi trong quá trình sinh câu trả lời: {e}")
