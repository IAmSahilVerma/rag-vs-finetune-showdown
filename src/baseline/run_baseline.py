"""
run_baseline.py
Zero-shot inference using Phi-2 with no RAG and no fine-tuning.
This sets the performance floor for the evaluation comparison.

Usage:
    python src/baseline/run_baseline.py
"""

import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

# --- Config ---
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_NAME = "microsoft/phi-2"
EVAL_PATH = BASE_DIR / "data" / "processed" / "eval.jsonl"
OUTPUT_PATH = BASE_DIR / "results" / "baseline_predictions.jsonl"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.3
TOP_P = 0.9

def build_prompt(question: str) -> str:
    """
    Zero-shot prompt - no context, no examples.
    """
    return f"""You are an expert machine learning researcher.
Answer the following question as clearly and accurately as possible.

Question: {question}

Answer:"""

def load_model():
    print(f"[Baseline] Loading model: {MODEL_NAME}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    print("[Baseline] Model loaded.")
    return tokenizer, model

def generate_answer(question: str, tokenizer, model) -> str:
    prompt = build_prompt(question)
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    prompt_length = inputs["input_ids"].shape[1]
    answer_ids = output_ids[0][prompt_length:]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

def load_eval_set(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records

def main():
    os.makedirs(name=str(OUTPUT_PATH.parent), exist_ok=True)
    
    print(">>> Loading eval set...")
    eval_set = load_eval_set(EVAL_PATH)
    print(f"    Loaded {len(eval_set)} eval pairs")
    
    tokenizer, model = load_model()
    
    print(f"\n>>> Running zero-shot inference on {len(eval_set)} questions...")
    
    with open(OUTPUT_PATH, "w") as f:
        for record in tqdm(eval_set, desc="Baseline inference"):
            answer = generate_answer(record["question"], tokenizer, model)
            
            output = {
                "arxiv_id":             record["arxiv_id"],
                "title":                record["title"],
                "question":             record["question"],
                "reference_answer":     record["answer"],
                "predicted_answer":     answer
            }
            
            f.write(json.dumps(output) + "\n")
    
    print(f"\n>>> Done. Predictions saved to {OUTPUT_PATH}")
    
    # Preview a couple of examples
    print("\n>>> Sample predictions:\n")
    with open(OUTPUT_PATH) as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            rec = json.loads(line)
            print(f"Q: {rec['question']}")
            print(f"Reference   : {rec['reference_answer'][:150]}...")
            print(f"Predicted   : {rec['predicted_answer'][:150]}...")
            print()
            
if __name__=="__main__":
    main()