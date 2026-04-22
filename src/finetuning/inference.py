"""
inference.py
Runs the fine-tuned Phi-2 + LoRA adapter on the eval set
and saves predictions in the same format as baseline_predictions.jsonl

Usage:
    python src/finetuning/inference.py
"""

import json
import os
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from src.finetuning.lora_config import (
    BASE_MODEL_NAME,
    ADAPTER_PATH,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    LOAD_IN_4BIT,
    BNB_4BIT_QUANT_TYPE,
    BNB_4BIT_DOUBLE_QUANT
)

# --- Config ---
BASE_PATH = Path(__file__).resolve().parents[2]
EVAL_PATH = BASE_PATH / "data" / "processed" / "eval.jsonl"
OUTPUT_PATH = BASE_PATH / "results" / "finetuned_predictions.jsonl"

def build_prompt(question: str, abstract: str) -> str:
    """
    Mirrors the Alpaca format used during training exactly.
    If the format differs from training, the model won't respond correctly.
    """
    return (
        f"### Instruction:\n"
        f"You are an expert in machine learning. "
        f"Answer the following question about an ML research paper.\n\n"
        f"### Input:\n"
        f"Abstract: {abstract}\n\n"
        f"Question: {question}\n\n"
        f"### Response:\n"
    )
    
def load_model():
    print(f"[FTInference] Loading base model: {BASE_MODEL_NAME}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=BNB_4BIT_DOUBLE_QUANT
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code = True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config = bnb_config,
        device_map = "auto",
        trust_remote_code = True
    )
    
    print(f"[FTInference] Loading LoRA adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    print("[FTInference] Model + adapter loaded.")
    return tokenizer, model

def generate_answer(question: str, abstract: str, tokenizer, model) -> str:
    prompt = build_prompt(question, abstract)
    
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
    os.makedirs(str(BASE_PATH / "results"), exist_ok=True)
    
    print(">>> Loading eval set...")
    eval_set = load_eval_set(EVAL_PATH)
    print(f"    Loaded {len(eval_set)} eval pairs")
    
    tokenizer, model = load_model()
    
    print(f"\n>>> Running fine-tuned inference on {len(eval_set)} questions...")
    
    with open(str(OUTPUT_PATH), "w") as f:
        for record in tqdm(eval_set, desc="Fine-tuned inference"):
            answer = generate_answer(
                record["question"],
                record["abstract"],
                tokenizer,
                model
            )
            
            output = {
                "arxiv_id": record["arxiv_id"],
                "title": record["title"],
                "question": record["question"],
                "reference_answer": record["answer"],
                "predicted_answer": answer
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
            print(f"Q:          {rec['question']}")
            print(f"Reference:  {rec['reference_answer'][:150]}...")
            print(f"Predicted:  {rec['predicted_answer'][:150]}...")
            print()
            
if __name__=="__main__":
    main()