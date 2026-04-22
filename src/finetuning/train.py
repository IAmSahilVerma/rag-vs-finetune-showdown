"""
train.py
QLoRA fine-tuning loop for Phi-2 on the ArXiv ML Q&A dataset.
Uses PEFT + TRL's SFTTrainer for clean, memory-efficient training.

Usage:
    python src/finetuning/train.py
"""

import json
import os
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import mlflow

from src.finetuning.lora_config import (
    BASE_MODEL_NAME,
    LOAD_IN_4BIT,
    BNB_4BIT_COMPUTE_DTYPE,
    BNB_4BIT_DOUBLE_QUANT,
    BNB_4BIT_QUANT_TYPE,
    TRAIN_PATH,
    OUTPUT_DIR,
    NUM_EPOCHS,
    BATCH_SIZE,
    GRAD_ACCUM,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    MAX_SEQ_LEN,
    LOGGING_STEPS,
    SAVE_STEPS,
    FP16,
    get_lora_config
)

# --- Data ---
def load_train_data(path: str) -> Dataset:
    """
    Load train.jsonl and convert to HuggingFace Dataset.
    """
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)

def format_prompt(example: dict) -> dict:
    """
    Format each record into a single 'text' field for SFTTrainer.
    Uses Alpaca-style instruction format.
    """
    text = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )
    return {"text": text}

# --- Model ---

def load_base_model():
    """
    Load Phi-2 in 4-bit quantisation, ready for LoRA.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=BNB_4BIT_DOUBLE_QUANT
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Required step before applying LoRA a quantised model
    model = prepare_model_for_kbit_training(model)
    
    return tokenizer, model

# --- Training ---

def main():
    os.makedirs(str(OUTPUT_DIR.parent), exist_ok=True)
    os.makedirs("mlruns", exist_ok=True)
    
    # MLflow setup
    mlflow.set_experiment("rag-vs-finetuning-showdown")
    
    with mlflow.start_run(run_name="phi2-qlora-finetune"):
        
        # Log hyperparameters
        mlflow.log_params({
            "base_model": BASE_MODEL_NAME,
            "lora_r": 16,
            "lora_alpha": 32,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "learning_rate": LEARNING_RATE,
            "max_seq_len": MAX_SEQ_LEN
        })
        
        # Load and format dataset
        print(">>> Loading training data...")
        dataset = load_train_data(TRAIN_PATH)
        dataset = dataset.map(format_prompt)
        print(f"    {len(dataset)} training examples")
        print(f"    Sample:\n{dataset[0]['text'][:300]}...\n")
        
        # Load model
        print(">>> Loading base model...")
        tokenizer, model = load_base_model()
        
        # Apply LoRA
        print(">>> Applying LoRA adapters...")
        lora_config = get_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir = str(OUTPUT_DIR),
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            fp16=FP16,
            logging_steps=LOGGING_STEPS,
            save_steps=SAVE_STEPS,
            save_total_limit=2,
            report_to="none",
            optim="paged_adamw_8bit"
        )
        
        # SFTTrainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LEN,
            args=training_args
        )
        
        # Train
        print(">>> Starting training...")
        train_result = trainer.train()
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            "train_loss": round(train_result.training_loss, 4),
            "train_runtime_secs": round(train_result.metrics["train_runtime"], 1),
            "samples_per_second": round(train_result.metrics["train_samples_per_second"], 2)
        })
        
        # Save adapter weights
        print(f"\n>>> Saving adapter to {OUTPUT_DIR}...")
        trainer.save_model(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))
        
        mlflow.log_param("adapter_path", str(OUTPUT_DIR))
        print(">>> Training complete.")
        print(f"    Loss    : {train_result.training_loss:.4f}")
        print(f"    Runtime : {train_result.metrics['train_runtime']:.0f}s")
        
if __name__=="__main__":
    main()