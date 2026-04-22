"""
lora_config.py
Central place for all LoRA / QLoRA hyperparameters.
Imported by train.py and inference.py.
"""

from pathlib import Path
from peft import LoraConfig, TaskType

# --- Base model ---
BASE_MODEL_NAME = "microsoft/phi-2"

# --- LoRA hyperparameters ---
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_BIAS = "none"

# Phi-2 attention projection layers to target
# These are the standard linear layers LoRA is applied to
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "dense"
]

# --- QLoRA quantisation ---
LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "float16"
BNB_4BIT_DOUBLE_QUANT = True

# --- Training ---
BASE_PATH = Path(__file__).resolve().parents[2]
TRAIN_PATH = BASE_PATH / "data" / "processed" / "train.jsonl"
OUTPUT_DIR = BASE_PATH / "models" / "phi2-qlora-arxiv"
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
MAX_SEQ_LEN = 512
LOGGING_STEPS = 10
SAVE_STEPS = 100
FP16 = True

# --- Inference ---
ADAPTER_PATH = "models/phi2-qlora-arxiv"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.3
TOP_P = 0.9

def get_lora_config() -> LoraConfig:
    """
    Returns the PEFT LoraConfig object ready for get_peft_model().
    """
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_MODULES
    )