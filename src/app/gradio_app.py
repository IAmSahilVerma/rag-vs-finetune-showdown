"""
gradio_app.py
Side-by-side demo of all three approaches: Baseline, RAG and Fine-tuned.
Loads all three models once at startup and runs inference on demand.

Usage:
    python -m src.app.gradio_app
"""

import json
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from src.rag.retriever import Retriever
from pathlib import Path
# --- Config ---
BASE_MODEL_NAME = "microsoft/phi-2"
BASE_DIR = Path(__file__).resolve().parents[2]
ADAPTER_PATH    = str(BASE_DIR / "models" / "phi2-qlora-arxiv")
MAX_NEW_TOKENS  = 256
TEMPERATURE     = 0.3
TOP_P           = 0.9
TOP_K           = 3


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
    )
    tokenizer.pad_token      = tokenizer.eos_token
    tokenizer.padding_side   = "right"
    return tokenizer


def load_base_model(tokenizer):
    """Loads Phi-2 in 4-bit. Shared across baseline and RAG."""
    print("[App] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=load_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("[App] Base model loaded.")
    return model


def load_finetuned_model(base_model):
    """Layers the LoRA adapter on top of the base model."""
    print("[App] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    print("[App] Fine-tuned model loaded.")
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def generate(prompt: str, model, tokenizer) -> str:
    """Shared generation function used by all three approaches."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    answer_ids    = output_ids[0][prompt_length:]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def baseline_prompt(question: str) -> str:
    return (
        f"You are an expert machine learning researcher.\n"
        f"Answer the following question as clearly and accurately as possible.\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def rag_prompt(question: str, context: str) -> str:
    return (
        f"You are an expert machine learning researcher.\n"
        f"Use the following research paper excerpts to answer the question.\n"
        f"Base your answer strictly on the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def finetuned_prompt(question: str) -> str:
    return (
        f"### Instruction:\n"
        f"You are an expert in machine learning. "
        f"Answer the following question about an ML research paper.\n\n"
        f"### Input:\n"
        f"Question: {question}\n\n"
        f"### Response:\n"
    )


# ---------------------------------------------------------------------------
# Startup — load everything once
# ---------------------------------------------------------------------------

print("[App] Initialising models and retriever...")
tokenizer   = load_tokenizer()
base_model  = load_base_model(tokenizer)
ft_model    = load_finetuned_model(base_model)
retriever   = Retriever(top_k=TOP_K)
print("[App] All models ready.\n")


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def run_all(question: str):
    """
    Runs all three approaches for a given question.
    Returns answers and RAG sources for display in Gradio.
    """
    if not question.strip():
        return "Please enter a question.", "", "", ""

    # --- Baseline ---
    baseline_answer = generate(baseline_prompt(question), base_model, tokenizer)

    # --- RAG ---
    retrieved       = retriever.retrieve(question)
    context         = retriever.format_context(retrieved)
    rag_answer      = generate(rag_prompt(question, context), base_model, tokenizer)
    sources         = "\n\n".join([
        f"📄 [{r['metadata']['title']}]({r['metadata']['url']})"
        for r in retrieved
    ])

    # --- Fine-tuned ---
    ft_answer = generate(finetuned_prompt(question), ft_model, tokenizer)

    return baseline_answer, rag_answer, ft_answer, sources


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="RAG vs Fine-tuning Showdown") as demo:

    gr.Markdown("""
    # 🔬 RAG vs Fine-tuning Showdown
    **Domain:** ArXiv ML/AI Papers · **Base model:** Phi-2 · **Fine-tuning:** QLoRA (LoRA rank 16)

    Enter a machine learning question and compare how each approach responds.
    """)

    with gr.Row():
        question_box = gr.Textbox(
            label="Your Question",
            placeholder="e.g. How does RLHF align language models with human preferences?",
            lines=2,
            scale=4,
        )
        submit_btn = gr.Button("Run All Three", variant="primary", scale=1)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🔵 Baseline\n*Zero-shot · No adaptation*")
            baseline_out = gr.Textbox(label="Answer", lines=8, interactive=False)

        with gr.Column():
            gr.Markdown("### 🟠 RAG\n*Retrieval-augmented · No fine-tuning*")
            rag_out      = gr.Textbox(label="Answer", lines=8, interactive=False)
            sources_out  = gr.Markdown(label="Sources Retrieved")

        with gr.Column():
            gr.Markdown("### 🟢 Fine-tuned\n*QLoRA adapted · No retrieval*")
            ft_out       = gr.Textbox(label="Answer", lines=8, interactive=False)

    # Example questions
    gr.Examples(
        examples=[
            ["How does RLHF align language models with human preferences?"],
            ["What are the main challenges in multi-modal learning?"],
            ["How does knowledge distillation work in neural networks?"],
            ["What methods are used to improve reasoning in large language models?"],
            ["How do diffusion models generate images?"],
        ],
        inputs=question_box,
    )

    submit_btn.click(
        fn=run_all,
        inputs=question_box,
        outputs=[baseline_out, rag_out, ft_out, sources_out],
    )

if __name__ == "__main__":
    demo.launch(share=True)