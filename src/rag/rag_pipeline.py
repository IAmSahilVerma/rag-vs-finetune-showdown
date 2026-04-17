"""
rag_pipeline.py
Combines retrieval + generation into a single callable pipeline.

Retrieves relevant chunks from ChromaDB, builds a prompt with context,
and generates an answer using a HuggingFace model.

Usage:
    python src/rag/rag_pipeline.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from src.rag.retriever import Retriever
from pathlib import Path

# ---Config---
MODEL_NAME = "microsoft/phi-2"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.3
TOP_P = 0.9
TOP_K = 3

def build_prompt(question: str, context: str) -> str:
    """
    Builds the RAG prompt by injecting retrieved context
    alongside the question.
    """
    return f"""You are an expert machine learning researcher.
Use the following research paper excerpts to answer the question.
Base your answer strictly on the provided context.
If the context does not contain enough information, say so clearly.

Context: {context}

Question: {question}

Answer:"""

class RAGPipeline:
    def __init__(self, top_k: int = TOP_K):
        self.retriever = Retriever(top_k=top_k)
        self.tokenizer, self.model = self._load_model()
        
    def _load_model(self):
        print(f"[RAGPipeline] Loading model: {MODEL_NAME}")
        
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
        
        print("[RAGPipelinne] Model loaded.")
        return tokenizer, model
    
    def generate(self, question: str) -> dict:
        """
        Full RAG pipeline: retrieve -> build prompt -> generate.
        
        Returns a dict with:
            - answer    : generated answer string
            - context   : formatted retrieved context
            - sources   : list of source metadata dicts
            - question  : original question
        """
        
        # Step 1: Retrieve
        retrieved = self.retriever.retrieve(question)
        context = self.retriever.format_context(retrieved)
        
        # Step 2: Build prompt
        prompt = build_prompt(question, context)
        
        # Step 3: Tokenise
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Step 4: Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Step 5: Decode - strip the prompt from the output
        prompt_length = inputs["input_ids"].shape[1]
        answer_ids = output_ids[0][prompt_length:]
        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        
        return {
            "question": question,
            "answer": answer,
            "context": context,
            "sources": [r["metadata"] for r in retrieved]
        }
        
if __name__=="__main__":
    pipeline = RAGPipeline(top_k=TOP_K)
    
    test_questions = [
        "What methods are used to align large language models with human preferences?",
        "How does knowledge distillation work in neural networks?",
        "What are the main challenges in multi-modal learning?"
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        result = pipeline.generate(question)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources used:")
        for s in result["sources"]:
            print(f" - {s['title']} ({s['url']})")
        