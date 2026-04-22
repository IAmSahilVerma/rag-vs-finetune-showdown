"""
run_rag_eval.py
Runs the RAG pipeline over the full eval set and saves predictions
to results/rag_predictions.jsonl for comparison with baseline and fine-tuned.

Usage:
    python src/rag/run_rag_eval.py
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
from src.rag.rag_pipeline import RAGPipeline

# --- Config ---
BASE_DIR = Path(__file__).resolve().parents[2]
EVAL_PATH   = BASE_DIR / "data" / "processed" / "eval.jsonl"
OUTPUT_PATH = BASE_DIR / "results" / "rag_predictions.jsonl"


def load_eval_set(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main():
    os.makedirs(str(Path(BASE_DIR / "results")), exist_ok=True)

    print(">>> Loading eval set...")
    eval_set = load_eval_set(str(EVAL_PATH))
    print(f"    Loaded {len(eval_set)} eval pairs")

    print("\n>>> Initialising RAG pipeline...")
    pipeline = RAGPipeline()

    print(f"\n>>> Running RAG inference on {len(eval_set)} questions...")

    with open(str(OUTPUT_PATH), "w") as f:
        for record in tqdm(eval_set, desc="RAG inference"):
            result = pipeline.generate(record["question"])

            output = {
                "arxiv_id":         record["arxiv_id"],
                "title":            record["title"],
                "question":         record["question"],
                "reference_answer": record["answer"],
                "predicted_answer": result["answer"],
                "sources":          result["sources"],   # bonus — RAG only
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
            print(f"Q:         {rec['question']}")
            print(f"Reference: {rec['reference_answer'][:150]}...")
            print(f"Predicted: {rec['predicted_answer'][:150]}...")
            print(f"Sources:   {[s['title'] for s in rec['sources']]}")
            print()


if __name__ == "__main__":
    main()