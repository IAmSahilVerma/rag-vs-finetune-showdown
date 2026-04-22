"""
run_eval.py
Loads predictions from all three approaches, scores them,
logs results to MLflow, and prints a comparison table.

Usage:
    python src/evaluation/run_eval.py
"""

import json
import os
import mlflow
import pandas as pd
from pathlib import Path
from src.evaluation.metrics import compute_all_metrics, print_metrics_table

# --- Config ---
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS = {
    "baseline": str(BASE_DIR / "results" / "baseline_predictions.jsonl"),
    "rag": str(BASE_DIR / "results" / "rag_predictions.jsonl"),
    "finetuned": str(BASE_DIR / "results" / "finetuned_predictions.jsonl") 
}
SUMMARY_PATH = str(BASE_DIR / "results" / "evaluation_summary.json")

def load_predictions(path: str) -> tuple[list[str], list[str]]:
    """
    Load predicted and reference answers from a predictions jsonl file.
    """
    predictions = []
    references = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            predictions.append(record["predicted_answer"])
            references.append(record["reference_answer"])
    return predictions, references

def load_full_records(path: str) -> list[dict]:
    """
    Load full records for per-question berakdown.
    """
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records

def build_per_question_breakdow(approach_records: dict[str, list[dict]]) -> pd.DataFrame:
    """
    Builds a pre-question DataFrame showing predicted answers
    from all three approaches side by side.
    """
    baseline_records = approach_records["baseline"]
    rows = []
    
    for i, base in enumerate(baseline_records):
        row = {
            "question": base["question"],
            "reference": base["reference_answer"],
            "baseline_answer": base["predicted_answer"],
            "rag_answer": approach_records["rag"][i]["predicted_answer"],
            "finetuned_answer": approach_records["finetuned"][i]["predicted_answer"]
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

def main():
    os.makedirs(str(BASE_DIR / "results"), exist_ok=True)
    mlflow.set_experiment("rag-vs-finetune-showdown")
    
    all_metrics = {}
    all_records = {}
    
    with mlflow.start_run(run_name="evaluation-all-approaches"):
        for approach, path in RESULTS.items():
            print(f"\n>>> Scoring: {approach.upper()}")
            
            predictions, references = load_predictions(path)
            all_records[approach] = load_full_records(path)
            
            metrics = compute_all_metrics(predictions, references)
            all_metrics[approach] = metrics
            
            # Log to MLflow with approach prefix
            mlflow.log_metrics({
                f"{approach}_rouge1":        metrics["rouge1"],
                f"{approach}_rouge2":        metrics["rouge2"],
                f"{approach}_rougeL":        metrics["rougeL"],
                f"{approach}_bertscore_f1":  metrics["bertscore_f1"],
            })

            print(f"    ROUGE-1    : {metrics['rouge1']}")
            print(f"    ROUGE-2    : {metrics['rouge2']}")
            print(f"    ROUGE-L    : {metrics['rougeL']}")
            print(f"    BERTScore  : {metrics['bertscore_f1']}")
            
        # Print comparison table
        print_metrics_table(all_metrics)
        
        # Save summary JSON
        with open(SUMMARY_PATH, "w") as f:
            json.dump(all_metrics, f, indent=2)
        mlflow.log_artifact(SUMMARY_PATH)
        print(f"\n>>> Summary saved to {SUMMARY_PATH}")
        
        # Save per-question breakdown as CSV
        breakdown_path = str(BASE_DIR / "results" / "per_question_breakdown.csv")
        df = build_per_question_breakdow(all_records)
        df.to_csv(breakdown_path, index=False)
        mlflow.log_artifact(breakdown_path)
        print(f">>> Per-question breakdown saved to {breakdown_path}")
        
        # Log which approach won each metric
        for metric in ["rouge1", "rouge2", "rougeL", "bertscore_f1"]:
            winner = max(all_metrics, key=lambda a: all_metrics[a][metric])
            mlflow.log_param(f"winner_{metric}", winner)
            print(f"    Base {metric}: {winner} ({all_metrics[winner][metric]})")
            
if __name__ == "__main__":
    main()