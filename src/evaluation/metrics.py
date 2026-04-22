"""
metrics.py
Scoring functions for comparing predicted answers against reference answers.
Computes ROUGE-1, ROUGE-2, ROUGE-L and BERTScore.

Not meant to be run directly - import only
"""

from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from pathlib import Path
import torch

# --- Config ---
BERTSCORE_MODEL = "distilbert-base-uncased"

def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """
    Computes ROUGE-1, ROUGE-2, ROUGE-L for a list of prediction/reference pairs.
    Returns mean F1 scores across the full list.
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)
        
    return {
        "rouge1": round(sum(rouge1_scores) / len(rouge1_scores), 4),
        "rouge2": round(sum(rouge2_scores) / len(rouge2_scores), 4),
        "rougeL": round(sum(rougeL_scores) / len(rougeL_scores), 4)
    }
    
def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
    """
    Computes BERTScore F1 for a list of prediction/reference pairs.
    Returns mean F1 across the full list.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    P, R, F1 = bert_score_fn(
        predictions,
        references,
        model_type=BERTSCORE_MODEL,
        device=device,
        verbose=False
    )
    
    return {
        "bertscore_precision": round(P.mean().item(), 4),
        "bertscore_recall": round(R.mean().item(), 4),
        "bertscore_f1": round(F1.mean().item(), 4)
    }
    
def compute_all_metrics(predictions: list[str], references: list[str]) -> dict:
    """
    Convenience wrapper - runs both ROUGE and BERTScore and merges results.
    This is the main entry point.
    """
    print("Computing ROUGE scores...")
    rouge = compute_rouge(predictions, references)
    
    print("Computing BERTScore...")
    bertscore = compute_bertscore(predictions, references)
    
    return {**rouge, **bertscore}

def print_metrics_table(results: dict[str, str]):
    """
    Pretty prints a comparison table of all three approaches.
    
    Args:
        results: {"baseline": {...metrics}, "rag": {...}, "finetuned": {...}}
    """
    
    metrics = ["rouge1", "rouge2", "rougeL", "bertscore_f1"]
    col_width = 12
    
    header = f"{'Metric':<20}" + "".join(f"{k:>{col_width}}" for k in results.keys())
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    
    for metric in metrics:
        row = f"{metric:<20}"
        for approach_metrics in results.values():
            value = approach_metrics.get(metric, 0.0)
            row += f"{value:>{col_width}.4f}"
        print(row)
        
    print("=" * len(header))