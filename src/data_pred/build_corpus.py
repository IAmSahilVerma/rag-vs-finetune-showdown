"""
build_corpus.py
Takes raw abstracts and formats them into a clean corpus for RAG ingestion.

Each record contains the text chunk that will be embedded + stored in ChromaDB,
along with metadata for filtering and citation in the Gradio app.

Usage:
    python src/data_pred/build_corpus.py
"""

import json
import os
from tqdm import tqdm
from pathlib import Path

# --- Config ---
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_PATH = BASE_DIR / "data" / "raw" / "abstracts.json"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "corpus.jsonl"

# IDs already used in eval/train - we exclude these from the RAG corpus
# so RAG isn't retrieving the exact papers the eval questions were generated from.
# This keeps the comparison fair.
EVAL_PATH = BASE_DIR / "data" / "processed" / "eval.jsonl"
TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.jsonl"

def load_used_ids(*paths: str) -> set[str]:
    """
    Collect arxiv_ids already used in eval and train splits.
    """
    used = set()
    for path in paths:
        if not os.path.exists(path):
            print(f"    Warning: {path} not found, skipping.")
            continue
        
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                used.add(record["arxiv_id"])
    return used

def format_chunk(paper: dict) -> str:
    """
    Format a paper into a single readable text chunk for embedding.
    Keeping title + abstract together gives the embedder full context.
    """
    return (
        f"Title: {paper['title']}\n\n"
        f"Abstract: {paper['abstract']}"
    )
    
def main():
    print(">>> Loading abstracts...")
    with open(RAW_PATH) as f:
        papers = json.load(f)
    print(f"    Loaded {len(papers)} papers")
    
    print("\n>>>    Loading user IDs from eval and train splits...")
    used_ids = load_used_ids(EVAL_PATH, TRAIN_PATH)
    print(f"    Excluding {len(used_ids)} IDs already used in eval/train")
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    included = 0
    excluded = 0
    
    with open(OUTPUT_PATH, "w") as f:
        for paper in tqdm(papers, desc="Building corpus"):
            if paper["arxiv_id"] in used_ids:
                excluded += 1
                continue
            
            record = {
                "arxiv_id": paper["arxiv_id"],
                "title":    paper["title"],
                "text":     format_chunk(paper),
                "metadata": {
                    "arxiv_id":     paper["arxiv_id"],
                    "title":        paper["title"],
                    "authors":      paper["authors"],
                    "categories":   paper["categories"],
                    "url":          paper["url"]
                }
            }
            
            f.write(json.dumps(record) + "\n")
            included += 1
    
    print(f"\n>>> Done.")
    print(f"    Included   : {included} papers")
    print(f"    Exlcuded   : {excluded} papers")
    print(f"    Saved to   : {OUTPUT_PATH}")
    print(f"\n>>> Sample corpus record:")
    with open(OUTPUT_PATH) as f:
        print(json.dumps(json.loads(f.readline()), indent=2))
        
if __name__ == "__main__":
    main()    