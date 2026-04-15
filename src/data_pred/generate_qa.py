"""
generate_qa.py
Uses GPT-3.5-turbo to generate Q&A pairs from ArXiv abstracts.

- 50 papers     -> eval.jsonl   (held-out evaluation set)
- 500 papers    -> train.jsonl  (fine-tuning instruction dataset)

Usage:
    python src/data_pred/generate_qa.py
"""

import json
import os
import time
import random
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_PATH = BASE_DIR / "data" / "raw" / "abstracts.json"
EVAL_PATH = BASE_DIR / "data" / "processed" / "eval.jsonl"
TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.jsonl"
EVAL_SIZE = 50
TRAIN_SIZE = 500
RANDOM_SEED = 42
SLEEP_BETWEEN = 1.0

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Prompts ---
EVAL_SYSTEM_PROMPT = """
You are a machine learning researcher creating evaluation questions.
Given an ArXiv abstract, generate ONE clear, specific question that:
- Can be answered using only the abstract
- Tests factual understanding (not opinion)
- Is answerable in 2-4 sentences

Respond ONLY with valid JSON. No preamble, no markdown, no backticks.
Format: {"question": "...", "answer": "..."}
"""

TRAIN_SYSTEM_PROMPT = """
You are a machine learning researcher creating training data.
Given an ArXiv abstract, generate ONE question-answer pair where:
- The question asks about the core contribution, method, or finding
- The answer is a clear 2-4 sentence explanation in the style of an ML expert
- The answer is grounded strictly in the abstract

Respond ONLY with valid JSON. No preamble, no markdown, no backticks.
Format: {"question": "...", "answer": "..."}
"""

def generate_qa_pair(abstract: str, title: str, mode: str) -> dict | None:
    """
    Call GPT-3.5-turbo to generate a Q&A pair for one abstract.
    """
    
    system_prompt = EVAL_SYSTEM_PROMPT if mode == "eval" else TRAIN_SYSTEM_PROMPT
    
    user_prompt = f"Title: {title}\n\nAbstract: {abstract}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        
        # Basic validation
        if "question" not in parsed or "answer" not in parsed:
            return None
        if len(parsed["question"]) < 10 or len(parsed["answer"]) < 20:
            return None
        
        return parsed
    
    except (json.JSONDecodeError, KeyError):
        return None
    except Exception as e:
        print(f"    API error: {e}")
        return None
    
def process_batch(papers: list[dict], mode: str, output_path: str) -> int:
    """
    Generate Q&A pairs for a list of papers and write to JSONL.
    """
    
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    success = 0
    skipped = 0
    
    with open(output_path, "w") as f:
        for paper in tqdm(papers, desc=f"Generating {mode} pairs"):
            qa = generate_qa_pair(paper["abstract"], paper["title"], mode)
            
            if qa is None:
                skipped += 1
                continue
            
            record = {
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "abstract": paper["abstract"],
                "question": qa["question"],
                "answer": qa["answer"]
            }
            
            # For training, wrap in instruction format for fine-tuning
            if mode == "train":
                record["instruction"] = (
                    "You are an expert in machine learning. "
                    "Answer the following question aboue an ML research paper."
                )
                record["input"] = (
                    f"Abstract: {paper['abstract']}\n\nQuestion: {qa['question']}"
                )
                record["output"] = qa["answer"]
            
            f.write(json.dumps(record) + "\n")
            success += 1
            time.sleep(SLEEP_BETWEEN)
    
    print(f"    Saved {success} pairs, skipped {skipped} (parse errors / too short)")
    return success
    
def main():
    print(">>> Loading abstracts...")
    with open(RAW_PATH) as f:
        papers = json.load(f)
        
        print(f"    Loaded {len(papers)} papers")
        
        random.seed(RANDOM_SEED)
        random.shuffle(papers)
        
        eval_papers = papers[:EVAL_SIZE]
        train_papers = papers[EVAL_SIZE : EVAL_SIZE + TRAIN_SIZE]
        
        print(f"\n>>> Generating EVAL set ({EVAL_SIZE} papers)...")
        eval_count = process_batch(eval_papers, mode='eval', output_path=EVAL_PATH)
        
        print(f"\n>>> Generating TRAIN set ({TRAIN_SIZE} papers)...")
        train_count = process_batch(train_papers, mode="train", output_path=TRAIN_PATH)
        
        print(f"\n>>> Done.")
        print(f"    eval.jsonl  :   {eval_count} pairs -> {EVAL_PATH}")
        print(f"    train.jsonl :   {train_count} pairs -> {TRAIN_PATH}")
        
        # Preview
        print("\n>>> Sample eval record:")
        with open(EVAL_PATH) as f:
            print(json.dumps(json.loads(f.readline()), indent=2))
            
        print("\n>>> Sample train record:")
        with open(TRAIN_PATH) as f:
            print(json.dumps(json.loads(f.readline()), indent=2))

if __name__ == "__main__":
    main()