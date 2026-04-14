"""
fetch_arxiv.py
Downloads abstracts from ArXiv across ML/AI categories and saves them
as raw JSON for downstream processing.
"""

import arxiv
import json
import time
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# --- Config ---
CATEGORIES = ["cs.AI", "cs.LG", "stat.ML"]
MAX_RESULTS_PER_CATEGORY = 400
DATE_FROM = "2022-01-01"

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_PATH = BASE_DIR / "data" / "raw" / "abstracts.json"

def fetch_abstracts(category: str, max_results: int) -> list[dict]:
    """
    Fetch abstracts from ArXiv for a given category.
    """
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    results = []
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
    
    for paper in tqdm(client.results(search), total=max_results, desc=f"Fetching {category}"):
        published = paper.published.strftime("%Y-%m-%d")
        if published < DATE_FROM:
            break
        
        results.append(
            {
                "arxiv_id": paper.entry_id.split("/")[-1],
                "title": paper.title.strip(),
                "abstract": paper.summary.strip().replace("\n", " "),
                "authors": [a.name for a in paper.authors[:5]],
                "categories": paper.categories,
                "url": paper.entry_id
            }
        )
        
        time.sleep(0.1)
        
    return results

def remove_duplicate(papers: list[dict]) -> list[dict]:
    """
    Remove duplicate papers by arxiv_id.
    """
    seen = set()
    unique = []
    for p in papers:
        if p["arxiv_id"] not in seen:
            seen.add(p["arxiv_id"])
            unique.append(p)
    
    return unique

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    all_papers = []
    
    for category in CATEGORIES:
        print(f"\n>>> Fetching category: {category}")
        papers = fetch_abstracts(category, MAX_RESULTS_PER_CATEGORY)
        print(f"    Retrieved {len(papers)} papers")
        all_papers.extend(papers)
        
    all_papers = remove_duplicate(all_papers)
    print(f"\n>>> Total unique papers after removing duplicates: {len(all_papers)}")
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_papers, f, indent=2)
        
    print(f">>> Saved to {OUTPUT_PATH}")
    print(f">>> Sampled entry:\n")
    print(json.dumps(all_papers[0], indent=2))
    
if __name__ == "__main__":
    main()