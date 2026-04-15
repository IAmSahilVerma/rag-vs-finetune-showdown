"""
ingest.py
Reards corpus.jsonl, embeds each chunk using BGE embeddings,
and stores everything in a persistent ChromaDB collection.

Run this once before using the RAG pipeline.

Usage:
    python src/rag/ingest.py
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# --- Config ---
BASE_DIR = Path(__file__).resolve().parents[2]
CORPUS_PATH = BASE_DIR / "data" / "processed" / "corpus.jsonl"
CHROMA_DB_PATH = BASE_DIR / "data" / "chromadb"
COLLECTION_NAME = "arxiv_ml_papers"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64

def load_corpus(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records

def build_collection(client, name: str, embed_fn):
    """
    Get or create a ChromaDB collection.
    """
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        print(f"    Collection '{name}' already exists - deleting and rebuilding.")
        client.delete_collection(name)
    return client.create_collection(name=name, embedding_function=embed_fn)

def batch(lst: list, size: int):
    """
    Yield successive chunks of a list.
    """
    for i in range(0, len(lst), size):
        yield lst[i : i + size]
        
def main():
    print(">>> Loading corpus...")
    records = load_corpus(CORPUS_PATH)
    print(f"    Loaded {len(records)} records")
    
    print(f"\n>>> Initialising ChromaDB at '{CHROMA_DB_PATH}'...")
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    chroma_client = PersistentClient(path=str(CHROMA_DB_PATH))
    
    print(f"\n>>> Loading embedding model: {EMBED_MODEL}")
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    
    collection = build_collection(chroma_client, COLLECTION_NAME, embed_fn)
    
    print(f"\n>>> Embedding and ingesting in batches of {BATCH_SIZE}...")
    
    for chunk in tqdm(list(batch(records, BATCH_SIZE)), desc="Ingesting batches"):
        ids = [r["arxiv_id"] for r in chunk]
        documents = [r["text"] for r in chunk]
        metadatas = [r["metadata"] for r in chunk]
        
        # ChromaDB requires metadata values to be str, int, float, or bool
        # Flatten list fields to comma-separated strings
        for m in metadatas:
            m["authors"] = ", ".join(m["authors"])
            m["categories"] = ", ".join(m["categories"])
            
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
    
    print(f"\n>>> Done.")
    print(f"    Collection '{COLLECTION_NAME}' now has {collection.count()} documents")
    
    # Sanity check - test query
    print("\n>>> Sanity check - querying: 'attention mechanism transformer'")
    results = collection.query(query_texts=["attention mechanism transformer"], n_results=2)
    for i, doc in enumerate(results["documents"][0]):
        print(f"\n Result {i+1}: {results['metadatas'][0][i]['title']}")
        print(f"    {doc[:150]}...")
        
if __name__ == "__main__":
    main()