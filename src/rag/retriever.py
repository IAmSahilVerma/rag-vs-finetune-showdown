"""
retriever.py
Queries ChromaDB and returns the top-k most relevant chunks
for a given question.

Not meant to be run directly. Import only.
"""

from pathlib import Path
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# --- Config ---
BASE_DIR = Path(__file__).resolve().parents[2]
CHROMA_DB_PATH = BASE_DIR / "data" / "chromadb"
COLLECTION_NAME = "arxiv_ml_papers"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_TOP_K = 3

class Retriever:
    def __init__(self, top_k: int = DEFAULT_TOP_K):
        self.top_k = top_k
        self.embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        self.client = PersistentClient(path=str(CHROMA_DB_PATH))
        self.collection = self.client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embed_fn
        )
        print(f"[Retriever] Connected to '{COLLECTION_NAME}' "
              f"({self.collection.count()} documents)")
        
    def retrieve(self, query: str) -> list[dict]:
        """
        Query ChromaDB and retrun top-k results.
        
        Returns a list of dicts, each with:
            - text      : the raw chunk text (title + abstract)
            - metadata  : arxiv_id, title, authors, categories, url
            - distance  : lower = more similar
        """
        
        results = self.collection.query(
            query_texts=[query],
            n_results=self.top_k
        )
        
        retrieved = []
        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            retrieved.append({
                "text": text,
                "metadata": metadata,
                "distance": round(distance, 4)
            })
            
        return retrieved
    
    def format_context(self, retrieved: list[dict]) -> str:
        """
        Formats retrieved chunks into a single context string
        to be injected into the LLM prompt.
        """
        sections = []
        for i, r in enumerate(retrieved, 1):
            sections.append(
                f"[Paper {i}] {r['metadata']['title']}\n"
                f"{r['text']}\n"
                f"Source: {r['metadata']['url']}"
            )
        return "\n\n---\n\n".join(sections)
        
if __name__=="__main__":
    # Quick test when run directly.
    retriever = Retriever(top_k=3)
    
    test_query = [
        "How do transformers use attention to process sequences?",
        "What are the limitations of large language models?",
        "Reinforcement learning from human feedback"
    ]
    
    for query in test_query:
        print(f"\n>>> Query: {query}")
        results = retriever.retrieve(query)
        for r in results:
            print(f"    [{r['distance']}] {r['metadata']['title']}")