"""
Run once (or whenever data/knowledge_base/*.json changes) to build the
FAISS index from the JSON knowledge base:

    python scripts/build_index.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.ingest import load_json, chunk_menu, chunk_policies
from app.rag.embeddings import SentenceTransformerEmbeddings
from app.rag.vectorstore import FAISSVectorStore

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge_base"
INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "faiss_index"


def main():
    menu_json = load_json(str(DATA_DIR / "menu.json"))
    policies_json = load_json(str(DATA_DIR / "policies.json"))

    chunks = chunk_menu(menu_json) + chunk_policies(policies_json)
    print(f"Built {len(chunks)} chunks from menu.json + policies.json")

    embedder = SentenceTransformerEmbeddings()
    vectors = embedder.embed([c["text"] for c in chunks])

    store = FAISSVectorStore(dim=embedder.dim)
    store.add(vectors, chunks)
    store.save(str(INDEX_DIR))
    print(f"Saved FAISS index ({len(store)} vectors) to {INDEX_DIR}")


if __name__ == "__main__":
    main()
