"""
FAISS-backed vector store.

Design choice: FAISS (IndexFlatIP over L2-normalized vectors, i.e. cosine
similarity) over Chroma/Pinecone/etc:
  - Knowledge base here is small (dozens-to-low-thousands of chunks per
    branch chain) -> an exact flat index is fast enough; no need for
    approximate-nearest-neighbor indexes (IVF/HNSW) or a managed service.
  - No external service/infra dependency -- index is a local file, which
    keeps the whole RAG agent self-contained and easy to grade/run.
  - Easy upgrade path: swapping IndexFlatIP for IndexIVFFlat/HNSW later is a
    one-line change if the KB grows to the "1M documents" scale discussed in
    the assessment, without touching the retriever/agent code above it.
"""
from pathlib import Path
from typing import List, Dict, Any
import json
import numpy as np
import faiss


class FAISSVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict[str, Any]] = []

    def add(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        if vectors.shape[0] != len(metadatas):
            raise ValueError("vectors and metadatas must be the same length")
        vectors = np.ascontiguousarray(vectors.astype("float32"))
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.metadata.extend(metadatas)

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        query_vector = np.ascontiguousarray(query_vector.astype("float32"))
        faiss.normalize_L2(query_vector)
        scores, idxs = self.index.search(query_vector, min(k, len(self.metadata) or 1))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append({"score": float(score), **self.metadata[idx]})
        return results

    def save(self, dir_path: str) -> None:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "metadata.json", "w") as f:
            json.dump({"dim": self.dim, "metadata": self.metadata}, f)

    @classmethod
    def load(cls, dir_path: str) -> "FAISSVectorStore":
        path = Path(dir_path)
        with open(path / "metadata.json") as f:
            payload = json.load(f)
        store = cls(dim=payload["dim"])
        store.index = faiss.read_index(str(path / "index.faiss"))
        store.metadata = payload["metadata"]
        return store

    def __len__(self) -> int:
        return len(self.metadata)
