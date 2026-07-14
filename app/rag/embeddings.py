"""
Embedding providers.

Design choice: we use a pluggable ABC so the vector store / retriever never
know or care which embedding backend is behind them. Default implementation
is local sentence-transformers (all-MiniLM-L6-v2):

Why all-MiniLM-L6-v2 over an API-based embedding model (e.g. OpenAI's
text-embedding-3-small):
  - Runs fully offline/local -> no per-query API cost or network dependency,
    which matters for a restaurant assistant that may run on-prem at branches.
  - 384-dim vectors keep the FAISS index small and search latency low, which
    is the right tradeoff for a knowledge base of a few hundred menu/policy
    chunks (not millions of documents).
  - "Good enough" semantic quality for short, structured factual text like
    menu items and policy clauses, where retrieval is closer to keyword/topic
    matching than deep semantic reasoning.
  - Deterministic and reproducible for grading/evaluation (recall@k etc.)
    since there's no model-version drift from a hosted API.

For a larger production knowledge base (thousands of long-form documents)
we'd swap in an API embedding model for higher retrieval quality -- the
EmbeddingProvider interface makes that a one-line change.
"""
from abc import ABC, abstractmethod
from typing import List
import numpy as np


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return a (len(texts), dim) float32 array."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError


class SentenceTransformerEmbeddings(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(texts, convert_to_numpy=True)
        return vectors.astype("float32")

    @property
    def dim(self) -> int:
        return self._dim


class HashEmbeddings(EmbeddingProvider):
    """
    Deterministic, dependency-free embedding fallback used for local dev,
    unit tests, and CI where downloading model weights isn't possible.
    NOT for production use -- purely a drop-in so the rest of the pipeline
    (chunking -> FAISS -> retrieval -> groundedness check) can be exercised
    and tested without network access to a model hub.
    """

    def __init__(self, dim: int = 128):
        self._dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self._dim), dtype="float32")
        for i, text in enumerate(texts):
            for token in text.lower().split():
                h = hash(token) % self._dim
                vectors[i, h] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    @property
    def dim(self) -> int:
        return self._dim
