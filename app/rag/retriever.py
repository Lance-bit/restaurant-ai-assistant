"""
Retrieval strategy: top-k dense retrieval with two filtering passes.

  1. Branch filter (metadata filter, not vector filter): if the user's query
     is scoped to a branch, we drop chunks that don't apply to that branch
     BEFORE truncating to k. This prevents e.g. recommending a Downtown-only
     dish to someone asking about the Maadi branch.
  2. Score threshold: chunks below a minimum cosine-similarity score are
     dropped even if they'd otherwise fill out the top-k. This is the first
     line of hallucination defense -- if nothing clears the bar, the agent
     is told "no relevant context" and must say it doesn't know, rather than
     stretching a weak match into an answer.

We overfetch (k * OVERFETCH_FACTOR) before filtering so that branch
filtering doesn't starve the result set.
"""
from typing import List, Dict, Any, Optional
from app.rag.embeddings import EmbeddingProvider
from app.rag.vectorstore import FAISSVectorStore

OVERFETCH_FACTOR = 4


class Retriever:
    def __init__(
        self,
        vectorstore: FAISSVectorStore,
        embedder: EmbeddingProvider,
        score_threshold: float = 0.25,
    ):
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.score_threshold = score_threshold

    def retrieve(
        self, query: str, branch: Optional[str] = None, k: int = 4
    ) -> List[Dict[str, Any]]:
        query_vector = self.embedder.embed([query])
        raw_results = self.vectorstore.search(query_vector, k=k * OVERFETCH_FACTOR)

        results = [r for r in raw_results if r["score"] >= self.score_threshold]

        if branch:
            branch_filtered = [
                r for r in results
                if branch in r.get("branches", []) or "all" in r.get("branches", [])
            ]
            # Only apply the branch filter if it doesn't wipe out everything --
            # otherwise fall back to unfiltered so we can at least surface a
            # "not available at your branch" style answer instead of silence.
            if branch_filtered:
                results = branch_filtered

        return results[:k]
