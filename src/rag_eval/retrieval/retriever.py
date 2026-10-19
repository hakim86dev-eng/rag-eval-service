"""Dense, sparse and hybrid (Reciprocal Rank Fusion) retrieval."""

from __future__ import annotations

from collections.abc import Sequence

from rag_eval.config import RetrievalMode
from rag_eval.embeddings.base import EmbeddingProvider
from rag_eval.models import ScoredChunk
from rag_eval.retrieval.bm25 import BM25Index
from rag_eval.retrieval.vector_store import InMemoryVectorStore


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], k: int = 60
) -> list[ScoredChunk]:
    """Fuse several ranked lists. Score = sum(1 / (k + rank)). Robust to score-scale mismatch."""
    fused: dict[str, float] = {}
    by_id: dict[str, ScoredChunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item.chunk.id] = fused.get(item.chunk.id, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(item.chunk.id, item)
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return [ScoredChunk(chunk=by_id[cid].chunk, score=score) for cid, score in ordered]


class Retriever:
    def __init__(
        self,
        store: InMemoryVectorStore,
        embedder: EmbeddingProvider,
        bm25: BM25Index | None = None,
        candidate_multiplier: int = 3,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25 or BM25Index(store.chunks)
        self.candidate_multiplier = candidate_multiplier

    def refresh(self) -> None:
        """Rebuild the sparse index after the store changes."""
        self.bm25.rebuild(self.store.chunks)

    def retrieve(self, query: str, k: int, mode: RetrievalMode = "hybrid") -> list[ScoredChunk]:
        if k <= 0:
            return []
        if mode == "sparse":
            return self.bm25.search(query, k)
        query_vec = self.embedder.embed([query])[0]
        if mode == "dense":
            return self.store.search(query_vec, k)
        pool = k * self.candidate_multiplier
        dense = self.store.search(query_vec, pool)
        sparse = self.bm25.search(query, pool)
        return reciprocal_rank_fusion([dense, sparse])[:k]
