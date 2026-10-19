"""A thread-safe in-memory vector store with JSON persistence.

Kept intentionally small: the :class:`Retriever` depends only on ``add``/``search``/``chunks``,
so replacing this with pgvector, Qdrant or OpenSearch is a contained change.
"""

from __future__ import annotations

import heapq
import json
import threading
from collections.abc import Sequence
from pathlib import Path

from rag_eval.embeddings.base import Vector, dot, l2_normalize
from rag_eval.models import Chunk, ScoredChunk


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, Vector] = {}
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        with self._lock:
            return list(self._chunks.values())

    def doc_ids(self) -> dict[str, int]:
        """Return ``{doc_id: chunk_count}``."""
        counts: dict[str, int] = {}
        for chunk in self.chunks:
            counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
        return counts

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        with self._lock:
            for chunk, vector in zip(chunks, vectors, strict=True):
                self._chunks[chunk.id] = chunk
                self._vectors[chunk.id] = l2_normalize(vector)

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            ids = [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]
            for cid in ids:
                del self._chunks[cid]
                del self._vectors[cid]
            return len(ids)

    def search(self, query: Vector, k: int) -> list[ScoredChunk]:
        q = l2_normalize(query)
        with self._lock:
            scored = ((dot(q, vec), cid) for cid, vec in self._vectors.items())
            top = heapq.nlargest(k, scored)
            return [ScoredChunk(chunk=self._chunks[cid], score=score) for score, cid in top]

    def save(self, path: Path) -> None:
        with self._lock:
            payload = [
                {"chunk": c.model_dump(), "vector": self._vectors[c.id]}
                for c in self._chunks.values()
            ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> InMemoryVectorStore:
        store = cls()
        items = json.loads(path.read_text(encoding="utf-8"))
        store.add([Chunk.model_validate(i["chunk"]) for i in items], [i["vector"] for i in items])
        return store
