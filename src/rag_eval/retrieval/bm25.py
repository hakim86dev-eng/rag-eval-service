"""Okapi BM25 sparse retrieval over chunks."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from rag_eval.models import Chunk, ScoredChunk
from rag_eval.text import tokenize


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk] = (), k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._tfs: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._df: Counter[str] = Counter()
        self._avgdl = 0.0
        self.rebuild(chunks)

    def rebuild(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._tfs = [Counter(tokenize(c.text)) for c in self._chunks]
        self._lengths = [sum(tf.values()) for tf in self._tfs]
        self._df = Counter(term for tf in self._tfs for term in tf)
        self._avgdl = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0

    def _idf(self, term: str) -> float:
        n = len(self._chunks)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        terms = tokenize(query)
        if not terms or not self._chunks:
            return []
        scores: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tfs):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * self._lengths[i] / (self._avgdl or 1.0))
            for term in terms:
                freq = tf.get(term, 0)
                if freq:
                    score += self._idf(term) * freq * (self.k1 + 1) / (freq + norm)
            if score > 0:
                scores.append((score, i))
        scores.sort(key=lambda s: (-s[0], s[1]))
        return [ScoredChunk(chunk=self._chunks[i], score=s) for s, i in scores[:k]]
