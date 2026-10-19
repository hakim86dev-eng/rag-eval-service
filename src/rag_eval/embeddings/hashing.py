"""Deterministic, dependency-free embeddings using signed feature hashing.

This is *not* a semantic model; it is a bag-of-words/bigram projection. It exists so the whole
service (and its CI) runs offline, reproducibly and in milliseconds. Swap in
:class:`~rag_eval.embeddings.openai.OpenAIEmbeddings` for production-quality vectors.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Sequence

from rag_eval.embeddings.base import Vector, l2_normalize
from rag_eval.text import tokenize


class HashingEmbeddings:
    def __init__(self, dim: int = 1024) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    @property
    def name(self) -> str:
        return f"hashing-{self.dim}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def _embed_one(self, text: str) -> Vector:
        vec = [0.0] * self.dim
        tokens = tokenize(text)
        features = tokens + [f"{a}_{b}" for a, b in itertools.pairwise(tokens)]
        for feature in features:
            idx, sign = self._bucket(feature)
            vec[idx] += sign
        return l2_normalize(vec)

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._embed_one(t) for t in texts]
