"""Embedding provider protocol."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

Vector = list[float]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Anything that turns texts into fixed-size vectors."""

    @property
    def name(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> list[Vector]: ...


def l2_normalize(vector: Sequence[float]) -> Vector:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return [0.0 for _ in vector]
    return [x / norm for x in vector]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))
