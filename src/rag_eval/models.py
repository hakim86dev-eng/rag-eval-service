"""Core domain models shared across ingestion, retrieval, generation and evaluation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A source document before chunking."""

    id: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._\-]+$")
    text: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable slice of a document."""

    id: str
    doc_id: str
    index: int
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    """A chunk together with its retrieval score (higher is better)."""

    chunk: Chunk
    score: float


class Citation(BaseModel):
    """A source referenced by a generated answer."""

    source: int = Field(description="1-based index used in the answer, e.g. [1].")
    chunk_id: str
    doc_id: str
    score: float
    snippet: str


class RAGAnswer(BaseModel):
    """Answer returned by the pipeline."""

    question: str
    answer: str
    citations: list[Citation]
    retrieval_mode: str
    latency_ms: float


class IngestStats(BaseModel):
    """Result of an ingestion call."""

    documents: int
    chunks: int
