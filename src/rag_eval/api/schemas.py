"""Request/response schemas for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from rag_eval.config import RetrievalMode
from rag_eval.eval.harness import EvalSummary
from rag_eval.models import Document, ScoredChunk


class IngestRequest(BaseModel):
    documents: list[Document] = Field(min_length=1, max_length=500)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    mode: RetrievalMode | None = None


class RetrieveResponse(BaseModel):
    question: str
    mode: RetrievalMode
    results: list[ScoredChunk]


class DocumentInfo(BaseModel):
    id: str
    chunks: int


class EvalRequest(BaseModel):
    top_k: int | None = Field(default=None, ge=1, le=20)
    mode: RetrievalMode | None = None
    include_results: bool = False


class EvalResponse(BaseModel):
    config: dict[str, str | int]
    summary: EvalSummary
    markdown: str
    results: list[dict[str, object]] | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    documents: int
    chunks: int
    embedder: str
    generator: str
