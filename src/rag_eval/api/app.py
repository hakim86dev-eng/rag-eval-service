"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status

from rag_eval import __version__
from rag_eval.api.schemas import (
    DocumentInfo,
    EvalRequest,
    EvalResponse,
    HealthResponse,
    IngestRequest,
    QueryRequest,
    RetrieveResponse,
)
from rag_eval.config import Settings, get_settings
from rag_eval.eval import load_dataset, run_evaluation
from rag_eval.ingestion import load_directory
from rag_eval.models import IngestStats, RAGAnswer
from rag_eval.pipeline import RAGPipeline

logger = logging.getLogger("rag_eval")


def get_pipeline(request: Request) -> RAGPipeline:
    pipeline: RAGPipeline = request.app.state.pipeline
    return pipeline


PipelineDep = Annotated[RAGPipeline, Depends(get_pipeline)]


def create_app(settings: Settings | None = None, pipeline: RAGPipeline | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pipeline = pipeline or RAGPipeline.from_settings(settings)
        store_is_empty = len(app.state.pipeline.store) == 0
        if settings.ingest_on_startup and settings.corpus_dir.is_dir() and store_is_empty:
            stats = app.state.pipeline.ingest(load_directory(settings.corpus_dir))
            logger.info("ingested %s documents (%s chunks)", stats.documents, stats.chunks)
        yield
        if settings.index_path:
            app.state.pipeline.store.save(settings.index_path)

    app = FastAPI(
        title="RAG Eval Service",
        version=__version__,
        description="Retrieval-augmented generation with hybrid search, citations and evaluation.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(p: PipelineDep) -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            documents=len(p.store.doc_ids()),
            chunks=len(p.store),
            embedder=p.embedder.name,
            generator=p.generator.name,
        )

    @app.post(
        "/v1/documents",
        response_model=IngestStats,
        status_code=status.HTTP_201_CREATED,
        tags=["documents"],
    )
    def ingest(body: IngestRequest, p: PipelineDep) -> IngestStats:
        return p.ingest(body.documents)

    @app.get("/v1/documents", response_model=list[DocumentInfo], tags=["documents"])
    def list_documents(p: PipelineDep) -> list[DocumentInfo]:
        return [DocumentInfo(id=d, chunks=n) for d, n in sorted(p.store.doc_ids().items())]

    @app.delete(
        "/v1/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["documents"]
    )
    def delete_document(doc_id: str, p: PipelineDep) -> None:
        if p.delete(doc_id) == 0:
            raise HTTPException(status_code=404, detail=f"document '{doc_id}' not found")

    @app.post("/v1/retrieve", response_model=RetrieveResponse, tags=["query"])
    def retrieve(body: QueryRequest, p: PipelineDep) -> RetrieveResponse:
        mode = body.mode or p.retrieval_mode
        return RetrieveResponse(
            question=body.question, mode=mode, results=p.retrieve(body.question, body.top_k, mode)
        )

    @app.post("/v1/query", response_model=RAGAnswer, tags=["query"])
    def query(body: QueryRequest, p: PipelineDep) -> RAGAnswer:
        return p.answer(body.question, body.top_k, body.mode)

    @app.post("/v1/eval", response_model=EvalResponse, tags=["evaluation"])
    def evaluate(body: EvalRequest, p: PipelineDep) -> EvalResponse:
        if not settings.eval_path.exists():
            raise HTTPException(status_code=404, detail="evaluation dataset not found")
        report = run_evaluation(p, load_dataset(settings.eval_path), k=body.top_k, mode=body.mode)
        return EvalResponse(
            config=report.config,
            summary=report.summary,
            markdown=report.to_markdown(),
            results=[r.model_dump() for r in report.results] if body.include_results else None,
        )

    return app
