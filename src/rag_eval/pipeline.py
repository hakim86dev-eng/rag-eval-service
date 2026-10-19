"""The RAG pipeline: ingest -> chunk -> embed -> index, and retrieve -> generate -> cite."""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Sequence

from rag_eval.config import RetrievalMode, Settings
from rag_eval.embeddings import EmbeddingProvider, HashingEmbeddings, OpenAIEmbeddings
from rag_eval.generation import (
    AnswerGenerator,
    ExtractiveAnswerGenerator,
    LLMAnswerGenerator,
    OpenAIChatClient,
)
from rag_eval.ingestion import Chunker
from rag_eval.models import Citation, Document, IngestStats, RAGAnswer, ScoredChunk
from rag_eval.retrieval import InMemoryVectorStore, Retriever

_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cited_sources(answer: str) -> list[int]:
    """Return the distinct 1-based source numbers cited in ``answer``, in order."""
    seen: list[int] = []
    for group in _CITE_RE.findall(answer):
        for raw in group.split(","):
            n = int(raw.strip())
            if n not in seen:
                seen.append(n)
    return seen


class RAGPipeline:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        generator: AnswerGenerator,
        chunker: Chunker | None = None,
        store: InMemoryVectorStore | None = None,
        top_k: int = 4,
        retrieval_mode: RetrievalMode = "hybrid",
    ) -> None:
        self.embedder = embedder
        self.generator = generator
        self.chunker = chunker or Chunker()
        self.store = store or InMemoryVectorStore()
        self.retriever = Retriever(self.store, embedder)
        self.top_k = top_k
        self.retrieval_mode: RetrievalMode = retrieval_mode
        self._write_lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> RAGPipeline:
        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
        embedder: EmbeddingProvider
        if settings.embedding_provider == "openai":
            embedder = OpenAIEmbeddings(
                api_key=key,
                model=settings.embedding_model,
                base_url=settings.openai_base_url,
                timeout_s=settings.request_timeout_s,
            )
        else:
            embedder = HashingEmbeddings(dim=settings.hashing_dim)

        generator: AnswerGenerator
        if settings.llm_provider == "openai":
            generator = LLMAnswerGenerator(
                OpenAIChatClient(
                    api_key=key,
                    model=settings.chat_model,
                    base_url=settings.openai_base_url,
                    timeout_s=settings.request_timeout_s,
                )
            )
        else:
            generator = ExtractiveAnswerGenerator()

        store = (
            InMemoryVectorStore.load(settings.index_path)
            if settings.index_path and settings.index_path.exists()
            else None
        )
        return cls(
            embedder=embedder,
            generator=generator,
            chunker=Chunker(settings.chunk_size, settings.chunk_overlap),
            store=store,
            top_k=settings.top_k,
            retrieval_mode=settings.retrieval_mode,
        )

    # ------------------------------------------------------------------ ingestion
    def ingest(self, documents: Sequence[Document]) -> IngestStats:
        """Chunk, embed and index documents. Re-ingesting a document id replaces it."""
        chunks = [chunk for doc in documents for chunk in self.chunker.split(doc)]
        vectors = self.embedder.embed([c.text for c in chunks]) if chunks else []
        with self._write_lock:
            for doc in documents:
                self.store.delete_document(doc.id)
            self.store.add(chunks, vectors)
            self.retriever.refresh()
        return IngestStats(documents=len(documents), chunks=len(chunks))

    def delete(self, doc_id: str) -> int:
        with self._write_lock:
            removed = self.store.delete_document(doc_id)
            self.retriever.refresh()
        return removed

    # ------------------------------------------------------------------ querying
    def retrieve(
        self, question: str, k: int | None = None, mode: RetrievalMode | None = None
    ) -> list[ScoredChunk]:
        return self.retriever.retrieve(question, k or self.top_k, mode or self.retrieval_mode)

    def answer_with_contexts(
        self, question: str, k: int | None = None, mode: RetrievalMode | None = None
    ) -> tuple[RAGAnswer, list[ScoredChunk]]:
        started = time.perf_counter()
        resolved_mode: RetrievalMode = mode or self.retrieval_mode
        contexts = self.retrieve(question, k, resolved_mode)
        text = self.generator.generate(question, contexts)
        citations = [
            Citation(
                source=n,
                chunk_id=contexts[n - 1].chunk.id,
                doc_id=contexts[n - 1].chunk.doc_id,
                score=round(contexts[n - 1].score, 6),
                snippet=contexts[n - 1].chunk.text[:240],
            )
            for n in cited_sources(text)
            if 1 <= n <= len(contexts)
        ]
        answer = RAGAnswer(
            question=question,
            answer=text,
            citations=citations,
            retrieval_mode=resolved_mode,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        return answer, contexts

    def answer(
        self, question: str, k: int | None = None, mode: RetrievalMode | None = None
    ) -> RAGAnswer:
        return self.answer_with_contexts(question, k, mode)[0]
