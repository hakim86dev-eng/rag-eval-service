from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval.embeddings import HashingEmbeddings
from rag_eval.generation import ExtractiveAnswerGenerator
from rag_eval.ingestion import Chunker, load_directory
from rag_eval.models import Document
from rag_eval.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
EVAL_PATH = ROOT / "data" / "eval" / "qa.jsonl"


@pytest.fixture
def corpus() -> list[Document]:
    return load_directory(CORPUS_DIR)


@pytest.fixture
def pipeline(corpus: list[Document]) -> RAGPipeline:
    p = RAGPipeline(
        embedder=HashingEmbeddings(dim=512),
        generator=ExtractiveAnswerGenerator(),
        chunker=Chunker(chunk_size=500, chunk_overlap=80),
        top_k=3,
    )
    p.ingest(corpus)
    return p
