from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from rag_eval.ingestion import Chunker, chunk_text, load_directory
from rag_eval.models import Document
from tests.conftest import CORPUS_DIR

LONG_TEXT = "\n\n".join(
    f"Paragraph {i}. " + " ".join(f"word{i}_{j}" for j in range(40)) + "." for i in range(10)
)


def test_short_text_is_single_chunk() -> None:
    assert chunk_text("Hello world.", chunk_size=100, chunk_overlap=10) == ["Hello world."]


@pytest.mark.parametrize(("size", "overlap"), [(120, 0), (200, 40), (500, 100)])
def test_chunks_respect_size_and_cover_all_words(size: int, overlap: int) -> None:
    chunks = chunk_text(LONG_TEXT, chunk_size=size, chunk_overlap=overlap)
    assert len(chunks) > 1
    assert all(len(c) <= size for c in chunks)
    joined = " ".join(chunks)
    assert all(word in joined for word in LONG_TEXT.split())


def test_overlap_shares_text_between_neighbours() -> None:
    chunks = chunk_text(LONG_TEXT, chunk_size=200, chunk_overlap=60)
    shared = sum(1 for a, b in itertools.pairwise(chunks) if set(a.split()) & set(b.split()))
    assert shared >= len(chunks) // 2


def test_unbreakable_token_is_hard_split() -> None:
    chunks = chunk_text("x" * 250, chunk_size=100, chunk_overlap=0)
    assert chunks == ["x" * 100, "x" * 100, "x" * 50]


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=size, chunk_overlap=overlap)


def test_chunker_assigns_stable_ids_and_metadata() -> None:
    doc = Document(id="doc-1", text=LONG_TEXT, metadata={"title": "T"})
    chunks = Chunker(chunk_size=300, chunk_overlap=50).split(doc)
    assert [c.id for c in chunks] == [f"doc-1::{i}" for i in range(len(chunks))]
    assert all(c.doc_id == "doc-1" and c.metadata["title"] == "T" for c in chunks)


def test_loader_reads_bundled_corpus() -> None:
    docs = load_directory(CORPUS_DIR)
    assert len(docs) == 6
    sla = next(d for d in docs if d.id == "service-level-agreement")
    assert sla.metadata["title"] == "Service Level Agreement"


def test_loader_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_directory(tmp_path / "missing")
