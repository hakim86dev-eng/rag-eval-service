from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval.embeddings import HashingEmbeddings
from rag_eval.models import Chunk, ScoredChunk
from rag_eval.pipeline import RAGPipeline
from rag_eval.retrieval import BM25Index, InMemoryVectorStore, Retriever, reciprocal_rank_fusion


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, doc_id=cid.split("::")[0], index=0, text=text)


CHUNKS = [
    _chunk("a::0", "Refunds are issued within 14 days of the first invoice."),
    _chunk("b::0", "Data is encrypted at rest with AES-256."),
    _chunk("c::0", "Rate limits return HTTP 429 with a Retry-After header."),
]


def test_vector_store_search_and_delete() -> None:
    emb = HashingEmbeddings(dim=256)
    store = InMemoryVectorStore()
    store.add(CHUNKS, emb.embed([c.text for c in CHUNKS]))
    top = store.search(emb.embed(["encrypted at rest"])[0], k=1)
    assert top[0].chunk.id == "b::0"
    assert store.delete_document("b") == 1
    assert len(store) == 2
    assert store.doc_ids() == {"a": 1, "c": 1}


def test_vector_store_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        InMemoryVectorStore().add(CHUNKS, [[1.0]])


def test_vector_store_persistence_roundtrip(tmp_path: Path) -> None:
    emb = HashingEmbeddings(dim=128)
    store = InMemoryVectorStore()
    store.add(CHUNKS, emb.embed([c.text for c in CHUNKS]))
    path = tmp_path / "index" / "store.json"
    store.save(path)
    loaded = InMemoryVectorStore.load(path)
    q = emb.embed(["HTTP 429"])[0]
    assert [s.chunk.id for s in loaded.search(q, 3)] == [s.chunk.id for s in store.search(q, 3)]


def test_bm25_ranks_exact_terms_first() -> None:
    index = BM25Index(CHUNKS)
    results = index.search("retry-after header 429", k=3)
    assert results[0].chunk.id == "c::0"
    assert index.search("zzz unknown", k=3) == []
    assert BM25Index().search("anything", k=3) == []


def test_rrf_rewards_agreement() -> None:
    a, b, c = (ScoredChunk(chunk=ch, score=1.0) for ch in CHUNKS)
    fused = reciprocal_rank_fusion([[a, b, c], [b, a, c], [b, c, a]])
    assert [s.chunk.id for s in fused] == ["b::0", "a::0", "c::0"]


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_retriever_modes(mode: str) -> None:
    emb = HashingEmbeddings(dim=256)
    store = InMemoryVectorStore()
    store.add(CHUNKS, emb.embed([c.text for c in CHUNKS]))
    retriever = Retriever(store, emb)
    results = retriever.retrieve("How are refunds issued?", k=2, mode=mode)  # type: ignore[arg-type]
    assert results[0].chunk.id == "a::0"
    assert retriever.retrieve("anything", k=0) == []


def test_pipeline_reingest_replaces_document(pipeline: RAGPipeline) -> None:
    from rag_eval.models import Document

    before = len(pipeline.store)
    pipeline.ingest([Document(id="billing-and-refunds", text="Short replacement text.")])
    assert pipeline.store.doc_ids()["billing-and-refunds"] == 1
    assert len(pipeline.store) < before
    hits = pipeline.retrieve("replacement", k=1, mode="sparse")
    assert hits[0].chunk.doc_id == "billing-and-refunds"
