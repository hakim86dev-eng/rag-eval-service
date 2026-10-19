from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from rag_eval.api import create_app
from rag_eval.config import Settings
from tests.conftest import CORPUS_DIR, EVAL_PATH


@pytest.fixture
def client() -> Iterator[TestClient]:
    settings = Settings(corpus_dir=CORPUS_DIR, eval_path=EVAL_PATH, _env_file=None)
    with TestClient(create_app(settings)) as c:
        yield c


def test_health_reports_ingested_corpus(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["documents"] == 6
    assert body["chunks"] > 6
    assert body["embedder"].startswith("hashing")


def test_query_returns_grounded_answer_with_citation(client: TestClient) -> None:
    r = client.post("/v1/query", json={"question": "Where is the Middle East region hosted?"})
    assert r.status_code == 200
    body = r.json()
    assert "Dubai" in body["answer"]
    assert body["citations"][0]["doc_id"] == "data-retention"
    assert body["retrieval_mode"] == "hybrid"


def test_query_unanswerable_refuses(client: TestClient) -> None:
    body = client.post(
        "/v1/query", json={"question": "Who is the chief executive officer of Nimbus Cloud?"}
    ).json()
    assert "don't know" in body["answer"]
    assert body["citations"] == []


def test_retrieve_endpoint_modes(client: TestClient) -> None:
    for mode in ("dense", "sparse", "hybrid"):
        r = client.post("/v1/retrieve", json={"question": "HTTP 429", "top_k": 2, "mode": mode})
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) == 2
        assert results[0]["chunk"]["doc_id"] == "api-rate-limits"


def test_document_lifecycle(client: TestClient) -> None:
    doc = {"id": "travel-policy", "text": "Employees fly economy on flights under six hours."}
    r = client.post("/v1/documents", json={"documents": [doc]})
    assert r.status_code == 201
    assert r.json() == {"documents": 1, "chunks": 1}
    ids = [d["id"] for d in client.get("/v1/documents").json()]
    assert "travel-policy" in ids
    answer = client.post("/v1/query", json={"question": "Do employees fly economy on flights?"})
    assert answer.json()["citations"][0]["doc_id"] == "travel-policy"
    assert client.delete("/v1/documents/travel-policy").status_code == 204
    assert client.delete("/v1/documents/travel-policy").status_code == 404


@pytest.mark.parametrize(
    "payload",
    [{"question": ""}, {"question": "x", "top_k": 0}, {"question": "x", "mode": "fuzzy"}],
)
def test_query_validation(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post("/v1/query", json=payload).status_code == 422


def test_ingest_validation(client: TestClient) -> None:
    bad = {"documents": [{"id": "has spaces", "text": "x"}]}
    assert client.post("/v1/documents", json=bad).status_code == 422
    assert client.post("/v1/documents", json={"documents": []}).status_code == 422


def test_eval_endpoint(client: TestClient) -> None:
    r = client.post("/v1/eval", json={"top_k": 3, "include_results": True})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["examples"] == 18
    assert body["summary"]["hit_rate"] >= 0.9
    assert len(body["results"]) == 18
    assert body["markdown"].startswith("## RAG evaluation report")


def test_openapi_schema_available(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/v1/query", "/v1/retrieve", "/v1/documents", "/v1/eval", "/health"} <= set(paths)


def test_settings_validation() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        Settings(chunk_size=200, chunk_overlap=200, _env_file=None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Settings(llm_provider="openai", _env_file=None)
