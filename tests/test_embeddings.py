from __future__ import annotations

import json
import math

import httpx
import pytest

from rag_eval.embeddings import EmbeddingProvider, HashingEmbeddings, OpenAIEmbeddings
from rag_eval.embeddings.base import dot


def test_hashing_is_deterministic_and_normalised() -> None:
    emb = HashingEmbeddings(dim=256)
    a1, a2 = emb.embed(["refund policy", "refund policy"])
    assert a1 == a2
    assert len(a1) == 256
    assert math.isclose(math.sqrt(dot(a1, a1)), 1.0, rel_tol=1e-9)
    assert isinstance(emb, EmbeddingProvider)


def test_hashing_similarity_orders_related_text_higher() -> None:
    emb = HashingEmbeddings(dim=1024)
    q, related, unrelated = emb.embed(
        ["refund within 14 days", "full refund within 14 days of invoice", "TLS encryption keys"]
    )
    assert dot(q, related) > dot(q, unrelated)


def test_empty_text_gives_zero_vector() -> None:
    assert set(HashingEmbeddings(dim=64).embed([""])[0]) == {0.0}


def test_invalid_dim() -> None:
    with pytest.raises(ValueError):
        HashingEmbeddings(dim=0)


def test_openai_embeddings_batches_and_orders_results() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer sk-test"
        body = json.loads(request.content)
        seen.append(body)
        data = [
            {"index": i, "embedding": [float(len(text)), 0.0]}
            for i, text in reversed(list(enumerate(body["input"])))
        ]
        return httpx.Response(200, json={"data": data})

    client = httpx.Client(base_url="https://api.test/v1", transport=httpx.MockTransport(handler))
    emb = OpenAIEmbeddings(api_key="sk-test", batch_size=2, client=client)
    vectors = emb.embed(["a", "bb", "ccc"])
    assert len(seen) == 2
    assert seen[0]["model"] == "text-embedding-3-small"
    assert vectors == [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]  # normalised
    assert emb.name == "openai:text-embedding-3-small"


def test_openai_embeddings_raises_on_http_error() -> None:
    client = httpx.Client(
        base_url="https://api.test/v1",
        transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"error": "bad key"})),
    )
    with pytest.raises(httpx.HTTPStatusError):
        OpenAIEmbeddings(api_key="x", client=client).embed(["hi"])
