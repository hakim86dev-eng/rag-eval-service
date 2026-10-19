from __future__ import annotations

import json

import httpx
import pytest

from rag_eval.generation import (
    REFUSAL,
    SYSTEM_PROMPT,
    ExtractiveAnswerGenerator,
    FakeChatClient,
    LLMAnswerGenerator,
    OpenAIChatClient,
    build_user_prompt,
)
from rag_eval.models import Chunk, ScoredChunk
from rag_eval.pipeline import RAGPipeline, cited_sources


def _ctx(text: str, cid: str = "d::0") -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(id=cid, doc_id=cid.split("::")[0], index=0, text=text), score=1)


def test_prompt_numbers_sources() -> None:
    prompt = build_user_prompt("Q?", [_ctx("first", "a::0"), _ctx("second", "b::0")])
    assert "[1] (doc: a)\nfirst" in prompt
    assert "[2] (doc: b)\nsecond" in prompt
    assert prompt.endswith("Question: Q?\nAnswer:")


def test_extractive_generator_cites_best_sentence() -> None:
    gen = ExtractiveAnswerGenerator()
    ctx = [_ctx("Unrelated sentence here. Refunds take 5 business days to process.")]
    answer = gen.generate("How long do refunds take to process?", ctx)
    assert answer == "Refunds take 5 business days to process. [1]"


def test_extractive_generator_refuses_without_support() -> None:
    gen = ExtractiveAnswerGenerator()
    assert gen.generate("Who founded the company?", [_ctx("Data is encrypted.")]) == REFUSAL
    assert gen.generate("anything", []) == REFUSAL


def test_llm_generator_uses_system_prompt_and_context() -> None:
    fake = FakeChatClient(responses=["Refunds take 5 days [1]."])
    gen = LLMAnswerGenerator(fake)
    assert gen.generate("refunds?", [_ctx("Refunds take 5 days.")]) == "Refunds take 5 days [1]."
    system, user = fake.calls[0]
    assert system == SYSTEM_PROMPT
    assert "Refunds take 5 days." in user
    assert gen.generate("refunds?", []) == REFUSAL
    assert len(fake.calls) == 1  # no LLM call without context


def test_fake_chat_without_script_raises() -> None:
    with pytest.raises(RuntimeError):
        FakeChatClient().complete("s", "u")


def test_openai_chat_client_request_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "gpt-4o-mini"
        assert body["temperature"] == 0.0
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
        return httpx.Response(200, json={"choices": [{"message": {"content": " Answer [1] "}}]})

    client = httpx.Client(base_url="https://api.test/v1", transport=httpx.MockTransport(handler))
    assert OpenAIChatClient(api_key="k", client=client).complete("s", "u") == "Answer [1]"


def test_cited_sources_parsing() -> None:
    assert cited_sources("A [1]. B [3, 1]. C [2]") == [1, 3, 2]
    assert cited_sources("no citations") == []


def test_pipeline_with_mock_llm_builds_citations(pipeline: RAGPipeline) -> None:
    pipeline.generator = LLMAnswerGenerator(
        FakeChatClient(responses=["Pro keys get 600 requests per minute [1]. Bogus [9]."])
    )
    answer = pipeline.answer("What is the Pro tier rate limit?", k=2)
    assert [c.source for c in answer.citations] == [1]  # out-of-range [9] is dropped
    assert answer.citations[0].doc_id == "api-rate-limits"
    assert answer.latency_ms >= 0
