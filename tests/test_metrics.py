from __future__ import annotations

import pytest

from rag_eval.eval import (
    LLMJudgeFaithfulness,
    answer_f1,
    faithfulness,
    hit_rate,
    is_refusal,
    recall_at_k,
    reciprocal_rank,
)
from rag_eval.generation import REFUSAL, FakeChatClient


def test_retrieval_metrics() -> None:
    retrieved = ["x", "a", "b"]
    assert hit_rate(retrieved, ["a"]) == 1.0
    assert hit_rate(retrieved, ["z"]) == 0.0
    assert recall_at_k(retrieved, ["a", "z"]) == 0.5
    assert recall_at_k(retrieved, []) == 0.0
    assert reciprocal_rank(retrieved, ["a"]) == 0.5
    assert reciprocal_rank(retrieved, ["z"]) == 0.0


def test_answer_f1() -> None:
    assert answer_f1("Refunds within 14 days [1]", "Refunds within 14 days") == 1.0
    assert answer_f1("completely different", "refunds within 14 days") == 0.0
    assert 0 < answer_f1("refunds take 14 days", "refunds within 14 days") < 1
    assert answer_f1("", "") == 1.0


def test_refusal_detection() -> None:
    assert is_refusal(REFUSAL)
    assert is_refusal("I don\u2019t know.")
    assert not is_refusal("The answer is 42.")


def test_faithfulness_lexical() -> None:
    ctx = ["Refunds are processed within 5 business days."]
    assert faithfulness("Refunds are processed within 5 business days [1].", ctx) == 1.0
    half = "Refunds are processed within 5 business days. The CEO is Alice Smith from Paris."
    assert faithfulness(half, ctx) == 0.5
    assert faithfulness(REFUSAL, ctx) == 1.0
    assert faithfulness("", ctx) == 0.0


@pytest.mark.parametrize(
    ("judge_output", "expected"),
    [('{"score": 0.8, "reason": "ok"}', 0.8), ('Sure! {"score": 7}', 1.0), ("garbage", 1.0)],
)
def test_llm_judge(judge_output: str, expected: float) -> None:
    fake = FakeChatClient(responses=[judge_output])
    judge = LLMJudgeFaithfulness(fake)
    # "garbage" falls back to the lexical metric, which fully supports this answer.
    assert judge("Refunds take 5 days.", ["Refunds take 5 days."]) == pytest.approx(expected)
    assert "CONTEXT" in fake.calls[0][1]


def test_llm_judge_skips_refusals() -> None:
    fake = FakeChatClient(responses=["unused"])
    assert LLMJudgeFaithfulness(fake)(REFUSAL, ["ctx"]) == 1.0
    assert fake.calls == []
