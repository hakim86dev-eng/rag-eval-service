"""Retrieval and generation metrics.

Retrieval: hit-rate@k, recall@k, reciprocal rank.
Generation: token-level F1 against a reference, refusal detection, and faithfulness
(the share of answer sentences supported by the retrieved context). Faithfulness has a fast
lexical implementation for CI and an optional LLM-as-judge implementation for deeper runs.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence

from rag_eval.generation.chat import ChatClient
from rag_eval.text import split_sentences, strip_citations, tokenize

_REFUSAL_PATTERNS = ("i don't know", "i do not know", "not in the provided", "cannot answer")


def hit_rate(retrieved_doc_ids: Sequence[str], expected: Sequence[str]) -> float:
    return 1.0 if set(retrieved_doc_ids) & set(expected) else 0.0


def recall_at_k(retrieved_doc_ids: Sequence[str], expected: Sequence[str]) -> float:
    if not expected:
        return 0.0
    return len(set(retrieved_doc_ids) & set(expected)) / len(set(expected))


def reciprocal_rank(retrieved_doc_ids: Sequence[str], expected: Sequence[str]) -> float:
    wanted = set(expected)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in wanted:
            return 1.0 / rank
    return 0.0


def is_refusal(answer: str) -> bool:
    lowered = answer.lower().replace("\u2019", "'")
    return any(p in lowered for p in _REFUSAL_PATTERNS)


def answer_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token F1 between prediction and reference (citations stripped)."""
    pred = tokenize(strip_citations(prediction))
    ref = tokenize(reference)
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def faithfulness(answer: str, contexts: Sequence[str], threshold: float = 0.6) -> float:
    """Fraction of answer sentences whose content tokens are mostly present in the context.

    A refusal is trivially faithful (it makes no claims). Returns a value in [0, 1].
    """
    if is_refusal(answer):
        return 1.0
    context_terms = set(tokenize(" ".join(contexts)))
    sentences = [strip_citations(s) for s in split_sentences(answer)]
    sentences = [s for s in sentences if tokenize(s)]
    if not sentences:
        return 0.0
    supported = 0
    for sentence in sentences:
        terms = tokenize(sentence)
        coverage = sum(1 for t in terms if t in context_terms) / len(terms)
        supported += coverage >= threshold
    return supported / len(sentences)


_JUDGE_SYSTEM = (
    "You grade whether an ANSWER is fully supported by the CONTEXT. "
    'Respond with JSON only: {"score": <float between 0 and 1>, "reason": "<short>"}.'
)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMJudgeFaithfulness:
    """LLM-as-judge faithfulness. Falls back to the lexical metric on malformed output."""

    def __init__(self, chat: ChatClient) -> None:
        self.chat = chat

    def __call__(self, answer: str, contexts: Sequence[str]) -> float:
        if is_refusal(answer):
            return 1.0
        prompt = "CONTEXT:\n" + "\n---\n".join(contexts) + f"\n\nANSWER:\n{answer}"
        raw = self.chat.complete(_JUDGE_SYSTEM, prompt)
        match = _JSON_RE.search(raw)
        try:
            score = float(json.loads(match.group(0))["score"]) if match else float("nan")
        except (ValueError, KeyError, TypeError):
            score = float("nan")
        if score != score:  # NaN -> malformed judge output
            return faithfulness(answer, contexts)
        return min(1.0, max(0.0, score))
