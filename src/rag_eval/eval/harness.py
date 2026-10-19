"""Runs the pipeline over an evaluation dataset and aggregates metrics into a report."""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence

from pydantic import BaseModel

from rag_eval.config import RetrievalMode
from rag_eval.eval.dataset import EvalExample
from rag_eval.eval.metrics import (
    answer_f1,
    faithfulness,
    hit_rate,
    is_refusal,
    recall_at_k,
    reciprocal_rank,
)
from rag_eval.pipeline import RAGPipeline

FaithfulnessFn = Callable[[str, Sequence[str]], float]


class ExampleResult(BaseModel):
    id: str
    question: str
    answerable: bool
    answer: str
    retrieved_doc_ids: list[str]
    hit: float | None
    recall: float | None
    reciprocal_rank: float | None
    answer_f1: float | None
    faithfulness: float
    refused: bool
    latency_ms: float


class EvalSummary(BaseModel):
    examples: int
    answerable: int
    hit_rate: float
    recall: float
    mrr: float
    answer_f1: float
    faithfulness: float
    refusal_accuracy: float | None
    false_refusal_rate: float
    latency_p50_ms: float
    latency_p95_ms: float


class EvalReport(BaseModel):
    config: dict[str, str | int]
    summary: EvalSummary
    results: list[ExampleResult]

    def to_markdown(self) -> str:
        s = self.summary
        rows = [
            ("Hit rate@k", s.hit_rate),
            ("Recall@k", s.recall),
            ("MRR", s.mrr),
            ("Answer F1", s.answer_f1),
            ("Faithfulness", s.faithfulness),
            ("False refusal rate", s.false_refusal_rate),
        ]
        if s.refusal_accuracy is not None:
            rows.append(("Refusal accuracy (unanswerable)", s.refusal_accuracy))
        cfg = ", ".join(f"{k}={v}" for k, v in self.config.items())
        lines = [
            "## RAG evaluation report",
            "",
            f"`{cfg}` - {s.examples} examples ({s.answerable} answerable)",
            "",
            "| Metric | Value |",
            "|---|---|",
            *[f"| {name} | {value:.3f} |" for name, value in rows],
            f"| Latency p50 / p95 (ms) | {s.latency_p50_ms:.1f} / {s.latency_p95_ms:.1f} |",
        ]
        return "\n".join(lines) + "\n"


def _mean(values: Sequence[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return round(ordered[idx], 3)


def run_evaluation(
    pipeline: RAGPipeline,
    dataset: Sequence[EvalExample],
    k: int | None = None,
    mode: RetrievalMode | None = None,
    faithfulness_fn: FaithfulnessFn = faithfulness,
) -> EvalReport:
    k = k or pipeline.top_k
    mode = mode or pipeline.retrieval_mode
    results: list[ExampleResult] = []
    for ex in dataset:
        answer, contexts = pipeline.answer_with_contexts(ex.question, k=k, mode=mode)
        doc_ids = list(dict.fromkeys(c.chunk.doc_id for c in contexts))
        refused = is_refusal(answer.answer)
        results.append(
            ExampleResult(
                id=ex.id,
                question=ex.question,
                answerable=ex.answerable,
                answer=answer.answer,
                retrieved_doc_ids=doc_ids,
                hit=hit_rate(doc_ids, ex.expected_doc_ids) if ex.answerable else None,
                recall=recall_at_k(doc_ids, ex.expected_doc_ids) if ex.answerable else None,
                reciprocal_rank=(
                    reciprocal_rank(doc_ids, ex.expected_doc_ids) if ex.answerable else None
                ),
                answer_f1=(
                    answer_f1(answer.answer, ex.reference_answer)
                    if ex.answerable and ex.reference_answer
                    else None
                ),
                faithfulness=faithfulness_fn(answer.answer, [c.chunk.text for c in contexts]),
                refused=refused,
                latency_ms=answer.latency_ms,
            )
        )

    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    latencies = [r.latency_ms for r in results]
    summary = EvalSummary(
        examples=len(results),
        answerable=len(answerable),
        hit_rate=_mean([r.hit for r in answerable if r.hit is not None]),
        recall=_mean([r.recall for r in answerable if r.recall is not None]),
        mrr=_mean([r.reciprocal_rank for r in answerable if r.reciprocal_rank is not None]),
        answer_f1=_mean([r.answer_f1 for r in answerable if r.answer_f1 is not None]),
        faithfulness=_mean([r.faithfulness for r in results if not r.refused]),
        refusal_accuracy=_mean([float(r.refused) for r in unanswerable]) if unanswerable else None,
        false_refusal_rate=_mean([float(r.refused) for r in answerable]),
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
    )
    config: dict[str, str | int] = {
        "k": k,
        "mode": mode,
        "embedder": pipeline.embedder.name,
        "generator": pipeline.generator.name,
        "chunk_size": pipeline.chunker.chunk_size,
    }
    return EvalReport(config=config, summary=summary, results=results)
