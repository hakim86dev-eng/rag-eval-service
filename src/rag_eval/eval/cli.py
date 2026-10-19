"""Command-line entry point: ``rag-eval`` / ``python -m rag_eval.eval``.

Exits non-zero when metrics fall below the given thresholds, so it can gate CI.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from rag_eval.config import Settings
from rag_eval.eval.dataset import load_dataset
from rag_eval.eval.harness import run_evaluation
from rag_eval.ingestion import load_directory
from rag_eval.pipeline import RAGPipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag-eval", description="Evaluate the RAG pipeline.")
    p.add_argument("--corpus", type=Path, default=None, help="Directory of .md/.txt documents")
    p.add_argument("--dataset", type=Path, default=None, help="JSONL evaluation dataset")
    p.add_argument("-k", "--top-k", type=int, default=None)
    p.add_argument("--mode", choices=["dense", "sparse", "hybrid"], default=None)
    p.add_argument("--output", type=Path, default=None, help="Write full JSON report here")
    p.add_argument("--min-hit-rate", type=float, default=0.0)
    p.add_argument("--min-faithfulness", type=float, default=0.0)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    pipeline = RAGPipeline.from_settings(settings)
    pipeline.ingest(load_directory(args.corpus or settings.corpus_dir))
    dataset = load_dataset(args.dataset or settings.eval_path)

    report = run_evaluation(pipeline, dataset, k=args.top_k, mode=args.mode)
    sys.stdout.write(report.to_markdown())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    failures = []
    if report.summary.hit_rate < args.min_hit_rate:
        failures.append(f"hit_rate {report.summary.hit_rate:.3f} < {args.min_hit_rate}")
    if report.summary.faithfulness < args.min_faithfulness:
        failures.append(f"faithfulness {report.summary.faithfulness:.3f} < {args.min_faithfulness}")
    for failure in failures:
        sys.stderr.write(f"QUALITY GATE FAILED: {failure}\n")
    return 1 if failures else 0
