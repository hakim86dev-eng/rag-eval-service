from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_eval.eval import load_dataset, run_evaluation
from rag_eval.eval.cli import main
from rag_eval.eval.metrics import LLMJudgeFaithfulness
from rag_eval.generation import FakeChatClient
from rag_eval.pipeline import RAGPipeline
from tests.conftest import CORPUS_DIR, EVAL_PATH


def test_dataset_loads() -> None:
    dataset = load_dataset(EVAL_PATH)
    assert len(dataset) == 18
    assert sum(not ex.answerable for ex in dataset) == 2


def test_dataset_rejects_bad_lines(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "x"}\n')
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        load_dataset(bad)


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_baseline_quality_on_bundled_dataset(pipeline: RAGPipeline, mode: str) -> None:
    """Regression gate: the offline baseline must keep these scores on the bundled data."""
    report = run_evaluation(pipeline, load_dataset(EVAL_PATH), k=3, mode=mode)  # type: ignore[arg-type]
    s = report.summary
    assert s.hit_rate >= 0.9
    assert s.mrr >= 0.85
    assert s.faithfulness >= 0.9
    assert s.answer_f1 >= 0.5
    assert s.refusal_accuracy == 1.0
    assert s.false_refusal_rate <= 0.1
    assert "| Hit rate@k |" in report.to_markdown()


def test_custom_faithfulness_judge(pipeline: RAGPipeline) -> None:
    dataset = load_dataset(EVAL_PATH)[:3]
    judge = LLMJudgeFaithfulness(FakeChatClient(responses=['{"score": 0.5}']))
    report = run_evaluation(pipeline, dataset, faithfulness_fn=judge)
    assert report.summary.faithfulness == 0.5


def test_cli_writes_report_and_enforces_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "report.json"
    args = ["--corpus", str(CORPUS_DIR), "--dataset", str(EVAL_PATH), "--output", str(out)]
    assert main([*args, "--min-hit-rate", "0.9"]) == 0
    assert json.loads(out.read_text())["summary"]["examples"] == 18
    assert "RAG evaluation report" in capsys.readouterr().out
    assert main([*args, "--min-faithfulness", "1.01"]) == 1
    assert "QUALITY GATE FAILED" in capsys.readouterr().err
