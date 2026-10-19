"""Evaluation dataset schema and loader (JSON Lines)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class EvalExample(BaseModel):
    id: str
    question: str
    expected_doc_ids: list[str] = Field(default_factory=list)
    reference_answer: str = ""
    answerable: bool = True


def load_dataset(path: Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            examples.append(EvalExample.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_no}: invalid example: {exc}") from exc
    return examples
