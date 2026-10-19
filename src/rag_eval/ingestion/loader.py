"""Load plain-text and markdown files into :class:`Document` objects."""

from __future__ import annotations

from pathlib import Path

from rag_eval.models import Document

SUPPORTED_SUFFIXES = frozenset({".md", ".markdown", ".txt"})


def _title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def load_file(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    return Document(
        id=path.stem,
        text=text,
        metadata={"source": path.name, "title": _title(text, path.stem)},
    )


def load_directory(directory: Path) -> list[Document]:
    """Load every supported file in ``directory`` (non-recursive), sorted by name."""
    if not directory.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {directory}")
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    return [load_file(p) for p in files]
