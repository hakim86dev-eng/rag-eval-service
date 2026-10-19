"""Recursive, separator-aware chunking with character overlap.

The splitter prefers to break on paragraph boundaries, then lines, then sentences and finally
words, so chunks stay semantically coherent. Neighbouring chunks share up to ``chunk_overlap``
characters (trimmed to a word boundary) so that facts spanning a boundary remain retrievable.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_eval.models import Chunk, Document

_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")


def _split(text: str, separators: tuple[str, ...], size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    if not separators:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        piece = part + (sep if i < len(parts) - 1 else "")
        if len(piece) > size:
            pieces.extend(_split(piece, rest, size))
        elif piece:
            pieces.append(piece)
    return pieces


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    if len(text) > overlap and " " in tail:
        tail = tail[tail.index(" ") + 1 :]
    return tail


def chunk_text(text: str, chunk_size: int = 600, chunk_overlap: int = 100) -> list[str]:
    """Split ``text`` into chunks of at most ``chunk_size`` characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    chunks: list[str] = []
    current = ""
    for piece in _split(text, _SEPARATORS, chunk_size):
        if len(current) + len(piece) <= chunk_size:
            current += piece
            continue
        if current.strip():
            chunks.append(current.strip())
        current = _overlap_tail(current, chunk_overlap) + piece
        if len(current) > chunk_size:
            current = piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


@dataclass(frozen=True, slots=True)
class Chunker:
    chunk_size: int = 600
    chunk_overlap: int = 100

    def split(self, document: Document) -> list[Chunk]:
        title = document.metadata.get("title")
        return [
            Chunk(
                id=f"{document.id}::{i}",
                doc_id=document.id,
                index=i,
                text=text,
                metadata={**document.metadata, **({"title": title} if title else {})},
            )
            for i, text in enumerate(chunk_text(document.text, self.chunk_size, self.chunk_overlap))
        ]
