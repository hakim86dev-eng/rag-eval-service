"""Prompt templates. Kept in one place so they can be versioned and evaluated."""

from __future__ import annotations

from collections.abc import Sequence

from rag_eval.models import ScoredChunk

SYSTEM_PROMPT = (
    "You are a precise assistant answering questions using ONLY the provided sources.\n"
    "Rules:\n"
    "1. Cite every factual statement with the source number in square brackets, e.g. [2].\n"
    "2. If the sources do not contain the answer, reply exactly: "
    '"I don\'t know based on the provided documents."\n'
    "3. Be concise: at most three sentences."
)


def format_context(contexts: Sequence[ScoredChunk]) -> str:
    return "\n\n".join(
        f"[{i}] (doc: {c.chunk.doc_id})\n{c.chunk.text}" for i, c in enumerate(contexts, start=1)
    )


def build_user_prompt(question: str, contexts: Sequence[ScoredChunk]) -> str:
    return f"Sources:\n{format_context(contexts)}\n\nQuestion: {question}\nAnswer:"
