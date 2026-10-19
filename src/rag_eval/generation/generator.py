"""Answer generators: an LLM-backed one and an offline extractive baseline."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from rag_eval.generation.chat import ChatClient
from rag_eval.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from rag_eval.models import ScoredChunk
from rag_eval.text import split_sentences, tokenize

REFUSAL = "I don't know based on the provided documents."


class AnswerGenerator(Protocol):
    @property
    def name(self) -> str: ...

    def generate(self, question: str, contexts: Sequence[ScoredChunk]) -> str: ...


class LLMAnswerGenerator:
    def __init__(self, chat: ChatClient) -> None:
        self.chat = chat

    @property
    def name(self) -> str:
        return f"llm:{self.chat.name}"

    def generate(self, question: str, contexts: Sequence[ScoredChunk]) -> str:
        if not contexts:
            return REFUSAL
        return self.chat.complete(SYSTEM_PROMPT, build_user_prompt(question, contexts))


class ExtractiveAnswerGenerator:
    """Offline baseline: picks the context sentences with the highest lexical overlap.

    Deterministic and always grounded, which makes it a useful *floor* for evaluation and lets
    the service run without any API key.
    """

    def __init__(
        self, max_sentences: int = 2, min_overlap: float = 0.4, relative_cutoff: float = 0.85
    ) -> None:
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap
        self.relative_cutoff = relative_cutoff

    @property
    def name(self) -> str:
        return "extractive"

    def generate(self, question: str, contexts: Sequence[ScoredChunk]) -> str:
        q_terms = set(tokenize(question))
        if not q_terms or not contexts:
            return REFUSAL
        sentences: list[tuple[int, int, str, set[str]]] = []
        seen: set[str] = set()
        for source, ctx in enumerate(contexts, start=1):
            for position, sentence in enumerate(split_sentences(ctx.chunk.text)):
                if sentence not in seen:
                    seen.add(sentence)
                    sentences.append((source, position, sentence, set(tokenize(sentence))))

        # IDF over the retrieved sentences: terms that appear everywhere ("company name")
        # carry little evidence, while rare question terms missing from the context
        # strongly suggest the question is unanswerable.
        n = len(sentences)
        idf = {
            t: math.log((n + 1) / (sum(t in terms for *_, terms in sentences) + 0.5))
            for t in q_terms
        }
        total = sum(idf.values()) or 1.0
        candidates: list[tuple[float, int, int, str]] = []
        for source, position, sentence, terms in sentences:
            overlap = sum(idf[t] for t in q_terms & terms) / total
            if overlap >= self.min_overlap:
                candidates.append((overlap, source, position, sentence))
        if not candidates:
            return REFUSAL
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
        top_score = candidates[0][0]
        chosen: list[tuple[float, int, int, str]] = []
        for cand in candidates:
            if len(chosen) == self.max_sentences or cand[0] < top_score * self.relative_cutoff:
                break
            # Skip fragments produced by chunk overlap (substring of an already chosen sentence).
            if any(cand[3] in c[3] or c[3] in cand[3] for c in chosen):
                continue
            chosen.append(cand)
        best = sorted(chosen, key=lambda c: (c[1], c[2]))
        return " ".join(f"{sentence} [{source}]" for _, source, _, sentence in best)
