"""Lightweight text utilities (tokenisation, sentence splitting) with no external deps."""

from __future__ import annotations

import re

STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have how i if in
    into is it its may me might must my no not of on or our shall should so than that the their
    them then there these they this those to too up us was we were what when where which who whom
    why will with would you your yours about after all also any each every just more most other
    over per some such via within without
    """.split()  # noqa: SIM905
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_CITATION_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")


def _stem(token: str) -> str:
    """Very small suffix stripper; good enough for lexical matching on short corpora."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    """Lower-case, split on non-alphanumerics, optionally drop stopwords, and stem."""
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [_stem(t) for t in tokens]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, dropping empty fragments and markdown headings."""
    sentences: list[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*").strip()
        if not line or line.startswith("#"):
            continue
        sentences.extend(s.strip() for s in _SENTENCE_RE.split(line) if s.strip())
    return sentences


def strip_citations(text: str) -> str:
    """Remove inline citation markers such as ``[1]`` or ``[1, 2]``."""
    return _CITATION_RE.sub("", text).strip()
