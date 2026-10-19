"""OpenAI-compatible embeddings client (works with OpenAI, Azure-style proxies, vLLM, Ollama)."""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from rag_eval.embeddings.base import Vector, l2_normalize


class OpenAIEmbeddings:
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        batch_size: int = 64,
        timeout_s: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        vectors: list[Vector] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = self._client.post(
                "/embeddings", json={"model": self.model, "input": batch}, headers=self._headers
            )
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda item: item["index"])
            vectors.extend(l2_normalize(item["embedding"]) for item in data)
        return vectors
