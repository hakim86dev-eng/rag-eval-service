"""Chat-completion clients behind a tiny protocol so they are trivial to fake in tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import httpx


@runtime_checkable
class ChatClient(Protocol):
    @property
    def name(self) -> str: ...

    def complete(self, system: str, user: str) -> str: ...


class OpenAIChatClient:
    """Minimal OpenAI-compatible ``/chat/completions`` client (OpenAI, vLLM, Ollama ``/v1``)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        timeout_s: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def complete(self, system: str, user: str) -> str:
        response = self._client.post(
            "/chat/completions",
            headers=self._headers,
            json={
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return str(content or "").strip()


@dataclass
class FakeChatClient:
    """Test double: returns scripted responses (or computes them) and records every call."""

    responses: Sequence[str] = ()
    handler: Callable[[str, str], str] | None = None
    calls: list[tuple[str, str]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "fake"

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.handler is not None:
            return self.handler(system, user)
        if not self.responses:
            raise RuntimeError("FakeChatClient has no scripted responses")
        return self.responses[min(len(self.calls), len(self.responses)) - 1]
