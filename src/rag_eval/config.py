"""Application settings loaded from environment variables (prefix ``RAG_``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RetrievalMode = Literal["dense", "sparse", "hybrid"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # Providers. "hashing" / "extractive" run fully offline and are the defaults.
    embedding_provider: Literal["hashing", "openai"] = "hashing"
    llm_provider: Literal["extractive", "openai"] = "extractive"

    openai_api_key: SecretStr | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    request_timeout_s: float = 30.0

    hashing_dim: int = Field(default=1024, ge=64, le=8192)

    # Chunking & retrieval
    chunk_size: int = Field(default=600, ge=100, le=8000)
    chunk_overlap: int = Field(default=100, ge=0)
    top_k: int = Field(default=4, ge=1, le=50)
    retrieval_mode: RetrievalMode = "hybrid"

    # Data
    corpus_dir: Path = Path("data/corpus")
    eval_path: Path = Path("data/eval/qa.jsonl")
    ingest_on_startup: bool = True
    index_path: Path | None = None

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        needs_key = self.embedding_provider == "openai" or self.llm_provider == "openai"
        if needs_key and self.openai_api_key is None:
            raise ValueError("RAG_OPENAI_API_KEY is required when an OpenAI provider is selected")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
