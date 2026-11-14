"""Real OpenAI embedding adapter compatible with Day 6 pgvector index (no mock fallback)."""
from __future__ import annotations

import time
from typing import Protocol

from openai import OpenAI

MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

MODEL_PRICE_PER_1M_TOKENS_USD: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


class QueryEmbedder(Protocol):
    model_name: str
    dimensions: int

    def embed_query(self, query: str) -> list[float]:
        ...


class OpenAIQueryEmbedder:
    """OpenAI query embedder with strict dimension checks and token/cost telemetry."""

    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small") -> None:
        if not api_key or not api_key.strip():
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Set OPENAI_API_KEY in .env before running dense retrieval."
            )
        resolved_model = (model_name or "text-embedding-3-small").strip()
        if resolved_model not in MODEL_DIMENSIONS:
            raise ValueError(
                f"Unsupported embedding model {resolved_model!r}; expected one of {sorted(MODEL_DIMENSIONS)}"
            )
        self.model_name = resolved_model
        self.dimensions = MODEL_DIMENSIONS[resolved_model]
        self.price_per_1m_usd = MODEL_PRICE_PER_1M_TOKENS_USD[resolved_model]
        self._client = OpenAI(api_key=api_key.strip(), timeout=30.0, max_retries=2)

        self.total_prompt_tokens = 0
        self.total_api_calls = 0
        self.total_api_time_sec = 0.0

    @property
    def total_cost_usd(self) -> float:
        return round((self.total_prompt_tokens / 1_000_000.0) * self.price_per_1m_usd, 8)

    def embed_query(self, query: str) -> list[float]:
        cleaned = (query or "").replace("\x00", " ").strip()
        if not cleaned:
            raise ValueError("Query text must not be empty")

        t0 = time.perf_counter()
        response = self._client.embeddings.create(model=self.model_name, input=cleaned)
        self.total_api_time_sec += time.perf_counter() - t0
        self.total_api_calls += 1
        if getattr(response, "usage", None) is not None:
            self.total_prompt_tokens += int(
                getattr(response.usage, "prompt_tokens", 0)
                or getattr(response.usage, "total_tokens", 0)
            )

        vector = list(response.data[0].embedding)
        if len(vector) != self.dimensions:
            raise RuntimeError(
                f"Embedding dimension mismatch: expected {self.dimensions}, got {len(vector)}"
            )
        return vector
