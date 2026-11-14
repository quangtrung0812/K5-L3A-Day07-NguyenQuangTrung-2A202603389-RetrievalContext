"""Real OpenAI embedding adapter compatible with Day 6 pgvector index (pure-Python HTTP, no mock fallback)."""
from __future__ import annotations

import json
import time
from typing import Protocol
import urllib.error
import urllib.request

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
    """OpenAI query embedder via REST API with strict dimension checks and token/cost telemetry."""

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
        self._api_key = api_key.strip()
        self._endpoint = "https://api.openai.com/v1/embeddings"

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

        payload = json.dumps({"model": self.model_name, "input": cleaned}).encode("utf-8")
        last_err: Exception | None = None
        for attempt in range(1, 4):
            t0 = time.perf_counter()
            req = urllib.request.Request(
                self._endpoint,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                self.total_api_time_sec += time.perf_counter() - t0
                self.total_api_calls += 1
                usage = body.get("usage") or {}
                self.total_prompt_tokens += int(
                    usage.get("prompt_tokens", usage.get("total_tokens", 0))
                )
                vector = list(body["data"][0]["embedding"])
                if len(vector) != self.dimensions:
                    raise RuntimeError(
                        f"Embedding dimension mismatch: expected {self.dimensions}, got {len(vector)}"
                    )
                return vector
            except urllib.error.HTTPError as http_err:
                err_text = http_err.read().decode("utf-8", errors="replace")
                last_err = RuntimeError(f"OpenAI Embeddings HTTP {http_err.code}: {err_text}")
                if http_err.code in (429, 500, 502, 503, 504) and attempt < 3:
                    time.sleep(1.5 * attempt)
                    continue
                raise last_err from http_err
            except Exception as exc:
                last_err = exc
                if attempt < 3:
                    time.sleep(1.5 * attempt)
                    continue
                raise RuntimeError(f"OpenAI embedding request failed: {exc}") from exc

        raise RuntimeError(f"OpenAI embedding failed after retries: {last_err}")
