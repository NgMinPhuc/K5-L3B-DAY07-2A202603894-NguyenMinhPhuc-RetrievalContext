"""OpenAI embedding adapter. Fails loudly: never falls back to mock/random vectors."""
import os
import time
from typing import Protocol

from .config import EMBEDDING_MODEL, MODEL_DIMENSIONS


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class OpenAIEmbedder:
    """Documents and queries use the same model with no prefix (symmetric model)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, batch_size: int = 100,
                 max_retries: int = 3) -> None:
        from openai import OpenAI  # imported here so steps 1-3 run without the SDK

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Thiếu OPENAI_API_KEY trong file .env ở thư mục gốc repo")
        if model_name not in MODEL_DIMENSIONS:
            raise ValueError(f"Chưa biết số chiều của model {model_name!r}")
        self.client = OpenAI()
        self.model_name = model_name
        self.dimensions = MODEL_DIMENSIONS[model_name]
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.total_tokens = 0          # usage reported by the API, for cost logging

    def _call(self, batch: list[str]) -> list[list[float]]:
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.embeddings.create(model=self.model_name, input=batch)
                break
            except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError):
                if attempt == self.max_retries:
                    raise            # surface the provider error; no silent fallback
                time.sleep(2 ** attempt)
        self.total_tokens += resp.usage.total_tokens
        vectors = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        if len(vectors) != len(batch) or any(len(v) != self.dimensions for v in vectors):
            raise RuntimeError("API trả về số vector hoặc số chiều không khớp")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._call(texts[i:i + self.batch_size]))
        return out

    def embed_query(self, question: str) -> list[float]:
        return self._call([question])[0]
