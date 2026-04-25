"""Embedding wrapper with batching and retry logic.

Supports two providers:
- "ollama"  — free, local, via Ollama REST API (default)
- "openai"  — paid, cloud, via OpenAI SDK
"""

from __future__ import annotations

import logging

import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from repoqa.tokenizer import TOKEN_COUNTER

logger = logging.getLogger(__name__)

# Token limits per provider (Ollama limit is conservative because we count
# tokens with the Qwen tokenizer, which may undercount for other models)
_TOKEN_LIMITS = {
    "openai": 8191,     # text-embedding-3-small
    "ollama": 2048,     # safe limit for nomic-embed-text
}


class Embedder:
    """Wraps embedding APIs with batching and retry."""

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "nomic-embed-text",
        api_key: str = "",
        ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        self.provider = provider
        self.model = model
        self._ollama_base_url = ollama_base_url
        self._max_tokens = _TOKEN_LIMITS.get(provider, 8192)

        if provider == "openai":
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def embed_batch(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """Embed a list of texts, returning a list of embedding vectors in the same order."""
        if not texts:
            return []

        if batch_size is None:
            batch_size = 10 if self.provider == "ollama" else 100

        # Token-based truncation for both providers. Char-based truncation is
        # unsafe for code (2-3 chars/token, not 4) and for LLM-expanded queries
        # that can blow past the embed model's context limit.
        safe_texts = [TOKEN_COUNTER.truncate(t, self._max_tokens) for t in texts]

        all_embeddings: list[list[float]] = []
        batches = [safe_texts[i: i + batch_size] for i in range(0, len(safe_texts), batch_size)]

        for batch in tqdm(batches, desc=f"Embedding ({self.model})", unit="batch", leave=False):
            embeddings = self._embed_one_batch(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    @retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(5))
    def _embed_one_batch(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "openai":
            return self._embed_openai(texts)
        else:
            return self._embed_ollama(texts)

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        response = self._openai_client.embeddings.create(model=self.model, input=texts)
        sorted_items = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_items]

    def _embed_ollama(self, texts: list[str]) -> list[list[float]]:
        # Ollama's /api/embed endpoint accepts multiple texts
        response = requests.post(
            f"{self._ollama_base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        if not response.ok:
            logger.error("Ollama embed error %d: %s", response.status_code, response.text[:500])
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_single(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]
