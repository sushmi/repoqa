"""Embedding wrapper with batching and retry logic."""

from __future__ import annotations

import logging

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from repoqa.tokenizer import truncate_to_limit

logger = logging.getLogger(__name__)

# text-embedding-3-small max input tokens
_MAX_INPUT_TOKENS = 8191


def _truncate_to_limit(text: str, max_tokens: int = _MAX_INPUT_TOKENS) -> str:
    return truncate_to_limit(text, max_tokens)


class Embedder:
    """Wraps the OpenAI embeddings API with batching and retry."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = "") -> None:
        self.model = model
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """Embed a list of texts, returning a list of embedding vectors in the same order."""
        if not texts:
            return []

        # Truncate each text to the model's token limit
        safe_texts = [_truncate_to_limit(t) for t in texts]

        all_embeddings: list[list[float]] = []
        batches = [safe_texts[i: i + batch_size] for i in range(0, len(safe_texts), batch_size)]

        for batch in tqdm(batches, desc=f"Embedding ({self.model})", unit="batch", leave=False):
            embeddings = self._embed_one_batch(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    @retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(5))
    def _embed_one_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts)
        # Response items are ordered by index
        sorted_items = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_items]

    def embed_single(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]
