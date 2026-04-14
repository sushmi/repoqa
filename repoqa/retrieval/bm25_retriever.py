"""BM25 keyword-based retrieval over chunks.

BM25 scores documents by term frequency, inverse document frequency,
and document length normalization — pure keyword matching with no
neural network involved.
"""

from __future__ import annotations

import re
import logging

from rank_bm25 import BM25Okapi

from repoqa.models import Chunk

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric, keep underscores."""
    return re.findall(r"[a-z_][a-z0-9_]*", text.lower())


class BM25Retriever:
    """Keyword-based retrieval using BM25Okapi over chunk contents."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        tokenized = [_tokenize(c.content) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built over %d chunks", len(chunks))

    def query(self, question: str, n_results: int = 15) -> list[tuple[Chunk, float]]:
        """Return the top-n chunks ranked by BM25 score, with scores."""
        tokenized_query = _tokenize(question)
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:n_results]
        return [(self._chunks[i], scores[i]) for i in top_indices if scores[i] > 0]
