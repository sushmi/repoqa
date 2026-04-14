"""Strategy router: selects retrieval configuration based on question type.

From the RepoQA paper (Section 3.2):
- Architecture → project/directory summaries first, then file-level chunks (top-down)
- Logic → function-level chunks via semantic search, then caller/callee context
- Deployment → non-code files (Dockerfiles, CI/CD, configs), then relevant code
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalStrategy:
    """Configuration for a single retrieval pass."""

    # Number of results to request from each retriever
    semantic_k: int
    bm25_k: int

    # ChromaDB metadata filters for semantic search
    semantic_filters: dict | None

    # RRF weights — multiplier on each retriever's RRF contribution
    semantic_weight: float
    bm25_weight: float

    # Whether to include summaries in semantic search
    include_summaries: bool


# ---- Per-question-type strategies (from paper Section 3.2) ----

ARCHITECTURE_STRATEGY = RetrievalStrategy(
    semantic_k=15,
    bm25_k=10,
    semantic_filters=None,  # search everything including summaries
    semantic_weight=1.5,    # boost semantic (captures conceptual relationships)
    bm25_weight=0.5,        # de-weight keywords (architecture terms are vague)
    include_summaries=True,  # summaries are critical for architecture questions
)

LOGIC_STRATEGY = RetrievalStrategy(
    semantic_k=15,
    bm25_k=15,
    semantic_filters={"doc_type": "chunk"},  # skip summaries, go straight to code
    semantic_weight=1.0,
    bm25_weight=1.5,    # boost keywords (function names, identifiers are exact)
    include_summaries=False,
)

DEPLOYMENT_STRATEGY = RetrievalStrategy(
    semantic_k=10,
    bm25_k=15,
    semantic_filters=None,
    semantic_weight=1.0,
    bm25_weight=1.0,
    include_summaries=True,
)

_STRATEGIES = {
    "architecture": ARCHITECTURE_STRATEGY,
    "logic": LOGIC_STRATEGY,
    "deployment": DEPLOYMENT_STRATEGY,
}


def get_strategy(question_type: str) -> RetrievalStrategy:
    """Return the retrieval strategy for a given question type."""
    return _STRATEGIES.get(question_type, LOGIC_STRATEGY)
