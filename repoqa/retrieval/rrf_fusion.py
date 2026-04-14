"""Reciprocal Rank Fusion (RRF) — merges multiple ranked lists into one.

From the paper (Section 3.2):
    RRF(d) = Σ  weight_r / (k + rank_r(d))
             r∈{S,B}

where S = semantic, B = BM25, and k = 60 is a smoothing constant.
"""

from __future__ import annotations


def rrf_fuse(
    ranked_lists: list[tuple[list[str], float]],
    k: int = 60,
) -> list[str]:
    """Fuse multiple ranked ID lists using weighted RRF.

    Args:
        ranked_lists: List of (ranked_ids, weight) tuples.
            Each ranked_ids is ordered best-first.
        k: Smoothing constant (default 60 per paper).

    Returns:
        Fused list of document IDs, ordered by combined RRF score.
    """
    scores: dict[str, float] = {}
    for ranked_ids, weight in ranked_lists:
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank + 1)
    return sorted(scores, key=lambda d: scores[d], reverse=True)
