"""Deterministic metrics computed WITHOUT an LLM.

These complement the Gemini judge's subjective scores (accuracy/completeness/
relevance/clarity) with objective signals:

  - retrieval_recall_at_k  : fraction of reference keywords found in retrieved chunks
  - citation_accuracy      : fraction of cited file paths that exist in the repo
"""

from __future__ import annotations

import re
from pathlib import Path

from repoqa.models import Chunk
from repoqa.qa.citation_verifier import Citation


# Stop words we exclude when pulling keywords from reference answers
_STOP_WORDS = {
    "the", "and", "for", "that", "this", "with", "from", "which", "when",
    "each", "into", "also", "using", "their", "than", "some", "like", "used",
    "need", "have", "not", "its", "your", "can", "set", "use", "you", "are",
    "has", "but", "all", "was", "one", "any", "how", "does", "own", "then",
    "these", "those", "they", "will", "them", "there", "here", "where",
    "what", "who", "why", "been", "being", "under", "over", "both", "only",
    "most", "other", "such", "should", "would", "could", "make", "made",
    "more", "many", "much",
}


def extract_reference_keywords(answer: str) -> list[str]:
    """Pull technical identifiers from a reference answer for recall scoring.

    Matches backtick-quoted terms, snake_case identifiers, and CamelCase names.
    """
    keywords: set[str] = set()

    for match in re.findall(r"`([^`]+)`", answer):
        cleaned = match.strip("@().,:;").lower()
        if len(cleaned) >= 2:
            keywords.add(cleaned)

    for match in re.findall(r"\b[a-z_][a-z0-9_]{2,}\b", answer):
        if match not in _STOP_WORDS and len(match) >= 3:
            keywords.add(match)

    for match in re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", answer):
        keywords.add(match.lower())

    return sorted(keywords)


def retrieval_recall_at_k(
    reference_answer: str,
    retrieved_chunks: list[Chunk],
) -> float:
    """Fraction of reference keywords that appear in any retrieved chunk.

    Returns a value in [0.0, 1.0]. Higher means retrieval found more of
    the content that a good answer should reference.
    """
    keywords = extract_reference_keywords(reference_answer)
    if not keywords:
        return 0.0

    corpus = "\n".join(c.content for c in retrieved_chunks).lower()
    hits = sum(1 for kw in keywords if kw in corpus)
    return hits / len(keywords)


def citation_accuracy(
    citations: list[Citation],
    repo_root: str | None = None,
    known_file_paths: set[str] | None = None,
) -> float:
    """Fraction of cited file paths that correspond to real files in the repo.

    Hallucination detector — an answer can cite a file that doesn't exist.
    Accepts either:
      - repo_root: check each cited path on disk, OR
      - known_file_paths: pre-built set of repo_path strings (faster, preferred
        when you already have the FileRecord list in memory).

    Returns value in [0.0, 1.0]. Higher = less hallucination.
    """
    if not citations:
        return 1.0  # no citations made → vacuously "accurate"

    if known_file_paths is None and repo_root is None:
        raise ValueError("Must supply either known_file_paths or repo_root")

    valid = 0
    for cite in citations:
        path = cite.file_path
        if known_file_paths is not None:
            if path in known_file_paths:
                valid += 1
        else:
            abs_path = Path(repo_root) / path
            if abs_path.exists():
                valid += 1

    return valid / len(citations)
