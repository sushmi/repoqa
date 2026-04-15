"""Citation verifier: checks that cited file paths/line ranges actually
appear in the retrieved context. Prevents hallucinated citations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from repoqa.models import Chunk

logger = logging.getLogger(__name__)

# Matches [path/to/file.py:10-42], [path/to/file.py:10], or [path/to/file.py]
_CITATION_RE = re.compile(
    r"\[([^\[\]\s]+?)(?::(\d+)(?:-(\d+))?)?\]"
)


@dataclass
class Citation:
    """A single citation extracted from an LLM answer."""
    file_path: str
    start_line: int | None
    end_line: int | None
    raw: str          # original bracketed text
    verified: bool    # matches a retrieved chunk?


@dataclass
class VerifiedAnswer:
    """An LLM answer with citations verified against retrieval context."""
    answer_text: str
    citations: list[Citation]
    confidence: str                  # "high" | "medium" | "low"
    verified_count: int
    unverified_count: int
    coverage_ratio: float             # verified / total citations
    used_chunk_ids: list[str]         # chunk IDs that were actually cited


def extract_citations(answer_text: str) -> list[Citation]:
    """Pull all [file:line-line] citations out of the answer text."""
    citations: list[Citation] = []
    for match in _CITATION_RE.finditer(answer_text):
        path, start, end = match.groups()
        # Skip obvious non-citations like [TODO] or [note]
        if "/" not in path and "." not in path:
            continue
        citations.append(
            Citation(
                file_path=path,
                start_line=int(start) if start else None,
                end_line=int(end) if end else (int(start) if start else None),
                raw=match.group(0),
                verified=False,
            )
        )
    return citations


def verify_citations(
    answer_text: str,
    retrieved_chunks: list[Chunk],
    confidence: str = "medium",
) -> VerifiedAnswer:
    """Check each citation in the answer against the retrieved chunks.

    A citation is "verified" if:
      - Its file path matches a retrieved chunk's repo_path, AND
      - Its line range overlaps with that chunk's line range
        (or no line range was given but the file matches)
    """
    citations = extract_citations(answer_text)
    used_ids: set[str] = set()

    for cite in citations:
        for chunk in retrieved_chunks:
            if chunk.file_record.repo_path != cite.file_path:
                continue
            # File matches — if no line range given, that's enough
            if cite.start_line is None:
                cite.verified = True
                used_ids.add(chunk.id)
                break
            # Check line range overlap
            if _ranges_overlap(
                cite.start_line, cite.end_line,
                chunk.start_line, chunk.end_line,
            ):
                cite.verified = True
                used_ids.add(chunk.id)
                break

    verified = sum(1 for c in citations if c.verified)
    unverified = len(citations) - verified
    coverage = verified / max(len(citations), 1)

    if unverified > 0:
        logger.warning(
            "Answer has %d unverified citation(s) out of %d",
            unverified, len(citations),
        )

    return VerifiedAnswer(
        answer_text=answer_text,
        citations=citations,
        confidence=confidence,
        verified_count=verified,
        unverified_count=unverified,
        coverage_ratio=coverage,
        used_chunk_ids=list(used_ids),
    )


def _ranges_overlap(a_start: int, a_end: int | None,
                    b_start: int, b_end: int) -> bool:
    """Check if two line ranges overlap."""
    a_end = a_end if a_end is not None else a_start
    return not (a_end < b_start or a_start > b_end)
