"""Context builder: formats retrieved chunks into a token-budgeted prompt context.

Takes retrieved chunks from Stage 2 and packs them into a single string
that fits within the LLM's context window, preserving the most relevant
chunks first and trimming overflow.
"""

from __future__ import annotations

import logging

from repoqa.models import Chunk
from repoqa.tokenizer import TOKEN_COUNTER

logger = logging.getLogger(__name__)


def build_context(
    chunks: list[Chunk],
    max_tokens: int = 4000,
) -> tuple[str, list[Chunk]]:
    """Pack retrieved chunks into a formatted context string.

    Each chunk is rendered as:

        [path/to/file.py:10-42 | function: my_func]
        ```python
        <content>
        ```

    Chunks are included in order of relevance until the token budget
    is exhausted. Returns the formatted context plus the list of chunks
    that were actually included (for citation verification downstream).
    """
    if not chunks:
        return "(no context retrieved)", []

    blocks: list[str] = []
    included: list[Chunk] = []
    used_tokens = 0

    for chunk in chunks:
        header = _format_header(chunk)
        fence_lang = chunk.language if chunk.language not in ("", "unknown") else ""
        block = f"{header}\n```{fence_lang}\n{chunk.content}\n```"

        block_tokens = TOKEN_COUNTER.count(block)

        # If this one chunk alone exceeds the whole budget, truncate its content
        if block_tokens > max_tokens and not included:
            truncated = TOKEN_COUNTER.truncate(chunk.content, max_tokens - 50)
            block = f"{header}\n```{fence_lang}\n{truncated}\n[...truncated...]\n```"
            blocks.append(block)
            included.append(chunk)
            break

        if used_tokens + block_tokens > max_tokens:
            logger.debug(
                "Context full at %d chunks (%d tokens used, %d budget)",
                len(included), used_tokens, max_tokens,
            )
            break

        blocks.append(block)
        included.append(chunk)
        used_tokens += block_tokens

    context = "\n\n".join(blocks)
    logger.info(
        "Built context: %d/%d chunks, ~%d tokens",
        len(included), len(chunks), used_tokens,
    )
    return context, included


def _format_header(chunk: Chunk) -> str:
    """Render the location header for a chunk."""
    path = chunk.file_record.repo_path
    loc = f"{path}:{chunk.start_line}-{chunk.end_line}"
    if chunk.symbol_name:
        return f"[{loc} | {chunk.chunk_type}: {chunk.symbol_name}]"
    return f"[{loc} | {chunk.chunk_type}]"
