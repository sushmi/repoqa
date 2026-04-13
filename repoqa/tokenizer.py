"""Shared Qwen tokenizer for token counting across the pipeline."""

from __future__ import annotations

from transformers import AutoTokenizer

# Qwen tokenizer — loaded once and shared across all modules.
_TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)


def count_tokens(text: str) -> int:
    """Return the number of tokens for *text* using the Qwen tokenizer."""
    return len(_TOKENIZER.encode(text))


def truncate_to_limit(text: str, max_tokens: int) -> str:
    """Truncate *text* to at most *max_tokens* Qwen tokens."""
    tokens = _TOKENIZER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _TOKENIZER.decode(tokens[:max_tokens])
