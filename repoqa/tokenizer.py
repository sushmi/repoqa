"""Shared Qwen tokenizer + LLM-call accounting.

All token counting, truncation, and per-call cost logging in the codebase
goes through the :class:`TokenCounter` singleton exposed as ``TOKEN_COUNTER``.
"""

from __future__ import annotations

import logging
from typing import Any

from transformers import AutoTokenizer


class TokenCounter:
    """Single source of truth for token counting in RepoQA.

    Wraps the Qwen2.5-7B tokenizer (the same one used by chunking and the
    answer generator) and provides count / truncate / message-counting /
    LLM-call logging in one place.
    """

    _MODEL_ID = "Qwen/Qwen2.5-7B"
    _instance: "TokenCounter | None" = None

    def __new__(cls) -> "TokenCounter":
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._tokenizer = AutoTokenizer.from_pretrained(
                cls._MODEL_ID, trust_remote_code=True
            )
            inst._llm_logger = logging.getLogger("repoqa.llm")
            cls._instance = inst
        return cls._instance

    # --- token counting -----------------------------------------------------

    def count(self, text: str) -> int:
        """Return the number of tokens in *text*."""
        return len(self._tokenizer.encode(text))

    def count_messages(self, prompt_input: Any) -> int:
        """Count tokens across a string or a list of LangChain messages/dicts."""
        if isinstance(prompt_input, str):
            return self.count(prompt_input)
        total = 0
        for m in prompt_input:
            if isinstance(m, dict):
                content = m.get("content", "")
            else:
                content = getattr(m, "content", "") or str(m)
            total += self.count(content)
        return total

    # --- truncation ---------------------------------------------------------

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate *text* to at most *max_tokens* tokens."""
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._tokenizer.decode(tokens[:max_tokens])

    # --- LLM-call accounting ------------------------------------------------

    def log_llm_call(
        self, label: str, prompt_input: Any, response_text: str
    ) -> dict[str, int]:
        """Count input + output tokens for an LLM call and log them.

        Returns a dict with ``input_tokens`` / ``output_tokens`` /
        ``total_tokens`` so callers can persist the figures (e.g. into
        experiment JSONL).
        """
        in_tokens = self.count_messages(prompt_input)
        out_tokens = self.count(response_text or "")
        total = in_tokens + out_tokens
        self._llm_logger.info(
            "LLM call [%s]: input=%d output=%d total=%d tokens",
            label, in_tokens, out_tokens, total,
        )
        return {
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "total_tokens": total,
        }


TOKEN_COUNTER = TokenCounter()
