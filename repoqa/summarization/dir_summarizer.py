"""Directory-level summarizer."""

from __future__ import annotations

import logging

from tenacity import retry, stop_after_attempt, wait_exponential

from repoqa.models import Summary
from repoqa.summarization.prompts import DIR_SUMMARY_PROMPT
from repoqa.tokenizer import TOKEN_COUNTER

logger = logging.getLogger(__name__)


class DirSummarizer:
    """Summarizes a directory from its child file summaries."""

    def __init__(self, llm, max_input_tokens: int = 3000) -> None:
        self.llm = llm
        self.max_input_tokens = max_input_tokens

    def summarize(self, dir_path: str, file_summaries: list[Summary]) -> Summary:
        bullet_list = self._build_bullet_list(file_summaries)
        content = self._invoke_with_retry(dir_path=dir_path, file_summaries_bullet_list=bullet_list)
        source_ids = [s.id for s in file_summaries]
        return Summary(
            id=Summary.make_id("directory", dir_path),
            level="directory",
            path=dir_path,
            content=content,
            source_ids=source_ids,
            token_count=TOKEN_COUNTER.count(content),
        )

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(3))
    def _invoke_with_retry(self, dir_path: str, file_summaries_bullet_list: str) -> str:
        messages = DIR_SUMMARY_PROMPT.format_messages(
            dir_path=dir_path,
            file_summaries_bullet_list=file_summaries_bullet_list,
        )
        response = self.llm.invoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        TOKEN_COUNTER.log_llm_call(f"dir_summary:{dir_path}", messages, text)
        return text

    def _build_bullet_list(self, file_summaries: list[Summary]) -> str:
        """Build a bullet list of file summaries, truncated to token budget."""
        lines: list[str] = []
        used_tokens = 0
        for s in file_summaries:
            bullet = f"- {s.path}: {s.content}"
            tokens = TOKEN_COUNTER.count(bullet)
            if used_tokens + tokens > self.max_input_tokens:
                # Include a truncated version
                lines.append(f"- {s.path}: <summary omitted for length>")
            else:
                lines.append(bullet)
                used_tokens += tokens
        return "\n".join(lines)
