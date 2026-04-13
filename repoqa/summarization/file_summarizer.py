"""File-level summarizer: generates an LLM summary for a single source file."""

from __future__ import annotations

import logging

from tenacity import retry, stop_after_attempt, wait_exponential

from repoqa.models import Chunk, FileRecord, Summary
from repoqa.summarization.prompts import FILE_SUMMARY_PROMPT
from repoqa.tokenizer import count_tokens as _count_tokens

logger = logging.getLogger(__name__)

# Priority order when selecting representative chunks for the LLM prompt
_CHUNK_PRIORITY = ["function", "class", "module", "config", "prose"]


class FileSummarizer:
    """Summarizes a single file from its chunks using an LLM."""

    def __init__(self, llm, max_input_tokens: int = 3000) -> None:
        """
        Args:
            llm: A LangChain BaseChatModel instance.
            max_input_tokens: Max tokens to include in the prompt's code excerpts.
        """
        self.llm = llm
        self.max_input_tokens = max_input_tokens

    def summarize(self, file_record: FileRecord, chunks: list[Chunk]) -> Summary:
        """Generate a Summary for *file_record* using its *chunks*."""
        excerpts = self._select_representative_chunks(chunks)
        code_text = "\n\n---\n\n".join(
            f"# {c.file_record.repo_path}:{c.start_line}-{c.end_line}\n{c.content}"
            for c in excerpts
        )

        content = self._invoke_with_retry(
            file_path=file_record.repo_path,
            language=file_record.language,
            code_excerpts=code_text,
        )

        source_ids = [c.id for c in chunks]
        return Summary(
            id=Summary.make_id("file", file_record.repo_path),
            level="file",
            path=file_record.repo_path,
            content=content,
            source_ids=source_ids,
            token_count=_count_tokens(content),
            language=file_record.language,
        )

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(3))
    def _invoke_with_retry(self, file_path: str, language: str, code_excerpts: str) -> str:
        chain = FILE_SUMMARY_PROMPT | self.llm
        response = chain.invoke(
            {
                "file_path": file_path,
                "language": language,
                "code_excerpts": code_excerpts,
            }
        )
        return response.content if hasattr(response, "content") else str(response)

    def _select_representative_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Greedily select chunks by priority until the token budget is exhausted."""
        sorted_chunks = sorted(chunks, key=lambda c: _CHUNK_PRIORITY.index(c.chunk_type) if c.chunk_type in _CHUNK_PRIORITY else 99)
        selected: list[Chunk] = []
        used_tokens = 0
        for chunk in sorted_chunks:
            if used_tokens + chunk.token_count <= self.max_input_tokens:
                selected.append(chunk)
                used_tokens += chunk.token_count
            if used_tokens >= self.max_input_tokens:
                break
        return selected
