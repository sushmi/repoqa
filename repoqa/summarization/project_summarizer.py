"""Project-level summarizer."""

from __future__ import annotations

import logging
from pathlib import Path

import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential

from repoqa.models import Chunk, Summary
from repoqa.summarization.prompts import PROJECT_SUMMARY_PROMPT

logger = logging.getLogger(__name__)

_ENCODER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


class ProjectSummarizer:
    """Summarizes an entire repository from directory summaries and the README."""

    def __init__(self, llm, max_input_tokens: int = 4000) -> None:
        self.llm = llm
        self.max_input_tokens = max_input_tokens

    def summarize(
        self,
        repo_root: str,
        dir_summaries: list[Summary],
        readme_chunk: Chunk | None = None,
    ) -> Summary:
        repo_name = Path(repo_root).name
        readme_excerpt = ""
        if readme_chunk:
            # Cap README excerpt at 500 tokens
            tokens = readme_chunk.token_count
            if tokens <= 500:
                readme_excerpt = readme_chunk.content
            else:
                lines = readme_chunk.content.splitlines()
                selected: list[str] = []
                used = 0
                for line in lines:
                    t = _count_tokens(line)
                    if used + t > 500:
                        break
                    selected.append(line)
                    used += t
                readme_excerpt = "\n".join(selected) + "\n...[truncated]"

        dir_list = self._build_dir_list(dir_summaries)
        content = self._invoke_with_retry(
            repo_name=repo_name,
            readme_excerpt=readme_excerpt or "(No README found)",
            dir_summaries_bullet_list=dir_list,
        )
        source_ids = [s.id for s in dir_summaries]
        return Summary(
            id=Summary.make_id("project", "."),
            level="project",
            path=".",
            content=content,
            source_ids=source_ids,
            token_count=_count_tokens(content),
        )

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(3))
    def _invoke_with_retry(
        self, repo_name: str, readme_excerpt: str, dir_summaries_bullet_list: str
    ) -> str:
        chain = PROJECT_SUMMARY_PROMPT | self.llm
        response = chain.invoke(
            {
                "repo_name": repo_name,
                "readme_excerpt": readme_excerpt,
                "dir_summaries_bullet_list": dir_summaries_bullet_list,
            }
        )
        return response.content if hasattr(response, "content") else str(response)

    def _build_dir_list(self, dir_summaries: list[Summary]) -> str:
        lines: list[str] = []
        used_tokens = 0
        for s in dir_summaries:
            bullet = f"- {s.path}/: {s.content}"
            tokens = _count_tokens(bullet)
            if used_tokens + tokens > self.max_input_tokens - 500:  # reserve space
                lines.append(f"- {s.path}/: <omitted>")
            else:
                lines.append(bullet)
                used_tokens += tokens
        return "\n".join(lines)
