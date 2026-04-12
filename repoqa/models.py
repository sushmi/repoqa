"""Shared dataclasses that flow through all three pipelines."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class FileRecord:
    """Represents a single file discovered during repo crawling."""

    repo_path: str      # relative path within the repo, e.g. "src/utils/helpers.py"
    abs_path: str       # absolute filesystem path
    language: str       # "python", "javascript", "markdown", "dockerfile", etc.
    size_bytes: int
    raw_content: str


@dataclass
class Chunk:
    """A semantically bounded piece of content (code or prose) ready for embedding."""

    id: str                     # deterministic sha256 hash of (repo_path + start_line)
    file_record: FileRecord
    content: str                # the actual text content of this chunk
    chunk_type: str             # "function" | "class" | "module" | "prose" | "config"
    start_line: int
    end_line: int
    symbol_name: str | None     # function/class name if applicable, else None
    token_count: int
    language: str = ""          # convenience copy from file_record.language

    def __post_init__(self) -> None:
        if not self.language:
            self.language = self.file_record.language

    @staticmethod
    def make_id(repo_path: str, start_line: int) -> str:
        key = f"{repo_path}:{start_line}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass
class Summary:
    """An LLM-generated summary at file, directory, or project level."""

    id: str
    level: str              # "file" | "directory" | "project"
    path: str               # file path, directory path, or "." for project root
    content: str            # LLM-generated summary text
    source_ids: list[str]   # chunk ids or child summary ids that fed this summary
    token_count: int
    language: str = ""      # primary language (for file summaries)

    @staticmethod
    def make_id(level: str, path: str) -> str:
        key = f"summary:{level}:{path}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
