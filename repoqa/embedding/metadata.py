"""Metadata schema for ChromaDB documents.

ChromaDB metadata values must be str | int | float | bool — no nested objects.
"""

from __future__ import annotations

from typing import TypedDict

from repoqa.models import Chunk, Summary


class ChunkMetadata(TypedDict):
    doc_type: str           # "chunk" or "summary"
    chunk_type: str         # "function" | "class" | "module" | "prose" | "config"
    file_path: str          # relative repo path (for summaries: the summarized path)
    language: str           # e.g. "python"
    start_line: int         # -1 for summaries
    end_line: int           # -1 for summaries
    symbol_name: str        # "" if none
    summary_level: str      # "file" | "directory" | "project" | "" for chunks
    token_count: int
    repo_name: str          # derived from repo root directory name


def chunk_to_metadata(chunk: Chunk, repo_name: str) -> ChunkMetadata:
    return ChunkMetadata(
        doc_type="chunk",
        chunk_type=chunk.chunk_type,
        file_path=chunk.file_record.repo_path,
        language=chunk.language,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        symbol_name=chunk.symbol_name or "",
        summary_level="",
        token_count=chunk.token_count,
        repo_name=repo_name,
    )


def summary_to_metadata(summary: Summary, repo_name: str) -> ChunkMetadata:
    return ChunkMetadata(
        doc_type="summary",
        chunk_type="",
        file_path=summary.path,
        language=summary.language,
        start_line=-1,
        end_line=-1,
        symbol_name="",
        summary_level=summary.level,
        token_count=summary.token_count,
        repo_name=repo_name,
    )
