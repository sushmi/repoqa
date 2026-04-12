"""Index non-code files: README, Dockerfiles, JSON/YAML configs, CI files, etc."""

from __future__ import annotations

import json
import logging
import re
import tomllib
from pathlib import Path

import tiktoken
import yaml

from repoqa.models import Chunk, FileRecord

logger = logging.getLogger(__name__)

_ENCODER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


class NonCodeChunker:
    """Splits non-code files into Chunk objects.

    Strategies are dispatch by file language:
    - markdown → split on heading boundaries
    - json      → one chunk per top-level key (or whole file if small)
    - yaml/toml → same as json
    - dockerfile → single or windowed chunks, chunk_type="config"
    - text/other → sliding window, chunk_type="prose"
    """

    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 64) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_file(self, file_record: FileRecord) -> list[Chunk]:
        lang = file_record.language
        dispatch = {
            "markdown": self._chunk_markdown,
            "json": self._chunk_json,
            "yaml": self._chunk_yaml,
            "toml": self._chunk_toml,
            "dockerfile": self._chunk_text_as_config,
        }
        handler = dispatch.get(lang, self._chunk_text)
        try:
            return handler(file_record)
        except Exception as exc:
            logger.warning("NonCodeChunker failed on %s: %s — falling back to text split", file_record.repo_path, exc)
            return self._chunk_text(file_record)

    # ------------------------------------------------------------------
    # Markdown — split on heading lines
    # ------------------------------------------------------------------

    def _chunk_markdown(self, file_record: FileRecord) -> list[Chunk]:
        lines = file_record.raw_content.splitlines()
        sections: list[tuple[int, list[str]]] = []  # (start_line_1indexed, lines)
        current_lines: list[str] = []
        current_start = 1

        for i, line in enumerate(lines, start=1):
            if re.match(r"^#{1,6}\s", line) and current_lines:
                sections.append((current_start, current_lines))
                current_lines = [line]
                current_start = i
            else:
                current_lines.append(line)
        if current_lines:
            sections.append((current_start, current_lines))

        chunks: list[Chunk] = []
        for start_line, sec_lines in sections:
            content = "\n".join(sec_lines).strip()
            if not content:
                continue
            # If a section is too long, further split it
            sub = self._window_lines(sec_lines, start_line, file_record, "prose")
            chunks.extend(sub)
        return chunks if chunks else self._chunk_text(file_record)

    # ------------------------------------------------------------------
    # JSON — one chunk per top-level key
    # ------------------------------------------------------------------

    def _chunk_json(self, file_record: FileRecord) -> list[Chunk]:
        data = json.loads(file_record.raw_content)
        if not isinstance(data, dict):
            return self._chunk_text(file_record)

        whole = json.dumps(data, indent=2)
        if _count_tokens(whole) <= self.max_tokens:
            return [self._make_chunk(file_record, whole, 1, len(whole.splitlines()), "config")]

        chunks: list[Chunk] = []
        for key, value in data.items():
            section = json.dumps({key: value}, indent=2)
            if _count_tokens(section) > self.max_tokens:
                # Truncate deeply nested objects
                section = json.dumps({key: str(value)[:200]}, indent=2)
            chunks.append(
                self._make_chunk(file_record, section, 1, len(section.splitlines()), "config")
            )
        return chunks

    # ------------------------------------------------------------------
    # YAML
    # ------------------------------------------------------------------

    def _chunk_yaml(self, file_record: FileRecord) -> list[Chunk]:
        data = yaml.safe_load(file_record.raw_content)
        if not isinstance(data, dict):
            return self._chunk_text(file_record)

        whole = yaml.dump(data, default_flow_style=False)
        if _count_tokens(whole) <= self.max_tokens:
            return [self._make_chunk(file_record, whole, 1, len(whole.splitlines()), "config")]

        chunks: list[Chunk] = []
        for key, value in data.items():
            section = yaml.dump({key: value}, default_flow_style=False)
            if _count_tokens(section) > self.max_tokens:
                section = f"{key}: <truncated>"
            chunks.append(
                self._make_chunk(file_record, section, 1, len(section.splitlines()), "config")
            )
        return chunks

    # ------------------------------------------------------------------
    # TOML
    # ------------------------------------------------------------------

    def _chunk_toml(self, file_record: FileRecord) -> list[Chunk]:
        data = tomllib.loads(file_record.raw_content)
        if not isinstance(data, dict):
            return self._chunk_text(file_record)

        whole = file_record.raw_content
        if _count_tokens(whole) <= self.max_tokens:
            return [self._make_chunk(file_record, whole, 1, len(whole.splitlines()), "config")]

        # Split by top-level keys (TOML sections)
        chunks: list[Chunk] = []
        for key, value in data.items():
            section = f"[{key}]\n{json.dumps(value, indent=2)}"
            if _count_tokens(section) > self.max_tokens:
                section = f"[{key}]\n# <truncated>"
            chunks.append(
                self._make_chunk(file_record, section, 1, len(section.splitlines()), "config")
            )
        return chunks

    # ------------------------------------------------------------------
    # Dockerfile and plain text
    # ------------------------------------------------------------------

    def _chunk_text_as_config(self, file_record: FileRecord) -> list[Chunk]:
        return self._chunk_text(file_record, chunk_type="config")

    def _chunk_text(self, file_record: FileRecord, chunk_type: str = "prose") -> list[Chunk]:
        lines = file_record.raw_content.splitlines()
        return self._window_lines(lines, 1, file_record, chunk_type)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _window_lines(
        self,
        lines: list[str],
        start_line_offset: int,
        file_record: FileRecord,
        chunk_type: str,
    ) -> list[Chunk]:
        """Sliding-window split of a list of lines respecting token budget."""
        chunks: list[Chunk] = []
        i = 0
        while i < len(lines):
            batch: list[str] = []
            token_count = 0
            j = i
            while j < len(lines) and token_count + _count_tokens(lines[j]) <= self.max_tokens:
                batch.append(lines[j])
                token_count += _count_tokens(lines[j])
                j += 1
            if not batch:
                batch = [lines[i]]
                j = i + 1

            content = "\n".join(batch).strip()
            if content:
                abs_start = start_line_offset + i
                abs_end = start_line_offset + j - 1
                chunks.append(self._make_chunk(file_record, content, abs_start, abs_end, chunk_type))

            overlap = max(1, self.overlap_tokens // 10)
            i = j - overlap if j - overlap > i else j

        return chunks

    def _make_chunk(
        self,
        file_record: FileRecord,
        content: str,
        start_line: int,
        end_line: int,
        chunk_type: str,
    ) -> Chunk:
        return Chunk(
            id=Chunk.make_id(file_record.repo_path, start_line),
            file_record=file_record,
            content=content,
            chunk_type=chunk_type,
            start_line=start_line,
            end_line=end_line,
            symbol_name=None,
            token_count=_count_tokens(content),
        )
