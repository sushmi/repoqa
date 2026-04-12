"""Walk a repository's filesystem and classify each file."""

from __future__ import annotations

import os
import logging
from pathlib import Path

from repoqa.models import FileRecord

logger = logging.getLogger(__name__)

# Directories to always skip
_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "build", "dist", "target", ".gradle", ".idea", ".vscode",
    "vendor", "third_party", ".mypy_cache", ".pytest_cache",
    "coverage", ".coverage", "htmlcov",
}

# File extensions → language name
_EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    # Non-code
    ".md": "markdown",
    ".rst": "text",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "text",
    ".cfg": "text",
    ".env": "text",
    ".sh": "text",
    ".bash": "text",
    ".zsh": "text",
}

# Special filenames regardless of extension
_SPECIAL_NAMES: dict[str, str] = {
    "Dockerfile": "dockerfile",
    "Makefile": "text",
    "Procfile": "text",
    "Gemfile": "text",
    "Rakefile": "text",
}


def classify_file(path: Path) -> str:
    """Return the language/type string for a given file path."""
    if path.name in _SPECIAL_NAMES:
        return _SPECIAL_NAMES[path.name]
    # Dockerfile variants: Dockerfile.prod, Dockerfile.dev, etc.
    if path.name.startswith("Dockerfile"):
        return "dockerfile"
    return _EXT_TO_LANGUAGE.get(path.suffix.lower(), "unknown")


def _is_binary(abs_path: str) -> bool:
    """Quick binary-file probe: look for null bytes in the first 8 KB."""
    try:
        with open(abs_path, "rb") as fh:
            chunk = fh.read(8192)
        return b"\x00" in chunk
    except OSError:
        logger.warning("Failed during is_binary file probe for %s: %s", abs_path, exc_info=True)
        return True

def crawl_repo(repo_root: str) -> list[FileRecord]:
    """Walk and read *repo_root* and return a FileRecord for every relevant file.

    Files are filtered by:
    - Not inside a skip directory.
    - Not binary.
    - Language must not be "unknown".
    - Readable as UTF-8.
    """
    root = Path(repo_root).resolve()
    records: list[FileRecord] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip dirs in-place so os.walk doesn't descend into them.
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]

        for fname in filenames:
            fpath = Path(dirpath) / fname
            language = classify_file(fpath)
            if language == "unknown":
                continue
            abs_path = str(fpath)
            if _is_binary(abs_path):
                continue
            try:
                raw_content = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                logger.warning("Failed during crawl_repo to read %s: %s", abs_path, exc_info=True)
                continue
            repo_path = str(fpath.relative_to(root))
            records.append(
                FileRecord(
                    repo_path=repo_path,
                    abs_path=abs_path,
                    language=language,
                    size_bytes=fpath.stat().st_size,
                    raw_content=raw_content,
                )
            )

    return records
