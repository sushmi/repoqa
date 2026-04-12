"""Smoke tests for the ingestion pipeline (no LLM calls required)."""

import textwrap
import pytest

from repoqa.models import Chunk, FileRecord
from repoqa.ingestion.repo_crawler import classify_file, crawl_repo
from repoqa.ingestion.ast_chunker import ASTChunker
from repoqa.ingestion.noncode_chunker import NonCodeChunker
from pathlib import Path


# ---------------------------------------------------------------------------
# FileRecord factory
# ---------------------------------------------------------------------------

def make_file_record(repo_path: str, language: str, content: str) -> FileRecord:
    return FileRecord(
        repo_path=repo_path,
        abs_path=f"/tmp/{repo_path}",
        language=language,
        size_bytes=len(content.encode()),
        raw_content=content,
    )


# ---------------------------------------------------------------------------
# repo_crawler tests
# ---------------------------------------------------------------------------

def test_classify_file_python():
    assert classify_file(Path("src/main.py")) == "python"

def test_classify_file_dockerfile():
    assert classify_file(Path("Dockerfile")) == "dockerfile"
    assert classify_file(Path("Dockerfile.prod")) == "dockerfile"

def test_classify_file_unknown():
    assert classify_file(Path("binary.bin")) == "unknown"

def test_crawl_repo_on_self(tmp_path):
    # Create a tiny fake repo
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def hello(): pass\n")
    (tmp_path / "README.md").write_text("# Test\nHello world\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lodash.js").write_text("// lodash\n")

    records = crawl_repo(str(tmp_path))
    paths = [r.repo_path for r in records]

    assert any("main.py" in p for p in paths), "Expected src/main.py"
    assert any("README.md" in p for p in paths), "Expected README.md"
    # node_modules should be excluded
    assert not any("lodash" in p for p in paths), "node_modules should be skipped"


# ---------------------------------------------------------------------------
# ASTChunker tests
# ---------------------------------------------------------------------------

SIMPLE_PYTHON = textwrap.dedent("""\
    import os

    CONSTANT = 42

    def add(a, b):
        \"\"\"Add two numbers.\"\"\"
        return a + b

    class Calculator:
        def multiply(self, x, y):
            return x * y
""")

def test_ast_chunker_python_functions():
    chunker = ASTChunker(max_tokens=200)
    fr = make_file_record("math.py", "python", SIMPLE_PYTHON)
    chunks = chunker.chunk_file(fr)
    assert len(chunks) >= 1
    chunk_types = {c.chunk_type for c in chunks}
    # Should have at least function and class chunks if tree-sitter is available
    all_content = "\n".join(c.content for c in chunks)
    assert "add" in all_content or "Calculator" in all_content

def test_ast_chunker_fallback_text():
    """Chunker must work even for unsupported languages (falls back to text split)."""
    chunker = ASTChunker(max_tokens=50)
    fr = make_file_record("data.rb", "unknown", "def greet\n  puts 'hello'\nend\n" * 20)
    chunks = chunker.chunk_file(fr)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.token_count <= 80  # overlap window can push slightly past max_tokens

def test_chunk_id_is_deterministic():
    chunker = ASTChunker()
    fr = make_file_record("foo.py", "python", "x = 1\n")
    chunks1 = chunker.chunk_file(fr)
    chunks2 = chunker.chunk_file(fr)
    assert [c.id for c in chunks1] == [c.id for c in chunks2]


# ---------------------------------------------------------------------------
# NonCodeChunker tests
# ---------------------------------------------------------------------------

def test_noncode_markdown_splits_on_headings():
    content = "# Section 1\nSome text\n## Subsection\nMore text\n# Section 2\nEnd\n"
    fr = make_file_record("README.md", "markdown", content)
    chunker = NonCodeChunker(max_tokens=100)
    chunks = chunker.chunk_file(fr)
    assert len(chunks) >= 2
    assert all(c.chunk_type == "prose" for c in chunks)

def test_noncode_json_single_chunk():
    content = '{"name": "my-app", "version": "1.0.0"}'
    fr = make_file_record("package.json", "json", content)
    chunker = NonCodeChunker(max_tokens=200)
    chunks = chunker.chunk_file(fr)
    assert len(chunks) >= 1
    assert chunks[0].chunk_type == "config"

def test_noncode_dockerfile():
    content = "FROM python:3.11\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\nCMD [\"python\", \"main.py\"]\n"
    fr = make_file_record("Dockerfile", "dockerfile", content)
    chunker = NonCodeChunker(max_tokens=100)
    chunks = chunker.chunk_file(fr)
    assert len(chunks) >= 1
    assert all(c.chunk_type == "config" for c in chunks)
