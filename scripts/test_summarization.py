"""Quick smoke test for the summarization pipeline against Ollama/Qwen.

Usage:
    python scripts/test_summarization.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from repoqa.models import Chunk, FileRecord

# --- Fake data: one file with two chunks ---
file_record = FileRecord(
    repo_path="src/calculator.py",
    abs_path="/tmp/src/calculator.py",
    language="python",
    size_bytes=200,
    raw_content="def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n",
)

chunks = [
    Chunk(
        id=Chunk.make_id("src/calculator.py", 1),
        file_record=file_record,
        content="def add(a, b):\n    return a + b",
        chunk_type="function",
        start_line=1,
        end_line=2,
        symbol_name="add",
        token_count=15,
    ),
    Chunk(
        id=Chunk.make_id("src/calculator.py", 4),
        file_record=file_record,
        content="def multiply(a, b):\n    return a * b",
        chunk_type="function",
        start_line=4,
        end_line=5,
        symbol_name="multiply",
        token_count=15,
    ),
]

# --- Run pipeline ---
from repoqa.summarization.pipeline import HierarchicalSummarizationPipeline

print("Building pipeline (connecting to Ollama)...")
pipeline = HierarchicalSummarizationPipeline()

print("Running summarization on 2 fake chunks...")
summaries = pipeline.run(chunks, repo_root="test-repo")

print(f"\n{'='*60}")
print(f"Got {len(summaries)} summaries:\n")
for s in summaries:
    print(f"[{s.level}] {s.path}")
    print(f"  {s.content[:200]}...")
    print()

print("All phases passed!")
