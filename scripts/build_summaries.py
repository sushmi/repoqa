#!/usr/bin/env python
"""02. CLI: run hierarchical summarization over ingested chunks.

Usage:
    python scripts/build_summaries.py \\
        --chunks ./data/chunks.json \\
        --repo-root /path/to/target-repo \\
        --output ./data/summaries.json
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from repoqa.ingestion.pipeline import IngestionPipeline
from repoqa.summarization.pipeline import HierarchicalSummarizationPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hierarchical summaries from chunks.")
    parser.add_argument("--chunks", required=True, help="Path to chunks JSON (from ingest_repo.py).")
    parser.add_argument("--repo-root", default=".", help="Repository root path (for project summary naming).")
    parser.add_argument("--output", default="./data/summaries.json", help="Output JSON file path.")
    args = parser.parse_args()

    chunks = IngestionPipeline.load_chunks(args.chunks)
    print(f"Loaded {len(chunks)} chunks from {args.chunks}")

    pipeline = HierarchicalSummarizationPipeline()
    summaries = pipeline.run_and_save(
        chunks=chunks,
        repo_root=args.repo_root,
        output_path=args.output,
    )

    level_counts = Counter(s.level for s in summaries)
    print("Summaries by level:", dict(level_counts))


if __name__ == "__main__":
    main()
