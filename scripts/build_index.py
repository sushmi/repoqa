#!/usr/bin/env python
"""CLI: embed chunks + summaries and store in ChromaDB.

Usage:
    python scripts/build_index.py \\
        --chunks ./data/chunks.json \\
        --summaries ./data/summaries.json \\
        --repo-name my-repo
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from repoqa.embedding.pipeline import EmbeddingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed and index chunks + summaries into ChromaDB.")
    parser.add_argument("--chunks", required=True, help="Path to chunks JSON.")
    parser.add_argument("--summaries", required=True, help="Path to summaries JSON.")
    parser.add_argument("--repo-name", required=True, help="Short identifier for this repository.")
    args = parser.parse_args()

    pipeline = EmbeddingPipeline()
    pipeline.run_from_cache(
        chunks_json=args.chunks,
        summaries_json=args.summaries,
        repo_name=args.repo_name,
    )
    stats = pipeline.store.get_collection_stats()
    print(f"Done. ChromaDB collection '{stats['collection_name']}' now has {stats['total_documents']} documents.")


if __name__ == "__main__":
    main()
