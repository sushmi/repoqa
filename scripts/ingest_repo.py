#!/usr/bin/env python
"""01. CLI: run repository ingestion.

Usage:
    python scripts/ingest_repo.py --repo-path /path/to/target-repo --output ./data/chunks.json
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from repoqa.ingestion.pipeline import IngestionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a repository into chunks.")
    parser.add_argument("--repo-path", required=True, help="Path to the target repository root.")
    parser.add_argument("--output", default="./data/chunks.json", help="Output JSON file path.")
    args = parser.parse_args()

    pipeline = IngestionPipeline()
    pipeline.run_and_save(repo_root=args.repo_path, output_path=args.output)


if __name__ == "__main__":
    main()
