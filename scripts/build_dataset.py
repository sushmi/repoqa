#!/usr/bin/env python3
"""Build a CoReQA-style evaluation dataset from GitHub repositories.

Usage
-----
    # Mine from default repos list
    python scripts/build_dataset.py --output data/evaluation/dataset.json

    # Mine specific repos
    python scripts/build_dataset.py \\
        --repos pallets/flask tiangolo/fastapi \\
        --output data/evaluation/dataset.json

    # Use a custom repos file
    python scripts/build_dataset.py \\
        --repos-file config/target_repos.json \\
        --output data/evaluation/dataset.json

Environment
-----------
    GITHUB_TOKEN   — required, a GitHub personal access token.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from repoqa.evaluation.dataset_builder import DatasetBuilder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a CoReQA-style evaluation dataset from GitHub repos.",
    )
    parser.add_argument("--repos", nargs="+", help="owner/repo list")
    parser.add_argument(
        "--repos-file",
        default="config/target_repos.json",
        help="JSON file listing target repos (default: config/target_repos.json)",
    )
    parser.add_argument(
        "--output",
        default="data/evaluation/dataset.json",
        help="Output JSON path (default: data/evaluation/dataset.json)",
    )
    parser.add_argument(
        "--min-reactions",
        type=int,
        default=3,
        help="Min positive reactions for a valid answer (default: 3)",
    )
    parser.add_argument("--name", default="repoqa-eval", help="Dataset name")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit(
            "ERROR: GITHUB_TOKEN env var is required.\n"
            "Create one at https://github.com/settings/personal-access-tokens"
            "and export GITHUB_TOKEN=github_pat_..."
        )

    if args.repos:
        repos = args.repos
    else:
        rf = Path(args.repos_file)
        if not rf.exists():
            sys.exit(f"ERROR: repos file not found: {rf}")
        repos = json.loads(rf.read_text())

    builder = DatasetBuilder(github_token=token, min_reactions=args.min_reactions)
    dataset = builder.build(repos, args.output, dataset_name=args.name)

    s = dataset.summary()
    print(f"\nDataset built successfully!")
    print(f"  Total QA pairs : {s['total']}")
    print(f"  Repositories   : {s['repos']}")
    print(f"  By language    : {s['by_language']}")
    print(f"  By type        : {s['by_question_type']}")
    print(f"  Saved to       : {args.output}")


if __name__ == "__main__":
    main()
