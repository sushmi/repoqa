"""Smoke test for Stage 3: structured QA generation per query.

Runs the full pipeline (Stage 2 retrieval + Stage 3 answer generation)
on a few sample questions and prints the structured output.

Usage:
    uv run scripts/test_qa.py \
        --chunks ./data/flask/transformers_chunks.json \
        --repo-name flask
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from repoqa.qa.pipeline import QAPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Stage 3 QA generation.")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--repo-name", default="flask")
    parser.add_argument("--n-results", type=int, default=8)
    parser.add_argument(
        "--save", default=None,
        help="Optional path to save all QA results as JSON.",
    )
    args = parser.parse_args()

    print("Initializing QA pipeline...")
    qa = QAPipeline.from_chunks_file(args.chunks)

    questions = [
        ("architecture", "How does Flask handle routing and URL dispatching?"),
        ("logic", "What does the make_response function do?"),
        ("deployment", "How is Flask configured for production deployment?"),
    ]

    results = []
    for expected_type, question in questions:
        print(f"\n{'='*70}")
        print(f"Q: {question}")
        print(f"   (expected type: {expected_type})")
        print(f"{'='*70}")

        result = qa.ask(question, n_results=args.n_results)
        results.append(result)

        print(f"\nType predicted: {result.question_type}")
        print(f"Confidence:    {result.confidence}")
        print(f"\n--- ANSWER ---")
        print(result.answer)

        print(f"\n--- CITATIONS ({result.verified.verified_count}/"
              f"{len(result.verified.citations)} verified) ---")
        for cite in result.verified.citations:
            mark = "OK" if cite.verified else "UNVERIFIED"
            loc = cite.file_path
            if cite.start_line:
                loc += f":{cite.start_line}"
                if cite.end_line and cite.end_line != cite.start_line:
                    loc += f"-{cite.end_line}"
            print(f"  [{mark}] {loc}")

        print(f"\nCoverage: {result.verified.coverage_ratio:.0%} "
              f"of citations verified against retrieved chunks")

    if args.save:
        QAPipeline.save_results(results, args.save)
        print(f"\nSaved {len(results)} results to {args.save}")

    print(f"\n{'='*70}")
    print("Stage 3 smoke test complete!")


if __name__ == "__main__":
    main()
