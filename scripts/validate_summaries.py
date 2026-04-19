"""Tier 1 summary validation — automated identifier grounding check.

Reads the hierarchical summaries produced during ingestion and verifies that
every identifier the LLM claimed exists in the corresponding source file(s).

Catches *gross* hallucination — fabricated function/class names that don't
appear anywhere in the file the summary claims to describe. Does not catch
subtle semantic errors (wrong relationship between two real entities, wrong
behavior description, etc.).

Usage:
    .venv/bin/python scripts/validate_summaries.py \\
        --summaries ./data/flask/summaries.json \\
        --repo-root ./flask \\
        --output ./data/flask/summary_validation.json
    
    
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Identifiers shorter than this are almost always false positives
# (language keywords, variable names like `i`, `n`)
MIN_IDENT_LEN = 3

# Common English/prose words that sneak into backticks and inflate the
# false-positive rate. Not an exhaustive stoplist — just the worst offenders.
_PROSE_STOPLIST = {
    "the", "and", "for", "with", "this", "that", "from", "into",
    "file", "path", "name", "code", "data", "list", "dict",
    "true", "false", "none", "null",
}


def extract_identifiers(summary_content: str) -> list[str]:
    """Extract identifier-like tokens the LLM explicitly claimed exist.

    Sources:
      - backtick-quoted tokens (`func_name`, `ClassName`)
      - inline snake_case with parens: foo_bar()
      - inline CamelCase with parens: FooBar()
    """
    idents: set[str] = set()

    for raw in re.findall(r"`([^`]+)`", summary_content):
        cleaned = raw.strip().strip("()@,.:;* ")
        if not cleaned or len(cleaned) < MIN_IDENT_LEN:
            continue
        if cleaned.lower() in _PROSE_STOPLIST:
            continue
        # Skip obvious non-identifiers (spaces, slashes — file paths are OK
        # but they're checked differently)
        if " " in cleaned:
            continue
        idents.add(cleaned)

    for raw in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\(\)", summary_content):
        if raw.lower() not in _PROSE_STOPLIST:
            idents.add(raw)

    return sorted(idents)


def check_grounding(
    identifier: str,
    source_text: str,
) -> bool:
    """Is this identifier present in the source text?

    Exact substring match — simple, fast, and sufficient for the gross-error
    case we care about (fabricated names that don't exist at all).
    """
    if identifier in source_text:
        return True
    # Try stripping trailing parens for function-call style mentions
    bare = identifier.rstrip("()")
    return bool(bare) and bare in source_text


def validate_file_summary(
    summary: dict,
    repo_root: Path,
) -> dict:
    """Check every identifier in a file-level summary against the source file."""
    source_path = repo_root / summary["path"]
    if not source_path.exists():
        return {
            "id": summary["id"],
            "path": summary["path"],
            "level": "file",
            "status": "source_missing",
            "n_identifiers": 0,
            "n_missing": 0,
            "missing": [],
        }

    try:
        source_text = source_path.read_text(errors="ignore")
    except OSError as exc:
        return {
            "id": summary["id"],
            "path": summary["path"],
            "level": "file",
            "status": f"read_error: {exc}",
            "n_identifiers": 0,
            "n_missing": 0,
            "missing": [],
        }

    identifiers = extract_identifiers(summary["content"])
    missing = [i for i in identifiers if not check_grounding(i, source_text)]

    return {
        "id": summary["id"],
        "path": summary["path"],
        "level": "file",
        "status": "grounded" if not missing else "has_missing",
        "n_identifiers": len(identifiers),
        "n_missing": len(missing),
        "missing": missing,
    }


def validate_dir_summary(
    summary: dict,
    repo_root: Path,
) -> dict:
    """Check identifiers against the concatenated text of every file in the dir.

    Directory summaries can legitimately reference identifiers from any file
    inside, so we build a per-directory corpus and check against that.
    """
    dir_path = repo_root / summary["path"]
    if not dir_path.exists() or not dir_path.is_dir():
        return {
            "id": summary["id"],
            "path": summary["path"],
            "level": "directory",
            "status": "source_missing",
            "n_identifiers": 0,
            "n_missing": 0,
            "missing": [],
        }

    # Corpus for a directory = filenames/subdir names + file contents.
    # Filenames must be included because summaries often reference files by
    # name (e.g. "contains quickstart.rst") and those names don't appear
    # inside the files themselves.
    corpus_parts: list[str] = []
    for child in dir_path.iterdir():
        corpus_parts.append(child.name)
        if child.is_file():
            try:
                corpus_parts.append(child.read_text(errors="ignore"))
            except OSError:
                continue
    corpus = "\n".join(corpus_parts)

    identifiers = extract_identifiers(summary["content"])
    missing = [i for i in identifiers if not check_grounding(i, corpus)]

    return {
        "id": summary["id"],
        "path": summary["path"],
        "level": "directory",
        "status": "grounded" if not missing else "has_missing",
        "n_identifiers": len(identifiers),
        "n_missing": len(missing),
        "missing": missing,
    }


def validate_project_summary(
    summary: dict,
    repo_root: Path,
) -> dict:
    """Check identifiers against the entire repo's text."""
    corpus_parts: list[str] = []
    for path in repo_root.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".rst", ".txt",
                                               ".yaml", ".yml", ".toml", ".cfg"}:
            try:
                corpus_parts.append(path.read_text(errors="ignore"))
            except OSError:
                continue
    corpus = "\n".join(corpus_parts)

    identifiers = extract_identifiers(summary["content"])
    missing = [i for i in identifiers if not check_grounding(i, corpus)]

    return {
        "id": summary["id"],
        "path": summary["path"],
        "level": "project",
        "status": "grounded" if not missing else "has_missing",
        "n_identifiers": len(identifiers),
        "n_missing": len(missing),
        "missing": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated grounding check for LLM-generated summaries."
    )
    parser.add_argument("--summaries", required=True,
                        help="Path to summaries.json")
    parser.add_argument("--repo-root", required=True,
                        help="Path to the checked-out repository")
    parser.add_argument("--output", required=True,
                        help="Per-summary validation results (JSON)")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    summaries = json.loads(Path(args.summaries).read_text())

    results: list[dict] = []
    for s in summaries:
        level = s.get("level")
        if level == "file":
            results.append(validate_file_summary(s, repo_root))
        elif level == "directory":
            results.append(validate_dir_summary(s, repo_root))
        elif level == "project":
            results.append(validate_project_summary(s, repo_root))

    # ── Aggregate stats ──
    by_level: dict[str, dict] = {}
    for r in results:
        lvl = r["level"]
        b = by_level.setdefault(lvl, {"n_summaries": 0, "n_identifiers": 0,
                                      "n_missing": 0, "n_with_any_miss": 0})
        b["n_summaries"] += 1
        b["n_identifiers"] += r["n_identifiers"]
        b["n_missing"] += r["n_missing"]
        if r["n_missing"] > 0:
            b["n_with_any_miss"] += 1

    # ── Report ──
    print(f"{'Level':<12} {'Summaries':>10} {'Idents':>8} {'Missing':>8} {'Grounded':>10} {'Clean':>10}")
    print("─" * 64)
    for lvl, b in by_level.items():
        grounding = (
            (b["n_identifiers"] - b["n_missing"]) / b["n_identifiers"]
            if b["n_identifiers"] else 1.0
        )
        clean = (b["n_summaries"] - b["n_with_any_miss"]) / b["n_summaries"]
        print(f"{lvl:<12} {b['n_summaries']:>10} {b['n_identifiers']:>8} "
              f"{b['n_missing']:>8} {grounding:>10.1%} {clean:>10.1%}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nPer-summary results → {out_path}")

    # Print the worst 5 for quick eyeballing
    worst = sorted(results, key=lambda r: r["n_missing"], reverse=True)[:5]
    if any(r["n_missing"] > 0 for r in worst):
        print("\nWorst 5 summaries (most ungrounded identifiers):")
        for r in worst:
            if r["n_missing"] == 0:
                continue
            print(f"  {r['path']:<60} {r['n_missing']:>3} missing: "
                  f"{', '.join(r['missing'][:5])}")


if __name__ == "__main__":
    sys.exit(main())
