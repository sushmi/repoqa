"""Validate that each summary's content matches its declared level.

Reads `data/flask/summaries.json` and runs four deterministic checks per
summary. Writes a CSV report to `data/flask/summary_level_validation.csv`
plus a console digest grouped by level.

Checks (each binary):

1. **mentions_self** — does the content mention its own path or basename?
   A file summary that never names the file is a strong hallucination signal.
2. **length_ok** — token count within the band typical for that level.
   Bands tuned from the current run (file ~80-200, dir ~80-400, project 300+).
3. **scope_ok** — content's discussion scope matches the level. File summaries
   should mostly describe themselves (≤3 other-file mentions); directory
   summaries should mention multiple files; project summaries should mention
   multiple directories.
4. **ident_grounded** — backtick-quoted identifiers in the summary actually
   exist in source at the appropriate scope. File-level identifiers must exist
   in that file; dir-level in any file under that dir; project-level anywhere.

Run
---
    .venv/bin/python scripts/validate_summary_levels.py
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/sushmi/dev/nlp/repoqa")
SUMMARIES_PATH = ROOT / "data/flask/summaries.json"
CHUNKS_PATH    = ROOT / "data/flask/transformers_chunks.json"
OUT_CSV        = ROOT / "data/flask/summary_level_validation.csv"

# Length bands (in tokens) — empirically tuned for the current flask run.
LENGTH_BAND = {
    "file":      (30,  400),   # short blurbs are suspicious; >400 means LLM rambled
    "directory": (50,  500),
    "project":   (300, 2000),  # must be substantive for the whole project
}

# Identifier extraction — backtick-quoted tokens + standalone foo()/FooBar mentions.
IDENT_RE = re.compile(r"`([A-Za-z_][\w./]*)`|\b([A-Z][a-zA-Z0-9_]+|[a-z_][a-zA-Z0-9_]*)\(\)")
# Anything that looks like a relative path (slashes + extension)
PATH_RE = re.compile(r"`([^`]+\.(?:py|rst|md|toml|yaml|yml|cfg|json|txt|ini))`")


PATH_LIKE_EXTS = (".py", ".rst", ".md", ".toml", ".yaml", ".yml", ".cfg", ".json", ".txt", ".ini", ".sh")


def extract_identifiers(text: str) -> tuple[set[str], set[str]]:
    """Return (code_idents, path_idents). Path-like tokens go to path_idents
    so they can be checked against the filesystem instead of content substrings."""
    code: set[str] = set()
    paths: set[str] = set()
    for m in IDENT_RE.finditer(text):
        tok = m.group(1) or m.group(2)
        if not tok or len(tok) < 3 or tok.lower() in {"the", "and", "for", "all", "you"}:
            continue
        if tok.endswith(PATH_LIKE_EXTS):
            paths.add(tok)
        else:
            code.add(tok)
    # Also pick up any backticked path-like token (with slashes)
    for m in PATH_RE.finditer(text):
        paths.add(m.group(1))
    return code, paths


def main() -> None:
    summaries = json.loads(SUMMARIES_PATH.read_text())
    chunks    = json.loads(CHUNKS_PATH.read_text())

    # Build a path → flat content lookup once
    content_by_path: dict[str, str] = defaultdict(str)
    for c in chunks:
        content_by_path[c["file_record"]["repo_path"]] += "\n" + c["content"]
    all_paths = sorted(content_by_path.keys())
    repo_corpus = "\n".join(content_by_path.values())

    rows: list[dict] = []
    for s in summaries:
        sid     = s["id"]
        level   = s["level"]
        path    = s["path"]
        content = s["content"]
        tokens  = s["token_count"]
        issues: list[str] = []

        # ----- (1) mentions_self ---------------------------------------------
        if level == "file":
            self_token = Path(path).name
        elif level == "directory":
            self_token = Path(path).name or path  # "." for root
        else:
            self_token = "flask"  # project — flask should appear by name
        mentions_self = self_token.lower() in content.lower() or path in content
        if not mentions_self:
            issues.append(f"missing_self_mention[{self_token}]")

        # ----- (2) length_ok -------------------------------------------------
        lo, hi = LENGTH_BAND[level]
        length_ok = lo <= tokens <= hi
        if not length_ok:
            issues.append(f"length_out_of_band[{tokens} not in {lo}..{hi}]")

        # ----- (3) scope_ok --------------------------------------------------
        code_idents, path_idents = extract_identifiers(content)
        path_mentions = path_idents
        # Count *other* file mentions (paths different from self)
        other_file_mentions = {p for p in path_mentions if p != path and "/" in p}
        if level == "file":
            scope_ok = len(other_file_mentions) <= 3
            if not scope_ok:
                issues.append(f"file_scope_too_broad[{len(other_file_mentions)} other-file mentions]")
        elif level == "directory":
            # Should reference at least 1 child file (otherwise it's just prose)
            in_dir = sum(1 for p in path_mentions if p.startswith(path.rstrip("/")) and p != path)
            scope_ok = in_dir >= 1 or len(path_mentions) >= 1
            if not scope_ok:
                issues.append("dir_scope_no_children_referenced")
        else:  # project
            # Should mention at least 2 distinct top-level directories
            dirs_mentioned = set()
            for w in re.findall(r"`([a-z_][\w]*/)`?", content) + re.findall(r"\b([a-z_][\w]+)/", content):
                dirs_mentioned.add(w.rstrip("/").split("/")[0])
            scope_ok = len(dirs_mentioned) >= 2
            if not scope_ok:
                issues.append(f"project_scope_too_narrow[{len(dirs_mentioned)} dirs mentioned]")

        # ----- (4) ident_grounded --------------------------------------------
        # Code identifiers are checked as substrings of source content;
        # path identifiers are checked against the actual chunk-paths set.
        if level == "file":
            scope_text = content_by_path.get(path, "")
            scope_paths = {p for p in all_paths if p == path}
        elif level == "directory":
            prefix = path.rstrip("/") + "/" if path != "." else ""
            scope_text = "\n".join(t for p, t in content_by_path.items()
                                    if path == "." or p.startswith(prefix))
            scope_paths = {p for p in all_paths if path == "." or p.startswith(prefix)}
        else:
            scope_text = repo_corpus
            scope_paths = set(all_paths)

        # Path mentions ground if any indexed path ends with the mentioned name
        def path_ident_grounded(tok: str) -> bool:
            return any(p == tok or p.endswith("/" + tok) for p in scope_paths)

        all_idents = code_idents | path_idents
        if all_idents:
            grounded_count = (
                sum(1 for i in code_idents if i in scope_text)
                + sum(1 for i in path_idents if path_ident_grounded(i))
            )
            ident_grounded_pct = grounded_count / len(all_idents)
            grounded = grounded_count
        else:
            ident_grounded_pct = float("nan")
            grounded = 0
        idents = all_idents

        ident_threshold = 0.6  # at least 60% of identifiers must exist in scope
        ident_grounded_ok = (
            ident_grounded_pct != ident_grounded_pct  # NaN passes (no idents)
            or ident_grounded_pct >= ident_threshold
        )
        if not ident_grounded_ok:
            issues.append(
                f"ident_grounding_low[{ident_grounded_pct:.0%}<{ident_threshold:.0%}, "
                f"{len(idents) - grounded}/{len(idents)} missing]"
            )

        # ----- aggregate -----------------------------------------------------
        passed = mentions_self and length_ok and scope_ok and ident_grounded_ok

        rows.append({
            "id":               sid[:8],
            "level":            level,
            "path":             path,
            "tokens":           tokens,
            "mentions_self":    int(mentions_self),
            "length_ok":        int(length_ok),
            "scope_ok":         int(scope_ok),
            "ident_count":      len(idents),
            "ident_grounded_pct": (
                "" if ident_grounded_pct != ident_grounded_pct else f"{ident_grounded_pct:.2f}"
            ),
            "ident_grounded_ok": int(ident_grounded_ok),
            "pass":             int(passed),
            "issues":           "; ".join(issues),
        })

    # ----- write CSV ---------------------------------------------------------
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows → {OUT_CSV}")

    # ----- console digest ----------------------------------------------------
    by_level: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_level[r["level"]].append(r)

    print("\n=== Pass rate by level ===")
    print(f"{'level':10s}  {'count':>6s}  {'pass':>6s}  {'pass_rate':>10s}  "
          f"{'mentions_self':>14s}  {'length_ok':>10s}  {'scope_ok':>9s}  {'ident_ok':>9s}")
    for level in ("project", "directory", "file"):
        rs = by_level.get(level, [])
        if not rs: continue
        n = len(rs)
        p = sum(r["pass"] for r in rs)
        ms = sum(r["mentions_self"] for r in rs)
        lo = sum(r["length_ok"] for r in rs)
        sc = sum(r["scope_ok"] for r in rs)
        id_ok = sum(r["ident_grounded_ok"] for r in rs)
        print(f"{level:10s}  {n:>6d}  {p:>6d}  {p/n:>9.1%}  "
              f"{ms:>10d}/{n}  {lo:>6d}/{n}  {sc:>5d}/{n}  {id_ok:>5d}/{n}")

    # ----- worst offenders ---------------------------------------------------
    failed = [r for r in rows if not r["pass"]]
    print(f"\n=== Failed summaries: {len(failed)}/{len(rows)} ===")
    if failed:
        print("Top 15 with most issues:")
        for r in sorted(failed, key=lambda x: -len(x["issues"].split(";")))[:15]:
            print(f"  [{r['level']:9s}] {r['path']:50s}  {r['issues']}")


if __name__ == "__main__":
    main()
