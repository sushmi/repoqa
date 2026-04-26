"""Gold-chunk mining + qualitative / error / distribution analysis for 12 flask questions."""
from __future__ import annotations

import json
import re
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path("/Users/sushmi/dev/nlp/repoqa")

# ---------------------------------------------------------------------------
# 1. Define gold (LLM-derived; user said "assume 100% matched with human")
# ---------------------------------------------------------------------------
# Each entry: question_id -> list of gold (path, optional line range, rationale).
# Paths grounded in current flask repo (verified via Read above).
GOLD: dict[str, list[dict]] = {
    # === ARCHITECTURE ===
    "56c0744de3723104": [  # #1902 — flask CLI ImportError flaskr (FLASK_APP env var)
        {"path": "src/flask/cli.py",                    "rationale": "FLASK_APP loading + locate_app traceback"},
        {"path": "examples/tutorial/flaskr/__init__.py","rationale": "the flaskr example the user is running"},
    ],
    "2dfdab95bae5e9fd": [  # #1045 — CSS mime-type text/plain
        {"path": "src/flask/helpers.py",                "rationale": "send_file / send_from_directory mimetype handling"},
    ],
    "90cd1566dc0b8e73": [  # #1829 — flask run debug FLASK_DEBUG file-not-found on Windows
        {"path": "src/flask/cli.py",                    "rationale": "run_command + reloader flow"},
    ],
    "4f50ce3ff2582730": [  # #2626 — official guide for Flask app structure (factory pattern)
        {"path": "docs/patterns/appfactories.rst",      "rationale": "the factory-pattern doc"},
        {"path": "docs/patterns/packages.rst",          "rationale": "larger-applications doc"},
        {"path": "docs/tutorial/factory.rst",           "rationale": "tutorial factory pattern (may not be indexed)"},
    ],
    # === LOGIC ===
    "b80d28a9aa87e555": [  # #593 — sub-blueprints / nested register_blueprint
        {"path": "src/flask/sansio/blueprints.py",      "rationale": "Blueprint class + register_blueprint method"},
        {"path": "src/flask/blueprints.py",             "rationale": "Blueprint subclass shim"},
    ],
    "21e270d71953438f": [  # #1907 — TEMPLATES_AUTO_RELOAD when debug set after init
        {"path": "src/flask/app.py", "lines": (469, 495), "rationale": "create_jinja_environment auto_reload logic"},
    ],
    "f4473ee7623f7fcc": [  # #2086 — large file uploads cached in memory
        {"path": "docs/patterns/fileuploads.rst",       "rationale": "official upload pattern + MAX_CONTENT_LENGTH"},
        {"path": "src/flask/app.py",                    "rationale": "MAX_CONTENT_LENGTH default config"},
    ],
    "d4dbd02ea3802ded": [  # #941 — HTTPException catch-all errorhandler not effective
        {"path": "src/flask/app.py", "lines": (820, 900), "rationale": "handle_http_exception + handle_user_exception"},
        {"path": "src/flask/sansio/scaffold.py",        "rationale": "errorhandler decorator"},
    ],
    # === DEPLOYMENT ===
    "1433443e3f89d1ea": [  # #2023 — default logging handler
        {"path": "src/flask/logging.py",                "rationale": "create_logger + default_handler setup"},
        {"path": "src/flask/app.py",                    "rationale": "Flask.logger property"},
    ],
    "c079b0ca784639b6": [  # #1847 — flask CLI sys.path / packaging confusion
        {"path": "src/flask/cli.py",                    "rationale": "ScriptInfo / locate_app sys.path handling"},
        {"path": "docs/patterns/packages.rst",          "rationale": "larger-application packaging guide"},
    ],
    "e0f9194336752194": [  # #4027 — install_requires version pins (Werkzeug 2.0 broke things)
        {"path": "pyproject.toml",                      "rationale": "dependency pins for werkzeug/jinja/click"},
    ],
    "b8428fcec753a9fe": [  # #3168 — `flask run` output confusing for beginners
        {"path": "src/flask/cli.py",                    "rationale": "run_command output strings"},
    ],
}

ALL_QIDS = list(GOLD.keys())
QTYPE = {
    "56c0744de3723104": "architecture", "2dfdab95bae5e9fd": "architecture",
    "90cd1566dc0b8e73": "architecture", "4f50ce3ff2582730": "architecture",
    "b80d28a9aa87e555": "logic",        "21e270d71953438f": "logic",
    "f4473ee7623f7fcc": "logic",        "d4dbd02ea3802ded": "logic",
    "1433443e3f89d1ea": "deployment",   "c079b0ca784639b6": "deployment",
    "e0f9194336752194": "deployment",   "b8428fcec753a9fe": "deployment",
}

# ---------------------------------------------------------------------------
# 2. Map gold paths to chunk IDs from transformers_chunks.json
# ---------------------------------------------------------------------------
chunks = json.load(open(ROOT / "data/flask/transformers_chunks.json"))
chunks_by_path = defaultdict(list)
for c in chunks:
    chunks_by_path[c["file_record"]["repo_path"]].append(c)

def chunk_ids_for(gold_entries: list[dict]) -> set[str]:
    ids: set[str] = set()
    for g in gold_entries:
        path = g["path"]
        for c in chunks_by_path.get(path, []):
            if "lines" in g:
                lo, hi = g["lines"]
                if not (c["end_line"] < lo or c["start_line"] > hi):
                    ids.add(c["id"])
            else:
                ids.add(c["id"])
    return ids

GOLD_PATHS = {qid: {g["path"] for g in entries} for qid, entries in GOLD.items()}
GOLD_CHUNK_IDS = {qid: chunk_ids_for(entries) for qid, entries in GOLD.items()}

# Coverage: how many gold paths actually have chunks indexed?
indexed_path_hits = {qid: sum(1 for p in paths if p in chunks_by_path) / max(1, len(paths))
                    for qid, paths in GOLD_PATHS.items()}
print("=== GOLD COVERAGE ===")
for qid in ALL_QIDS:
    n_paths = len(GOLD_PATHS[qid])
    n_indexed = sum(1 for p in GOLD_PATHS[qid] if p in chunks_by_path)
    n_chunks = len(GOLD_CHUNK_IDS[qid])
    print(f"  {QTYPE[qid]:14s} {qid[:8]}  paths={n_paths}  indexed={n_indexed}  chunks={n_chunks}")

# ---------------------------------------------------------------------------
# 3. Load run JSONL + extract cited paths from each answer
# ---------------------------------------------------------------------------
CITE_RE = re.compile(r"\[([^\]\n]+?)(?::(\d+)(?:-(\d+))?)?\]")

def extract_cited_paths(answer: str) -> set[str]:
    paths: set[str] = set()
    for m in CITE_RE.finditer(answer):
        candidate = m.group(1).strip()
        if any(candidate.endswith(ext) for ext in (".py", ".rst", ".md", ".toml", ".cfg", ".yml", ".yaml", ".js", ".ts")):
            paths.add(candidate)
    return paths

with open(ROOT / "data/experiments/run_2026-04-15_n23.jsonl") as f:
    rows = [json.loads(l) for l in f]
results = [r for r in rows if r.get("type") == "result" and r.get("qa_id") in set(ALL_QIDS)]
print(f"\nRun rows for our 12 questions across all configs: {len(results)} (expected {12 * 6})")

# ---------------------------------------------------------------------------
# 4. Per-row scoring: gold path overlap + carry forward existing metrics
# ---------------------------------------------------------------------------
def normalize(p: str) -> str:
    """Match path suffixes — flask repo uses 'src/flask/...' but answers may say 'flask/...'."""
    return p.lstrip("./").replace("flask/src/", "src/")

def has_gold_path(answer_paths: set[str], gold_paths: set[str]) -> bool:
    for ap in {normalize(p) for p in answer_paths}:
        for gp in gold_paths:
            if ap.endswith(gp) or gp.endswith(ap):
                return True
    return False

# attach analysis fields per row
for r in results:
    cited = extract_cited_paths(r.get("answer", ""))
    r["_cited_paths"] = cited
    r["_gold_path_hit"] = has_gold_path(cited, GOLD_PATHS[r["qa_id"]])

# ---------------------------------------------------------------------------
# 5. Error bucketing per (config, question)
# ---------------------------------------------------------------------------
# Buckets (mutually exclusive, evaluated in order):
# - retrieval_miss: keyword retrieval_recall == 0
# - empty_context: n_citations == 0  (model didn't cite anything)
# - wrong_routing: classifier_correct is False
# - cited_wrong_path: n_citations > 0 but gold_path_hit is False
# - unverified_citation: cite_acc == 0 with n_citations > 0
# - low_judge_quality: judge.accuracy_mean < 6
# - clean_pass: everything else
def bucket(r) -> str:
    if r.get("retrieval_recall", 0) == 0:
        return "retrieval_miss"
    if r.get("n_citations", 0) == 0:
        return "empty_context"
    if not r.get("classifier_correct", True):
        return "wrong_routing"
    if not r["_gold_path_hit"]:
        return "cited_wrong_path"
    if r.get("citation_accuracy", 1.0) == 0 and r.get("n_citations", 0) > 0:
        return "unverified_citation"
    judge = r.get("judge") or {}
    if judge.get("accuracy_mean", 10) < 6:
        return "low_judge_quality"
    return "clean_pass"

for r in results:
    r["_bucket"] = bucket(r)

# ---------------------------------------------------------------------------
# 6. Distribution
# ---------------------------------------------------------------------------
print("\n=== ERROR DISTRIBUTION (per config, n=12 questions) ===")
configs = sorted({r["config"] for r in results})
buckets_order = ["retrieval_miss", "empty_context", "wrong_routing",
                 "cited_wrong_path", "unverified_citation", "low_judge_quality", "clean_pass"]

header = f"{'config':16s} | " + " | ".join(f"{b[:10]:>10s}" for b in buckets_order) + " | total"
print(header)
print("-" * len(header))
for cfg in configs:
    row = [r for r in results if r["config"] == cfg]
    counts = Counter(r["_bucket"] for r in row)
    cells = " | ".join(f"{counts.get(b, 0):>10d}" for b in buckets_order)
    print(f"{cfg:16s} | {cells} | {len(row):>5d}")

print("\n=== ERROR DISTRIBUTION BY QUESTION TYPE (full_repoqa only) ===")
fr = [r for r in results if r["config"] == "full_repoqa"]
by_type = defaultdict(Counter)
for r in fr:
    by_type[QTYPE[r["qa_id"]]][r["_bucket"]] += 1
for t in ("architecture", "logic", "deployment"):
    print(f"  {t:14s}: {dict(by_type[t])}")

# ---------------------------------------------------------------------------
# 7. Aggregate metrics for the 12-question subset
# ---------------------------------------------------------------------------
print("\n=== AGGREGATE METRICS (12-question subset) ===")
print(f"{'config':16s} | recall@k | gold_hit | cite_acc | judge_acc | n_clean")
print("-" * 76)
for cfg in configs:
    row = [r for r in results if r["config"] == cfg]
    recall = sum(r.get("retrieval_recall", 0) for r in row) / len(row)
    gold_hit = sum(r["_gold_path_hit"] for r in row) / len(row)
    cite_acc = sum(r.get("citation_accuracy", 0) for r in row) / len(row)
    judge_acc = [(r.get("judge") or {}).get("accuracy_mean") for r in row]
    judge_acc = [j for j in judge_acc if j is not None]
    judge_acc_mean = sum(judge_acc) / len(judge_acc) if judge_acc else float("nan")
    n_clean = sum(1 for r in row if r["_bucket"] == "clean_pass")
    print(f"{cfg:16s} | {recall:7.2%} | {gold_hit:7.2%} | {cite_acc:7.2%} | {judge_acc_mean:8.2f} | {n_clean:>6d}")

# ---------------------------------------------------------------------------
# 8. Qualitative cases — pick 1 case per bucket from full_repoqa
# ---------------------------------------------------------------------------
print("\n=== QUALITATIVE CASES (full_repoqa) ===")
fr_by_bucket = defaultdict(list)
for r in fr:
    fr_by_bucket[r["_bucket"]].append(r)

for b in buckets_order:
    if not fr_by_bucket[b]:
        continue
    case = fr_by_bucket[b][0]
    qid = case["qa_id"]
    print(f"\n--- [{b}] qid={qid[:8]} type={QTYPE[qid]} ---")
    print(f"Q: {case['question'][:200].replace(chr(10), ' ')}...")
    print(f"Answer (truncated): {case.get('answer','')[:300].replace(chr(10), ' ')}...")
    print(f"Cited paths: {sorted(case['_cited_paths'])}")
    print(f"Gold paths : {sorted(GOLD_PATHS[qid])}")
    print(f"Recall@k={case.get('retrieval_recall',0):.2f}  cite_acc={case.get('citation_accuracy',0):.2f}  "
          f"judge_acc={(case.get('judge') or {}).get('accuracy_mean','?')}  "
          f"classifier_correct={case.get('classifier_correct')}")

# ---------------------------------------------------------------------------
# 9. Save gold dataset
# ---------------------------------------------------------------------------
src = json.load(open(ROOT / "data/evaluation/flask_curated.json"))
keep = [q for q in src["qa_pairs"] if q["id"] in set(ALL_QIDS)]
for q in keep:
    q["reference_contexts"] = [
        {"path": g["path"], "lines": list(g.get("lines", [])), "rationale": g["rationale"],
         "chunk_ids": sorted(c["id"] for c in chunks_by_path.get(g["path"], [])
                             if "lines" not in g or not (c["end_line"] < g["lines"][0] or c["start_line"] > g["lines"][1]))}
        for g in GOLD[q["id"]]
    ]
out = {**{k: src[k] for k in ("name", "version", "metadata")},
       "description": "12-question gold subset (4 architecture, 4 logic, 4 deployment)",
       "qa_pairs": keep}
out_path = ROOT / "data/evaluation/flask_gold12.json"
json.dump(out, open(out_path, "w"), indent=2)
print(f"\nSaved gold dataset → {out_path}")
