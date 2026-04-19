"""Aggregate experiment JSONL results into a shareable Markdown report.

Reads:
    data/experiments/run_YYYY-MM-DD.jsonl

Writes:
    data/experiments/report_YYYY-MM-DD.md
    data/experiments/aggregated_scores.csv

Produces:
  - Ablation table (mean ± std per config × metric)
  - Per-question-type breakdown
  - Classifier accuracy
  - Citation accuracy
  - Latency comparison
  - Key findings block computed from the deltas
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Data loading ─────────────────────────────────────────────────

def load_jsonl(path: str) -> tuple[dict, list[dict]]:
    """Return (metadata, list_of_result_records)."""
    metadata = {}
    results: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "metadata":
                metadata = rec
            elif rec.get("type") == "result" and "error" not in rec:
                results.append(rec)
    return metadata, results


# ── Aggregation ──────────────────────────────────────────────────

def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


def aggregate_by_config(results: list[dict]) -> dict[str, dict]:
    """For each config, compute mean ± std per metric."""
    by_config: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_config[r["config"]].append(r)

    summary: dict[str, dict] = {}
    for config, recs in by_config.items():
        has_judge = any(r.get("judge") for r in recs)

        agg = {
            "n_questions": len(recs),
            "retrieval_recall": _mean_std([r["retrieval_recall"] for r in recs]),
            "citation_accuracy": _mean_std([r["citation_accuracy"] for r in recs]),
            "classifier_acc": sum(1 for r in recs if r["classifier_correct"]) / len(recs),
            "latency_s": _mean_std([r["gen_latency_s"] for r in recs]),
        }
        if has_judge:
            agg["accuracy"] = _mean_std([r["judge"]["accuracy_mean"] for r in recs if r.get("judge")])
            agg["completeness"] = _mean_std([r["judge"]["completeness_mean"] for r in recs if r.get("judge")])
            agg["relevance"] = _mean_std([r["judge"]["relevance_mean"] for r in recs if r.get("judge")])
            agg["clarity"] = _mean_std([r["judge"]["clarity_mean"] for r in recs if r.get("judge")])
        summary[config] = agg
    return summary


def aggregate_by_config_and_type(results: list[dict]) -> dict:
    """config × question_type breakdown for retrieval recall."""
    bucket = defaultdict(lambda: defaultdict(list))
    for r in results:
        bucket[r["config"]][r["question_type_expected"]].append(r["retrieval_recall"])
    return {
        cfg: {qt: _mean_std(vals) for qt, vals in types.items()}
        for cfg, types in bucket.items()
    }


# ── Rendering ────────────────────────────────────────────────────

def _fmt(mean_std: tuple[float, float], as_pct: bool = False) -> str:
    m, s = mean_std
    if as_pct:
        return f"{m*100:.1f}% ± {s*100:.1f}"
    return f"{m:.2f} ± {s:.2f}"


def render_ablation_table(summary: dict, has_judge: bool) -> str:
    """Markdown table: config rows × metric columns."""
    if has_judge:
        header = (
            "| Configuration | Accuracy | Completeness | Relevance | Clarity "
            "| Recall@k | Cite Acc | Latency (s) |"
        )
        sep = "|" + "|".join(["---"] * 8) + "|"
    else:
        header = (
            "| Configuration | Recall@k | Cite Acc | Classifier Acc | Latency (s) |"
        )
        sep = "|" + "|".join(["---"] * 5) + "|"

    rows = [header, sep]

    preferred = ["full_repoqa", "no_summaries", "semantic_only",
                 "no_expansion", "no_routing", "bm25_baseline"]
    order = [c for c in preferred if c in summary] + [c for c in summary if c not in preferred]

    for cfg in order:
        s = summary[cfg]
        if has_judge and "accuracy" in s:
            rows.append(
                f"| {cfg} | {_fmt(s['accuracy'])} | {_fmt(s['completeness'])} "
                f"| {_fmt(s['relevance'])} | {_fmt(s['clarity'])} "
                f"| {_fmt(s['retrieval_recall'], as_pct=True)} "
                f"| {_fmt(s['citation_accuracy'], as_pct=True)} "
                f"| {_fmt(s['latency_s'])} |"
            )
        else:
            rows.append(
                f"| {cfg} | {_fmt(s['retrieval_recall'], as_pct=True)} "
                f"| {_fmt(s['citation_accuracy'], as_pct=True)} "
                f"| {s['classifier_acc']:.0%} "
                f"| {_fmt(s['latency_s'])} |"
            )
    return "\n".join(rows)


def render_type_breakdown(by_type: dict) -> str:
    q_types = set()
    for cfg_dict in by_type.values():
        q_types.update(cfg_dict.keys())
    q_types = sorted(q_types)

    header = "| Configuration | " + " | ".join(q_types) + " |"
    sep = "|" + "|".join(["---"] * (len(q_types) + 1)) + "|"
    rows = [header, sep]

    for cfg in sorted(by_type):
        cells = [cfg]
        for qt in q_types:
            stats = by_type[cfg].get(qt, (0.0, 0.0))
            cells.append(_fmt(stats, as_pct=True))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def compute_findings(summary: dict) -> list[str]:
    """Auto-generate key-finding bullets from the aggregated numbers."""
    findings: list[str] = []

    full = summary.get("full_repoqa")
    bm25 = summary.get("bm25_baseline")

    if full and bm25 and "accuracy" in full and "accuracy" in bm25:
        delta_acc = full["accuracy"][0] - bm25["accuracy"][0]
        findings.append(
            f"**Full RepoQA vs BM25:** accuracy delta = **+{delta_acc:.2f}** "
            f"({full['accuracy'][0]:.2f} vs {bm25['accuracy'][0]:.2f}). "
            f"{'Hypothesis validated' if delta_acc > 0.3 else 'Marginal gain'}."
        )

    if full:
        r_full = full["retrieval_recall"][0]
        if bm25:
            r_bm25 = bm25["retrieval_recall"][0]
            findings.append(
                f"**Retrieval recall:** full RepoQA = {r_full:.1%}, "
                f"BM25 baseline = {r_bm25:.1%} "
                f"(delta: **{(r_full-r_bm25)*100:+.1f}pp**)."
            )

    # Ablation contributions
    for ablation in ["no_summaries", "semantic_only", "no_expansion", "no_routing"]:
        abl = summary.get(ablation)
        if full and abl and "accuracy" in full and "accuracy" in abl:
            drop = full["accuracy"][0] - abl["accuracy"][0]
            if abs(drop) > 0.05:
                component = {
                    "no_summaries": "summaries",
                    "semantic_only": "hybrid (BM25+RRF)",
                    "no_expansion": "query expansion",
                    "no_routing": "question-type routing",
                }[ablation]
                findings.append(
                    f"**{component}** contributes **{drop:+.2f}** to accuracy "
                    f"({full['accuracy'][0]:.2f} with vs {abl['accuracy'][0]:.2f} without)."
                )

    if full and "classifier_acc" in full:
        findings.append(
            f"**Classifier accuracy:** {full['classifier_acc']:.0%} "
            f"on full-RepoQA config."
        )

    return findings


def render_report(metadata: dict, summary: dict, by_type: dict) -> str:
    has_judge = any("accuracy" in s for s in summary.values())

    lines = [
        f"# RepoQA Experiment Report — {metadata.get('run_date', 'unknown date')[:10]}",
        "",
        "## Setup",
        "",
        f"- **Repo:** `{metadata.get('repo_filter', '?')}` "
        f"@ `{metadata.get('repo_git_sha', '?')}`",
        f"- **Generator model:** `{metadata.get('generator_model', '?')}` (Qwen 2.5, local via Ollama)",
        f"- **Embedding model:** `{metadata.get('embedding_model', '?')}` "
        f"(nomic-embed-text, local via Ollama)",
        f"- **Judge model:** `{metadata.get('judge_model', 'none')}` "
        f"(different model family → avoids self-evaluation bias)",
        f"- **Questions evaluated:** {metadata.get('n_questions', '?')}",
        f"- **Configurations tested:** {metadata.get('n_configs', '?')}",
        f"- **Judge runs per question:** {metadata.get('judge_runs_per_question', 0)} "
        "(scores are mean ± std across runs)",
        f"- **Top-K chunks retrieved:** {metadata.get('n_results', '?')}",
        "",
        "## Ablation Results",
        "",
        render_ablation_table(summary, has_judge),
        "",
        "## Retrieval Recall by Question Type",
        "",
        render_type_breakdown(by_type),
        "",
        "## Key Findings",
        "",
    ]

    findings = compute_findings(summary)
    if findings:
        for f in findings:
            lines.append(f"- {f}")
    else:
        lines.append("_(No findings auto-computed — check raw data.)_")

    lines.extend([
        "",
        "## Caveats",
        "",
        "- Absolute scores reflect our choice of generator (Qwen 2.5 7B, free/local) "
        "and embedder (nomic-embed-text). Comparisons in published CoReQA work use "
        "GPT-4o and will show higher absolute numbers.",
        "- The research hypothesis concerns **relative improvement** of the "
        "combined architecture over BM25, which is unaffected by model choice.",
        "- Each configuration uses the same embedding/generator model; only the "
        "retrieval architecture varies across rows, so component ablations are fair.",
        "",
        "## Reproducibility",
        "",
        f"- Raw results: this file's companion JSONL (one line per (config, question) pair)",
        f"- Re-run with the same JSONL to regenerate this report",
        f"- Repo git SHA: `{metadata.get('repo_git_sha', '?')}`",
        "",
    ])
    return "\n".join(lines)


def write_csv(summary: dict, path: str) -> None:
    """Flat CSV for importing into slides / external tools."""
    has_judge = any("accuracy" in s for s in summary.values())
    headers = ["config", "n_questions"]
    if has_judge:
        headers += [
            "accuracy_mean", "accuracy_std",
            "completeness_mean", "completeness_std",
            "relevance_mean", "relevance_std",
            "clarity_mean", "clarity_std",
        ]
    headers += [
        "retrieval_recall_mean", "retrieval_recall_std",
        "citation_accuracy_mean", "citation_accuracy_std",
        "classifier_acc",
        "latency_s_mean", "latency_s_std",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for cfg, s in summary.items():
            row: list = [cfg, s["n_questions"]]
            if has_judge:
                for key in ("accuracy", "completeness", "relevance", "clarity"):
                    m, std = s.get(key, (0.0, 0.0))
                    row.extend([f"{m:.3f}", f"{std:.3f}"])
            for key in ("retrieval_recall", "citation_accuracy"):
                m, std = s[key]
                row.extend([f"{m:.3f}", f"{std:.3f}"])
            row.append(f"{s['classifier_acc']:.3f}")
            m, std = s["latency_s"]
            row.extend([f"{m:.3f}", f"{std:.3f}"])
            writer.writerow(row)


# ── Main ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True,
                        help="Experiment results JSONL from run_experiments.py")
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    md_path = Path(args.output_md) if args.output_md else jsonl_path.with_name(
        jsonl_path.stem.replace("run_", "report_") + ".md"
    )
    csv_path = Path(args.output_csv) if args.output_csv else jsonl_path.with_name(
        "aggregated_scores.csv"
    )

    print(f"Reading: {jsonl_path}")
    metadata, results = load_jsonl(str(jsonl_path))
    print(f"  metadata: {metadata.get('run_date', '?')}")
    print(f"  results: {len(results)} records")

    summary = aggregate_by_config(results)
    by_type = aggregate_by_config_and_type(results)

    report_md = render_report(metadata, summary, by_type)
    md_path.write_text(report_md)
    print(f"Wrote: {md_path}")

    write_csv(summary, str(csv_path))
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
