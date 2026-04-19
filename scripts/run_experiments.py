"""Run the full ablation experiment grid.

For each configuration × each question × N runs:
  1. Construct a QAPipeline with the config's ablation flags
  2. Ask the question, get an answer
  3. Compute deterministic metrics (retrieval recall, citation accuracy)
  4. Score with Gemini judge (4 dimensions × 3 runs → mean ± std)
  5. Log raw results to JSONL

Configurations match the ablation table from the project slides:
  - full_repoqa    : all 4 components enabled
  - no_summaries   : summaries disabled
  - semantic_only  : hybrid (BM25+RRF) disabled
  - no_expansion   : query expansion disabled
  - no_routing     : question-type routing disabled
  - bm25_baseline  : all components disabled (semantic off via hybrid → BM25 only)
  - no_retrieval   : retrieval skipped entirely

Usage:
    export GEMINI_API_KEY=your_key_here
    uv run scripts/run_experiments.py \\
        --chunks ./data/flask/transformers_chunks.json \\
        --repo-root ./flask \\
        --dataset ./data/evaluation/seed_dataset.json \\
        --repo-filter pallets/flask \\
        --output ./data/experiments/run_$(date +%Y-%m-%d).jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings
from repoqa.evaluation.judge import GeminiJudge, OllamaJudge
from repoqa.evaluation.metrics import (
    citation_accuracy,
    retrieval_recall_at_k,
)
from repoqa.ingestion.pipeline import IngestionPipeline
from repoqa.qa.pipeline import QAPipeline

logging.basicConfig(level=logging.WARNING)


# ── Ablation configurations ──────────────────────────────────────

CONFIGS = [
    {
        "name": "full_repoqa",
        "flags": {
            "enable_summaries": True, "enable_hybrid": True,
            "enable_query_expansion": True, "enable_routing": True,
            "enable_retrieval": True,
        },
    },
    {
        "name": "no_summaries",
        "flags": {
            "enable_summaries": False, "enable_hybrid": True,
            "enable_query_expansion": True, "enable_routing": True,
            "enable_retrieval": True,
        },
    },
    {
        "name": "semantic_only",
        "flags": {
            "enable_summaries": True, "enable_hybrid": False,
            "enable_query_expansion": True, "enable_routing": True,
            "enable_retrieval": True,
        },
    },
    {
        "name": "no_expansion",
        "flags": {
            "enable_summaries": True, "enable_hybrid": True,
            "enable_query_expansion": False, "enable_routing": True,
            "enable_retrieval": True,
        },
    },
    {
        "name": "no_routing",
        "flags": {
            "enable_summaries": True, "enable_hybrid": True,
            "enable_query_expansion": True, "enable_routing": False,
            "enable_retrieval": True,
        },
    },
    {
        "name": "bm25_baseline",
        # All LLM-powered retrieval enhancements off; semantic effectively
        # disabled because the strategy weights favor BM25 under no-routing.
        # Semantic ChromaDB is still queried but BM25 dominates RRF.
        "flags": {
            "enable_summaries": False, "enable_hybrid": True,
            "enable_query_expansion": False, "enable_routing": False,
            "enable_retrieval": True,
        },
    },
]


# ── Helpers ──────────────────────────────────────────────────────

def build_settings(overrides: dict) -> Settings:
    """Create a Settings instance with the given ablation overrides."""
    # Settings reads from env vars. Overlay flags on top of the current env.
    # We use model_construct path by instantiating with explicit overrides.
    s = Settings()
    for key, value in overrides.items():
        setattr(s, key, value)
    return s


def get_git_sha(path: str) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=path, capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ── Main ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation experiments.")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--repo-root", required=True,
                        help="Path to checked-out repo (for citation accuracy)")
    parser.add_argument("--dataset", default="data/evaluation/seed_dataset.json")
    parser.add_argument("--repo-filter", default="pallets/flask",
                        help="Only use QA pairs where repo_name matches this")
    parser.add_argument("--output", required=True,
                        help="JSONL output path, e.g. data/experiments/run.jsonl")
    parser.add_argument("--n-results", type=int, default=8,
                        help="Top-K chunks to retrieve per query")
    parser.add_argument("--judge-runs", type=int, default=3,
                        help="Gemini judge runs per question (mean ± std)")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Subset of config names to run (default: all)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Cap N questions (for debugging)")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM judge scoring (deterministic metrics only)")
    parser.add_argument("--judge-provider", choices=["gemini", "ollama"], default=None,
                        help="Override JUDGE_PROVIDER setting (gemini=API, ollama=local)")
    args = parser.parse_args()

    # ── Load data ──
    print("Loading chunks...")
    chunks = IngestionPipeline.load_chunks(args.chunks)
    known_paths = {c.file_record.repo_path for c in chunks}
    print(f"  {len(chunks)} chunks, {len(known_paths)} unique files")

    with open(args.dataset) as f:
        raw = json.load(f)
    qa_pairs = [qa for qa in raw["qa_pairs"] if qa["repo_name"] == args.repo_filter]
    if args.max_questions:
        qa_pairs = qa_pairs[:args.max_questions]
    print(f"  {len(qa_pairs)} questions for {args.repo_filter}")

    if not qa_pairs:
        print(f"ERROR: no questions found for repo_filter={args.repo_filter}")
        sys.exit(1)

    # ── Judge setup ──
    judge = None
    if not args.skip_judge:
        s = Settings()
        provider = (args.judge_provider or s.judge_provider).lower()
        if provider == "ollama":
            judge = OllamaJudge(model=s.judge_model_local, base_url=s.ollama_base_url)
            print(f"Judge: Ollama local ({s.judge_model_local})")
        else:
            api_key = s.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                print("WARNING: GEMINI_API_KEY not set — running with --skip-judge")
                args.skip_judge = True
            else:
                judge = GeminiJudge(api_key=api_key, model=s.judge_model)
                print(f"Judge: Gemini API ({s.judge_model})")

    # ── Filter configs ──
    configs_to_run = CONFIGS
    if args.configs:
        configs_to_run = [c for c in CONFIGS if c["name"] in args.configs]
    print(f"\nRunning {len(configs_to_run)} configs × {len(qa_pairs)} questions")
    if not args.skip_judge:
        print(f"  × {args.judge_runs} judge runs per answer")
    total_qa_calls = len(configs_to_run) * len(qa_pairs)
    total_judge_calls = total_qa_calls * args.judge_runs if not args.skip_judge else 0
    print(f"  → {total_qa_calls} QA calls, {total_judge_calls} judge calls")

    # ── Metadata header ──
    settings = Settings()
    metadata = {
        "run_date": dt.datetime.now().isoformat(),
        "repo_filter": args.repo_filter,
        "repo_root": args.repo_root,
        "repo_git_sha": get_git_sha(args.repo_root),
        "n_questions": len(qa_pairs),
        "n_configs": len(configs_to_run),
        "judge_runs_per_question": args.judge_runs if not args.skip_judge else 0,
        "generator_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "judge_provider": (
            None if args.skip_judge
            else (args.judge_provider or settings.judge_provider).lower()
        ),
        "judge_model": (
            None if args.skip_judge
            else (settings.judge_model_local
                  if (args.judge_provider or settings.judge_provider).lower() == "ollama"
                  else settings.judge_model)
        ),
        "n_results": args.n_results,
    }

    # ── Output file ──
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = open(out_path, "w", encoding="utf-8")
    out_file.write(json.dumps({"type": "metadata", **metadata}) + "\n")

    # ── Main loop ──
    t_start = time.time()
    try:
        for config_idx, config in enumerate(configs_to_run, 1):
            print(f"\n{'='*70}")
            print(f"CONFIG {config_idx}/{len(configs_to_run)}: {config['name']}")
            print(f"{'='*70}")

            cfg_settings = build_settings(config["flags"])
            pipeline = QAPipeline(chunks=chunks, settings=cfg_settings)

            for q_idx, qa in enumerate(qa_pairs, 1):
                question = qa["question"]
                reference = qa["answer"]
                print(f"\n  [{q_idx}/{len(qa_pairs)}] {question[:70]}...")

                # ── Generate answer ──
                t_gen = time.time()
                try:
                    result = pipeline.ask(question, n_results=args.n_results)
                except Exception as exc:
                    print(f"    QA generation failed: {exc}")
                    out_file.write(json.dumps({
                        "type": "result",
                        "config": config["name"],
                        "qa_id": qa["id"],
                        "error": str(exc),
                    }) + "\n")
                    continue
                gen_latency = time.time() - t_gen

                # ── Deterministic metrics ──
                recall = retrieval_recall_at_k(reference, result.verified.used_chunk_ids and
                                               [c for c in pipeline.retriever._chunks_by_id.values()
                                                if c.id in result.retrieved_chunk_ids])
                # Recompute with actual chunks (clearer)
                retrieved_chunk_objs = [
                    pipeline.retriever._chunks_by_id[cid]
                    for cid in result.retrieved_chunk_ids
                    if cid in pipeline.retriever._chunks_by_id
                ]
                recall = retrieval_recall_at_k(reference, retrieved_chunk_objs)
                cite_acc = citation_accuracy(
                    result.verified.citations,
                    known_file_paths=known_paths,
                )

                # ── LLM-as-Judge ──
                judge_block = None
                if judge:
                    try:
                        aggregated = judge.score_n_times(
                            question=question,
                            reference_answer=reference,
                            candidate_answer=result.answer,
                            n=args.judge_runs,
                        )
                        judge_block = {
                            "accuracy_mean": aggregated.accuracy_mean,
                            "accuracy_std": aggregated.accuracy_std,
                            "completeness_mean": aggregated.completeness_mean,
                            "completeness_std": aggregated.completeness_std,
                            "relevance_mean": aggregated.relevance_mean,
                            "relevance_std": aggregated.relevance_std,
                            "clarity_mean": aggregated.clarity_mean,
                            "clarity_std": aggregated.clarity_std,
                            "n_runs": aggregated.n_runs,
                            "per_run_scores": [
                                {
                                    "accuracy": r.accuracy,
                                    "completeness": r.completeness,
                                    "relevance": r.relevance,
                                    "clarity": r.clarity,
                                    "reasoning": r.reasoning,
                                }
                                for r in aggregated.per_run
                            ],
                        }
                        print(
                            f"    judge: acc={aggregated.accuracy_mean:.1f}±{aggregated.accuracy_std:.1f}, "
                            f"comp={aggregated.completeness_mean:.1f}±{aggregated.completeness_std:.1f}, "
                            f"rel={aggregated.relevance_mean:.1f}, clar={aggregated.clarity_mean:.1f}"
                        )
                    except Exception as exc:
                        print(f"    judge failed: {exc}")

                # ── Record ──
                record = {
                    "type": "result",
                    "config": config["name"],
                    "qa_id": qa["id"],
                    "question": question,
                    "question_type_expected": qa["question_type"],
                    "question_type_predicted": result.question_type,
                    "classifier_correct":
                        result.question_type == qa["question_type"],
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "retrieval_recall": recall,
                    "citation_accuracy": cite_acc,
                    "n_citations": len(result.verified.citations),
                    "n_citations_verified": result.verified.verified_count,
                    "gen_latency_s": gen_latency,
                    "judge": judge_block,
                }
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()

                print(
                    f"    recall={recall:.2f}, cite_acc={cite_acc:.2f}, "
                    f"latency={gen_latency:.1f}s"
                )

    finally:
        out_file.close()

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Done in {elapsed/60:.1f} min. Results saved to: {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
