"""Evaluate Stage 2 hybrid retrieval without needing Stage 3.

Runs Flask questions from the seed dataset through the retrieval pipeline
and measures how well the system finds relevant chunks.

Evaluates three things:
  1. Question classifier accuracy (does it pick the right type?)
  2. Retrieval quality (do the returned chunks contain relevant content?)
  3. Component comparison (semantic-only vs BM25-only vs hybrid)

Usage:
    uv run scripts/eval_retrieval.py \
        --chunks ./data/flask/transformers_chunks.json \
        --repo-name flask
"""

import argparse
import json
import logging
import sys
import re
import time
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from repoqa.embedding.embedder import Embedder
from repoqa.embedding.chroma_store import ChromaStore
from repoqa.ingestion.pipeline import IngestionPipeline
from repoqa.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult
from repoqa.retrieval.bm25_retriever import BM25Retriever
from repoqa.retrieval.question_classifier import QuestionClassifier
from repoqa.retrieval.rrf_fusion import rrf_fuse
from repoqa.summarization.pipeline import _build_llm

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ── Evaluation metrics ────────────────────────────────────────────

@dataclass
class RetrievalMetrics:
    """Metrics for a single query. 
    A container to store per-question results so we can aggregate at the end. 
    """
    question: str
    question_type_expected: str
    question_type_predicted: str
    classifier_correct: bool
    keyword_hits: int       # how many answer keywords found in retrieved chunks
    total_keywords: int     # total answer keywords checked
    keyword_recall: float   # keyword_hits / total_keywords
    top_files: list[str]    # file paths of top retrieved chunks
    latency_ms: float


def extract_keywords(answer: str) -> list[str]:
    """Extract important technical keywords from a reference answer.

    Pulls out: function/class names (word_with_underscores),
    backtick-quoted terms, and CamelCase identifiers.
    """
    keywords = set()

    # Backtick-quoted terms: `flask.g`, `@app.route`
    for match in re.findall(r"`([^`]+)`", answer):
        cleaned = match.strip("@().,:;")
        if len(cleaned) >= 2:
            keywords.add(cleaned.lower())

    # snake_case identifiers: url_map, get_db, before_request
    for match in re.findall(r"\b[a-z_][a-z0-9_]{2,}\b", answer):
        # Skip common English words
        if match not in {"the", "and", "for", "that", "this", "with", "from",
                         "which", "when", "each", "into", "also", "using",
                         "their", "than", "some", "like", "used", "need",
                         "have", "not", "its", "your", "can", "set",
                         "use", "you", "are", "has", "but", "all",
                         "was", "one", "any", "how", "does", "own"}:
            keywords.add(match)

    # CamelCase: MapAdapter, RequestContext, ValidationError
    for match in re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", answer):
        keywords.add(match.lower())

    return list(keywords)


def compute_keyword_recall(keywords: list[str], chunks_text: str) -> tuple[int, int]:
    """Count how many answer keywords appear in the retrieved chunks."""
    chunks_lower = chunks_text.lower()
    hits = sum(1 for kw in keywords if kw in chunks_lower)
    return hits, len(keywords)


# ── Individual component evaluation ──────────────────────────────

def eval_semantic_only(query: str, embedder: Embedder, store: ChromaStore,
                       n_results: int = 10) -> list[str]:
    """Run semantic search alone, return chunk IDs."""
    vec = embedder.embed_single(query)
    results = store.query(query_embedding=vec, n_results=n_results)
    return results["ids"][0] if results["ids"] else []


def eval_bm25_only(query: str, bm25: BM25Retriever,
                   n_results: int = 10) -> list[str]:
    """Run BM25 search alone, return chunk IDs."""
    results = bm25.query(query, n_results=n_results)
    return [chunk.id for chunk, _score in results]


# ── Main evaluation ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 retrieval.")
    parser.add_argument("--chunks", required=True, help="Path to chunks JSON.")
    parser.add_argument("--repo-name", default="flask")
    parser.add_argument("--dataset", default="data/evaluation/seed_dataset.json")
    parser.add_argument("--n-results", type=int, default=10)
    args = parser.parse_args()

    settings = get_settings()

    # ── Load data ──
    print("Loading chunks...")
    chunks = IngestionPipeline.load_chunks(args.chunks)
    chunks_by_id = {c.id: c for c in chunks}
    print(f"  {len(chunks)} chunks loaded")

    with open(args.dataset) as f:
        dataset = json.load(f)

    # Filter to questions for the target repo
    flask_questions = [
        qa for qa in dataset["qa_pairs"]
        if qa["repo_name"] == "pallets/flask"
    ]
    print(f"  {len(flask_questions)} Flask evaluation questions")

    if not flask_questions:
        print("ERROR: No Flask questions found in dataset.")
        sys.exit(1)

    # ── Initialize components ──
    print("\nInitializing retrieval components...")
    llm = _build_llm(settings)
    store = ChromaStore(persist_dir=settings.chroma_persist_dir)
    embedder = Embedder(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        ollama_base_url=settings.ollama_base_url,
    )

    retriever = HybridRetriever(llm=llm, store=store, embedder=embedder, chunks=chunks)
    bm25 = BM25Retriever(chunks)
    classifier = QuestionClassifier(llm)

    stats = store.get_collection_stats()
    print(f"  ChromaDB: {stats['total_documents']} documents")
    print(f"  BM25: {len(chunks)} chunks indexed")

    # ── Run evaluation ──
    print(f"\n{'='*70}")
    print("EVALUATION: Stage 2 Hybrid Retrieval")
    print(f"{'='*70}\n")

    all_metrics: list[RetrievalMetrics] = []
    semantic_keyword_recalls = []
    bm25_keyword_recalls = []
    hybrid_keyword_recalls = []

    for i, qa in enumerate(flask_questions, 1):
        question = qa["question"]
        expected_type = qa["question_type"]
        answer = qa["answer"]

        print(f"\n--- Q{i}: [{expected_type}] {question[:80]}...")

        keywords = extract_keywords(answer)

        # ── 1. Hybrid retrieval (full pipeline) ──
        t0 = time.time()
        result = retriever.query(question, n_results=args.n_results)
        latency = (time.time() - t0) * 1000

        hybrid_text = "\n".join(c.content for c in result.chunks)
        hybrid_hits, total_kw = compute_keyword_recall(keywords, hybrid_text)
        hybrid_recall = hybrid_hits / max(total_kw, 1)
        hybrid_keyword_recalls.append(hybrid_recall)

        # ── 2. Semantic-only ──
        sem_ids = eval_semantic_only(question, embedder, store, args.n_results)
        sem_text = "\n".join(chunks_by_id[cid].content for cid in sem_ids if cid in chunks_by_id)
        sem_hits, _ = compute_keyword_recall(keywords, sem_text)
        sem_recall = sem_hits / max(total_kw, 1)
        semantic_keyword_recalls.append(sem_recall)

        # ── 3. BM25-only ──
        bm25_ids = eval_bm25_only(question, bm25, args.n_results)
        bm25_text = "\n".join(chunks_by_id[cid].content for cid in bm25_ids if cid in chunks_by_id)
        bm25_hits, _ = compute_keyword_recall(keywords, bm25_text)
        bm25_recall_val = bm25_hits / max(total_kw, 1)
        bm25_keyword_recalls.append(bm25_recall_val)

        # ── Classifier accuracy ──
        classifier_correct = result.question_type == expected_type

        top_files = list(dict.fromkeys(
            c.file_record.repo_path for c in result.chunks
        ))[:5]

        metrics = RetrievalMetrics(
            question=question[:80],
            question_type_expected=expected_type,
            question_type_predicted=result.question_type,
            classifier_correct=classifier_correct,
            keyword_hits=hybrid_hits,
            total_keywords=total_kw,
            keyword_recall=hybrid_recall,
            top_files=top_files,
            latency_ms=latency,
        )
        all_metrics.append(metrics)

        # ── Print per-question results ──
        type_mark = "OK" if classifier_correct else "WRONG"
        print(f"    Classifier: {result.question_type} ({type_mark})")
        print(f"    Keywords:  {hybrid_hits}/{total_kw} = {hybrid_recall:.0%} "
              f"(sem={sem_recall:.0%}, bm25={bm25_recall_val:.0%})")
        print(f"    Latency:   {latency:.0f}ms")
        print(f"    Top files: {', '.join(top_files[:3])}")

    # ── Summary report ──
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    total = len(all_metrics)
    classifier_acc = sum(1 for m in all_metrics if m.classifier_correct) / total

    avg_hybrid = sum(hybrid_keyword_recalls) / total
    avg_semantic = sum(semantic_keyword_recalls) / total
    avg_bm25 = sum(bm25_keyword_recalls) / total
    avg_latency = sum(m.latency_ms for m in all_metrics) / total

    print(f"\n  Questions evaluated:    {total}")
    print(f"\n  Classifier accuracy:    {classifier_acc:.0%} ({sum(1 for m in all_metrics if m.classifier_correct)}/{total})")

    # Per-type classifier breakdown
    for qtype in ("architecture", "logic", "deployment"):
        type_qs = [m for m in all_metrics if m.question_type_expected == qtype]
        if type_qs:
            correct = sum(1 for m in type_qs if m.classifier_correct)
            print(f"    {qtype:15s}: {correct}/{len(type_qs)}")

    print(f"\n  Keyword recall (avg):   higher = better, finds more relevant content")
    print(f"    Hybrid (full):    {avg_hybrid:.1%}")
    print(f"    Semantic-only:    {avg_semantic:.1%}")
    print(f"    BM25-only:        {avg_bm25:.1%}")

    improvement = avg_hybrid - max(avg_semantic, avg_bm25)
    best_single = "semantic" if avg_semantic >= avg_bm25 else "BM25"
    print(f"\n  Hybrid vs best single ({best_single}): {improvement:+.1%}")

    print(f"\n  Avg latency:           {avg_latency:.0f}ms per query")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    if avg_hybrid > max(avg_semantic, avg_bm25):
        print("  GOOD: Hybrid retrieval outperforms both individual methods.")
        print("        RRF fusion is adding value.")
    elif avg_hybrid == max(avg_semantic, avg_bm25):
        print("  NEUTRAL: Hybrid matches best single method.")
        print("           Try adjusting strategy weights in strategy_router.py")
    else:
        print("  WARNING: Hybrid is WORSE than a single method.")
        print("           Check RRF weights and query expansion quality.")

    if classifier_acc >= 0.8:
        print(f"  GOOD: Classifier accuracy {classifier_acc:.0%} is solid.")
    else:
        print(f"  WARNING: Classifier accuracy {classifier_acc:.0%} is low.")
        print("           Improve the classifier prompt in question_classifier.py")

    if avg_hybrid >= 0.4:
        print(f"  GOOD: Keyword recall {avg_hybrid:.0%} — retrieval finds relevant content.")
    elif avg_hybrid >= 0.2:
        print(f"  OK: Keyword recall {avg_hybrid:.0%} — decent but room for improvement.")
    else:
        print(f"  WARNING: Keyword recall {avg_hybrid:.0%} — retrieval is struggling.")
        print("           Check: are embeddings indexed? Is ChromaDB populated?")

    print(f"\n  Ready for Stage 3?")
    if avg_hybrid >= 0.2 and classifier_acc >= 0.6:
        print("  YES — retrieval quality is sufficient to build QA generation on top.")
    else:
        print("  NOT YET — fix retrieval first, or Stage 3 answers will be poor.")

    print()


if __name__ == "__main__":
    main()
