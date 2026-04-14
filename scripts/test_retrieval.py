"""Smoke test for Stage 2: hybrid retrieval with question-type routing.

Usage:
    uv pip install rank-bm25
    uv run scripts/test_retrieval.py --chunks ./data/flask/transformers_chunks.json --repo-name flask
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from repoqa.embedding.embedder import Embedder
from repoqa.embedding.chroma_store import ChromaStore
from repoqa.ingestion.pipeline import IngestionPipeline
from repoqa.retrieval.hybrid_retriever import HybridRetriever
from repoqa.summarization.pipeline import _build_llm


def main() -> None:
    parser = argparse.ArgumentParser(description="Test hybrid retrieval.")
    parser.add_argument("--chunks", required=True, help="Path to chunks JSON.")
    parser.add_argument("--repo-name", default="flask", help="Repo name in ChromaDB.")
    args = parser.parse_args()

    settings = get_settings()
    chunks = IngestionPipeline.load_chunks(args.chunks)
    print(f"Loaded {len(chunks)} chunks")

    llm = _build_llm(settings)
    store = ChromaStore(persist_dir=settings.chroma_persist_dir)
    embedder = Embedder(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        ollama_base_url=settings.ollama_base_url,
    )

    retriever = HybridRetriever(llm=llm, store=store, embedder=embedder, chunks=chunks)

    # Test with one question per type
    test_questions = [
        "How does Flask handle routing and URL dispatching?",
        "What does the make_response function do?",
        "How is Docker configured for this project?",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        result = retriever.query(question, n_results=5)
        print(f"Type: {result.question_type}")
        print(f"Expanded: {result.expanded_query[:120]}...")
        print(f"Semantic hits: {len(result.semantic_ids)}, BM25 hits: {len(result.bm25_ids)}")
        print(f"Top {len(result.chunks)} fused results:")
        for i, chunk in enumerate(result.chunks, 1):
            print(f"  {i}. [{chunk.chunk_type}] {chunk.file_record.repo_path}:{chunk.start_line} "
                  f"({chunk.symbol_name or 'n/a'})")

    print(f"\n{'='*60}")
    print("Stage 2 retrieval test complete!")


if __name__ == "__main__":
    main()
