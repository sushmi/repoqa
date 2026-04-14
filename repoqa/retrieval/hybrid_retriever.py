"""Hybrid retriever with question-type routing.

Implements the Stage 2 pipeline from the RepoQA paper (Section 3.2):

    Developer question
        │
        ├── Question classifier → architecture / logic / deployment
        │
        ├── Query expander → enriched query with extra search terms
        │
        ├── Strategy router → retrieval config per question type
        │
        ├── Semantic search (ChromaDB cosine similarity)
        │
        ├── BM25 keyword search (rank_bm25)
        │
        └── RRF fusion → merged & re-ranked results
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from repoqa.models import Chunk, Summary
from repoqa.embedding.embedder import Embedder
from repoqa.embedding.chroma_store import ChromaStore
from repoqa.retrieval.bm25_retriever import BM25Retriever
from repoqa.retrieval.question_classifier import QuestionClassifier
from repoqa.retrieval.query_expander import QueryExpander
from repoqa.retrieval.strategy_router import RetrievalStrategy, get_strategy
from repoqa.retrieval.rrf_fusion import rrf_fuse

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result of a hybrid retrieval query."""

    question: str
    question_type: str
    expanded_query: str
    chunks: list[Chunk]
    semantic_ids: list[str]
    bm25_ids: list[str]
    fused_ids: list[str]


class HybridRetriever:
    """Orchestrates question classification, query expansion, multi-signal
    retrieval, and RRF fusion per the RepoQA paper architecture."""

    def __init__(
        self,
        llm,
        store: ChromaStore,
        embedder: Embedder,
        chunks: list[Chunk],
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._classifier = QuestionClassifier(llm)
        self._expander = QueryExpander(llm)
        self._bm25 = BM25Retriever(chunks)
        self._chunks_by_id = {c.id: c for c in chunks}

    def query(self, question: str, n_results: int = 10) -> RetrievalResult:
        """Full Stage 2 pipeline: classify → expand → retrieve → fuse."""

        # Step 1: Classify question type
        question_type = self._classifier.classify(question)
        logger.info("Question classified as: %s", question_type)

        # Step 2: Get retrieval strategy for this question type
        strategy = get_strategy(question_type)

        # Step 3: Expand query with additional search terms
        expanded_query = self._expander.expand(question)
        logger.info("Expanded query: %s", expanded_query[:200])

        # Step 4: Semantic search (ChromaDB)
        semantic_ids = self._semantic_search(expanded_query, strategy)

        # Step 5: BM25 keyword search
        bm25_ids = self._bm25_search(expanded_query, strategy)

        # Step 6: RRF fusion with strategy weights
        fused_ids = rrf_fuse(
            [
                (semantic_ids, strategy.semantic_weight),
                (bm25_ids, strategy.bm25_weight),
            ]
        )

        # Step 7: Resolve IDs to Chunk objects (top n_results)
        result_chunks = []
        for doc_id in fused_ids[:n_results]:
            if doc_id in self._chunks_by_id:
                result_chunks.append(self._chunks_by_id[doc_id])

        logger.info(
            "Retrieval: type=%s, semantic=%d, bm25=%d, fused=%d, returned=%d",
            question_type,
            len(semantic_ids),
            len(bm25_ids),
            len(fused_ids),
            len(result_chunks),
        )

        return RetrievalResult(
            question=question,
            question_type=question_type,
            expanded_query=expanded_query,
            chunks=result_chunks,
            semantic_ids=semantic_ids,
            bm25_ids=bm25_ids,
            fused_ids=fused_ids[:n_results],
        )

    def _semantic_search(self, query: str, strategy: RetrievalStrategy) -> list[str]:
        """Embed the query and search ChromaDB."""
        query_vector = self._embedder.embed_single(query)
        results = self._store.query(
            query_embedding=query_vector,
            n_results=strategy.semantic_k,
            where=strategy.semantic_filters,
        )
        return results["ids"][0] if results["ids"] else []

    def _bm25_search(self, query: str, strategy: RetrievalStrategy) -> list[str]:
        """Run BM25 keyword search over chunks."""
        results = self._bm25.query(query, n_results=strategy.bm25_k)
        return [chunk.id for chunk, _score in results]
