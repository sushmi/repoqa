"""Stage 3 QA pipeline: retrieves context and generates a verified answer.

Wires together Stage 2 (HybridRetriever) and Stage 3 (AnswerGenerator):

    question
        │
        ├── HybridRetriever.query()          (Stage 2)
        │     → classify, expand, retrieve, fuse
        │
        ├── AnswerGenerator.generate()        (Stage 3)
        │     → build context, prompt, invoke LLM,
        │       parse structured output, verify citations
        │
        └── QAResult (answer + citations + confidence)
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import asdict
from pathlib import Path

from config.settings import Settings, get_settings
from repoqa.embedding.chroma_store import ChromaStore
from repoqa.embedding.embedder import Embedder
from repoqa.ingestion.pipeline import IngestionPipeline
from repoqa.models import Chunk
from repoqa.qa.answer_generator import AnswerGenerator, QAResult
from repoqa.retrieval.hybrid_retriever import HybridRetriever
from repoqa.summarization.pipeline import _build_llm

logger = logging.getLogger(__name__)


class QAPipeline:
    """Answers developer questions about a repository end-to-end."""

    def __init__(
        self,
        chunks: list[Chunk],
        settings: Settings | None = None,
        max_context_tokens: int = 4000,
    ) -> None:
        self.settings = settings or get_settings()
        llm = _build_llm(self.settings)
        store = ChromaStore(persist_dir=self.settings.chroma_persist_dir)
        embedder = Embedder(
            provider=self.settings.embedding_provider,
            model=self.settings.embedding_model,
            ollama_base_url=self.settings.ollama_base_url,
        )

        self.retriever = HybridRetriever(
            llm=llm, store=store, embedder=embedder, chunks=chunks,
        )
        self.generator = AnswerGenerator(
            llm=llm, max_context_tokens=max_context_tokens,
        )

    def ask(self, question: str, n_results: int = 10) -> QAResult:
        """Answer a single question end-to-end."""
        retrieval = self.retriever.query(question, n_results=n_results)
        result = self.generator.generate(
            question=question,
            question_type=retrieval.question_type,
            retrieved_chunks=retrieval.chunks,
        )
        logger.info(
            "QA: type=%s, confidence=%s, citations=%d/%d verified",
            result.question_type,
            result.confidence,
            result.verified.verified_count,
            len(result.verified.citations),
        )
        return result

    def ask_batch(self, questions: list[str], n_results: int = 10) -> list[QAResult]:
        """Answer a list of questions."""
        return [self.ask(q, n_results=n_results) for q in questions]

    @classmethod
    def from_chunks_file(
        cls,
        chunks_path: str,
        settings: Settings | None = None,
        max_context_tokens: int = 4000,
    ) -> QAPipeline:
        """Convenience constructor that loads chunks from a JSON file."""
        chunks = IngestionPipeline.load_chunks(chunks_path)
        return cls(chunks=chunks, settings=settings,
                   max_context_tokens=max_context_tokens)

    @staticmethod
    def save_results(results: list[QAResult], output_path: str) -> None:
        """Persist QA results to JSON."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        serializable = [_qa_to_dict(r) for r in results]
        with open(out, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)


def _qa_to_dict(result: QAResult) -> dict:
    """Serialize a QAResult (with nested dataclasses) to JSON-safe dict."""
    data = asdict(result)
    # asdict handles nested dataclasses (VerifiedAnswer, Citation) automatically
    return data
