"""Answer generator: takes a RetrievalResult and produces a verified answer.

Ties together:
  - context_builder (format chunks into prompt-ready text)
  - prompts (question-type-specific templates)
  - LLM invocation (Qwen via Ollama)
  - citation_verifier (ensure citations ground in retrieved chunks)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from repoqa.models import Chunk
from repoqa.qa.context_builder import build_context
from repoqa.qa.prompts import build_prompt
from repoqa.qa.citation_verifier import verify_citations, VerifiedAnswer

logger = logging.getLogger(__name__)

# Parse "ANSWER: ..." / "CITATIONS: ..." / "CONFIDENCE: ..." blocks
_ANSWER_RE = re.compile(r"ANSWER:\s*(.*?)(?=CITATIONS:|CONFIDENCE:|$)", re.DOTALL | re.IGNORECASE)
_CITATIONS_RE = re.compile(r"CITATIONS:\s*(.*?)(?=CONFIDENCE:|$)", re.DOTALL | re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*(\w+)", re.IGNORECASE)


@dataclass
class QAResult:
    """End-to-end QA output for one question."""
    question: str
    question_type: str
    answer: str
    confidence: str
    raw_llm_output: str
    verified: VerifiedAnswer
    retrieved_chunk_ids: list[str]
    used_chunk_ids: list[str]


class AnswerGenerator:
    """Generates an answer from a question + retrieved chunks."""

    def __init__(self, llm, max_context_tokens: int = 4000) -> None:
        """
        Args:
            llm: A LangChain BaseChatModel (e.g. ChatOllama with Qwen 2.5).
            max_context_tokens: Token budget for retrieved context.
        """
        self.llm = llm
        self.max_context_tokens = max_context_tokens

    def generate(
        self,
        question: str,
        question_type: str,
        retrieved_chunks: list[Chunk],
    ) -> QAResult:
        """Generate an answer for *question* using *retrieved_chunks*."""
        context, included_chunks = build_context(
            retrieved_chunks, max_tokens=self.max_context_tokens
        )

        prompt = build_prompt(question_type)
        raw_output = self._invoke_with_retry(
            prompt=prompt,
            question=question,
            context=context,
            num_chunks=len(included_chunks),
        )

        answer_text, confidence = self._parse_structured_output(raw_output)
        verified = verify_citations(
            answer_text=answer_text,
            retrieved_chunks=included_chunks,
            confidence=confidence,
        )

        return QAResult(
            question=question,
            question_type=question_type,
            answer=answer_text,
            confidence=confidence,
            raw_llm_output=raw_output,
            verified=verified,
            retrieved_chunk_ids=[c.id for c in retrieved_chunks],
            used_chunk_ids=verified.used_chunk_ids,
        )

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
    )
    def _invoke_with_retry(self, prompt, question, context, num_chunks) -> str:
        chain = prompt | self.llm
        response = chain.invoke(
            {
                "question": question,
                "context": context,
                "num_chunks": num_chunks,
            }
        )
        return response.content if hasattr(response, "content") else str(response)

    @staticmethod
    def _parse_structured_output(raw: str) -> tuple[str, str]:
        """Pull ANSWER / CITATIONS / CONFIDENCE blocks out of the LLM output.

        If the model ignores the format, we fall back to returning the
        whole output as the answer with medium confidence.
        """
        answer_match = _ANSWER_RE.search(raw)
        conf_match = _CONFIDENCE_RE.search(raw)

        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            logger.debug("LLM did not use structured format, using raw output")
            answer = raw.strip()

        confidence = "medium"
        if conf_match:
            raw_conf = conf_match.group(1).strip().lower()
            if raw_conf in ("high", "medium", "low"):
                confidence = raw_conf

        return answer, confidence
