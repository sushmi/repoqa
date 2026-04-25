"""Classify developer questions into architecture / logic / deployment.

Uses a lightweight LLM prompt to categorize the question type,
which determines the retrieval strategy.
"""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate

from repoqa.tokenizer import TOKEN_COUNTER

logger = logging.getLogger(__name__)

QUESTION_TYPES = ("architecture", "logic", "deployment")

CLASSIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a question classifier for code repositories. "
            "Classify the developer question into exactly ONE category. "
            "Reply with a single word: architecture, logic, or deployment.",
        ),
        (
            "human",
            """\
Categories:
- architecture: project structure, module relationships, design patterns, how components connect
  Examples: "How does authentication work?", "What is the project structure?", "How do modules interact?"
- logic: specific function behavior, algorithms, data transformations, bug investigation
  Examples: "What does validate_token() do?", "How is the tax calculated?", "Why does this return null?"
- deployment: configuration, build process, environment setup, Docker, CI/CD, infrastructure
  Examples: "How is Docker configured?", "How do I run this locally?", "What environment variables are needed?"

Question: {question}

Category:""",
        ),
    ]
)


class QuestionClassifier:
    """Classifies a developer question into architecture / logic / deployment."""

    def __init__(self, llm) -> None:
        self._llm = llm

    def classify(self, question: str) -> str:
        """Return one of: 'architecture', 'logic', 'deployment'."""
        messages = CLASSIFY_PROMPT.format_messages(question=question)
        response = self._llm.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)
        TOKEN_COUNTER.log_llm_call("question_classifier", messages, raw)
        label = raw.strip().lower().split()[0].rstrip(".,;:")

        if label not in QUESTION_TYPES:
            logger.warning("Classifier returned unknown type '%s', defaulting to logic", label)
            return "logic"
        return label
