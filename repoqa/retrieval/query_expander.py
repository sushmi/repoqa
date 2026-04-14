"""Query expansion: use the LLM to generate additional search terms.

Example: "How does auth work?" → "authentication login token session middleware"
This helps both BM25 (more keywords to match) and semantic search (richer query).
"""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

EXPAND_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a search query expander for code repositories. "
            "Given a developer question, output 5-10 additional search terms "
            "(function names, class names, keywords, synonyms) that would help "
            "find relevant code. Output ONLY the terms, space-separated, no explanation.",
        ),
        (
            "human",
            "{question}",
        ),
    ]
)


class QueryExpander:
    """Expands a developer question with additional search terms."""

    def __init__(self, llm) -> None:
        self._chain = EXPAND_PROMPT | llm

    def expand(self, question: str) -> str:
        """Return the original question + expanded terms."""
        response = self._chain.invoke({"question": question})
        raw = response.content if hasattr(response, "content") else str(response)
        expanded_terms = raw.strip()
        logger.debug("Query expansion: '%s' → '%s'", question, expanded_terms)
        return f"{question} {expanded_terms}"
