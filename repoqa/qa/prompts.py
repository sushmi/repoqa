"""LangChain prompt templates for structured QA generation.

Each question type (architecture / logic / deployment) uses a specialized
prompt that matches the retrieval strategy it was paired with.
"""

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# System prompts per question type
# ---------------------------------------------------------------------------

_SYSTEM_ARCHITECTURE = (
    "You are an expert software architect explaining how a codebase is "
    "organized. Answer based ONLY on the provided context. Cite file paths "
    "and line ranges in square brackets, e.g. [src/app.py:10-42]. "
    "If the context does not contain enough information, say so explicitly."
)

_SYSTEM_LOGIC = (
    "You are an expert software engineer explaining how specific code works. "
    "Answer based ONLY on the provided context. Quote function/class names "
    "exactly as they appear in the code. Cite the file path and line range "
    "for every claim using [path:start-end] format. "
    "If the context does not contain enough information, say so explicitly."
)

_SYSTEM_DEPLOYMENT = (
    "You are a deployment and infrastructure expert. Answer based ONLY on "
    "the provided context from Dockerfiles, CI/CD configs, and documentation. "
    "Cite file paths using [path] format. If the context does not contain "
    "enough information to answer, say so explicitly."
)

_SYSTEM_BY_TYPE = {
    "architecture": _SYSTEM_ARCHITECTURE,
    "logic": _SYSTEM_LOGIC,
    "deployment": _SYSTEM_DEPLOYMENT,
}


# ---------------------------------------------------------------------------
# Human prompt — shared template with type-specific guidance injected
# ---------------------------------------------------------------------------

_HUMAN_TEMPLATE = """\
Question: {question}

--- Retrieved context ({num_chunks} chunks) ---
{context}
--- End of context ---

Instructions:
{type_guidance}

Write your answer in this exact structure:

ANSWER:
<your answer, 3-8 sentences, with inline citations>

CITATIONS:
- <file_path>:<start_line>-<end_line>
- <file_path>:<start_line>-<end_line>
...

CONFIDENCE: <high | medium | low>
"""


_TYPE_GUIDANCE = {
    "architecture": (
        "- Focus on how components interact and where boundaries lie.\n"
        "- Name the key classes, modules, or directories involved.\n"
        "- Prefer high-level explanation over line-by-line detail."
    ),
    "logic": (
        "- Focus on what specific functions/methods do, step by step.\n"
        "- Quote function signatures and key lines from the code.\n"
        "- Explain any non-obvious control flow or edge cases."
    ),
    "deployment": (
        "- Focus on configuration, environment, and runtime setup.\n"
        "- Reference specific Dockerfile directives, CI steps, or env vars.\n"
        "- Call out any prerequisites or ordering requirements."
    ),
}


def build_prompt(question_type: str) -> ChatPromptTemplate:
    """Build a ChatPromptTemplate specialized for the given question type."""
    system = _SYSTEM_BY_TYPE.get(question_type, _SYSTEM_LOGIC)
    guidance = _TYPE_GUIDANCE.get(question_type, _TYPE_GUIDANCE["logic"])

    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", _HUMAN_TEMPLATE),
        ]
    ).partial(type_guidance=guidance)
