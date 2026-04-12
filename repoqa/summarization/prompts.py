"""LangChain prompt templates for hierarchical summarization.

All templates live here so they can be iterated independently of logic.
"""

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# File-level summary
# ---------------------------------------------------------------------------

FILE_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert software engineer. Summarize source code files clearly and concisely.",
        ),
        (
            "human",
            """\
File: {file_path}
Language: {language}

--- Source code (representative excerpts) ---
{code_excerpts}
--- End of excerpts ---

Write a concise summary (3–6 sentences) covering:
1. What this file does / its purpose.
2. Key classes, functions, or exports it provides.
3. Notable dependencies or patterns.
4. Any important side effects or configuration.

Be specific — mention actual names from the code.""",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Directory-level summary
# ---------------------------------------------------------------------------

DIR_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert software engineer. Summarize directories of source code clearly.",
        ),
        (
            "human",
            """\
Directory: {dir_path}

File summaries within this directory:
{file_summaries_bullet_list}

Write a concise summary (2–4 sentences) covering:
1. The overall purpose / responsibility of this directory.
2. How the files relate to each other.
3. How this directory fits into the broader project.

Be specific — mention actual file names and key symbols.""",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Project-level summary
# ---------------------------------------------------------------------------

PROJECT_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert software architect. Produce a comprehensive project overview.",
        ),
        (
            "human",
            """\
Repository: {repo_name}

README excerpt:
{readme_excerpt}

Directory summaries:
{dir_summaries_bullet_list}

Write a comprehensive project summary (1–2 paragraphs) covering:
1. What the project does and who it's for.
2. High-level architecture and key components.
3. Technology stack and notable design decisions.
4. Entry points and deployment approach (if discernible).

Be specific — mention actual directory and module names.""",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Symbol-level one-liner (used for function/class metadata enrichment)
# ---------------------------------------------------------------------------

SYMBOL_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a concise technical writer.",
        ),
        (
            "human",
            """\
Summarize the following {symbol_type} in one sentence (max 25 words).
Mention its name, what it does, and any key parameters or return values.

{code}""",
        ),
    ]
)
