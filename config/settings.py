"""Application configuration via pydantic-settings.

All values can be overridden by environment variables or a .env file.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")
    chat_model: str = Field(default="qwen2.5:7b", alias="CHAT_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    #llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    #chat_model: str = Field(default="gpt-4o-mini", alias="CHAT_MODEL")

    # Embeddings
    embedding_provider: str = Field(default="ollama", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="nomic-embed-text", alias="EMBEDDING_MODEL")

    # ChromaDB
    chroma_persist_dir: str = Field(default="./data/chroma", alias="CHROMA_PERSIST_DIR")

    # Chunking
    chunk_max_tokens: int = Field(default=512, alias="CHUNK_MAX_TOKENS")
    chunk_overlap_tokens: int = Field(default=64, alias="CHUNK_OVERLAP_TOKENS")

    # Summarization
    summary_max_tokens: int = Field(default=4096, alias="SUMMARY_MAX_TOKENS")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # ── Ablation switches (experiment mode) ──
    # All default True = full RepoQA. Flip individually via .env to run ablations.
    enable_summaries: bool = Field(default=True, alias="ENABLE_SUMMARIES")
    enable_hybrid: bool = Field(default=True, alias="ENABLE_HYBRID")
    enable_query_expansion: bool = Field(default=True, alias="ENABLE_QUERY_EXPANSION")
    enable_routing: bool = Field(default=True, alias="ENABLE_ROUTING")
    enable_retrieval: bool = Field(default=True, alias="ENABLE_RETRIEVAL")

    # ── Evaluation / judge ──
    # judge_provider: "gemini" (API, capped at 20 req/day on free tier) or
    # "ollama" (local, unlimited but slower & weaker). Use ollama when the
    # Gemini quota is exhausted.
    judge_provider: str = Field(default="gemini", alias="JUDGE_PROVIDER")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    judge_model: str = Field(default="gemini-2.5-flash", alias="JUDGE_MODEL")
    judge_model_local: str = Field(default="llama3.1:8b", alias="JUDGE_MODEL_LOCAL")
    judge_runs_per_question: int = Field(default=3, alias="JUDGE_RUNS_PER_QUESTION")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
