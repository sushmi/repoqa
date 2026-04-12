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
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    chat_model: str = Field(default="gpt-4o-mini", alias="CHAT_MODEL")

    # Embeddings
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")

    # ChromaDB
    chroma_persist_dir: str = Field(default="./data/chroma", alias="CHROMA_PERSIST_DIR")

    # Chunking
    chunk_max_tokens: int = Field(default=512, alias="CHUNK_MAX_TOKENS")
    chunk_overlap_tokens: int = Field(default=64, alias="CHUNK_OVERLAP_TOKENS")

    # Summarization
    summary_max_tokens: int = Field(default=4096, alias="SUMMARY_MAX_TOKENS")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
