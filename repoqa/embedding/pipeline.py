"""Embedding pipeline: embed all chunks and summaries, store in ChromaDB."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

from rich.console import Console

from config.settings import Settings, get_settings
from repoqa.models import Chunk, FileRecord, Summary
from repoqa.embedding.embedder import Embedder
from repoqa.embedding.chroma_store import ChromaStore

logger = logging.getLogger(__name__)
console = Console()


class EmbeddingPipeline:
    """Orchestrates embedding of chunks + summaries and storage in ChromaDB."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._embedder = Embedder(
            model=self.settings.embedding_model,
            api_key=self.settings.openai_api_key,
        )
        self._store = ChromaStore(persist_dir=self.settings.chroma_persist_dir)

    def run(
        self,
        chunks: list[Chunk],
        summaries: list[Summary],
        repo_name: str,
    ) -> None:
        """Embed *chunks* and *summaries*, then upsert into ChromaDB."""
        console.print(f"[bold cyan]Embedding {len(chunks)} chunks + {len(summaries)} summaries for repo=[bold]{repo_name}[/bold][/bold cyan]")

        # Combine all texts so we make the minimum number of API calls
        chunk_texts = [c.content for c in chunks]
        summary_texts = [s.content for s in summaries]
        all_texts = chunk_texts + summary_texts

        all_embeddings = self._embedder.embed_batch(all_texts)

        # Split back
        chunk_embeddings = all_embeddings[: len(chunks)]
        summary_embeddings = all_embeddings[len(chunks):]

        self._store.upsert_chunks(chunks, chunk_embeddings, repo_name)
        self._store.upsert_summaries(summaries, summary_embeddings, repo_name)

        stats = self._store.get_collection_stats()
        console.print(f"[bold green]ChromaDB total documents: {stats['total_documents']}[/bold green]")

    def run_from_cache(
        self,
        chunks_json: str,
        summaries_json: str,
        repo_name: str,
    ) -> None:
        """Load chunks and summaries from JSON files (written by upstream pipelines)."""
        chunks = _load_chunks(chunks_json)
        summaries = _load_summaries(summaries_json)
        self.run(chunks, summaries, repo_name)

    @property
    def store(self) -> ChromaStore:
        return self._store

    @property
    def embedder(self) -> Embedder:
        return self._embedder


# ------------------------------------------------------------------
# JSON deserialization helpers (mirror the save logic in ingestion/summarization pipelines)
# ------------------------------------------------------------------

def _load_chunks(json_path: str) -> list[Chunk]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    chunks: list[Chunk] = []
    for item in raw:
        fr_data = item.pop("file_record")
        fr = FileRecord(**fr_data)
        chunks.append(Chunk(file_record=fr, **item))
    return chunks


def _load_summaries(json_path: str) -> list[Summary]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Summary(**item) for item in raw]
