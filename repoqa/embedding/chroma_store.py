"""ChromaDB vector store management."""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb

from repoqa.models import Chunk, Summary
from repoqa.embedding.metadata import chunk_to_metadata, summary_to_metadata

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "repoqa"


class ChromaStore:
    """Manages a single ChromaDB collection for all chunks and summaries.

    Uses pre-computed embeddings (passed explicitly) so we have full control
    over batching, retries, and cost.
    """

    def __init__(self, persist_dir: str, collection_name: str = _COLLECTION_NAME) -> None:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        # Collection with no built-in embedding function — we supply embeddings ourselves
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaStore ready: collection=%s, existing docs=%d",
            collection_name,
            self._collection.count(),
        )

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        repo_name: str,
    ) -> None:
        """Upsert code/prose chunks into the collection."""
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.content for c in chunks],
            metadatas=[chunk_to_metadata(c, repo_name) for c in chunks],
        )
        logger.info("Upserted %d chunks for repo=%s", len(chunks), repo_name)

    def upsert_summaries(
        self,
        summaries: list[Summary],
        embeddings: list[list[float]],
        repo_name: str,
    ) -> None:
        """Upsert hierarchical summaries into the collection."""
        if not summaries:
            return
        self._collection.upsert(
            ids=[s.id for s in summaries],
            embeddings=embeddings,
            documents=[s.content for s in summaries],
            metadatas=[summary_to_metadata(s, repo_name) for s in summaries],
        )
        logger.info("Upserted %d summaries for repo=%s", len(summaries), repo_name)

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
    ) -> dict:
        """Semantic similarity query. Returns ChromaDB QueryResult dict."""
        kwargs: dict = dict(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where
        return self._collection.query(**kwargs)

    def get_collection_stats(self) -> dict:
        count = self._collection.count()
        return {"total_documents": count, "collection_name": self._collection.name}

    def delete_repo(self, repo_name: str) -> None:
        """Remove all documents belonging to *repo_name*."""
        result = self._collection.get(where={"repo_name": repo_name}, include=[])
        ids_to_delete = result.get("ids", [])
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
            logger.info("Deleted %d documents for repo=%s", len(ids_to_delete), repo_name)
        else:
            logger.info("No documents found for repo=%s", repo_name)
