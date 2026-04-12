"""Ingestion pipeline orchestrator.

Ties together repo_crawler → (ast_chunker | noncode_indexer) → list[Chunk].
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from config.settings import Settings, get_settings
from repoqa.models import Chunk, FileRecord
from repoqa.ingestion.repo_crawler import crawl_repo
from repoqa.ingestion.ast_chunker import ASTChunker
from repoqa.ingestion.noncode_indexer import NonCodeIndexer

logger = logging.getLogger(__name__)
console = Console()

# Languages handled by the AST chunker
_CODE_LANGUAGES = {"python", "javascript", "typescript", "java", "go", "rust", "c", "cpp"}


class IngestionPipeline:
    """Orchestrates full repository ingestion into a flat list of Chunk objects."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._ast_chunker = ASTChunker(
            max_tokens=self.settings.chunk_max_tokens,
            overlap_tokens=self.settings.chunk_overlap_tokens,
        )
        self._noncode_indexer = NonCodeIndexer(
            max_tokens=self.settings.chunk_max_tokens,
            overlap_tokens=self.settings.chunk_overlap_tokens,
        )

    def run(self, repo_root: str) -> list[Chunk]:
        """Crawl *repo_root* and return all chunks."""
        console.print(f"[bold cyan]Ingesting repository:[/bold cyan] {repo_root}")

        records = crawl_repo(repo_root)
        console.print(f"  Found [green]{len(records)}[/green] files")

        all_chunks: list[Chunk] = []
        for record in tqdm(records, desc="Chunking files", unit="file"):
            try:
                if record.language in _CODE_LANGUAGES:
                    chunks = self._ast_chunker.chunk_file(record)
                else:
                    chunks = self._noncode_indexer.index_file(record)
                all_chunks.extend(chunks)
            except Exception as exc:
                logger.warning("Failed to chunk %s: %s", record.repo_path, exc)

        self._print_stats(all_chunks)
        return all_chunks

    def run_and_save(self, repo_root: str, output_path: str) -> list[Chunk]:
        """Run ingestion and save chunks to *output_path* as JSON."""
        chunks = self.run(repo_root)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump([dataclasses.asdict(c) for c in chunks], f, indent=2)
        console.print(f"[bold green]Saved {len(chunks)} chunks to {output_path}[/bold green]")
        return chunks

    @staticmethod
    def load_chunks(json_path: str) -> list[Chunk]:
        """Deserialize chunks previously saved with run_and_save."""
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        chunks: list[Chunk] = []
        for item in raw:
            fr = FileRecord(**item["file_record"])
            del item["file_record"]
            chunks.append(Chunk(file_record=fr, **item))
        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_stats(chunks: list[Chunk]) -> None:
        # By language
        lang_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        for c in chunks:
            lang_counts[c.language] = lang_counts.get(c.language, 0) + 1
            type_counts[c.chunk_type] = type_counts.get(c.chunk_type, 0) + 1

        table = Table(title="Ingestion Summary", show_header=True)
        table.add_column("Language", style="cyan")
        table.add_column("Chunks", justify="right")
        for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
            table.add_row(lang, str(count))

        console.print(table)
        console.print(f"Total chunks: [bold]{len(chunks)}[/bold]")
        console.print("By type: " + ", ".join(f"{t}={n}" for t, n in sorted(type_counts.items())))
