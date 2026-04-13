"""Hierarchical summarization pipeline.

Runs bottom-up: file summaries → directory summaries → project summary.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
from rich.console import Console

from config.settings import Settings, get_settings
from repoqa.models import Chunk, Summary
from repoqa.summarization.file_summarizer import FileSummarizer
from repoqa.summarization.dir_summarizer import DirSummarizer
from repoqa.summarization.project_summarizer import ProjectSummarizer

logger = logging.getLogger(__name__)
console = Console()


def _build_llm(settings: Settings):
    """Instantiate the appropriate LangChain chat model from settings."""
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.chat_model,
            anthropic_api_key=settings.anthropic_api_key,
            max_tokens=settings.summary_max_tokens,
        )
    elif settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.chat_model,
            openai_api_key=settings.openai_api_key,
            max_tokens=settings.summary_max_tokens,
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.chat_model,
            base_url=settings.ollama_base_url,
            num_predict=settings.summary_max_tokens,
            num_ctx=8192,
        )


class HierarchicalSummarizationPipeline:
    """Generates file, directory, and project summaries from a list of Chunk objects."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        llm = _build_llm(self.settings)
        self._file_summarizer = FileSummarizer(llm)
        self._dir_summarizer = DirSummarizer(llm)
        self._project_summarizer = ProjectSummarizer(llm)

    def run(self, chunks: list[Chunk], repo_root: str = ".") -> list[Summary]:
        """Generate all summaries and return them as a flat list."""
        all_summaries: list[Summary] = []

        # --- Phase 1: File summaries ---
        chunks_by_file: dict[str, list[Chunk]] = defaultdict(list)
        for c in chunks:
            chunks_by_file[c.file_record.repo_path].append(c)

        console.print(f"[bold cyan]Summarizing {len(chunks_by_file)} files...[/bold cyan]")
        file_summaries: dict[str, Summary] = {}  # repo_path → Summary
        for repo_path, file_chunks in tqdm(chunks_by_file.items(), desc="File summaries", unit="file"):
            try:
                summary = self._file_summarizer.summarize(file_chunks[0].file_record, file_chunks)
                file_summaries[repo_path] = summary
                all_summaries.append(summary)
            except Exception as exc:
                logger.warning("File summarization failed for %s: %s", repo_path, exc)

        # --- Phase 2: Directory summaries (bottom-up tree traversal) ---
        # Group file summaries by their parent directory
        dir_to_file_summaries: dict[str, list[Summary]] = defaultdict(list)
        for repo_path, summary in file_summaries.items():
            parent = str(Path(repo_path).parent)
            dir_to_file_summaries[parent].append(summary)

        # Walk up the tree: generate dir summaries level by level
        dir_summaries: dict[str, Summary] = {}  # dir_path → Summary
        pending_dirs = set(dir_to_file_summaries.keys())

        console.print(f"[bold cyan]Summarizing {len(pending_dirs)} directories...[/bold cyan]")
        for dir_path in tqdm(sorted(pending_dirs), desc="Dir summaries", unit="dir"):
            child_summaries = dir_to_file_summaries[dir_path]
            # Also include summaries of subdirectories we've already computed
            child_summaries += [
                s for d, s in dir_summaries.items()
                if str(Path(d).parent) == dir_path
            ]
            if not child_summaries:
                continue
            try:
                summary = self._dir_summarizer.summarize(dir_path, child_summaries)
                dir_summaries[dir_path] = summary
                all_summaries.append(summary)
                # Propagate up to parent
                parent = str(Path(dir_path).parent)
                if parent != dir_path:  # avoid infinite loop at root
                    dir_to_file_summaries[parent].append(summary)
            except Exception as exc:
                logger.warning("Dir summarization failed for %s: %s", dir_path, exc)

        # --- Phase 3: Project summary ---
        console.print("[bold cyan]Generating project summary...[/bold cyan]")
        readme_chunk = self._find_readme_chunk(chunks)
        top_level_dirs = [s for path, s in dir_summaries.items() if str(Path(path).parent) in (".", "")]
        if not top_level_dirs:
            top_level_dirs = list(dir_summaries.values())[:20]  # fallback
        try:
            project_summary = self._project_summarizer.summarize(
                repo_root=repo_root,
                dir_summaries=top_level_dirs,
                readme_chunk=readme_chunk,
            )
            all_summaries.append(project_summary)
            console.print("[green]Project summary generated.[/green]")
        except Exception as exc:
            logger.error("Project summarization failed: %s", exc)

        console.print(f"[bold green]Total summaries: {len(all_summaries)}[/bold green]")
        return all_summaries

    def run_and_save(self, chunks: list[Chunk], repo_root: str, output_path: str) -> list[Summary]:
        summaries = self.run(chunks, repo_root=repo_root)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump([dataclasses.asdict(s) for s in summaries], f, indent=2)
        console.print(f"[bold green]Saved {len(summaries)} summaries to {output_path}[/bold green]")
        return summaries

    @staticmethod
    def load_summaries(json_path: str) -> list[Summary]:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [Summary(**item) for item in raw]

    @staticmethod
    def _find_readme_chunk(chunks: list[Chunk]) -> Chunk | None:
        """Return the first chunk from a README file, if any."""
        for c in chunks:
            if "readme" in c.file_record.repo_path.lower():
                return c
        return None
