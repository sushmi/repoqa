"""Orchestrate end-to-end construction of a CoReQA-style eval dataset."""

from __future__ import annotations

import logging
from pathlib import Path

from .github_miner import GitHubMiner, MinerConfig
from .models import EvaluationDataset, QAPair
from .question_classifier import classify_question

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Mine issues → classify questions → persist dataset JSON."""

    def __init__(self, github_token: str, min_reactions: int = 3) -> None:
        self.miner = GitHubMiner(
            MinerConfig(token=github_token, min_reactions=min_reactions)
        )

    def build(
        self,
        repos: list[str],
        output_path: str | Path,
        dataset_name: str = "repoqa-eval",
    ) -> EvaluationDataset:
        all_pairs: list[QAPair] = []
        for repo in repos:
            pairs = self.miner.mine_repo(repo)
            for p in pairs:
                p.question_type = classify_question(p.question)
            all_pairs.extend(pairs)

        dataset = EvaluationDataset(
            name=dataset_name,
            version="1.0.0",
            description=(
                f"CoReQA-style evaluation dataset with {len(all_pairs)} QA pairs "
                f"mined from {len(repos)} repositories."
            ),
            qa_pairs=all_pairs,
            metadata={
                "source_repos": repos,
                "min_reactions": self.miner.config.min_reactions,
            },
        )
        dataset.save(output_path)
        stats = dataset.summary()
        logger.info("Saved %d QA pairs to %s", stats["total"], output_path)
        logger.info("  Languages: %s", stats["by_language"])
        logger.info("  Types:     %s", stats["by_question_type"])
        return dataset
