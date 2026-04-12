"""Data models for the evaluation dataset, following CoReQA schema."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class QAPair:
    """A single question-answer pair from a GitHub repository.

    Mirrors CoReQA's schema: each pair is sourced from a real GitHub
    issue, has a community-validated answer, and carries BM25-retrieved
    reference contexts for baseline comparison.
    """

    id: str
    repo_name: str                  # "owner/repo"
    repo_url: str                   # https://github.com/owner/repo
    language: str                   # python | java | go | typescript
    question: str                   # developer question (issue body)
    answer: str                     # reference answer (top-voted comment)
    question_type: str              # architecture | logic | deployment
    issue_url: str                  # source GitHub issue URL
    issue_number: int
    answer_score: int               # sum of positive reactions on the answer
    reference_contexts: list[str] = field(default_factory=list)

    @staticmethod
    def make_id(repo_name: str, issue_number: int) -> str:
        raw = f"{repo_name}:{issue_number}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class EvaluationDataset:
    """A collection of QA pairs for evaluation."""

    name: str
    version: str
    description: str
    qa_pairs: list[QAPair]
    metadata: dict = field(default_factory=dict)

    # -- persistence -----------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "metadata": self.metadata,
            "qa_pairs": [asdict(qa) for qa in self.qa_pairs],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> EvaluationDataset:
        raw = json.loads(Path(path).read_text())
        qa_pairs = [QAPair(**qp) for qp in raw["qa_pairs"]]
        return cls(
            name=raw["name"],
            version=raw["version"],
            description=raw["description"],
            qa_pairs=qa_pairs,
            metadata=raw.get("metadata", {}),
        )

    # -- analytics -------------------------------------------------------------

    def summary(self) -> dict:
        by_lang: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for qa in self.qa_pairs:
            by_lang[qa.language] = by_lang.get(qa.language, 0) + 1
            by_type[qa.question_type] = by_type.get(qa.question_type, 0) + 1
        return {
            "total": len(self.qa_pairs),
            "by_language": dict(sorted(by_lang.items())),
            "by_question_type": dict(sorted(by_type.items())),
            "repos": len({qa.repo_name for qa in self.qa_pairs}),
        }
