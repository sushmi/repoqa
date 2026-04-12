"""Mine QA pairs from GitHub issues using the REST API.

Follows the CoReQA methodology: extract questions from issue bodies
and answers from top-voted comments (≥ *min_reactions* positive reactions).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import requests

from .models import QAPair

logger = logging.getLogger(__name__)
GITHUB_API = "https://api.github.com"


@dataclass
class MinerConfig:
    token: str
    min_reactions: int = 3
    max_issues_per_repo: int = 100
    labels: list[str] = field(default_factory=list)


class GitHubMiner:
    """Mines Q&A pairs from GitHub repository issues."""

    def __init__(self, config: MinerConfig) -> None:
        self.config = config
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"token {config.token}",
            "Accept": "application/vnd.github.v3+json",
        })
        self._lang_cache: dict[str, str] = {}

    def mine_repo(self, repo: str) -> list[QAPair]:
        """Mine QA pairs from *repo* (``'owner/repo'`` format)."""
        logger.info("Mining issues from %s …", repo)
        issues = self._fetch_issues(repo)
        pairs: list[QAPair] = []
        for issue in issues:
            pair = self._process_issue(repo, issue)
            if pair is not None:
                pairs.append(pair)
        logger.info("  → %d QA pairs from %s", len(pairs), repo)
        return pairs

    # -- GitHub API helpers ---------------------------------------------------

    def _fetch_issues(self, repo: str) -> list[dict]:
        all_issues: list[dict] = []
        page = 1
        per_page = 100
        while len(all_issues) < self.config.max_issues_per_repo:
            params: dict = {
                "state": "closed",
                "sort": "comments",
                "direction": "desc",
                "per_page": per_page,
                "page": page,
            }
            if self.config.labels:
                params["labels"] = ",".join(self.config.labels)

            data = self._get(f"{GITHUB_API}/repos/{repo}/issues", params=params)
            if not data:
                break
            batch = [i for i in data if "pull_request" not in i]
            if not batch:
                break
            all_issues.extend(batch)
            page += 1
            if len(data) < per_page:
                break
        return all_issues[: self.config.max_issues_per_repo]

    def _process_issue(self, repo: str, issue: dict) -> QAPair | None:
        body = (issue.get("body") or "").strip()
        if len(body) < 30:
            return None
        comments = self._get(
            f"{GITHUB_API}/repos/{repo}/issues/{issue['number']}/comments",
            params={"per_page": 30},
        ) or []
        best = self._best_answer(comments)
        if best is None:
            return None

        return QAPair(
            id=QAPair.make_id(repo, issue["number"]),
            repo_name=repo,
            repo_url=f"https://github.com/{repo}",
            language=self._repo_language(repo),
            question=body,
            answer=best["body"],
            question_type="",       # classified later by DatasetBuilder
            issue_url=issue["html_url"],
            issue_number=issue["number"],
            answer_score=self._positive_reactions(best),
        )

    def _best_answer(self, comments: list[dict]) -> dict | None:
        scored = [
            (self._positive_reactions(c), c)
            for c in comments
            if self._positive_reactions(c) >= self.config.min_reactions
        ]
        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    @staticmethod
    def _positive_reactions(comment: dict) -> int:
        r = comment.get("reactions", {})
        return r.get("+1", 0) + r.get("heart", 0) + r.get("hooray", 0) + r.get("rocket", 0)

    def _repo_language(self, repo: str) -> str:
        if repo not in self._lang_cache:
            data = self._get(f"{GITHUB_API}/repos/{repo}")
            raw = ((data or {}).get("language") or "").lower()
            self._lang_cache[repo] = {
                "python": "python", "java": "java", "go": "go",
                "typescript": "typescript", "javascript": "typescript",
            }.get(raw, raw or "unknown")
        return self._lang_cache[repo]

    def _get(self, url: str, params: dict | None = None) -> dict | list | None:
        try:
            resp = self._session.get(url, params=params, timeout=30)
            if resp.status_code == 403:
                reset = int(resp.headers.get("X-RateLimit-Reset", 0))
                wait = max(reset - int(time.time()), 10)
                logger.warning("Rate-limited — sleeping %ds …", wait)
                time.sleep(wait)
                return self._get(url, params)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("GitHub API error: %s", exc)
            return None
