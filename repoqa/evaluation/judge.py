"""LLM-as-Judge evaluator using Gemini 1.5 Flash.

Scores a generated answer against a reference answer on 4 dimensions
(CoReQA schema): accuracy, completeness, relevance, clarity.

Each evaluation can be run N times and aggregated (mean ± std) to reduce
variance. Gemini 1.5 Flash is used as the judge because:
  - Different model family than the generator (Qwen) → avoids self-bias
  - Free tier: 1,500 requests/day, 15 requests/minute
  - Strong at structured JSON output with chain-of-thought

Requires: GEMINI_API_KEY in .env
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import time
from dataclasses import dataclass, field

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)

_OLLAMA_DEFAULT_URL = "http://localhost:11434"


_JUDGE_PROMPT = """\
You are an expert code reviewer evaluating a Q&A system's answer about a \
software repository. Score the candidate answer against the reference \
answer on 4 dimensions, each from 1 to 10.

USE CHAIN OF THOUGHT. Think through each dimension briefly, then output \
a single JSON object.

---
QUESTION:
{question}

---
REFERENCE ANSWER (ground truth, community-validated):
{reference_answer}

---
CANDIDATE ANSWER (system under evaluation):
{candidate_answer}

---
SCORING DIMENSIONS:

1. ACCURACY (1-10): Is the candidate factually correct vs. the reference?
   Are specific names (functions, classes, files) used correctly?
   10 = every technical claim matches reference or repo reality
   1  = contradicts the reference or hallucinates facts

2. COMPLETENESS (1-10): Does the candidate cover all key points of the reference?
   10 = covers every major aspect of the reference answer
   1  = misses almost everything of substance

3. RELEVANCE (1-10): Does the candidate address the core concern of the question?
   10 = directly answers what was asked
   1  = answers a different question or goes off-topic

4. CLARITY (1-10): Is the candidate well-structured, readable, and clear?
   10 = logical flow, well-organized, easy to follow
   1  = confused, disorganized, or hard to read

---
FIRST reason step-by-step (briefly, 1-2 sentences per dimension).
THEN output a single JSON object on its own line with this exact schema:

{{"accuracy": <int 1-10>, "completeness": <int 1-10>, "relevance": <int 1-10>, \
"clarity": <int 1-10>, "reasoning": "<one-sentence overall justification>"}}

Do not output any text after the JSON.
"""


@dataclass
class JudgeScore:
    """Scores from a single judge run."""
    accuracy: int
    completeness: int
    relevance: int
    clarity: int
    reasoning: str
    raw_response: str = ""


@dataclass
class AggregatedScore:
    """Aggregated scores across N runs (mean + std per dimension)."""
    accuracy_mean: float
    accuracy_std: float
    completeness_mean: float
    completeness_std: float
    relevance_mean: float
    relevance_std: float
    clarity_mean: float
    clarity_std: float
    n_runs: int
    per_run: list[JudgeScore] = field(default_factory=list)


class GeminiJudge:
    """Gemini 2.5 Flash as an LLM-as-Judge evaluator."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash",
                 min_request_interval_s: float = 13.0) -> None:
        """
        Args:
            api_key: Google AI Studio key from https://aistudio.google.com/
            model: Gemini model name (default gemini-2.5-flash)
            min_request_interval_s: Min seconds between calls. gemini-2.5 Flash
                free tier is 5 RPM → 12s floor + 1s slack = 13s default.
                For higher-RPM models pass a smaller value (e.g. 4.5 for 15 RPM).
        """
        if not api_key:
            raise ValueError(
                "Gemini API key is empty. Set GEMINI_API_KEY in .env. "
                "Get a free key at https://aistudio.google.com/app/apikey"
            )
        self.api_key = api_key
        self.model = model
        self._min_interval = min_request_interval_s
        self._last_request_ts = 0.0

    def score(
        self,
        question: str,
        reference_answer: str,
        candidate_answer: str,
    ) -> JudgeScore:
        """Run one judge evaluation."""
        prompt = _JUDGE_PROMPT.format(
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
        )
        raw = self._invoke(prompt)
        return self._parse_score(raw)

    def score_n_times(
        self,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        n: int = 3,
    ) -> AggregatedScore:
        """Run N independent judge evaluations and aggregate."""
        runs: list[JudgeScore] = []
        for i in range(n):
            try:
                runs.append(self.score(question, reference_answer, candidate_answer))
            except Exception as exc:
                logger.warning("Judge run %d/%d failed: %s", i + 1, n, exc)

        if not runs:
            raise RuntimeError("All judge runs failed")

        def _stats(values: list[int]) -> tuple[float, float]:
            return (
                statistics.mean(values),
                statistics.stdev(values) if len(values) > 1 else 0.0,
            )

        acc_m, acc_s = _stats([r.accuracy for r in runs])
        com_m, com_s = _stats([r.completeness for r in runs])
        rel_m, rel_s = _stats([r.relevance for r in runs])
        cla_m, cla_s = _stats([r.clarity for r in runs])

        return AggregatedScore(
            accuracy_mean=acc_m, accuracy_std=acc_s,
            completeness_mean=com_m, completeness_std=com_s,
            relevance_mean=rel_m, relevance_std=rel_s,
            clarity_mean=cla_m, clarity_std=cla_s,
            n_runs=len(runs),
            per_run=runs,
        )

    # ── Internal ─────────────────────────────────────────────────

    @retry(
        wait=wait_exponential(multiplier=2, min=12, max=120),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _invoke(self, prompt: str) -> str:
        """Call Gemini REST API with rate limiting and 429-aware retries."""
        self._throttle()

        url = _GEMINI_URL.format(model=self.model, key=self.api_key)
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            },
        }
        response = requests.post(url, json=payload, timeout=60)
        self._last_request_ts = time.time()

        if response.status_code == 429:
            wait_s = self._parse_retry_after(response)
            logger.warning(
                "Gemini 429 rate-limited. Sleeping %.1fs before retry.", wait_s
            )
            time.sleep(wait_s)
            response.raise_for_status()  # raises → tenacity retries

        if not response.ok:
            logger.error("Gemini error %d: %s", response.status_code, response.text[:300])
        response.raise_for_status()

        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected Gemini response: {data}") from exc

    def _throttle(self) -> None:
        """Block until min interval since last call has elapsed."""
        elapsed = time.time() - self._last_request_ts
        if elapsed < self._min_interval:
            sleep_s = self._min_interval - elapsed
            logger.debug("Throttling: sleeping %.1fs to respect RPM limit", sleep_s)
            time.sleep(sleep_s)

    @staticmethod
    def _parse_retry_after(response: requests.Response) -> float:
        """Extract suggested retry delay from a 429 response.

        Gemini returns it two ways:
          - Retry-After header (seconds, RFC 7231)
          - error.details[].retryDelay = "Ns" in the JSON body
        Falls back to 30s when neither is present.
        """
        # Header form
        ra = response.headers.get("Retry-After")
        if ra and ra.isdigit():
            return float(ra) + 1.0  # +1s slack

        # Google API error body form
        try:
            body = response.json()
            for detail in body.get("error", {}).get("details", []):
                delay = detail.get("retryDelay")
                if isinstance(delay, str) and delay.endswith("s"):
                    return float(delay[:-1]) + 1.0
        except (ValueError, KeyError):
            pass

        return 30.0  # conservative default for free-tier 5 RPM

    @staticmethod
    def _parse_score(raw: str) -> JudgeScore:
        """Extract the JSON object from the judge's reasoning + JSON output."""
        return _parse_score(raw)


def _parse_score(raw: str) -> JudgeScore:
    """Extract the scoring JSON from a judge's reasoning + JSON response.

    Shared by both GeminiJudge and OllamaJudge.
    """
    matches = re.findall(r"\{[^{}]*\"accuracy\"[^{}]*\}", raw, re.DOTALL)
    if not matches:
        raise ValueError(f"No scoring JSON found in judge output:\n{raw[:500]}")
    parsed = json.loads(matches[-1])

    def _clip(v) -> int:
        v = int(v)
        return max(1, min(10, v))

    return JudgeScore(
        accuracy=_clip(parsed["accuracy"]),
        completeness=_clip(parsed["completeness"]),
        relevance=_clip(parsed["relevance"]),
        clarity=_clip(parsed["clarity"]),
        reasoning=parsed.get("reasoning", "")[:500],
        raw_response=raw,
    )


def _aggregate_runs(runs: list[JudgeScore]) -> AggregatedScore:
    """Compute mean ± std across N independent judge runs."""
    if not runs:
        raise RuntimeError("All judge runs failed")

    def _stats(values: list[int]) -> tuple[float, float]:
        return (
            statistics.mean(values),
            statistics.stdev(values) if len(values) > 1 else 0.0,
        )

    acc_m, acc_s = _stats([r.accuracy for r in runs])
    com_m, com_s = _stats([r.completeness for r in runs])
    rel_m, rel_s = _stats([r.relevance for r in runs])
    cla_m, cla_s = _stats([r.clarity for r in runs])

    return AggregatedScore(
        accuracy_mean=acc_m, accuracy_std=acc_s,
        completeness_mean=com_m, completeness_std=com_s,
        relevance_mean=rel_m, relevance_std=rel_s,
        clarity_mean=cla_m, clarity_std=cla_s,
        n_runs=len(runs),
        per_run=runs,
    )


class OllamaJudge:
    """Local LLM-as-Judge via Ollama (no API key, no rate limits).

    Default model is llama3.1:8b — different family from the Qwen generator,
    so we still avoid self-evaluation bias. Quality is lower than Gemini Flash
    but the cost is zero and unlimited, which matters when the free tier is
    capped at 20 requests/day.

    Same interface as GeminiJudge: score() and score_n_times().
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = _OLLAMA_DEFAULT_URL,
        temperature: float = 0.1,
        timeout_s: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout_s = timeout_s

    def score(
        self,
        question: str,
        reference_answer: str,
        candidate_answer: str,
    ) -> JudgeScore:
        prompt = _JUDGE_PROMPT.format(
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
        )
        raw = self._invoke(prompt)
        return _parse_score(raw)

    def score_n_times(
        self,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        n: int = 3,
    ) -> AggregatedScore:
        runs: list[JudgeScore] = []
        for i in range(n):
            try:
                runs.append(self.score(question, reference_answer, candidate_answer))
            except Exception as exc:
                logger.warning("Local judge run %d/%d failed: %s", i + 1, n, exc)
        return _aggregate_runs(runs)

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _invoke(self, prompt: str) -> str:
        """Call Ollama /api/generate (non-streaming)."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": 0.95,
                "num_predict": 1024,
                # Ollama defaults to 2048 — too small for our ~2-3k token
                # judge prompts (instruction + Q + ref answer + candidate).
                # 8192 gives generous headroom without exploding RAM use.
                "num_ctx": 8192,
            },
        }
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        text = data.get("response", "")
        if not text:
            raise RuntimeError(f"Empty response from Ollama: {data}")
        return text
