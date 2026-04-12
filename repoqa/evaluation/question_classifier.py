"""Classify developer questions into architecture / logic / deployment."""

from __future__ import annotations

import re

_ARCHITECTURE_PATTERNS: list[str] = [
    r"\barchitect",
    r"\bstructur",
    r"\bdesign.pattern",
    r"\bmodule",
    r"\bcomponent",
    r"\borganiz",
    r"\blayer",
    r"\brelationship",
    r"\bhow\s+(?:does|do|is)\s+(?:the\s+)?(?:project|app|system|code)",
    r"\bproject\s+(?:structure|layout|organization)",
    r"\bdirectory",
    r"\boverview",
    r"\bhigh[\s-]level",
    r"\bwhere\s+(?:is|are|does|do)",
    r"\bwhat\s+(?:is|are)\s+the\s+(?:main|key|core)",
    r"\bhow\s+.*\bwork",
    r"\bflow\b",
    r"\bpipeline",
    r"\bmiddleware",
    r"\brouting",
    r"\bdependenc",
]

_LOGIC_PATTERNS: list[str] = [
    r"\bfunction\b",
    r"\bmethod\b",
    r"\bclass\b",
    r"\bimplement",
    r"\balgorithm",
    r"\bbug\b",
    r"\berror\b",
    r"\bfix\b",
    r"\breturn\b",
    r"\bparameter",
    r"\bargument",
    r"\btype[\s-](?:hint|annotation|error|check)",
    r"\bwhat\s+does\s+(?:this|the)\s+(?:function|method|code)",
    r"\bhow\s+does\s+(?:this|the)\s+(?:function|method|code)",
    r"\bwhy\s+(?:does|do|is)",
    r"\bexception",
    r"\bvalidat",
    r"\bpars",
    r"\btransform",
    r"\bcallback",
    r"\basync\b",
    r"\bawait\b",
    r"\brecurs",
    r"\bloop\b",
]

_DEPLOYMENT_PATTERNS: list[str] = [
    r"\bdeploy",
    r"\bdocker",
    r"\bcontainer",
    r"\bci[\s/]cd",
    r"\bgithub.action",
    r"\bkubernetes",
    r"\bk8s",
    r"\benvironment",
    r"\bconfig(?:ur)?",
    r"\binstall\b",
    r"\bsetup\b",
    r"\bbuild\b",
    r"\bpackage",
    r"\bpip\b",
    r"\bnpm\b",
    r"\byarn\b",
    r"\bgradle",
    r"\bmaven",
    r"\bmakefile",
    r"\bvenv\b",
    r"\brequirements",
    r"\bserver\b",
    r"\bversion\b",
    r"\bmigrat",
    r"\bdatabase\b",
    r"\b\.env\b",
]


def classify_question(text: str) -> str:
    """Return ``'architecture'``, ``'logic'``, or ``'deployment'``."""
    low = text.lower()
    scores = {
        "architecture": _score(low, _ARCHITECTURE_PATTERNS),
        "logic": _score(low, _LOGIC_PATTERNS),
        "deployment": _score(low, _DEPLOYMENT_PATTERNS),
    }
    if max(scores.values()) == 0:
        return "logic"          # conservative default
    return max(scores, key=scores.get)  # type: ignore[arg-type]


def _score(text: str, patterns: list[str]) -> int:
    return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
