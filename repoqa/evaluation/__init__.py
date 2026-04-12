"""Evaluation module — CoReQA-style dataset generation and management."""

from .models import QAPair, EvaluationDataset
from .question_classifier import classify_question

__all__ = ["QAPair", "EvaluationDataset", "classify_question"]
