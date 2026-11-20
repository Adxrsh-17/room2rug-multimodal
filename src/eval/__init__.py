"""Evaluation harness, metrics, and baseline rankers."""

from src.eval.base import BaseRanker, CandidateRug, Query, RankResult
from src.eval.metrics import compute_bootstrap_ci, evaluate_rankings, paired_permutation_test

__all__ = [
    "BaseRanker",
    "CandidateRug",
    "Query",
    "RankResult",
    "evaluate_rankings",
    "compute_bootstrap_ci",
    "paired_permutation_test",
]
