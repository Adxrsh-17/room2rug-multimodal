"""Grounded explanation generation and faithfulness evaluation package."""

from src.explain.faithfulness import (
    FaithfulnessResult,
    FaithfulnessValidator,
    evaluate_faithfulness_suite,
)
from src.explain.generator import (
    DeterministicGroundedGenerator,
    ExplanationResult,
    GroundedExplanationGenerator,
)

__all__ = [
    "FaithfulnessResult",
    "FaithfulnessValidator",
    "evaluate_faithfulness_suite",
    "DeterministicGroundedGenerator",
    "ExplanationResult",
    "GroundedExplanationGenerator",
]
