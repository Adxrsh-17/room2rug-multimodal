"""Contrastive retrieval models, projection adapters, and hybrid fusion."""

from src.retrieval.adapter import ContrastiveRetrievalModel, ResidualProjectionHead
from src.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from src.retrieval.loss import InfoNCELoss

__all__ = [
    "ResidualProjectionHead",
    "ContrastiveRetrievalModel",
    "InfoNCELoss",
    "HybridRetriever",
    "reciprocal_rank_fusion",
]
