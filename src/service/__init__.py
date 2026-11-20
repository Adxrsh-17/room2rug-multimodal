"""Production FastAPI serving package for Room2Rug Multimodal Search."""

from src.service.pipeline import ProductionSearchPipeline
from src.service.schemas import (
    HealthResponse,
    MetricsResponse,
    RugCandidateResponse,
    SearchRequest,
    SearchResponse,
)

__all__ = [
    "ProductionSearchPipeline",
    "HealthResponse",
    "MetricsResponse",
    "RugCandidateResponse",
    "SearchRequest",
    "SearchResponse",
]
