"""Pydantic v2 schemas for the Room2Rug Multimodal Search REST API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Input payload for multimodal search."""

    text_query: str = Field(
        default="",
        description="Free-form user search query (e.g. '8x10 warm beige living room rug')",
        examples=["8x10 warm beige living room rug"],
    )
    room_image_b64: str | None = Field(
        default=None,
        description="Optional base64-encoded room photograph.",
    )
    room_image_path: str | None = Field(
        default=None,
        description="Optional local filesystem path to room image for profiling/eval.",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of ranked rug recommendations to return.",
    )
    mode: Literal["two_stage", "stage1_hybrid", "baseline_v1"] = Field(
        default="two_stage",
        description="Retrieval/ranking pipeline mode.",
    )
    explain: bool = Field(
        default=True,
        description="Whether to generate grounded natural language explanations for results.",
    )
    explanation_backend: Literal["deterministic", "llm"] = Field(
        default="deterministic",
        description="Backend generator for explanations.",
    )


class RugCandidateResponse(BaseModel):
    """Ranked rug candidate with catalog metadata, scores, and grounded rationale."""

    handle: str
    title: str
    final_rank: int
    score: float
    stage1_rank: int | None = None
    price: float | None = None
    color: str | None = None
    style: str | None = None
    size: str | None = None
    sizes: list[str] = Field(default_factory=list)
    image_url: str | None = None
    cielab_delta_e: float | None = None
    explanation: str | None = None
    explanation_faithful: bool | None = None


class SearchResponse(BaseModel):
    """Complete response payload for a multimodal search request."""

    query_id: str
    text_query: str
    mode: str
    results_count: int
    latency_breakdown_ms: dict[str, float]
    candidates: list[RugCandidateResponse]


class HealthResponse(BaseModel):
    """System and model readiness status."""

    status: str
    catalog_size: int
    models_loaded: dict[str, bool]
    memory_rss_mb: float
    uptime_seconds: float
    version: str = "0.2.0"


class MetricsResponse(BaseModel):
    """Operational latency and throughput metrics."""

    total_requests: int
    error_count: int
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    average_rps: float
    uptime_seconds: float
