"""FastAPI production REST API for Room2Rug Multimodal Search."""

from __future__ import annotations

import collections
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import psutil
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.service.pipeline import ProductionSearchPipeline
from src.service.schemas import (
    HealthResponse,
    MetricsResponse,
    SearchRequest,
    SearchResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("room2rug.api")

# Global operational state
START_TIME = time.time()
REQUEST_LATENCIES = collections.deque(maxlen=10000)
TOTAL_REQUESTS = 0
ERROR_COUNT = 0
PIPELINE: ProductionSearchPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan manager for pre-loading models and initializing index resources."""
    global PIPELINE
    logger.info("Initializing Room2Rug Production Search Pipeline...")
    catalog_path = os.getenv("CATALOG_PATH", "data/products.json")
    reranker_path = os.getenv("RERANKER_PATH", "models/lambdamart_reranker.txt")

    try:
        PIPELINE = ProductionSearchPipeline(
            catalog_path=catalog_path,
            reranker_model_path=reranker_path,
        )
        logger.info("Pipeline successfully initialized and ready to serve traffic.")
    except Exception as e:
        logger.error("Failed to initialize pipeline during startup: %s", e)
        raise e

    yield

    logger.info("Shutting down Room2Rug API service. Releasing resources.")


app = FastAPI(
    title="Room2Rug Multimodal Search & Recommendation API",
    description="Production-grade two-stage multimodal retrieval and learning-to-rank serving engine for interior rugs.",
    version="0.2.0",
    lifespan=lifespan,
)

# Enable CORS for frontend clients (e.g. Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Observability"])
async def health_check() -> HealthResponse:
    """Service health check, catalog readiness, and process memory status."""
    process = psutil.Process(os.getpid())
    mem_rss_mb = round(process.memory_info().rss / (1024 * 1024), 2)
    uptime_sec = round(time.time() - START_TIME, 1)

    catalog_len = len(PIPELINE.catalog_list) if PIPELINE else 0
    models_status = {
        "hybrid_retriever": PIPELINE is not None and PIPELINE.stage1_retriever is not None,
        "lambdamart_reranker": PIPELINE is not None and PIPELINE.reranker.is_fitted,
        "grounded_explainer": PIPELINE is not None and PIPELINE.explainer is not None,
    }

    return HealthResponse(
        status="healthy" if PIPELINE is not None else "degraded",
        catalog_size=catalog_len,
        models_loaded=models_status,
        memory_rss_mb=mem_rss_mb,
        uptime_seconds=uptime_sec,
    )


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_endpoint(request: SearchRequest) -> SearchResponse:
    """Multimodal search endpoint executing Stage 1 hybrid retrieval and Stage 2 LTR reranking."""
    global TOTAL_REQUESTS, ERROR_COUNT
    TOTAL_REQUESTS += 1

    if PIPELINE is None:
        ERROR_COUNT += 1
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search pipeline is not initialized or still starting up.",
        )

    t0 = time.perf_counter()
    try:
        response = PIPELINE.search(request)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        REQUEST_LATENCIES.append(latency_ms)
        return response
    except Exception as e:
        ERROR_COUNT += 1
        logger.exception("Error processing search query '%s': %s", request.text_query, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search pipeline execution error: {str(e)}",
        )


@app.get("/metrics", response_model=MetricsResponse, tags=["Observability"])
async def metrics_endpoint() -> MetricsResponse:
    """Operational latency percentiles (P50, P90, P95, P99) and throughput metrics."""
    uptime_sec = max(time.time() - START_TIME, 1.0)
    avg_rps = round(TOTAL_REQUESTS / uptime_sec, 3)

    if len(REQUEST_LATENCIES) > 0:
        sorted_latencies = sorted(REQUEST_LATENCIES)
        n = len(sorted_latencies)
        p50 = round(sorted_latencies[int(n * 0.50)], 2)
        p90 = round(sorted_latencies[int(n * 0.90)], 2)
        p95 = round(sorted_latencies[min(int(n * 0.95), n - 1)], 2)
        p99 = round(sorted_latencies[min(int(n * 0.99), n - 1)], 2)
    else:
        p50 = p90 = p95 = p99 = 0.0

    return MetricsResponse(
        total_requests=TOTAL_REQUESTS,
        error_count=ERROR_COUNT,
        latency_p50_ms=p50,
        latency_p90_ms=p90,
        latency_p95_ms=p95,
        latency_p99_ms=p99,
        average_rps=avg_rps,
        uptime_seconds=round(uptime_sec, 1),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.service.api:app", host="0.0.0.0", port=8000, reload=False)
