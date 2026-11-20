"""Integration tests for the FastAPI serving endpoints using TestClient."""

import pytest
from fastapi.testclient import TestClient

from src.service.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["catalog_size"] > 0
    assert "hybrid_retriever" in data["models_loaded"]
    assert data["memory_rss_mb"] > 0


def test_search_endpoint_two_stage(client):
    payload = {
        "text_query": "8x10 warm beige living room rug",
        "top_k": 5,
        "mode": "two_stage",
        "explain": True,
    }
    response = client.post("/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["results_count"] == 5
    assert len(data["candidates"]) == 5
    assert data["mode"] == "two_stage"

    # Verify candidate attributes
    cand0 = data["candidates"][0]
    assert cand0["final_rank"] == 1
    assert "handle" in cand0
    assert "title" in cand0
    assert cand0["explanation"] is not None
    assert cand0["explanation_faithful"] is True

    # Verify latency breakdown
    latencies = data["latency_breakdown_ms"]
    assert "parsing_ms" in latencies
    assert "retrieval_ms" in latencies
    assert "reranking_ms" in latencies
    assert "explanation_ms" in latencies
    assert "total_latency_ms" in latencies


def test_search_endpoint_baseline_v1_mode(client):
    payload = {
        "text_query": "modern grey office runner",
        "top_k": 3,
        "mode": "baseline_v1",
    }
    response = client.post("/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["results_count"] == 3
    assert data["mode"] == "baseline_v1"


def test_search_validation_error(client):
    # top_k > 50 should trigger Pydantic validation error (422)
    payload = {
        "text_query": "rug",
        "top_k": 100,
    }
    response = client.post("/search", json=payload)
    assert response.status_code == 422


def test_metrics_endpoint(client):
    # After prior searches, total_requests should be >= 2
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_requests"] >= 2
    assert data["latency_p50_ms"] >= 0.0
    assert data["uptime_seconds"] > 0
