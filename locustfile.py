"""Locust load-testing user definition for Room2Rug Multimodal Search API."""

import random

from locust import HttpUser, between, task

SAMPLE_QUERIES = [
    "8x10 warm beige living room rug",
    "modern minimalist geometric home office rug",
    "5x7 muted blue soft bedroom rug",
    "traditional vintage oriental dining room rug",
    "clean durable grey workspace rug for office desk",
    "contemporary abstract area rug",
    "boho chic runner rug",
    "large neutral area rug for hardwood floor",
]


class SearchUser(HttpUser):
    """Simulates realistic user traffic against the FastAPI search and health endpoints."""

    wait_time = between(0.1, 0.5)

    @task(5)
    def search_two_stage(self):
        query = random.choice(SAMPLE_QUERIES)
        payload = {
            "text_query": query,
            "top_k": 10,
            "mode": "two_stage",
            "explain": True,
        }
        self.client.post("/search", json=payload, name="/search [two_stage]")

    @task(2)
    def search_stage1(self):
        query = random.choice(SAMPLE_QUERIES)
        payload = {
            "text_query": query,
            "top_k": 10,
            "mode": "stage1_hybrid",
            "explain": False,
        }
        self.client.post("/search", json=payload, name="/search [stage1_hybrid]")

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")

    @task(1)
    def metrics_check(self):
        self.client.get("/metrics", name="/metrics")
