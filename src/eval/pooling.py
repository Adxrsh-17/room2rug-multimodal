"""Multi-system candidate pooling for unbiased evaluation harness."""

from __future__ import annotations

import random
from typing import Any

from src.eval.base import BaseRanker, CandidateRug, Query


def pool_candidates(
    query: Query,
    candidates: list[CandidateRug],
    systems: list[BaseRanker],
    pool_depth_per_system: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Pool top candidates across multiple retrieval systems for a query.

    Blinds system identity to prevent annotator bias while retaining provenance.

    Args:
        query: Query object.
        candidates: Catalog candidates.
        systems: List of BaseRanker instances.
        pool_depth_per_system: How many top items to take from each system.
        seed: Random seed for shuffling.

    Returns:
        dict containing:
            query_id: str
            pooled_rug_ids: list of unique rug IDs in randomized order
            provenance: dict mapping rug_id -> list of system names that retrieved it
            system_ranks: dict mapping system_name -> list of top rug_ids
    """
    provenance: dict[str, list[str]] = {}
    system_ranks: dict[str, list[str]] = {}

    for sys in systems:
        ranked = sys.rank(query, candidates, top_k=pool_depth_per_system)
        top_ids = [res.rug_id for res in ranked]
        system_ranks[sys.name] = top_ids

        for item_id in top_ids:
            provenance.setdefault(item_id, []).append(sys.name)

    unique_pooled = list(provenance.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_pooled)

    return {
        "query_id": query.query_id,
        "pooled_rug_ids": unique_pooled,
        "provenance": provenance,
        "system_ranks": system_ranks,
        "n_unique_candidates": len(unique_pooled),
    }


def pool_dataset_queries(
    queries: list[Query],
    candidates: list[CandidateRug],
    systems: list[BaseRanker],
    pool_depth_per_system: int = 5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate pooled candidate sets for a collection of queries."""
    pools = []
    for idx, query in enumerate(queries):
        pool = pool_candidates(
            query=query,
            candidates=candidates,
            systems=systems,
            pool_depth_per_system=pool_depth_per_system,
            seed=seed + idx,
        )
        pools.append(pool)
    return pools
