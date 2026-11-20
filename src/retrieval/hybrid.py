"""Hybrid multimodal retrieval: dense + sparse (BM25) + attribute constraints.

Implements:
1. Reciprocal Rank Fusion (RRF)
2. Weighted Linear Late Fusion
3. Validation-set hyperparameter tuning (zero test leakage)
"""

from __future__ import annotations

import logging

from src.eval.base import BaseRanker, CandidateRug, Query, RankResult
from src.eval.baselines import AttributeFilterBaseline, BM25Baseline

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    system_rankings: dict[str, list[RankResult]],
    weights: dict[str, float] | None = None,
    k_rrf: int = 60,
    top_k: int = 10,
) -> list[RankResult]:
    """Combine rankings using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(d) = sum_{m in models} w_m / (k_rrf + rank_m(d))
    """
    if weights is None:
        weights = {sys_name: 1.0 for sys_name in system_rankings}

    scores: dict[str, float] = {}
    provenance: dict[str, list[str]] = {}

    for sys_name, ranked_items in system_rankings.items():
        w = weights.get(sys_name, 1.0)
        for res in ranked_items:
            rid = res.rug_id
            rrf_val = w / (k_rrf + res.rank)
            scores[rid] = scores.get(rid, 0.0) + rrf_val
            provenance.setdefault(rid, []).append(f"{sys_name}@{res.rank}")

    # Sort descending by fused RRF score
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results: list[RankResult] = []
    for rank_idx, (rid, score) in enumerate(sorted_items[:top_k], start=1):
        results.append(
            RankResult(
                rug_id=rid,
                score=round(float(score), 5),
                rank=rank_idx,
                system_name="hybrid_rrf",
                explanation=f"RRF fusion (k={k_rrf}): {', '.join(provenance.get(rid, []))}",
            )
        )
    return results


def weighted_linear_fusion(
    system_rankings: dict[str, list[RankResult]],
    weights: dict[str, float],
    top_k: int = 10,
) -> list[RankResult]:
    """Combine rankings using Min-Max normalized linear weighted fusion."""
    # Normalize scores per system to [0, 1]
    norm_scores: dict[str, dict[str, float]] = {}
    all_rug_ids: set[str] = set()

    for sys_name, ranked_items in system_rankings.items():
        sys_dict: dict[str, float] = {}
        if ranked_items:
            raw_scores = [r.score for r in ranked_items]
            min_s, max_s = min(raw_scores), max(raw_scores)
            span = max_s - min_s if max_s > min_s else 1.0
            for r in ranked_items:
                sys_dict[r.rug_id] = (r.score - min_s) / span
                all_rug_ids.add(r.rug_id)
        norm_scores[sys_name] = sys_dict

    total_weight = sum(weights.values()) if sum(weights.values()) > 0 else 1.0
    fused_scores: dict[str, float] = {}

    for rid in all_rug_ids:
        score = 0.0
        for sys_name, w in weights.items():
            sc = norm_scores.get(sys_name, {}).get(rid, 0.0)
            score += (w / total_weight) * sc
        fused_scores[rid] = score

    sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    results: list[RankResult] = []
    for rank_idx, (rid, score) in enumerate(sorted_items[:top_k], start=1):
        results.append(
            RankResult(
                rug_id=rid,
                score=round(float(score), 4),
                rank=rank_idx,
                system_name="hybrid_linear",
                explanation=f"Linear fusion score: {score:.4f}",
            )
        )
    return results


class HybridRetriever(BaseRanker):
    """Hybrid ranker fusing dense embeddings, BM25 text, and metadata constraints."""

    name = "hybrid_retriever"

    def __init__(
        self,
        dense_ranker: BaseRanker,
        bm25_ranker: BaseRanker | None = None,
        attr_ranker: BaseRanker | None = None,
        fusion_mode: str = "rrf",  # 'rrf' or 'linear'
        weights: dict[str, float] | None = None,
        k_rrf: int = 60,
    ) -> None:
        self.dense_ranker = dense_ranker
        self.bm25_ranker = bm25_ranker or BM25Baseline()
        self.attr_ranker = attr_ranker or AttributeFilterBaseline()
        self.fusion_mode = fusion_mode
        self.weights = weights or {"dense": 0.50, "bm25": 0.25, "attr": 0.25}
        self.k_rrf = k_rrf

    def rank(
        self,
        query: Query,
        candidates: list[CandidateRug],
        top_k: int = 10,
    ) -> list[RankResult]:
        pool_k = min(len(candidates), max(top_k * 3, 20))

        dense_ranked = self.dense_ranker.rank(query, candidates, top_k=pool_k)
        bm25_ranked = self.bm25_ranker.rank(query, candidates, top_k=pool_k)
        attr_ranked = self.attr_ranker.rank(query, candidates, top_k=pool_k)

        sub_systems = {
            "dense": dense_ranked,
            "bm25": bm25_ranked,
            "attr": attr_ranked,
        }

        if self.fusion_mode == "rrf":
            return reciprocal_rank_fusion(
                sub_systems,
                weights=self.weights,
                k_rrf=self.k_rrf,
                top_k=top_k,
            )
        else:
            return weighted_linear_fusion(
                sub_systems,
                weights=self.weights,
                top_k=top_k,
            )


def tune_fusion_weights_on_val(
    val_queries: list[Query],
    val_candidates: list[CandidateRug],
    val_gold_labels: dict[str, dict[str, float]],
    dense_ranker: BaseRanker,
    fusion_mode: str = "rrf",
) -> tuple[dict[str, float], float]:
    """Tuning fusion weights strictly on the validation split to maximize NDCG@5.

    ZERO LEAKAGE: Never evaluates on test queries!
    """
    from src.eval.metrics import evaluate_rankings

    weight_candidates = [
        {"dense": 0.80, "bm25": 0.10, "attr": 0.10},
        {"dense": 0.60, "bm25": 0.20, "attr": 0.20},
        {"dense": 0.50, "bm25": 0.25, "attr": 0.25},
        {"dense": 0.40, "bm25": 0.30, "attr": 0.30},
        {"dense": 0.33, "bm25": 0.33, "attr": 0.33},
    ]

    best_weights = weight_candidates[0]
    best_ndcg5 = -1.0

    for weights in weight_candidates:
        retriever = HybridRetriever(
            dense_ranker=dense_ranker,
            fusion_mode=fusion_mode,
            weights=weights,
        )

        sys_rankings = {}
        for q in val_queries:
            ranked = retriever.rank(q, val_candidates, top_k=10)
            sys_rankings[q.query_id] = [r.rug_id for r in ranked]

        eval_res = evaluate_rankings(sys_rankings, val_gold_labels, k_values=[5])
        ndcg5 = eval_res["means"].get("ndcg@5", 0.0)

        if ndcg5 > best_ndcg5:
            best_ndcg5 = ndcg5
            best_weights = weights

    logger.info("Validation weight tuning complete: best_weights=%s, val_ndcg@5=%.4f", best_weights, best_ndcg5)
    return best_weights, best_ndcg5
