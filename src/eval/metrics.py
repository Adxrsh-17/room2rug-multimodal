"""Evaluation metrics suite for information retrieval and ranking.

Includes:
- Graded relevance NDCG@k
- Binary & graded Recall@k, MRR@k, MAP@k
- Bootstrap 95% Confidence Intervals (cluster bootstrap by query)
- Two-sided paired permutation significance testing
- Slice evaluation (by scene category, query length, etc.)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def dcg_at_k(relevances: list[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank k with 2^rel - 1 gain."""
    k = min(k, len(relevances))
    if k == 0:
        return 0.0
    dcg = 0.0
    for i in range(k):
        rel = relevances[i]
        dcg += (math.pow(2, rel) - 1.0) / math.log2(i + 2)  # i+2 because i starts at 0 -> rank 1
    return dcg


def idcg_at_k(gold_relevances: list[float], k: int) -> float:
    """Compute Ideal Discounted Cumulative Gain at rank k."""
    sorted_rels = sorted(gold_relevances, reverse=True)
    return dcg_at_k(sorted_rels, k)


def ndcg_at_k(
    ranked_ids: list[str],
    gold_relevances: dict[str, float],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k.

    Args:
        ranked_ids: List of retrieved item IDs in predicted rank order.
        gold_relevances: Dict mapping item_id -> relevance score (e.g. 0.0, 1.0, 2.0).
        k: Cutoff rank.
    """
    if not gold_relevances:
        return 0.0

    retrieved_rels = [gold_relevances.get(item_id, 0.0) for item_id in ranked_ids[:k]]
    all_gold_rels = list(gold_relevances.values())

    idcg = idcg_at_k(all_gold_rels, k)
    if idcg <= 0.0:
        return 0.0

    dcg = dcg_at_k(retrieved_rels, k)
    return dcg / idcg


def recall_at_k(
    ranked_ids: list[str],
    gold_relevances: dict[str, float],
    k: int = 10,
    rel_threshold: float = 1.0,
) -> float:
    """Compute Recall at rank k (fraction of relevant items retrieved in top k)."""
    relevant_gold = {item_id for item_id, rel in gold_relevances.items() if rel >= rel_threshold}
    if not relevant_gold:
        return 0.0

    retrieved_top_k = set(ranked_ids[:k])
    hits = len(retrieved_top_k.intersection(relevant_gold))
    return hits / len(relevant_gold)


def mrr_at_k(
    ranked_ids: list[str],
    gold_relevances: dict[str, float],
    k: int = 10,
    rel_threshold: float = 1.0,
) -> float:
    """Compute Reciprocal Rank at rank k (reciprocal of rank of first relevant item)."""
    for rank_idx, item_id in enumerate(ranked_ids[:k], start=1):
        if gold_relevances.get(item_id, 0.0) >= rel_threshold:
            return 1.0 / rank_idx
    return 0.0


def map_at_k(
    ranked_ids: list[str],
    gold_relevances: dict[str, float],
    k: int = 10,
    rel_threshold: float = 1.0,
) -> float:
    """Compute Mean Average Precision at rank k."""
    relevant_gold = {item_id for item_id, rel in gold_relevances.items() if rel >= rel_threshold}
    if not relevant_gold:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for rank_idx, item_id in enumerate(ranked_ids[:k], start=1):
        if item_id in relevant_gold:
            hits += 1
            sum_precisions += hits / rank_idx

    denominator = min(len(relevant_gold), k)
    return sum_precisions / denominator if denominator > 0 else 0.0


def compute_query_metrics(
    ranked_ids: list[str],
    gold_relevances: dict[str, float],
    k_values: list[int] | None = None,
    rel_threshold: float = 1.0,
) -> dict[str, float]:
    """Compute full metric suite for a single query."""
    if k_values is None:
        k_values = [1, 3, 5, 10]

    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, gold_relevances, k=k)
        metrics[f"recall@{k}"] = recall_at_k(ranked_ids, gold_relevances, k=k, rel_threshold=rel_threshold)

    metrics["mrr@10"] = mrr_at_k(ranked_ids, gold_relevances, k=10, rel_threshold=rel_threshold)
    metrics["map@10"] = map_at_k(ranked_ids, gold_relevances, k=10, rel_threshold=rel_threshold)
    return metrics


def evaluate_rankings(
    system_rankings: dict[str, list[str]],
    gold_labels: dict[str, dict[str, float]],
    k_values: list[int] | None = None,
    rel_threshold: float = 1.0,
) -> dict[str, Any]:
    """Evaluate a system across all queries.

    Args:
        system_rankings: query_id -> list of retrieved item IDs.
        gold_labels: query_id -> {item_id: relevance_score}.
        k_values: List of k cutoffs.
        rel_threshold: Threshold for binary relevance.

    Returns:
        Dict containing aggregated mean metrics and per-query vectors.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    query_ids = [qid for qid in gold_labels if qid in system_rankings]
    if not query_ids:
        return {"n_queries": 0, "means": {}, "per_query": {}}

    per_query: dict[str, list[float]] = {f"ndcg@{k}": [] for k in k_values}
    for k in k_values:
        per_query[f"recall@{k}"] = []
    per_query["mrr@10"] = []
    per_query["map@10"] = []

    for qid in query_ids:
        ranked = system_rankings[qid]
        gold = gold_labels[qid]
        q_metrics = compute_query_metrics(ranked, gold, k_values=k_values, rel_threshold=rel_threshold)
        for m_name, val in q_metrics.items():
            per_query[m_name].append(val)

    means = {m_name: float(np.mean(vals)) for m_name, vals in per_query.items()}

    return {
        "n_queries": len(query_ids),
        "query_ids": query_ids,
        "means": means,
        "per_query": per_query,
    }


def compute_bootstrap_ci(
    values: list[float],
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval using percentile method.

    Args:
        values: Per-query metric values.
        n_bootstraps: Number of bootstrap resamples.
        ci: Confidence level (default 0.95 for 95% CI).
        seed: Random seed for exact reproducibility.

    Returns:
        (mean, ci_lower, ci_upper)
    """
    if not values:
        return 0.0, 0.0, 0.0

    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean_val = float(np.mean(arr))

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_bootstraps, dtype=np.float64)

    for b in range(n_bootstraps):
        sample_indices = rng.randint(0, n, size=n)
        boot_means[b] = np.mean(arr[sample_indices])

    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))

    return mean_val, ci_lower, ci_upper


def paired_permutation_test(
    values_a: list[float],
    values_b: list[float],
    n_permutations: int = 2000,
    seed: int = 42,
) -> float:
    """Compute two-sided paired permutation significance test p-value.

    Null hypothesis: no difference in metric distribution between System A and System B.
    """
    assert len(values_a) == len(values_b), "Systems must be evaluated on identical queries"
    n = len(values_a)
    if n == 0:
        return 1.0

    diffs = np.array(values_a, dtype=np.float64) - np.array(values_b, dtype=np.float64)
    observed_mean_diff = abs(float(np.mean(diffs)))

    if observed_mean_diff == 0.0:
        return 1.0

    rng = np.random.RandomState(seed)
    # Generate random +1 / -1 sign flips
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    permuted_diff_means = np.abs(np.mean(diffs * signs, axis=1))

    # P-value is fraction of permuted statistics >= observed
    p_val = float(np.mean(permuted_diff_means >= observed_mean_diff))
    return p_val


def evaluate_slices(
    system_rankings: dict[str, list[str]],
    gold_labels: dict[str, dict[str, float]],
    query_slices: dict[str, str],  # query_id -> slice_name (e.g. "living_room", "bedroom")
    target_metric: str = "ndcg@5",
    k_values: list[int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute slice-level performance breakdown.

    Returns:
        dict: slice_name -> {"n_queries": int, "mean": float, "ci_low": float, "ci_high": float}
    """
    slices: dict[str, list[str]] = {}
    for qid, slice_name in query_slices.items():
        if qid in gold_labels and qid in system_rankings:
            slices.setdefault(slice_name, []).append(qid)

    slice_reports: dict[str, dict[str, Any]] = {}
    for slice_name, qids in slices.items():
        slice_rankings = {qid: system_rankings[qid] for qid in qids}
        slice_gold = {qid: gold_labels[qid] for qid in qids}

        eval_res = evaluate_rankings(slice_rankings, slice_gold, k_values=k_values)
        vals = eval_res["per_query"].get(target_metric, [])
        mean, ci_low, ci_high = compute_bootstrap_ci(vals)

        slice_reports[slice_name] = {
            "n_queries": len(qids),
            "metric": target_metric,
            "mean": round(mean, 4),
            "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        }

    return slice_reports
