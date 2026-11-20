"""Unit tests for IR evaluation metrics, bootstrap CIs, and significance testing."""

import math

from src.eval.metrics import (
    compute_bootstrap_ci,
    dcg_at_k,
    evaluate_rankings,
    evaluate_slices,
    idcg_at_k,
    map_at_k,
    mrr_at_k,
    ndcg_at_k,
    paired_permutation_test,
    recall_at_k,
)


def test_dcg_and_idcg_known_values():
    # Relevances: [2.0, 1.0, 0.0]
    # DCG = (2^2 - 1)/log2(2) + (2^1 - 1)/log2(3) + (2^0 - 1)/log2(4)
    #     = 3/1 + 1/1.5849625 + 0 = 3.0 + 0.63092975 = 3.63093
    rels = [2.0, 1.0, 0.0]
    expected_dcg = 3.0 + (1.0 / math.log2(3))
    assert abs(dcg_at_k(rels, k=3) - expected_dcg) < 1e-5

    # IDCG of same list is identical
    assert abs(idcg_at_k(rels, k=3) - expected_dcg) < 1e-5


def test_ndcg_perfect_and_worst():
    gold = {"doc1": 2.0, "doc2": 1.0, "doc3": 0.0}

    # Perfect ranking: doc1, doc2, doc3
    assert abs(ndcg_at_k(["doc1", "doc2", "doc3"], gold, k=3) - 1.0) < 1e-5

    # Reverse ranking: doc3, doc2, doc1
    ndcg_rev = ndcg_at_k(["doc3", "doc2", "doc1"], gold, k=3)
    assert 0.0 < ndcg_rev < 1.0

    # All irrelevant retrieved
    assert ndcg_at_k(["docX", "docY"], gold, k=2) == 0.0


def test_recall_at_k():
    gold = {"doc1": 2.0, "doc2": 1.0, "doc3": 0.0}
    # Relevant items (rel >= 1.0): doc1, doc2 (total 2)

    assert recall_at_k(["doc1"], gold, k=1, rel_threshold=1.0) == 0.5
    assert recall_at_k(["doc1", "doc2"], gold, k=2, rel_threshold=1.0) == 1.0
    assert recall_at_k(["doc3"], gold, k=1, rel_threshold=1.0) == 0.0
    assert recall_at_k(["doc1", "doc3"], gold, k=2, rel_threshold=1.0) == 0.5


def test_mrr_at_k():
    gold = {"doc1": 0.0, "doc2": 2.0, "doc3": 1.0}

    # First relevant at rank 2 -> RR = 0.5
    assert mrr_at_k(["doc1", "doc2", "doc3"], gold, k=3, rel_threshold=1.0) == 0.5

    # First relevant at rank 1 -> RR = 1.0
    assert mrr_at_k(["doc2", "doc1"], gold, k=2, rel_threshold=1.0) == 1.0

    # No relevant retrieved
    assert mrr_at_k(["doc1", "docX"], gold, k=2, rel_threshold=1.0) == 0.0


def test_map_at_k():
    gold = {"doc1": 1.0, "doc2": 2.0, "doc3": 0.0}
    # Relevant: doc1, doc2 (total 2)
    # Order: doc1 (hit, P@1=1.0), doc3 (miss, P@2=0.5), doc2 (hit, P@3=2/3)
    # AP = (1.0 + 2/3) / 2 = 1.6666 / 2 = 0.8333
    expected_ap = (1.0 + (2.0 / 3.0)) / 2.0
    val = map_at_k(["doc1", "doc3", "doc2"], gold, k=3, rel_threshold=1.0)
    assert abs(val - expected_ap) < 1e-5


def test_bootstrap_ci():
    values = [0.8, 0.85, 0.9, 0.75, 0.88, 0.92, 0.79, 0.84]
    mean, ci_low, ci_high = compute_bootstrap_ci(values, n_bootstraps=500, seed=42)

    sample_mean = sum(values) / len(values)
    assert abs(mean - sample_mean) < 1e-5
    assert ci_low <= mean <= ci_high
    assert 0.70 < ci_low < 0.85
    assert 0.83 < ci_high < 0.95


def test_paired_permutation_test():
    # Identical systems -> p-value = 1.0
    sys_a = [0.8, 0.9, 0.7, 0.85]
    sys_b = [0.8, 0.9, 0.7, 0.85]
    p_same = paired_permutation_test(sys_a, sys_b, n_permutations=500, seed=42)
    assert p_same == 1.0

    # System A strongly dominates System B -> p-value < 0.05
    dom_a = [0.95, 0.92, 0.98, 0.90, 0.94, 0.96, 0.93, 0.97]
    dom_b = [0.20, 0.25, 0.15, 0.30, 0.22, 0.18, 0.28, 0.24]
    p_diff = paired_permutation_test(dom_a, dom_b, n_permutations=500, seed=42)
    assert p_diff <= 0.05


def test_evaluate_rankings_and_slices():
    gold = {
        "q1": {"r1": 2.0, "r2": 1.0, "r3": 0.0},
        "q2": {"r4": 2.0, "r5": 0.0},
    }
    system_ranks = {
        "q1": ["r1", "r2", "r3"],
        "q2": ["r5", "r4"],
    }
    res = evaluate_rankings(system_ranks, gold, k_values=[1, 3])
    assert res["n_queries"] == 2
    assert "ndcg@3" in res["means"]
    assert "recall@3" in res["means"]

    # Slices
    slices = {"q1": "living_room", "q2": "bedroom"}
    slice_rep = evaluate_slices(system_ranks, gold, slices, target_metric="ndcg@3", k_values=[1, 3])
    assert "living_room" in slice_rep
    assert "bedroom" in slice_rep
    assert slice_rep["living_room"]["mean"] > slice_rep["bedroom"]["mean"]
