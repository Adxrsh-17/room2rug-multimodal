"""Unit tests for hybrid retrieval, RRF, and late fusion."""

from src.eval.base import CandidateRug, Query, RankResult
from src.eval.baselines import AttributeFilterBaseline
from src.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion, weighted_linear_fusion


def test_reciprocal_rank_fusion():
    sys1 = [
        RankResult(rug_id="r1", score=0.9, rank=1),
        RankResult(rug_id="r2", score=0.7, rank=2),
    ]
    sys2 = [
        RankResult(rug_id="r2", score=0.95, rank=1),
        RankResult(rug_id="r1", score=0.85, rank=2),
    ]

    # Both systems rank r1 and r2 in top 2 (symmetric ties)
    fused = reciprocal_rank_fusion({"sys1": sys1, "sys2": sys2}, k_rrf=60, top_k=2)
    assert len(fused) == 2
    assert fused[0].rank == 1
    assert fused[1].rank == 2
    assert fused[0].score == fused[1].score  # 1/61 + 1/62 for both


def test_weighted_linear_fusion():
    sys1 = [
        RankResult(rug_id="r1", score=10.0, rank=1),
        RankResult(rug_id="r2", score=0.0, rank=2),
    ]
    sys2 = [
        RankResult(rug_id="r1", score=1.0, rank=1),
        RankResult(rug_id="r2", score=0.0, rank=2),
    ]
    weights = {"sys1": 0.5, "sys2": 0.5}
    fused = weighted_linear_fusion({"sys1": sys1, "sys2": sys2}, weights=weights, top_k=2)
    assert fused[0].rug_id == "r1"
    assert fused[0].score == 1.0
    assert fused[1].rug_id == "r2"
    assert fused[1].score == 0.0


def test_hybrid_retriever_interface():
    attr_ranker = AttributeFilterBaseline()
    hybrid = HybridRetriever(dense_ranker=attr_ranker, fusion_mode="rrf")

    cands = [
        CandidateRug(rug_id="r1", title="Red Persian Rug", color="Red", style="Traditional"),
        CandidateRug(rug_id="r2", title="Blue Geometric Rug", color="Blue", style="Modern"),
    ]
    q = Query(query_id="q1", text_query="red traditional rug")
    results = hybrid.rank(q, cands, top_k=2)

    assert len(results) == 2
    assert results[0].rug_id == "r1"
    assert results[0].rank == 1
