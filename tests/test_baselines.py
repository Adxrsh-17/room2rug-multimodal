"""Unit tests for baseline rankers."""

from src.eval.base import BaseRanker, CandidateRug, Query
from src.eval.baselines import (
    AttributeFilterBaseline,
    BaselineV1Wrapper,
    BM25Baseline,
)


def _make_dummy_candidates() -> list[CandidateRug]:
    return [
        CandidateRug(
            rug_id="r1",
            title="Vintage Persian Red Wool Area Rug",
            color="Red",
            style="Traditional",
            pattern="Oriental",
            sizes=["8x10", "5x7"],
            tags="traditional, wool, red",
            body="Hand-knotted classic oriental carpet.",
        ),
        CandidateRug(
            rug_id="r2",
            title="Contemporary Minimalist Grey Geometric Rug",
            color="Grey",
            style="Modern",
            pattern="Geometric",
            sizes=["8x10"],
            tags="modern, grey, geometric",
            body="Sleek low-pile living room rug.",
        ),
        CandidateRug(
            rug_id="r3",
            title="Non-Slip Cushioned Rug Pad",
            color="Beige",
            style="Rug Pad",
            pattern="Solid",
            sizes=["8x10"],
            tags="rug pad, non-slip",
            body="Polymer coated gripper pad.",
        ),
    ]


def test_attribute_filter_baseline():
    ranker = AttributeFilterBaseline()
    assert isinstance(ranker, BaseRanker)
    candidates = _make_dummy_candidates()
    q = Query(query_id="q1", text_query="8x10 red traditional oriental rug", scene_type="living_room")

    results = ranker.rank(q, candidates, top_k=2)
    assert len(results) == 2
    assert results[0].rug_id == "r1"
    assert results[0].rank == 1
    assert results[0].score > results[1].score


def test_bm25_baseline():
    ranker = BM25Baseline()
    assert isinstance(ranker, BaseRanker)
    candidates = _make_dummy_candidates()
    q = Query(query_id="q2", text_query="grey geometric modern carpet")

    results = ranker.rank(q, candidates, top_k=3)
    assert len(results) == 3
    assert results[0].rug_id == "r2"
    assert results[0].rank == 1


def test_baseline_v1_wrapper():
    wrapper = BaselineV1Wrapper()
    assert isinstance(wrapper, BaseRanker)
    candidates = _make_dummy_candidates()
    q = Query(query_id="q3", text_query="modern grey rug")

    results = wrapper.rank(q, candidates, top_k=2)
    assert len(results) == 2
    assert results[0].rug_id == "r2"
