"""Unit tests for multi-system candidate pooling."""

from src.eval.base import BaseRanker, CandidateRug, Query, RankResult
from src.eval.pooling import pool_candidates


class MockRankerA(BaseRanker):
    name = "mock_a"

    def rank(self, query: Query, candidates: list[CandidateRug], top_k: int = 5) -> list[RankResult]:
        return [
            RankResult(rug_id=f"rug_{i}", score=1.0 - (i * 0.1), rank=i + 1, system_name=self.name)
            for i in range(min(top_k, len(candidates)))
        ]


class MockRankerB(BaseRanker):
    name = "mock_b"

    def rank(self, query: Query, candidates: list[CandidateRug], top_k: int = 5) -> list[RankResult]:
        # Ranks in reverse
        reversed_cands = list(reversed(candidates))
        return [
            RankResult(rug_id=r.rug_id, score=1.0 - (i * 0.1), rank=i + 1, system_name=self.name)
            for i, r in enumerate(reversed_cands[:top_k])
        ]


def test_candidate_pooling_deduplication_and_provenance():
    cands = [CandidateRug(rug_id=f"rug_{i}") for i in range(10)]
    q = Query(query_id="q_test", text_query="neutral rug")

    systems = [MockRankerA(), MockRankerB()]
    pool = pool_candidates(q, cands, systems, pool_depth_per_system=3, seed=42)

    # Ranker A gives rug_0, rug_1, rug_2
    # Ranker B gives rug_9, rug_8, rug_7
    assert pool["query_id"] == "q_test"
    assert pool["n_unique_candidates"] == 6
    assert set(pool["pooled_rug_ids"]) == {"rug_0", "rug_1", "rug_2", "rug_7", "rug_8", "rug_9"}

    # Provenance tracking
    assert "mock_a" in pool["provenance"]["rug_0"]
    assert "mock_b" in pool["provenance"]["rug_9"]
