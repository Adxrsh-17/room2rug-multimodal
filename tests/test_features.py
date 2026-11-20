"""Unit tests for LTR feature extraction."""

from src.eval.base import CandidateRug, Query, RankResult
from src.rerank.features import FEATURE_NAMES, RerankingFeatureExtractor


def test_feature_extractor_pair_and_matrix():
    extractor = RerankingFeatureExtractor()

    q = Query(
        query_id="q1",
        text_query="8x10 beige modern living room rug",
        scene_type="living_room",
    )
    rug_dec = CandidateRug(
        rug_id="r_dec",
        title="Modern Beige Wool Rug",
        color="Beige",
        style="Modern",
        sizes=["8x10"],
        tags="modern, beige, wool",
        price=250.0,
    )
    rug_pad = CandidateRug(
        rug_id="r_pad",
        title="Non-Slip Grip Rug Pad",
        color="Grey",
        style="Rug Pad",
        sizes=["8x10"],
        tags="pad, gripper",
        price=30.0,
    )

    stage1_res = RankResult(rug_id="r_dec", score=0.85, rank=1)
    feats_dec = extractor.extract_pair_features(q, rug_dec, stage1_res=stage1_res)

    # Check all feature names present
    for fname in FEATURE_NAMES:
        assert fname in feats_dec

    assert feats_dec["is_rug_pad"] == 0.0
    assert feats_dec["color_exact_match"] == 1.0
    assert feats_dec["size_exact_match"] == 1.0
    assert feats_dec["stage1_rank"] == 1.0

    feats_pad = extractor.extract_pair_features(q, rug_pad)
    assert feats_pad["is_rug_pad"] == 1.0

    # Test matrix extraction
    X, rug_ids = extractor.extract_matrix_for_query(q, [rug_dec, rug_pad])
    assert X.shape == (2, len(FEATURE_NAMES))
    assert rug_ids == ["r_dec", "r_pad"]
