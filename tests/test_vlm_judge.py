"""Unit tests for VLM-as-Judge labeler and validation logic."""

from src.eval.vlm_judge import OfflineVLMJudge, validate_vlm_against_gold


def test_offline_vlm_judge_rubric_adherence():
    judge = OfflineVLMJudge(seed=42)

    # High match test
    res_high = judge.judge_pair(
        query_id="q_liv",
        room_image_path=None,
        scene_type="living_room",
        query_text="8x10 warm beige contemporary living room rug",
        rug_id="rug_beige",
        rug_title="Soft Wool Beige Contemporary Rug",
        rug_color="Beige",
        rug_style="Contemporary",
        rug_tags="contemporary, warm, beige, 8x10",
    )
    assert res_high.is_weak_label is True  # Enforced invariant
    assert res_high.rating in [1, 2]
    assert res_high.confidence > 0.5
    assert len(res_high.reasoning) > 10

    # Low match test (rug pad in decorative living room)
    res_low = judge.judge_pair(
        query_id="q_liv",
        room_image_path=None,
        scene_type="living_room",
        query_text="colorful bohemian living room rug",
        rug_id="rug_pad",
        rug_title="Non-Slip Gripper Rug Pad",
        rug_color="Grey",
        rug_style="Rug Pad",
        rug_tags="pad, gripper",
    )
    assert res_low.rating == 0


def test_validate_vlm_against_gold():
    gold = {
        ("q1", "r1"): 2,
        ("q1", "r2"): 1,
        ("q1", "r3"): 0,
        ("q2", "r4"): 2,
        ("q2", "r5"): 0,
    }
    vlm = {
        ("q1", "r1"): 2,
        ("q1", "r2"): 1,
        ("q1", "r3"): 0,
        ("q2", "r4"): 1,  # minor disagreement
        ("q2", "r5"): 0,
    }

    rep = validate_vlm_against_gold(vlm, gold)
    assert rep["n_pairs"] == 5
    assert rep["exact_agreement"] == 0.8
    assert rep["adjacent_agreement"] == 1.0
    assert rep["spearman_rho"] > 0.8
    assert rep["severe_disagreements"]["vlm_overoptimistic (gold=0, vlm=2)"] == 0
