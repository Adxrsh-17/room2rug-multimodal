import pandas as pd
import pytest

from src.data.splits import create_group_splits, verify_no_leakage


def test_group_split_zero_leakage():
    # Synthetic multi-item pairs dataset
    rugs = [f"rug_{i}" for i in range(100)]
    rooms = [f"room_{j}" for j in range(80)]

    pairs = []
    for r in rugs:
        for rm in rooms[:10]:
            pairs.append({"rug_id": r, "room_id": rm, "relevance": 1})

    df = pd.DataFrame(pairs)
    train_df, val_df, test_df = create_group_splits(df, rug_col="rug_id", room_col="room_id")

    report = verify_no_leakage(train_df, val_df, test_df, rug_col="rug_id", room_col="room_id")
    assert not report["has_leakage"]
    assert report["overlaps"]["rug_train_test"] == 0
    assert report["overlaps"]["room_train_test"] == 0
    assert report["overlaps"]["rug_train_val"] == 0
    assert report["overlaps"]["room_train_val"] == 0


def test_leakage_detector_raises_on_leak():
    train_df = pd.DataFrame([{"rug_id": "rug_1", "room_id": "room_A"}])
    val_df = pd.DataFrame([{"rug_id": "rug_2", "room_id": "room_B"}])
    test_df = pd.DataFrame([{"rug_id": "rug_1", "room_id": "room_C"}])  # Leaked rug_1!

    with pytest.raises(ValueError, match="Data leakage detected"):
        verify_no_leakage(train_df, val_df, test_df)
