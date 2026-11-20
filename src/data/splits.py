from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)


def create_group_splits(
    pairs_df: pd.DataFrame,
    rug_col: str = "rug_id",
    room_col: str = "room_id",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create strict group-aware train/val/test splits ensuring zero leakage
    across both rug entities and room entities.
    """
    assert abs((train_size + val_size + test_size) - 1.0) < 1e-5, "Split fractions must sum to 1.0"

    # Step 1: Split rugs into train_rugs, val_rugs, test_rugs
    unique_rugs = pd.Series(pairs_df[rug_col].unique())
    gss1 = GroupShuffleSplit(n_splits=1, test_size=(val_size + test_size), random_state=random_state)
    train_rug_idx, temp_rug_idx = next(gss1.split(unique_rugs, groups=unique_rugs))

    train_rugs = set(unique_rugs.iloc[train_rug_idx])
    temp_rugs = unique_rugs.iloc[temp_rug_idx]

    val_ratio_in_temp = val_size / (val_size + test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=(1.0 - val_ratio_in_temp), random_state=random_state)
    val_rug_idx, test_rug_idx = next(gss2.split(temp_rugs, groups=temp_rugs))

    val_rugs = set(temp_rugs.iloc[val_rug_idx])
    test_rugs = set(temp_rugs.iloc[test_rug_idx])

    # Step 2: Split rooms similarly into disjoint sets
    unique_rooms = pd.Series(pairs_df[room_col].unique())
    gss3 = GroupShuffleSplit(n_splits=1, test_size=(val_size + test_size), random_state=random_state + 1)
    train_room_idx, temp_room_idx = next(gss3.split(unique_rooms, groups=unique_rooms))

    train_rooms = set(unique_rooms.iloc[train_room_idx])
    temp_rooms = unique_rooms.iloc[temp_room_idx]

    gss4 = GroupShuffleSplit(n_splits=1, test_size=(1.0 - val_ratio_in_temp), random_state=random_state + 1)
    val_room_idx, test_room_idx = next(gss4.split(temp_rooms, groups=temp_rooms))

    val_rooms = set(temp_rooms.iloc[val_room_idx])
    test_rooms = set(temp_rooms.iloc[test_room_idx])

    # Step 3: Assign pairs strictly where both rug and room belong to the partition
    train_df = pairs_df[pairs_df[rug_col].isin(train_rugs) & pairs_df[room_col].isin(train_rooms)].copy()
    val_df = pairs_df[pairs_df[rug_col].isin(val_rugs) & pairs_df[room_col].isin(val_rooms)].copy()
    test_df = pairs_df[pairs_df[rug_col].isin(test_rugs) & pairs_df[room_col].isin(test_rooms)].copy()

    logger.info(
        f"Splits created: Train={len(train_df)} pairs ({len(train_rugs)} rugs, {len(train_rooms)} rooms), "
        f"Val={len(val_df)} pairs ({len(val_rugs)} rugs, {len(val_rooms)} rooms), "
        f"Test={len(test_df)} pairs ({len(test_rugs)} rugs, {len(test_rooms)} rooms)"
    )

    return train_df, val_df, test_df


def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rug_col: str = "rug_id",
    room_col: str = "room_id",
) -> dict[str, Any]:
    """Assert zero leakage across train, validation, and test splits."""
    train_rugs = set(train_df[rug_col].dropna().unique())
    val_rugs = set(val_df[rug_col].dropna().unique())
    test_rugs = set(test_df[rug_col].dropna().unique())

    train_rooms = set(train_df[room_col].dropna().unique())
    val_rooms = set(val_df[room_col].dropna().unique())
    test_rooms = set(test_df[room_col].dropna().unique())

    rug_overlap_train_test = train_rugs.intersection(test_rugs)
    rug_overlap_train_val = train_rugs.intersection(val_rugs)
    rug_overlap_val_test = val_rugs.intersection(test_rugs)

    room_overlap_train_test = train_rooms.intersection(test_rooms)
    room_overlap_train_val = train_rooms.intersection(val_rooms)
    room_overlap_val_test = val_rooms.intersection(test_rooms)

    has_leakage = any(
        [
            rug_overlap_train_test,
            rug_overlap_train_val,
            rug_overlap_val_test,
            room_overlap_train_test,
            room_overlap_train_val,
            room_overlap_val_test,
        ]
    )

    report = {
        "has_leakage": bool(has_leakage),
        "rug_counts": {"train": len(train_rugs), "val": len(val_rugs), "test": len(test_rugs)},
        "room_counts": {"train": len(train_rooms), "val": len(val_rooms), "test": len(test_rooms)},
        "overlaps": {
            "rug_train_test": len(rug_overlap_train_test),
            "rug_train_val": len(rug_overlap_train_val),
            "rug_val_test": len(rug_overlap_val_test),
            "room_train_test": len(room_overlap_train_test),
            "room_train_val": len(room_overlap_train_val),
            "room_val_test": len(room_overlap_val_test),
        },
    }

    if has_leakage:
        raise ValueError(f"CRITICAL: Data leakage detected across groups! Overlaps: {report['overlaps']}")

    return report
