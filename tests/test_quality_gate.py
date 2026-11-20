"""Unit and regression tests verifying that CI quality gates catch intentional degradations."""

import pandas as pd
import pytest

from scripts.ci_quality_gate import (
    MIN_ALLOWED_NDCG5,
    check_leakage_gate,
    verify_lineage,
)
from src.data.splits import verify_no_leakage


def test_quality_gate_lineage():
    assert verify_lineage(), "Artifact lineage verification must pass on clean repo"


def test_quality_gate_leakage_passes_clean():
    assert check_leakage_gate(), "Leakage gate must pass on valid group-aware splits"


def test_quality_gate_leakage_catches_intentional_leak():
    # Intentionally corrupt splits by leaking a product into test
    train_df = pd.DataFrame([{"rug_id": "item_alpha", "room_id": "r1"}])
    val_df = pd.DataFrame([{"rug_id": "item_beta", "room_id": "r2"}])
    test_df = pd.DataFrame([{"rug_id": "item_alpha", "room_id": "r3"}])  # LEAK!

    with pytest.raises(ValueError, match="Data leakage detected"):
        verify_no_leakage(train_df, val_df, test_df)


def test_quality_gate_ndcg_threshold_constant():
    # Ensure threshold constant matches specification
    assert MIN_ALLOWED_NDCG5 == 0.58
