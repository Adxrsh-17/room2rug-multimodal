from pathlib import Path

import pytest

from src.utils.config import load_config


def test_load_default_config():
    config_path = Path("configs/default.yaml")
    config = load_config(config_path)

    assert isinstance(config, dict)
    assert config["seed"] == 42
    assert "data" in config
    assert "baseline_v1" in config
    assert config["baseline_v1"]["top_k"] == 10
    assert config["baseline_v1"]["weights"]["fusion_mode_b_image"] == 0.6


def test_load_baseline_v1_config():
    config_path = Path("configs/baseline_v1.yaml")
    config = load_config(config_path)

    assert isinstance(config, dict)
    assert config["model"]["name"] == "baseline_v1_heuristic_clip"
    assert config["retrieval"]["fusion_weights"]["image"] == 0.6
    assert config["retrieval"]["fusion_weights"]["text"] == 0.4


def test_load_nonexistent_config():
    with pytest.raises(FileNotFoundError):
        load_config("configs/nonexistent.yaml")
