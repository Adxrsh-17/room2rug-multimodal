"""Unit tests for CIELAB transformations and color palette distances."""

import numpy as np

from src.rerank.palette import (
    cielab_delta_e_distance,
    extract_dominant_cielab_palette,
    rgb_to_cielab,
)


def test_rgb_to_cielab_known_values():
    # Pure white [255, 255, 255] -> L* = 100.0, a* = 0.0, b* = 0.0
    white_rgb = np.array([255, 255, 255], dtype=np.uint8)
    white_lab = rgb_to_cielab(white_rgb)
    assert abs(white_lab[0] - 100.0) < 0.5
    assert abs(white_lab[1]) < 0.5
    assert abs(white_lab[2]) < 0.5

    # Pure black [0, 0, 0] -> L* = 0.0, a* = 0.0, b* = 0.0
    black_rgb = np.array([0, 0, 0], dtype=np.uint8)
    black_lab = rgb_to_cielab(black_rgb)
    assert abs(black_lab[0]) < 0.1
    assert abs(black_lab[1]) < 0.1
    assert abs(black_lab[2]) < 0.1


def test_cielab_delta_e_distance():
    # Distance between identical palettes must be 0.0
    p1 = np.array([[50.0, 10.0, 20.0]])
    min_d, w_d = cielab_delta_e_distance(p1, p1)
    assert min_d == 0.0
    assert w_d == 0.0

    # Distance between white and black is approx 100
    white_lab = np.array([[100.0, 0.0, 0.0]])
    black_lab = np.array([[0.0, 0.0, 0.0]])
    min_d_wb, _ = cielab_delta_e_distance(white_lab, black_lab)
    assert abs(min_d_wb - 100.0) < 0.5


def test_extract_dominant_cielab_palette():
    # Synthetic 2-color image: half red, half blue
    img_arr = np.zeros((20, 20, 3), dtype=np.uint8)
    img_arr[:10, :] = [255, 0, 0]
    img_arr[10:, :] = [0, 0, 255]

    centroids, weights = extract_dominant_cielab_palette(img_arr, n_colors=2)
    assert len(centroids) == 2
    assert len(weights) == 2
    assert abs(sum(weights) - 1.0) < 1e-4
