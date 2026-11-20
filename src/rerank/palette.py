"""CIELAB color space transformation and palette distance computation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger(__name__)


def rgb_to_cielab(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB array [0, 255] to CIELAB space [L*, a*, b*] under D65 illuminant.

    Args:
        rgb: (..., 3) array of RGB values in range [0, 255].

    Returns:
        (..., 3) array of [L*, a*, b*] values.
    """
    arr = np.asarray(rgb, dtype=np.float64) / 255.0

    # 1. sRGB gamma correction to linear RGB
    mask = arr > 0.04045
    linear = np.empty_like(arr)
    linear[mask] = np.power((arr[mask] + 0.055) / 1.055, 2.4)
    linear[~mask] = arr[~mask] / 12.92

    # 2. Linear RGB to XYZ (D65 matrix)
    x = linear[..., 0] * 0.4124564 + linear[..., 1] * 0.3575761 + linear[..., 2] * 0.1804375
    y = linear[..., 0] * 0.2126729 + linear[..., 1] * 0.7151522 + linear[..., 2] * 0.0721750
    z = linear[..., 0] * 0.0193339 + linear[..., 1] * 0.1191920 + linear[..., 2] * 0.9503041

    # Normalize by D65 reference white
    x = x / 0.95047
    y = y / 1.00000
    z = z / 1.08883

    # 3. XYZ to CIELAB
    delta = 6.0 / 29.0
    threshold = delta**3  # ~0.008856

    def f(t):
        t_mask = t > threshold
        out = np.empty_like(t)
        out[t_mask] = np.cbrt(t[t_mask])
        out[~t_mask] = t[~t_mask] / (3.0 * delta**2) + (4.0 / 29.0)
        return out

    fx, fy, fz = f(x), f(y), f(z)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return np.stack([L, a, b], axis=-1)


def extract_dominant_cielab_palette(
    image_source: str | Path | Image.Image | np.ndarray,
    n_colors: int = 3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract dominant color centroids and weights in CIELAB space using K-Means.

    Returns:
        centroids: (n_colors, 3) CIELAB coordinates.
        weights: (n_colors,) normalized frequency weights summing to 1.0.
    """
    try:
        if isinstance(image_source, (str, Path)):
            if not Path(image_source).exists():
                # Fallback neutral palette
                return np.array([[75.0, 5.0, 15.0]]), np.array([1.0])
            img = Image.open(image_source).convert("RGB")
        elif isinstance(image_source, Image.Image):
            img = image_source.convert("RGB")
        elif isinstance(image_source, np.ndarray):
            img = Image.fromarray(image_source.astype(np.uint8)).convert("RGB")
        else:
            return np.array([[75.0, 5.0, 15.0]]), np.array([1.0])

        # Resize for fast clustering
        img = img.resize((64, 64))
        rgb_pixels = np.array(img).reshape(-1, 3)

        kmeans = MiniBatchKMeans(
            n_clusters=n_colors,
            random_state=random_state,
            batch_size=256,
            n_init=1,
        )
        kmeans.fit(rgb_pixels)

        # Count frequencies
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        weights = counts.astype(np.float64) / len(kmeans.labels_)

        # Convert RGB centroids to CIELAB
        lab_centroids = rgb_to_cielab(kmeans.cluster_centers_)

        # Sort descending by frequency
        sort_order = np.argsort(weights)[::-1]
        return lab_centroids[sort_order], weights[sort_order]

    except Exception as e:
        logger.debug("Palette extraction exception: %s. Using default neutral.", e)
        return np.array([[75.0, 5.0, 15.0]]), np.array([1.0])


def cielab_delta_e_distance(
    palette1: np.ndarray,
    palette2: np.ndarray,
    weights1: np.ndarray | None = None,
    weights2: np.ndarray | None = None,
) -> tuple[float, float]:
    """Calculate CIE76 Delta E color distance between two palettes.

    Returns:
        (min_delta_e, weighted_delta_e)
    """
    p1 = np.asarray(palette1)
    p2 = np.asarray(palette2)

    # Compute pairwise Euclidean distance matrix: (N1, N2)
    diff = p1[:, np.newaxis, :] - p2[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    min_delta_e = float(np.min(dist_matrix))

    if weights1 is not None and weights2 is not None:
        # Distance from each cluster in p1 to nearest in p2
        min_to_p2 = np.min(dist_matrix, axis=1)
        weighted_delta_e = float(np.sum(min_to_p2 * weights1))
    else:
        weighted_delta_e = float(np.mean(dist_matrix))

    return round(min_delta_e, 3), round(weighted_delta_e, 3)
