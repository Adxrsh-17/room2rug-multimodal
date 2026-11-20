"""Inter-rater agreement metrics for human and VLM annotations."""

from __future__ import annotations

from typing import Any

import numpy as np


def cohens_kappa(
    rater1: list[int],
    rater2: list[int],
    weights: str | None = "quadratic",
) -> float:
    """Compute Cohen's kappa coefficient between two raters.

    Args:
        rater1: List of integer ratings (e.g. 0, 1, 2).
        rater2: List of integer ratings from second rater.
        weights: None (unweighted), 'linear', or 'quadratic'.

    Returns:
        float: Kappa statistic between -1.0 and 1.0.
    """
    assert len(rater1) == len(rater2), "Rater arrays must be of identical length"
    if len(rater1) == 0:
        return 0.0

    r1 = np.array(rater1, dtype=int)
    r2 = np.array(rater2, dtype=int)

    categories = np.unique(np.concatenate([r1, r2]))
    if len(categories) <= 1:
        return 1.0  # Perfect trivial agreement

    min_cat, max_cat = int(categories.min()), int(categories.max())
    n_classes = max_cat - min_cat + 1

    # Confusion matrix O
    conf = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(r1, r2):
        conf[a - min_cat, b - min_cat] += 1.0

    n_total = len(r1)
    if n_total == 0:
        return 0.0

    # Marginals
    hist1 = conf.sum(axis=1)
    hist2 = conf.sum(axis=0)

    # Expected matrix E under independence
    expected = np.outer(hist1, hist2) / n_total

    if weights is None:
        p_o = np.trace(conf) / n_total
        p_e = np.trace(expected) / n_total
        if p_e >= 1.0:
            return 1.0
        return float((p_o - p_e) / (1.0 - p_e))

    # Weight matrix
    w = np.zeros((n_classes, n_classes), dtype=np.float64)
    denom = float(n_classes - 1)
    for i in range(n_classes):
        for j in range(n_classes):
            diff = abs(i - j)
            if weights == "linear":
                w[i, j] = diff / denom
            elif weights == "quadratic":
                w[i, j] = (diff / denom) ** 2
            else:
                raise ValueError(f"Unknown weight type: {weights}")

    obs_dist = np.sum(w * conf)
    exp_dist = np.sum(w * expected)

    if exp_dist == 0.0:
        return 1.0
    return float(1.0 - (obs_dist / exp_dist))


def compute_confusion_matrix(
    rater1: list[int],
    rater2: list[int],
    classes: list[int] | None = None,
) -> list[list[int]]:
    """Compute 2D confusion matrix between rater1 (rows) and rater2 (cols)."""
    if classes is None:
        classes = sorted(list(set(rater1).union(set(rater2))))

    class_to_idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    mat = [[0] * n for _ in range(n)]

    for a, b in zip(rater1, rater2):
        if a in class_to_idx and b in class_to_idx:
            mat[class_to_idx[a]][class_to_idx[b]] += 1

    return mat


def agreement_report(
    rater1: list[int],
    rater2: list[int],
    rater1_name: str = "Rater 1",
    rater2_name: str = "Rater 2",
) -> dict[str, Any]:
    """Generate comprehensive inter-rater agreement statistics."""
    assert len(rater1) == len(rater2), "Length mismatch"
    n = len(rater1)
    if n == 0:
        return {"n_pairs": 0}

    r1 = np.array(rater1)
    r2 = np.array(rater2)

    raw_agreement = float(np.mean(r1 == r2))
    adjacent_agreement = float(np.mean(np.abs(r1 - r2) <= 1))

    unweighted_k = cohens_kappa(rater1, rater2, weights=None)
    quadratic_k = cohens_kappa(rater1, rater2, weights="quadratic")

    conf = compute_confusion_matrix(rater1, rater2, classes=[0, 1, 2])

    return {
        "n_pairs": n,
        "rater1": rater1_name,
        "rater2": rater2_name,
        "exact_agreement": round(raw_agreement, 4),
        "adjacent_agreement": round(adjacent_agreement, 4),
        "cohens_kappa_unweighted": round(unweighted_k, 4),
        "cohens_kappa_quadratic": round(quadratic_k, 4),
        "confusion_matrix_3x3": conf,
    }
