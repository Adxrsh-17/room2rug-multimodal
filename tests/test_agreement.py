"""Unit tests for inter-rater reliability metrics (Cohen's Kappa)."""

from src.eval.agreement import agreement_report, cohens_kappa, compute_confusion_matrix


def test_cohens_kappa_perfect_agreement():
    r1 = [0, 1, 2, 0, 1, 2, 2, 1]
    r2 = [0, 1, 2, 0, 1, 2, 2, 1]

    k_unweighted = cohens_kappa(r1, r2, weights=None)
    k_quadratic = cohens_kappa(r1, r2, weights="quadratic")

    assert abs(k_unweighted - 1.0) < 1e-5
    assert abs(k_quadratic - 1.0) < 1e-5


def test_cohens_kappa_chance_agreement():
    # Independent distributions
    r1 = [0, 0, 0, 0, 2, 2, 2, 2]
    r2 = [2, 2, 2, 2, 0, 0, 0, 0]

    k = cohens_kappa(r1, r2, weights=None)
    assert k <= 0.0  # Perfect disagreement is negative


def test_confusion_matrix():
    r1 = [0, 1, 2]
    r2 = [0, 2, 1]
    mat = compute_confusion_matrix(r1, r2, classes=[0, 1, 2])
    assert mat[0][0] == 1
    assert mat[1][2] == 1
    assert mat[2][1] == 1


def test_agreement_report():
    r1 = [0, 1, 2, 1, 2]
    r2 = [0, 1, 2, 2, 2]
    rep = agreement_report(r1, r2)
    assert rep["n_pairs"] == 5
    assert rep["exact_agreement"] == 0.8
    assert rep["adjacent_agreement"] == 1.0
    assert rep["cohens_kappa_quadratic"] > 0.5
