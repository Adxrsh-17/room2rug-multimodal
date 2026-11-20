"""Unit tests for LightGBM LambdaMART ranker and SHAP integration."""

import numpy as np

from src.eval.base import CandidateRug, Query
from src.rerank.features import FEATURE_NAMES
from src.rerank.lambdamart import LambdaMARTRanker


def test_lambdamart_fit_predict_and_shap():
    # Synthetic dataset: 2 queries with 4 items each
    # Feature 0 positively correlates with target relevance
    rng = np.random.RandomState(42)
    feature_cols = FEATURE_NAMES[:5]
    X = rng.randn(8, 5).astype(np.float32)
    y = np.array([2, 1, 0, 0, 2, 2, 1, 0])
    X[:, 0] = y * 1.5 + rng.randn(8) * 0.1  # strong signal on feature 0
    groups = [4, 4]

    ranker = LambdaMARTRanker(
        feature_subset=feature_cols,
        stage1_retriever=None,
        n_estimators=10,
        learning_rate=0.1,
        random_state=42,
    )
    ranker.fit(X, y, groups)
    assert ranker.is_fitted is True

    # Feature importance
    importances = ranker.get_feature_importances()
    assert feature_cols[0] in importances
    assert importances[feature_cols[0]] > 0.0

    # TreeSHAP contributions
    shap_vals = ranker.get_shap_contributions(X)
    assert shap_vals.shape == (8, 6)  # 5 features + 1 bias

    # Rank inference
    q = Query(query_id="test_q", text_query="neutral rug")
    cands = [CandidateRug(rug_id=f"r_{i}", title=f"Rug {i}") for i in range(4)]
    results = ranker.rank(q, cands, top_k=3)

    assert len(results) == 3
    assert results[0].rank == 1
    assert results[1].rank == 2
    assert results[0].score >= results[1].score
