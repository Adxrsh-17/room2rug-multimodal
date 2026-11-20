"""LightGBM LambdaMART ranker for Learning-to-Rank (LTR)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from src.eval.base import BaseRanker, CandidateRug, Query, RankResult
from src.rerank.features import (
    FEATURE_NAMES,
    RerankingFeatureExtractor,
)
from src.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


class LambdaMARTRanker(BaseRanker):
    """Learning-to-Rank reranker using LightGBM LambdaMART with TreeSHAP explanations."""

    name = "lambdamart_reranker"

    def __init__(
        self,
        model_path: str | Path | None = None,
        feature_subset: list[str] | None = None,
        stage1_retriever: BaseRanker | None = None,
        n_estimators: int = 80,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        random_state: int = 42,
    ) -> None:
        self.feature_names = feature_subset or FEATURE_NAMES
        self.feature_extractor = RerankingFeatureExtractor()
        self.stage1_retriever = stage1_retriever or HybridRetriever(dense_ranker=None)
        self.random_state = random_state

        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            eval_at=[1, 3, 5, 10],
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=3,
            random_state=random_state,
            importance_type="gain",
            verbose=-1,
        )
        self.is_fitted = False

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray | None = None,
        groups_train: list[int] | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        groups_val: list[int] | None = None,
        **kwargs: Any,
    ) -> LambdaMARTRanker:
        """Fit LambdaMART ranker with optional early stopping on validation NDCG."""
        if y_train is None:
            y_train = kwargs.get("y")
        if groups_train is None:
            groups_train = kwargs.get("groups") or kwargs.get("group")
        assert y_train is not None, "y_train must be provided"
        assert groups_train is not None, "groups_train must be provided"

        eval_set = None
        eval_group = None
        callbacks = []

        if X_val is not None and y_val is not None and groups_val is not None:
            eval_set = [(X_val, y_val)]
            eval_group = [groups_val]
            callbacks.append(lgb.early_stopping(stopping_rounds=10, verbose=False))

        self.model.fit(
            X=X_train,
            y=y_train,
            group=groups_train,
            eval_set=eval_set,
            eval_group=eval_group,
            callbacks=callbacks,
        )
        self.is_fitted = True
        return self

    def get_feature_importances(self) -> dict[str, float]:
        """Return normalized feature importances based on split gain."""
        if not self.is_fitted:
            return {f: 0.0 for f in self.feature_names}
        gains = self.model.booster_.feature_importance(importance_type="gain")
        total = sum(gains) if sum(gains) > 0 else 1.0
        return {name: round(float(g / total), 4) for name, g in zip(self.feature_names, gains)}

    def get_shap_contributions(self, X: np.ndarray) -> np.ndarray:
        """Compute exact TreeSHAP feature contributions using LightGBM native C++ engine."""
        if not self.is_fitted:
            return np.zeros((len(X), len(self.feature_names) + 1))
        # predict with pred_contrib=True returns (N, num_features + 1 bias)
        return self.model.booster_.predict(X, pred_contrib=True)

    def rank(
        self,
        query: Query,
        candidates: list[CandidateRug],
        top_k: int = 10,
    ) -> list[RankResult]:
        """Rerank candidate pool using trained LambdaMART model."""
        if not candidates:
            return []

        # Stage 1: Retrieve candidate pool (e.g. top 25-50)
        stage1_k = min(len(candidates), max(top_k * 3, 25))
        if self.stage1_retriever is not None and hasattr(self.stage1_retriever, "rank"):
            try:
                stage1_ranked = self.stage1_retriever.rank(query, candidates, top_k=stage1_k)
            except Exception:
                stage1_ranked = None
        else:
            stage1_ranked = None

        cand_subset = candidates
        if stage1_ranked:
            pool_ids = {r.rug_id for r in stage1_ranked}
            cand_subset = [c for c in candidates if c.rug_id in pool_ids] or candidates

        # Stage 2: Feature extraction & scoring
        X, rug_ids = self.feature_extractor.extract_matrix_for_query(
            query=query,
            candidates=cand_subset,
            stage1_rankings=stage1_ranked,
            feature_subset=self.feature_names,
        )

        if self.is_fitted:
            scores = self.model.predict(X)
            # Compute SHAP for explanations
            shap_vals = self.get_shap_contributions(X)
        else:
            # Fallback to stage1 score or simple heuristic
            scores = X[:, 0] if X.shape[1] > 0 else np.zeros(len(cand_subset))
            shap_vals = None

        # Build results
        sorted_indices = np.argsort(scores)[::-1]

        results: list[RankResult] = []
        for rank_idx, idx in enumerate(sorted_indices[:top_k], start=1):
            rid = rug_ids[idx]
            sc = float(scores[idx])

            # Generate concise SHAP explanation of top positive feature
            explanation = "LambdaMART score"
            if shap_vals is not None and len(self.feature_names) > 0:
                feat_contribs = shap_vals[idx, :-1]  # exclude bias
                top_feat_idx = int(np.argmax(feat_contribs))
                top_feat_name = self.feature_names[top_feat_idx]
                top_val = float(feat_contribs[top_feat_idx])
                explanation = f"Top SHAP driver: +{top_val:.2f} ({top_feat_name})"

            results.append(
                RankResult(
                    rug_id=rid,
                    score=round(sc, 4),
                    rank=rank_idx,
                    system_name=self.name,
                    explanation=explanation,
                )
            )

        return results

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(str(path))
        logger.info("Saved LambdaMART model to %s", path)

    def load(self, path: str | Path) -> None:
        booster = lgb.Booster(model_file=str(path))
        self.model._Booster = booster
        self.is_fitted = True
        logger.info("Loaded LambdaMART model from %s", path)
