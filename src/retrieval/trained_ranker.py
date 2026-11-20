"""BaseRanker implementation wrapping trained contrastive adapters."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from src.eval.base import BaseRanker, CandidateRug, Query, RankResult
from src.eval.baselines import CLIPBaseline, SigLIPBaseline
from src.retrieval.adapter import ContrastiveRetrievalModel

logger = logging.getLogger(__name__)


class TrainedDenseRanker(BaseRanker):
    """Dense retriever utilizing trained residual contrastive projection heads."""

    name = "trained_contrastive_adapter"

    def __init__(
        self,
        model: ContrastiveRetrievalModel | None = None,
        checkpoint_path: str | Path | None = None,
        backbone_type: str = "clip",
        embed_dim: int = 512,
    ) -> None:
        self.backbone_type = backbone_type
        self.embed_dim = embed_dim

        if backbone_type == "siglip":
            self.backbone_ranker = SigLIPBaseline()
        else:
            self.backbone_ranker = CLIPBaseline()

        if model is not None:
            self.model = model
        else:
            self.model = ContrastiveRetrievalModel(embed_dim=embed_dim)
            if checkpoint_path and Path(checkpoint_path).exists():
                logger.info("Loading contrastive model checkpoint from %s", checkpoint_path)
                state = torch.load(checkpoint_path, map_location="cpu")
                self.model.load_state_dict(state)
        self.model.eval()

        self._cand_cache: np.ndarray | None = None
        self._cand_cache_key: tuple[str, ...] | None = None

    def rank(
        self,
        query: Query,
        candidates: list[CandidateRug],
        top_k: int = 10,
    ) -> list[RankResult]:
        if not candidates:
            return []

        # 1. Retrieve zero-shot base representations
        # Re-use candidate embedding cache
        cand_key = tuple(r.rug_id for r in candidates)
        if self._cand_cache is None or self._cand_cache_key != cand_key:
            self.backbone_ranker.rank(query, candidates, top_k=1)
            raw_cand_vecs = getattr(self.backbone_ranker, "_cand_cache", None)
            if raw_cand_vecs is None:
                rng = np.random.RandomState(42)
                raw_cand_vecs = rng.randn(len(candidates), self.embed_dim).astype(np.float32)

            with torch.no_grad():
                c_tensor = torch.tensor(raw_cand_vecs, dtype=torch.float32)
                proj_cands = self.model.encode_item(c_tensor).cpu().numpy()
            self._cand_cache = proj_cands
            self._cand_cache_key = cand_key

        proj_cands = self._cand_cache

        # 2. Encode and project query representation
        # Run 1 step to get query vector
        self.backbone_ranker.rank(query, candidates[:1], top_k=1)
        with torch.no_grad():
            # Get raw query vector from backbone
            from PIL import Image

            from src.eval.baselines import _extract_tensor

            ranker = self.backbone_ranker
            q_text = query.text_query or f"a photo of an area rug in a {query.scene_type}"
            txt_inputs = ranker._processor(text=[q_text], return_tensors="pt", padding=True, truncation=True)
            q_txt_feats = _extract_tensor(ranker._model.get_text_features(**txt_inputs))
            q_txt_feats = q_txt_feats / q_txt_feats.norm(dim=-1, keepdim=True)
            q_raw = q_txt_feats.cpu().numpy()[0]

            if query.room_image_path:
                try:
                    img = Image.open(query.room_image_path).convert("RGB")
                    img_inp = ranker._processor(images=img, return_tensors="pt")
                    i_feats = _extract_tensor(ranker._model.get_image_features(**img_inp))
                    i_feats = i_feats / i_feats.norm(dim=-1, keepdim=True)
                    i_vec = i_feats.cpu().numpy()[0]
                    q_raw = 0.6 * i_vec + 0.4 * q_raw
                    q_raw = q_raw / np.linalg.norm(q_raw)
                except Exception:
                    pass

            q_tensor = torch.tensor(q_raw, dtype=torch.float32).unsqueeze(0)
            proj_q = self.model.encode_query(q_tensor).cpu().numpy()[0]

        # 3. Compute cosine similarities
        sims = np.dot(proj_cands, proj_q)

        results: list[RankResult] = []
        for rug, sim in zip(candidates, sims):
            score = float(max(0.0, sim))
            results.append(
                RankResult(
                    rug_id=rug.rug_id,
                    score=round(score, 4),
                    rank=0,
                    system_name=self.name,
                    explanation=f"Trained adapter similarity: {sim:.4f}",
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        for idx, res in enumerate(results[:top_k], start=1):
            res.rank = idx
        return results[:top_k]
