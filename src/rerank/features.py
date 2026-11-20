"""Feature extraction pipeline for Learning-to-Rank (LTR) reranking."""

from __future__ import annotations

import logging
import math
import re

import numpy as np

from src.eval.base import CandidateRug, Query, RankResult
from src.eval.baselines import (
    COLOR_KEYWORDS,
    SIZE_PATTERN,
    STYLE_KEYWORDS,
    AttributeFilterBaseline,
    BM25Baseline,
    CLIPBaseline,
    MiniLMBaseline,
    SigLIPBaseline,
)
from src.rerank.palette import (
    cielab_delta_e_distance,
    extract_dominant_cielab_palette,
)

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "clip_sim",
    "siglip_sim",
    "minilm_sim",
    "stage1_score",
    "stage1_rrf",
    "stage1_rank",
    "bm25_score",
    "cielab_min_delta_e",
    "cielab_weighted_delta_e",
    "color_exact_match",
    "color_overlap_score",
    "style_exact_match",
    "style_overlap_score",
    "size_exact_match",
    "is_rug_pad",
    "log_price",
    "title_token_overlap",
]

FEATURE_GROUPS = {
    "dense_only": ["clip_sim", "siglip_sim", "minilm_sim", "stage1_score", "stage1_rrf", "stage1_rank"],
    "dense_plus_color": [
        "clip_sim",
        "siglip_sim",
        "minilm_sim",
        "stage1_score",
        "stage1_rrf",
        "stage1_rank",
        "cielab_min_delta_e",
        "cielab_weighted_delta_e",
        "color_exact_match",
        "color_overlap_score",
    ],
    "dense_plus_color_plus_metadata": [
        "clip_sim",
        "siglip_sim",
        "minilm_sim",
        "stage1_score",
        "stage1_rrf",
        "stage1_rank",
        "cielab_min_delta_e",
        "cielab_weighted_delta_e",
        "color_exact_match",
        "color_overlap_score",
        "style_exact_match",
        "style_overlap_score",
        "size_exact_match",
        "bm25_score",
        "log_price",
        "title_token_overlap",
    ],
    "all_features": FEATURE_NAMES,
}


class RerankingFeatureExtractor:
    """Extracts dense, colorimetric, semantic, and structural ranking features."""

    def __init__(self) -> None:
        self.clip_ranker = CLIPBaseline()
        self.siglip_ranker = SigLIPBaseline()
        self.minilm_ranker = MiniLMBaseline()
        self.bm25_ranker = BM25Baseline()
        self.attr_ranker = AttributeFilterBaseline()

        self._palette_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _get_palette(self, image_source: str | None) -> tuple[np.ndarray, np.ndarray]:
        if not image_source:
            return np.array([[75.0, 5.0, 15.0]]), np.array([1.0])
        if image_source not in self._palette_cache:
            p, w = extract_dominant_cielab_palette(image_source, n_colors=3)
            self._palette_cache[image_source] = (p, w)
        return self._palette_cache[image_source]

    def extract_pair_features(
        self,
        query: Query,
        candidate: CandidateRug,
        stage1_res: RankResult | None = None,
        precomputed_scores: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Extract all 17 ranking features for a single (query, candidate) pair."""
        scores = precomputed_scores or {}

        # 1. Dense similarities
        clip_sim = scores.get("clip", 0.5)
        siglip_sim = scores.get("siglip", 0.5)
        minilm_sim = scores.get("minilm", 0.5)
        bm25_score = scores.get("bm25", 0.0)

        # 2. Stage 1 signals
        if stage1_res is not None:
            stage1_score = float(stage1_res.score)
            stage1_rank = float(stage1_res.rank)
            stage1_rrf = float(1.0 / (60.0 + stage1_res.rank))
        else:
            stage1_score = 0.5
            stage1_rank = 10.0
            stage1_rrf = float(1.0 / 70.0)

        # 3. CIELAB palette Delta E
        room_pal, room_w = self._get_palette(query.room_image_path)
        # Use candidate image or color default
        rug_pal, rug_w = self._get_palette(candidate.image_url if candidate.image_url else None)
        min_delta_e, weighted_delta_e = cielab_delta_e_distance(room_pal, rug_pal, room_w, rug_w)

        # 4. Color text overlap
        q_text = f"{query.text_query} {query.scene_type}".lower()
        cand_text = f"{candidate.title} {candidate.color} {candidate.tags}".lower()

        target_colors = {c for c in COLOR_KEYWORDS if c in q_text}
        if target_colors:
            matched_colors = sum(1 for c in target_colors if c in cand_text)
            color_exact_match = 1.0 if matched_colors > 0 else 0.0
            color_overlap_score = matched_colors / len(target_colors)
        else:
            color_exact_match = 0.5
            color_overlap_score = 0.5

        # 5. Style text overlap
        target_styles = {s for s in STYLE_KEYWORDS if s in q_text}
        if target_styles:
            matched_styles = sum(1 for s in target_styles if s in cand_text)
            style_exact_match = 1.0 if matched_styles > 0 else 0.0
            style_overlap_score = matched_styles / len(target_styles)
        else:
            style_exact_match = 0.5
            style_overlap_score = 0.5

        # 6. Size match
        size_match = SIZE_PATTERN.search(q_text)
        if size_match:
            t_size = f"{size_match.group(1)}x{size_match.group(2)}"
            cand_sizes = [s.lower().replace(" ", "") for s in candidate.sizes]
            size_exact_match = 1.0 if t_size in cand_sizes or t_size in candidate.size.lower() else 0.0
        else:
            size_exact_match = 0.5

        # 7. Utilitarian / Rug Pad guardrail
        is_rug_pad = 1.0 if ("pad" in candidate.title.lower() or "pad" in candidate.tags.lower()) else 0.0

        # 8. Price & Title Overlap
        price = candidate.price if candidate.price > 0 else 100.0
        log_price = math.log1p(price)

        q_tokens = set(re.findall(r"\b\w{3,}\b", q_text)) - {"the", "and", "rug", "room", "area", "for", "with"}
        t_tokens = set(re.findall(r"\b\w{3,}\b", candidate.title.lower()))
        if q_tokens:
            title_overlap = len(q_tokens.intersection(t_tokens)) / len(q_tokens)
        else:
            title_overlap = 0.0

        return {
            "clip_sim": round(float(clip_sim), 4),
            "siglip_sim": round(float(siglip_sim), 4),
            "minilm_sim": round(float(minilm_sim), 4),
            "stage1_score": round(stage1_score, 4),
            "stage1_rrf": round(stage1_rrf, 5),
            "stage1_rank": round(stage1_rank, 1),
            "bm25_score": round(float(bm25_score), 4),
            "cielab_min_delta_e": round(min_delta_e, 3),
            "cielab_weighted_delta_e": round(weighted_delta_e, 3),
            "color_exact_match": round(color_exact_match, 2),
            "color_overlap_score": round(color_overlap_score, 4),
            "style_exact_match": round(style_exact_match, 2),
            "style_overlap_score": round(style_overlap_score, 4),
            "size_exact_match": round(size_exact_match, 2),
            "is_rug_pad": round(is_rug_pad, 1),
            "log_price": round(log_price, 4),
            "title_token_overlap": round(title_overlap, 4),
        }

    def extract_matrix_for_query(
        self,
        query: Query,
        candidates: list[CandidateRug],
        stage1_rankings: list[RankResult] | None = None,
        feature_subset: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Extract 2D feature matrix (N_candidates, N_features) for a query."""
        stage1_map = {r.rug_id: r for r in stage1_rankings} if stage1_rankings else {}

        # Precompute dense & bm25 scores across candidates in batch
        clip_ranks = {r.rug_id: r.score for r in self.clip_ranker.rank(query, candidates, top_k=len(candidates))}
        siglip_ranks = {r.rug_id: r.score for r in self.siglip_ranker.rank(query, candidates, top_k=len(candidates))}
        minilm_ranks = {r.rug_id: r.score for r in self.minilm_ranker.rank(query, candidates, top_k=len(candidates))}
        bm25_ranks = {r.rug_id: r.score for r in self.bm25_ranker.rank(query, candidates, top_k=len(candidates))}

        feature_cols = feature_subset or FEATURE_NAMES
        rows = []
        rug_ids = []

        for cand in candidates:
            precomputed = {
                "clip": clip_ranks.get(cand.rug_id, 0.5),
                "siglip": siglip_ranks.get(cand.rug_id, 0.5),
                "minilm": minilm_ranks.get(cand.rug_id, 0.5),
                "bm25": bm25_ranks.get(cand.rug_id, 0.0),
            }
            feat_dict = self.extract_pair_features(
                query=query,
                candidate=cand,
                stage1_res=stage1_map.get(cand.rug_id),
                precomputed_scores=precomputed,
            )
            rows.append([feat_dict.get(col, 0.0) for col in feature_cols])
            rug_ids.append(cand.rug_id)


        return np.array(rows, dtype=np.float32), rug_ids
