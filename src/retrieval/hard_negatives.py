"""Hard negative mining for contrastive multimodal retrieval."""

from __future__ import annotations

import logging

from src.eval.base import CandidateRug, Query

logger = logging.getLogger(__name__)


def mine_hard_negatives_for_query(
    query: Query,
    positive_rug: CandidateRug,
    candidate_catalog: list[CandidateRug],
    dense_scores: dict[str, float] | None = None,
    num_hard_negatives: int = 3,
) -> list[CandidateRug]:
    """Mine hard negative items for a query-positive pair.

    Selects items from the catalog that:
    1. Are distinct from the positive rug
    2. Have conflicting style or utilitarian mismatch (e.g. rug pad vs decorative rug)
    3. Have high dense similarity or lexical attribute overlap (confusing negatives).
    """
    q_text = f"{query.text_query} {query.scene_type}".lower()
    pos_id = positive_rug.rug_id

    candidates_with_scores: list[tuple[CandidateRug, float]] = []

    for rug in candidate_catalog:
        if rug.rug_id == pos_id:
            continue

        rug_text = f"{rug.title} {rug.color} {rug.style} {rug.tags}".lower()

        # Hard negative heuristic:
        # 1. Non-slip pads are hard lexical negatives when user seeks area rug
        is_pad = "pad" in rug.title.lower() or "pad" in rug.tags.lower()
        # 2. Style clash: modern vs traditional
        style_clash = (
            ("traditional" in q_text and "modern" in rug_text)
            or ("modern" in q_text and "traditional" in rug_text)
            or ("minimalist" in q_text and "oriental" in rug_text)
        )
        # 3. Dense score if provided
        dense_sim = dense_scores.get(rug.rug_id, 0.5) if dense_scores else 0.5

        # Hardness score: higher means harder (more confusing to model)
        hardness = dense_sim
        if is_pad and "pad" not in q_text:
            hardness += 0.4
        if style_clash:
            hardness += 0.3

        candidates_with_scores.append((rug, hardness))

    # Sort descending by hardness
    candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in candidates_with_scores[:num_hard_negatives]]
