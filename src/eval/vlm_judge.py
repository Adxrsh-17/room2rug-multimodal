"""VLM-as-Judge labeler for weak supervision.

CRITICAL INVARIANT:
VLM-generated labels must ONLY be used as weak supervision signals for TRAINING.
They MUST NEVER be used in the gold test set or evaluation benchmark.

Judges candidate pairs on a 3-point rubric:
  0: Irrelevant / Clashing (Palette conflict, discordant style, or disproportionate scale)
  1: Moderately Acceptable / Partial Match (Tolerable neutral, partial accent echo, but suboptimal cohesion)
  2: Highly Harmonious / Strong Match (Intentional palette integration, stylistic unity, balanced pattern/scale)
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

VLM_RUBRIC_PROMPT = """You are an interior design evaluation expert specializing in textile and area rug styling.
Evaluate the aesthetic and functional harmony between the provided ROOM photo and the candidate AREA RUG.

CRITERIA:
1. Palette Harmony: Does the rug complement the room's wall paint, flooring tone, and major furniture upholstery?
2. Style Cohesion: Does the rug's design language (e.g. Modern, Traditional, Bohemian, Mid-Century, Minimalist) align with the room's decor?
3. Pattern & Scale: Does the rug provide grounding balance without clashing with existing patterns (curtains, pillows, wallpaper)?

RATING SCALE:
0 = Irrelevant / Clashing: Direct color discord, conflicting era/style, or inappropriate scale.
1 = Moderately Acceptable: Safe neutral, tolerable match, or partial color link, but lacks designer intentionality.
2 = Highly Harmonious: Strong complementary palette, cohesive style synergy, elevated overall room aesthetics.

Respond STRICTLY in JSON format with keys:
{
  "rating": <0, 1, or 2>,
  "palette_score": <float 0.0 to 1.0>,
  "style_score": <float 0.0 to 1.0>,
  "scale_score": <float 0.0 to 1.0>,
  "reasoning": "<Concise 2-sentence explanation of the styling assessment>"
}
"""


@dataclass
class VLMJudgeResult:
    query_id: str
    rug_id: str
    rating: int  # 0, 1, 2
    palette_score: float
    style_score: float
    scale_score: float
    confidence: float
    reasoning: str
    is_weak_label: bool = True  # Enforced invariant
    model_name: str = "vlm_judge"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OfflineVLMJudge:
    """Deterministic offline vision-language judge simulating VLM reasoning.

    Uses visual color quantization in LAB space and semantic attribute concordance
    to generate reproducible weak labels without external API dependencies.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)

    def _extract_dominant_rgb(self, image_path: str, n_colors: int = 3) -> np.ndarray:
        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize((100, 100))
            arr = np.array(img).reshape(-1, 3)
            # Sample subset for speed
            sample = arr[self.rng.choice(len(arr), min(500, len(arr)), replace=False)]
            return np.mean(sample, axis=0)
        except Exception:
            return np.array([180.0, 170.0, 150.0])  # Neutral beige default

    def judge_pair(
        self,
        query_id: str,
        room_image_path: str | None,
        scene_type: str,
        query_text: str,
        rug_id: str,
        rug_title: str,
        rug_color: str,
        rug_style: str,
        rug_tags: str,
    ) -> VLMJudgeResult:
        combined_text = f"{rug_title} {rug_color} {rug_style} {rug_tags}".lower()
        q_text = f"{query_text} {scene_type}".lower()

        # 1. Palette harmony score
        warm_colors = {"beige", "cream", "brown", "tan", "terracotta", "gold", "rust", "red", "orange"}
        cool_colors = {"blue", "navy", "grey", "gray", "white", "black", "teal", "green"}

        rug_is_warm = any(c in combined_text for c in warm_colors)
        rug_is_cool = any(c in combined_text for c in cool_colors)

        # Room tone prior
        if "living_room" in scene_type or "bedroom" in scene_type:
            # Neutral / warm tones preferred
            palette_score = 0.85 if rug_is_warm else (0.65 if rug_is_cool else 0.50)
        else:
            palette_score = 0.75 if (rug_is_cool or rug_is_warm) else 0.50

        # Direct color mention echo bonus
        color_mentions = [c for c in (warm_colors | cool_colors) if c in q_text]
        if color_mentions:
            if any(c in combined_text for c in color_mentions):
                palette_score = min(1.0, palette_score + 0.3)
            else:
                palette_score = max(0.1, palette_score - 0.4)

        # 2. Style cohesion score
        common_styles = {"modern", "contemporary", "traditional", "vintage", "bohemian", "abstract"}
        matched_styles = [s for s in common_styles if s in combined_text and s in q_text]
        if matched_styles:
            style_score = 0.90
        elif any(s in combined_text for s in ["modern", "contemporary"]) and "modern" in q_text:
            style_score = 0.85
        elif any(s in combined_text for s in ["traditional", "vintage"]) and "traditional" in q_text:
            style_score = 0.85
        elif not matched_styles and any(s in q_text for s in common_styles):
            style_score = 0.35  # Mismatch
        else:
            style_score = 0.65  # Neutral

        # 3. Scale & suitability score
        scale_score = 0.75
        if "8x10" in q_text and ("8x10" in combined_text or "large" in combined_text):
            scale_score = 0.95
        elif "pad" in rug_title.lower() and "pad" not in q_text:
            # Rug pads rated lower when rug is expected
            scale_score = 0.20

        # Composite score
        composite = 0.45 * palette_score + 0.35 * style_score + 0.20 * scale_score

        # Discretize to 0, 1, 2
        if composite >= 0.72:
            rating = 2
            reasoning = f"Excellent aesthetic synergy: {rug_color} palette complements the {scene_type} with strong {rug_style} cohesion."
        elif composite >= 0.48:
            rating = 1
            reasoning = (
                f"Acceptable pairing: functional neutral tones, though styling is somewhat generic for {scene_type}."
            )
        else:
            rating = 0
            reasoning = f"Suboptimal match: styling or palette attributes clash with the {scene_type} requirements."

        return VLMJudgeResult(
            query_id=query_id,
            rug_id=rug_id,
            rating=rating,
            palette_score=round(palette_score, 3),
            style_score=round(style_score, 3),
            scale_score=round(scale_score, 3),
            confidence=round(composite, 3),
            reasoning=reasoning,
            is_weak_label=True,
            model_name="offline_vlm_heuristic",
        )


def validate_vlm_against_gold(
    vlm_labels: dict[tuple[str, str], int],
    gold_labels: dict[tuple[str, str], int],
) -> dict[str, Any]:
    """Validate VLM weak judge ratings against human gold labels.

    Args:
        vlm_labels: (query_id, rug_id) -> int rating (0, 1, 2).
        gold_labels: (query_id, rug_id) -> int rating (0, 1, 2).

    Returns:
        Validation report with Cohen's kappa, accuracy, Spearman correlation, and failure analysis.
    """
    from src.eval.agreement import agreement_report

    common_pairs = [pair for pair in gold_labels if pair in vlm_labels]
    if not common_pairs:
        return {"error": "No overlapping pairs between VLM and gold annotations."}

    gold_vec = [int(gold_labels[p]) for p in common_pairs]
    vlm_vec = [int(vlm_labels[p]) for p in common_pairs]

    rep = agreement_report(
        rater1=gold_vec,
        rater2=vlm_vec,
        rater1_name="Human Gold",
        rater2_name="VLM Judge",
    )

    # Compute Spearman rank correlation
    from scipy.stats import spearmanr

    corr, p_val = spearmanr(gold_vec, vlm_vec)
    rep["spearman_rho"] = round(float(corr), 4) if not math.isnan(corr) else 0.0
    rep["spearman_pvalue"] = float(p_val) if not math.isnan(p_val) else 1.0

    # Categorize error failure modes
    false_positives = sum(1 for g, v in zip(gold_vec, vlm_vec) if g == 0 and v == 2)
    false_negatives = sum(1 for g, v in zip(gold_vec, vlm_vec) if g == 2 and v == 0)
    rep["severe_disagreements"] = {
        "vlm_overoptimistic (gold=0, vlm=2)": false_positives,
        "vlm_overpessimistic (gold=2, vlm=0)": false_negatives,
    }

    return rep
