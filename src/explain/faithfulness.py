"""Automated faithfulness validator and contradiction detector for catalog-grounded explanations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

KNOWN_FIBER_TYPES = [
    "wool",
    "polyester",
    "polypropylene",
    "cotton",
    "jute",
    "silk",
    "nylon",
    "viscose",
    "chenille",
    "microfiber",
    "sisal",
]

KNOWN_COLORS = [
    "beige",
    "grey",
    "gray",
    "blue",
    "navy",
    "red",
    "cream",
    "ivory",
    "brown",
    "black",
    "white",
    "green",
    "gold",
    "yellow",
    "orange",
    "pink",
    "purple",
    "rust",
    "teal",
    "turquoise",
    "silver",
]

COMMON_DIMENSIONS = [
    "2x3",
    "3x5",
    "4x6",
    "5x7",
    "5x8",
    "6x9",
    "8x10",
    "9x12",
    "10x13",
    "10x14",
    "12x15",
]


@dataclass
class FaithfulnessResult:
    """Detailed diagnostic output of faithfulness validation."""

    is_faithful: bool
    contradictions: list[str] = field(default_factory=list)
    hallucinations: list[str] = field(default_factory=list)
    verified_claims: list[str] = field(default_factory=list)
    grounding_precision: float = 1.0


class FaithfulnessValidator:
    """Automated validator detecting attribute contradictions and ungrounded claims in explanations."""

    def __init__(self) -> None:
        pass

    @classmethod
    def extract_catalog_ground_truth(cls, product: dict[str, Any]) -> dict[str, Any]:
        """Extracts canonical ground truth attributes from a product dictionary."""
        full_text = " ".join(
            [
                str(product.get("title", "")),
                str(product.get("body", "")),
                str(product.get("body_html", "")),
                str(product.get("tags", "")),
                str(product.get("style", "")),
                str(product.get("color", "")),
            ]
        ).lower()

        # Material extraction
        materials = set()
        for fiber in KNOWN_FIBER_TYPES:
            if re.search(rf"\b{fiber}\b", full_text):
                materials.add(fiber)

        # Color extraction
        colors = set()
        primary_color = str(product.get("color", "")).lower().strip()
        if primary_color:
            colors.add(primary_color)
            if primary_color == "gray":
                colors.add("grey")
            elif primary_color == "grey":
                colors.add("gray")

        for c in KNOWN_COLORS:
            if re.search(rf"\b{c}\b", full_text):
                colors.add(c)
                if c == "gray":
                    colors.add("grey")
                elif c == "grey":
                    colors.add("gray")

        # Size extraction
        sizes = set()
        raw_sizes = product.get("sizes", [])
        if isinstance(raw_sizes, list):
            for s in raw_sizes:
                for dim in COMMON_DIMENSIONS:
                    if dim in str(s).lower():
                        sizes.add(dim)
        raw_size = str(product.get("size", "")).lower()
        for dim in COMMON_DIMENSIONS:
            if dim in raw_size:
                sizes.add(dim)

        is_rug_pad = (
            "rug pad" in full_text or str(product.get("style", "")).lower() == "rug pads" or "padsecure" in full_text
        )

        return {
            "materials": materials,
            "colors": colors,
            "sizes": sizes,
            "is_rug_pad": is_rug_pad,
            "title": str(product.get("title", "")),
            "style": str(product.get("style", "")).lower(),
        }

    def validate(
        self,
        explanation: str,
        product: dict[str, Any],
        room_context: dict[str, Any] | None = None,
        query: str = "",
    ) -> FaithfulnessResult:
        """Validates an explanation string against product metadata and room context.

        Detects:
        1. Material contradictions (asserts material not matching product)
        2. Color contradictions (asserts color alien to product)
        3. Dimension contradictions (claims dimensions not offered by product)
        4. Functional mismatch (describing a utilitarian pad as a luxury decorative rug)
        """
        gt = self.extract_catalog_ground_truth(product)
        text_lower = explanation.lower()

        contradictions: list[str] = []
        hallucinations: list[str] = []
        verified_claims: list[str] = []

        # 1. Material verification
        mentioned_fibers = {f for f in KNOWN_FIBER_TYPES if re.search(rf"\b{f}\b", text_lower)}
        if mentioned_fibers:
            if gt["materials"]:
                # Check for direct conflicts
                conflicts = mentioned_fibers - gt["materials"]
                matches = mentioned_fibers & gt["materials"]
                for c in conflicts:
                    contradictions.append(
                        f"Material contradiction: claims '{c}', but catalog specifies {list(gt['materials'])}"
                    )
                for m in matches:
                    verified_claims.append(f"Material: {m}")
            else:
                # Catalog doesn't specify material, but text asserts one -> ungrounded hallucination
                for c in mentioned_fibers:
                    hallucinations.append(
                        f"Ungrounded material claim: asserts '{c}', but catalog does not state fiber composition"
                    )

        # 2. Color verification
        mentioned_colors = {c for c in KNOWN_COLORS if re.search(rf"\b{c}\b", text_lower)}
        # Filter out room colors if mentioned as part of room description
        if room_context and "target_color" in room_context:
            target_col = str(room_context["target_color"]).lower()
            # If color matches room context, it can be valid in the explanation context
            if target_col in mentioned_colors:
                mentioned_colors.discard(target_col)

        # Check if remaining asserted rug colors conflict with catalog
        if mentioned_colors:
            if gt["colors"]:
                conflicts = set()
                for mc in mentioned_colors:
                    if mc not in gt["colors"]:
                        # Check phrase context: "blue room" vs "blue rug"
                        if re.search(rf"\b{mc}\s+(rug|carpet|pad|tones|palette|weave|pile)\b", text_lower) or re.search(
                            rf"\b(this|the)\s+{mc}\b", text_lower
                        ):
                            conflicts.add(mc)
                        elif mc in text_lower and not any(
                            rc in text_lower for rc in ["room", "wall", "floor", "space", "furniture"]
                        ):
                            conflicts.add(mc)

                for c in conflicts:
                    contradictions.append(
                        f"Color contradiction: claims '{c}' for rug, but catalog color is {list(gt['colors'])}"
                    )
                for m in mentioned_colors & gt["colors"]:
                    verified_claims.append(f"Color: {m}")

        # 3. Size / Dimension verification
        mentioned_dims = {d for d in COMMON_DIMENSIONS if re.search(rf"\b{d}\b", text_lower)}
        if mentioned_dims:
            if gt["sizes"]:
                for d in mentioned_dims:
                    if d in gt["sizes"]:
                        verified_claims.append(f"Dimension: {d}")
                    else:
                        contradictions.append(
                            f"Dimension contradiction: claims size '{d}', but catalog only offers {sorted(gt['sizes'])}"
                        )
            else:
                for d in mentioned_dims:
                    # If user queried this dimension, it's contextual
                    if query and d in query.lower():
                        verified_claims.append(f"Query Dimension: {d}")
                    else:
                        hallucinations.append(f"Ungrounded dimension: claims '{d}' without catalog verification")

        # 4. Rug Pad Utilitarian Functional Check
        if gt["is_rug_pad"]:
            decorative_terms = ["plush", "luxurious", "decorative", "oriental", "vintage medallion", "cozy pile"]
            for term in decorative_terms:
                if term in text_lower:
                    contradictions.append(f"Product type contradiction: describes utilitarian rug pad with '{term}'")

        total_claims = len(verified_claims) + len(contradictions) + len(hallucinations)
        grounding_precision = len(verified_claims) / total_claims if total_claims > 0 else 1.0
        is_faithful = (len(contradictions) == 0) and (len(hallucinations) == 0)

        return FaithfulnessResult(
            is_faithful=is_faithful,
            contradictions=contradictions,
            hallucinations=hallucinations,
            verified_claims=verified_claims,
            grounding_precision=round(grounding_precision, 4),
        )


def evaluate_faithfulness_suite(
    pairs: list[dict[str, Any]],
    validator: FaithfulnessValidator,
) -> dict[str, Any]:
    """Computes aggregate faithfulness metrics across a dataset of (explanation, product) pairs."""
    total = len(pairs)
    if total == 0:
        return {
            "total_samples": 0,
            "faithful_count": 0,
            "contradiction_count": 0,
            "hallucination_count": 0,
            "faithfulness_rate": 1.0,
            "contradiction_rate": 0.0,
            "hallucination_rate": 0.0,
            "mean_grounding_precision": 1.0,
        }

    faithful_count = 0
    contradiction_count = 0
    hallucination_count = 0
    precisions = []

    for item in pairs:
        explanation = item.get("explanation", "")
        product = item.get("product", {})
        room_context = item.get("room_context", None)
        query = item.get("query", "")

        res = validator.validate(explanation, product, room_context=room_context, query=query)
        if res.is_faithful:
            faithful_count += 1
        if len(res.contradictions) > 0:
            contradiction_count += 1
        if len(res.hallucinations) > 0:
            hallucination_count += 1
        precisions.append(res.grounding_precision)

    return {
        "total_samples": total,
        "faithful_count": faithful_count,
        "contradiction_count": contradiction_count,
        "hallucination_count": hallucination_count,
        "faithfulness_rate": round(faithful_count / total, 4),
        "contradiction_rate": round(contradiction_count / total, 4),
        "hallucination_rate": round(hallucination_count / total, 4),
        "mean_grounding_precision": round(sum(precisions) / total, 4),
    }
