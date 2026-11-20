"""Grounded explanation generation combining constrained LLM reasoning with guaranteed deterministic fallback."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.explain.faithfulness import FaithfulnessResult, FaithfulnessValidator

logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """Result of an explanation generation call with provenance and validation metadata."""

    explanation: str
    backend_used: str
    fallback_triggered: bool = False
    fallback_reason: str | None = None
    generation_latency_ms: float = 0.0
    faithfulness: FaithfulnessResult | None = None
    cited_attributes: dict[str, Any] = field(default_factory=dict)


class DeterministicGroundedGenerator:
    """High-speed, zero-hallucination deterministic template synthesizer grounded directly in catalog metadata."""

    @classmethod
    def generate(
        cls,
        product: dict[str, Any],
        room_context: dict[str, Any] | None = None,
        query: str = "",
        visual_metrics: dict[str, Any] | None = None,
    ) -> str:
        """Constructs a factual, attribute-verified explanation string."""
        title = str(product.get("title", "This rug")).strip()
        color = str(product.get("color", "")).capitalize().strip() or "neutral"
        style = str(product.get("style", "")).strip() or "contemporary"
        price = product.get("price")

        room_type = "room"
        if room_context and "scene_type" in room_context:
            room_type = str(room_context["scene_type"]).replace("_", " ")

        gt = FaithfulnessValidator.extract_catalog_ground_truth(product)
        materials_list = sorted(gt["materials"])
        material_str = ", ".join(materials_list) if materials_list else ""

        # Rug pad specific explanation
        if gt["is_rug_pad"]:
            mat_clause = f" featuring {material_str} construction" if material_str else ""
            price_clause = f" at ${price:.2f}" if price is not None and price > 0 else ""
            return (
                f"Designed as a specialized functional rug pad{mat_clause}, this {color.lower()} foundation provides essential "
                f"cushioning and non-slip floor protection for {room_type} settings{price_clause}."
            )

        # Standard decorative rug explanation
        sentences = []

        # 1. Palette harmony sentence
        cielab_de = None
        if visual_metrics and "cielab_min_delta_e" in visual_metrics:
            cielab_de = visual_metrics["cielab_min_delta_e"]

        if cielab_de is not None and cielab_de < 30.0:
            sentences.append(
                f"The {color.lower()} palette of {title} provides strong color harmony with the {room_type} "
                f"(measured CIELAB color distance: {cielab_de:.1f})."
            )
        else:
            sentences.append(
                f"The {color.lower()} tones of {title} establish a coherent visual anchor within the {room_type}."
            )

        # 2. Material & Style grounding
        style_clause = f"{style.lower()} styling" if style.lower() != "rug pads" else "clean styling"
        mat_clause = f"woven from durable {material_str}" if material_str else "crafted for daily residential use"

        # Check size alignment
        sizes = sorted(gt["sizes"])
        matched_size = None
        if query:
            for s in sizes:
                if s in query.lower():
                    matched_size = s
                    break

        if matched_size:
            sentences.append(
                f"Offered in the requested {matched_size} dimension, it aligns with {style_clause} and is {mat_clause}."
            )
        else:
            sentences.append(
                f"Its {style_clause} complements the interior aesthetic while being {mat_clause}."
            )

        return " ".join(sentences)


class GroundedExplanationGenerator:
    """Production grounded explanation generator.

    Provides:
    - Structured prompt engineering enforcing catalog attribute boundaries.
    - Automated post-generation faithfulness verification.
    - Automatic graceful degradation to DeterministicGroundedGenerator upon hallucination/timeout.
    """

    def __init__(
        self,
        llm_callable: Callable[[str], str] | None = None,
        fallback_on_unfaithful: bool = True,
        validator: FaithfulnessValidator | None = None,
    ) -> None:
        self.llm_callable = llm_callable
        self.fallback_on_unfaithful = fallback_on_unfaithful
        self.validator = validator or FaithfulnessValidator()
        self.deterministic_gen = DeterministicGroundedGenerator()

    def build_grounded_prompt(
        self,
        product: dict[str, Any],
        room_context: dict[str, Any] | None = None,
        query: str = "",
        visual_metrics: dict[str, Any] | None = None,
    ) -> str:
        """Constructs an unbreachable constraint prompt for LLM generation."""
        gt = self.validator.extract_catalog_ground_truth(product)

        catalog_spec = {
            "title": product.get("title", ""),
            "color": product.get("color", ""),
            "style": product.get("style", ""),
            "verified_fiber_materials": sorted(gt["materials"]),
            "available_sizes": sorted(gt["sizes"]),
            "price_usd": product.get("price"),
            "is_rug_pad": gt["is_rug_pad"],
        }

        room_info = {
            "scene_type": room_context.get("scene_type", "interior") if room_context else "interior",
            "user_query": query,
            "target_color": room_context.get("target_color", "") if room_context else "",
            "target_style": room_context.get("target_style", "") if room_context else "",
        }

        visual_info = visual_metrics or {}

        prompt = (
            "You are an expert interior design recommendation system. Generate a concise 2-sentence rationale "
            "explaining why the retrieved rug fits the user's room query.\n\n"
            "CRITICAL FAITHFULNESS CONSTRAINTS:\n"
            "1. ONLY reference materials, colors, styles, and sizes explicitly declared in the CATALOG_METADATA below.\n"
            "2. NEVER invent unlisted materials (e.g. do NOT say 'wool' or 'silk' if the catalog says 'polyester').\n"
            "3. If is_rug_pad is true, describe it strictly as a functional floor pad, NOT a decorative plush area rug.\n"
            "4. Format response strictly as JSON with keys 'explanation' (string) and 'cited_attributes' (dictionary).\n\n"
            f"CATALOG_METADATA: {json.dumps(catalog_spec)}\n"
            f"ROOM_AND_QUERY_CONTEXT: {json.dumps(room_info)}\n"
            f"VISUAL_SIMILARITY_METRICS: {json.dumps(visual_info)}\n\n"
            "JSON Response:"
        )
        return prompt

    def generate_explanation(
        self,
        product: dict[str, Any],
        room_context: dict[str, Any] | None = None,
        query: str = "",
        visual_metrics: dict[str, Any] | None = None,
        force_deterministic: bool = False,
    ) -> ExplanationResult:
        """Generates a verified, grounded explanation with automatic fallback."""
        start_time = time.perf_counter()

        if force_deterministic or self.llm_callable is None:
            text = self.deterministic_gen.generate(
                product, room_context=room_context, query=query, visual_metrics=visual_metrics
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            val_res = self.validator.validate(text, product, room_context=room_context, query=query)
            return ExplanationResult(
                explanation=text,
                backend_used="deterministic",
                fallback_triggered=False,
                generation_latency_ms=round(elapsed_ms, 2),
                faithfulness=val_res,
                cited_attributes={"color": product.get("color"), "style": product.get("style")},
            )

        # Attempt LLM generation
        try:
            prompt = self.build_grounded_prompt(product, room_context, query, visual_metrics)
            raw_response = self.llm_callable(prompt)
            if not raw_response:
                raise ValueError("Empty LLM response received")

            # Parse JSON if returned
            parsed_text = raw_response.strip()
            cited_attrs: dict[str, Any] = {}
            if "{" in parsed_text and "}" in parsed_text:
                try:
                    json_str = parsed_text[parsed_text.find("{") : parsed_text.rfind("}") + 1]
                    data = json.loads(json_str)
                    if "explanation" in data:
                        parsed_text = str(data["explanation"])
                    if "cited_attributes" in data:
                        cited_attrs = data["cited_attributes"]
                except Exception:
                    pass  # Keep raw text if JSON parse fails

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            val_res = self.validator.validate(parsed_text, product, room_context=room_context, query=query)

            # Check if faithful
            if self.fallback_on_unfaithful and not val_res.is_faithful:
                reason = "; ".join(val_res.contradictions + val_res.hallucinations)
                logger.warning("Unfaithful explanation detected (%s). Falling back to deterministic generator.", reason)

                det_text = self.deterministic_gen.generate(
                    product, room_context=room_context, query=query, visual_metrics=visual_metrics
                )
                det_val = self.validator.validate(det_text, product, room_context=room_context, query=query)
                return ExplanationResult(
                    explanation=det_text,
                    backend_used="deterministic_fallback",
                    fallback_triggered=True,
                    fallback_reason=reason,
                    generation_latency_ms=round(elapsed_ms, 2),
                    faithfulness=det_val,
                    cited_attributes=cited_attrs,
                )

            return ExplanationResult(
                explanation=parsed_text,
                backend_used="llm_grounded",
                fallback_triggered=False,
                generation_latency_ms=round(elapsed_ms, 2),
                faithfulness=val_res,
                cited_attributes=cited_attrs,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error("LLM generation failed: %s. Triggering deterministic fallback.", e)
            det_text = self.deterministic_gen.generate(
                product, room_context=room_context, query=query, visual_metrics=visual_metrics
            )
            det_val = self.validator.validate(det_text, product, room_context=room_context, query=query)
            return ExplanationResult(
                explanation=det_text,
                backend_used="deterministic_fallback",
                fallback_triggered=True,
                fallback_reason=str(e),
                generation_latency_ms=round(elapsed_ms, 2),
                faithfulness=det_val,
            )
