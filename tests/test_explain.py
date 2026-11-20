"""Unit tests for Phase 5 Grounded Explanation generation and Faithfulness Validator."""

import pytest

from src.explain.faithfulness import (
    FaithfulnessValidator,
    evaluate_faithfulness_suite,
)
from src.explain.generator import (
    DeterministicGroundedGenerator,
    GroundedExplanationGenerator,
)


@pytest.fixture
def sample_polyester_rug():
    return {
        "handle": "test-poly-beige-01",
        "title": "Modern Geometric Beige Area Rug",
        "color": "Beige",
        "style": "Contemporary",
        "size": "5x8 (5' 5\" x 7'10\")",
        "sizes": ["4x6 (3' 4\" x 5')", "5x8 (5' 5\" x 7'10')", "8x10 (7' 6\" x 9' 8\")"],
        "body": "Primary Color: Beige. Fiber Type: Polyester. Weave: Machine-Woven.",
        "tags": "Contemporary, Polyester, Machine-Woven",
        "price": 89.99,
    }


@pytest.fixture
def sample_rug_pad():
    return {
        "handle": "test-pad-01",
        "title": "SuperGrip Protective Rug Pad",
        "color": "Grey",
        "style": "Rug Pads",
        "size": "SCATTER (2' x 3')",
        "sizes": ["SCATTER (2' x 3')", "5x8", "8x10"],
        "body": "Non-slip polymer coated polyester pad for hardwood protection.",
        "tags": "Rug Pad, Non-Slip, Polyester",
        "price": 14.50,
    }


def test_validator_extract_ground_truth(sample_polyester_rug, sample_rug_pad):
    validator = FaithfulnessValidator()
    gt_rug = validator.extract_catalog_ground_truth(sample_polyester_rug)
    assert "polyester" in gt_rug["materials"]
    assert "wool" not in gt_rug["materials"]
    assert "beige" in gt_rug["colors"]
    assert "5x8" in gt_rug["sizes"]
    assert not gt_rug["is_rug_pad"]

    gt_pad = validator.extract_catalog_ground_truth(sample_rug_pad)
    assert gt_pad["is_rug_pad"]


def test_validator_detects_material_contradiction(sample_polyester_rug):
    validator = FaithfulnessValidator()
    # Hallucinates wool on a polyester rug
    explanation = "This beautiful rug is woven from 100% natural wool, providing soft cushioning for your bedroom."
    result = validator.validate(explanation, sample_polyester_rug)
    assert not result.is_faithful
    assert len(result.contradictions) > 0
    assert any("wool" in c for c in result.contradictions)


def test_validator_detects_color_contradiction(sample_polyester_rug):
    validator = FaithfulnessValidator()
    # Claims vibrant blue for a beige rug
    explanation = "The vibrant blue tones of this contemporary rug create a dramatic accent in your room."
    result = validator.validate(explanation, sample_polyester_rug)
    assert not result.is_faithful
    assert any("blue" in c for c in result.contradictions)


def test_validator_detects_rug_pad_type_contradiction(sample_rug_pad):
    validator = FaithfulnessValidator()
    # Describing a rug pad as a plush decorative rug
    explanation = "This plush decorative rug adds luxurious warmth and elegant aesthetic charm to your floor."
    result = validator.validate(explanation, sample_rug_pad)
    assert not result.is_faithful
    assert any("Product type contradiction" in c for c in result.contradictions)


def test_validator_passes_faithful_explanation(sample_polyester_rug):
    validator = FaithfulnessValidator()
    explanation = (
        "The beige tones of Modern Geometric Beige Area Rug align with the living room palette. "
        "Available in 8x10, it is crafted from durable polyester for everyday use."
    )
    result = validator.validate(explanation, sample_polyester_rug, query="8x10 living room")
    assert result.is_faithful
    assert len(result.contradictions) == 0
    assert result.grounding_precision == 1.0


def test_deterministic_generator_is_always_faithful(sample_polyester_rug, sample_rug_pad):
    validator = FaithfulnessValidator()

    # Rug test
    rug_exp = DeterministicGroundedGenerator.generate(
        sample_polyester_rug,
        room_context={"scene_type": "living_room"},
        query="8x10 beige rug",
        visual_metrics={"cielab_min_delta_e": 12.5},
    )
    res_rug = validator.validate(rug_exp, sample_polyester_rug, query="8x10")
    assert res_rug.is_faithful
    assert len(res_rug.contradictions) == 0

    # Rug pad test
    pad_exp = DeterministicGroundedGenerator.generate(
        sample_rug_pad,
        room_context={"scene_type": "dining_room"},
        query="rug pad",
    )
    res_pad = validator.validate(pad_exp, sample_rug_pad)
    assert res_pad.is_faithful
    assert "rug pad" in pad_exp.lower()


def test_grounded_generator_fallback_on_unfaithful(sample_polyester_rug):
    # Mock LLM that hallucinates wool
    hallucinating_llm = lambda prompt: "This rug is made of 100% fine New Zealand wool."  # noqa: E731

    gen = GroundedExplanationGenerator(llm_callable=hallucinating_llm, fallback_on_unfaithful=True)
    res = gen.generate_explanation(sample_polyester_rug, room_context={"scene_type": "bedroom"})

    # Should have triggered fallback to deterministic generator
    assert res.fallback_triggered
    assert res.backend_used == "deterministic_fallback"
    assert res.faithfulness.is_faithful
    assert "wool" not in res.explanation.lower()


def test_evaluate_faithfulness_suite():
    validator = FaithfulnessValidator()
    pairs = [
        {
            "explanation": "This polyester rug has subtle beige hues.",
            "product": {"title": "Beige Rug", "color": "Beige", "body": "Polyester"},
        },
        {
            "explanation": "This 100% pure wool rug has royal blue patterns.",
            "product": {"title": "Beige Rug", "color": "Beige", "body": "Polyester"},
        },
    ]
    summary = evaluate_faithfulness_suite(pairs, validator)
    assert summary["total_samples"] == 2
    assert summary["faithful_count"] == 1
    assert summary["contradiction_count"] == 1
    assert summary["faithfulness_rate"] == 0.5
    assert summary["contradiction_rate"] == 0.5
