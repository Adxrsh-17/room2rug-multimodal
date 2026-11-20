from src.baseline_v1.filter_catalog import metadata_score
from src.baseline_v1.fusion import compute_fusion_score, determine_mode
from src.baseline_v1.query_parser import parse_query


def test_baseline_v1_query_parser():
    parsed = parse_query("8x10 beige traditional rug")
    assert parsed["dimensions"] == {"width": 8.0, "length": 10.0}
    assert "beige" in parsed["color"]
    assert "traditional" in parsed["style"]
    assert parsed["shape"] == "rectangle"


def test_baseline_v1_fusion_score():
    assert compute_fusion_score(0.8, None, "A") == 0.8
    assert round(compute_fusion_score(0.8, 0.6, "B"), 2) == 0.72
    assert compute_fusion_score(None, 0.7, "C") == 0.7


def test_baseline_v1_determine_mode():
    assert determine_mode(True, False) == "A"
    assert determine_mode(True, True) == "B"
    assert determine_mode(False, True) == "C"


def test_baseline_v1_metadata_score_empty():
    product = {
        "title": "Vintage Persian Rug",
        "tags": "traditional, beige, wool",
        "searchable_text": "vintage persian rug traditional beige wool",
        "sizes": ["8x10"],
    }
    parsed = parse_query("8x10 beige traditional rug")
    score = metadata_score(product, parsed)
    assert score > 0.8
