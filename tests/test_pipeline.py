from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embedder import embed_image, embed_text
from src.filter_catalog import filter_products
from src.fusion import compute_fusion_score
from src.indexer import load_index, search_index
from src.preprocess import load_catalog
from src.query_parser import parse_query
from src.ranker import rank_results
from src.search_part1 import search_structured
from src.search_part2 import search_multimodal

try:
    from src.filter_catalog import _normalize_size as _parse_catalog_size
except Exception:  # pragma: no cover - defensive fallback for test utility only
    _parse_catalog_size = None

CATALOG = load_catalog("data/products.csv")
RESULTS_LOG: list[dict[str, str]] = []


def log(test_name: str, status: str, details: str = "") -> None:
    icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⏭️"
    print(f"{icon} [{test_name}] {details}")
    RESULTS_LOG.append({"test": test_name, "status": status, "details": details})


def _as_float_pair(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    if _parse_catalog_size is not None:
        return _parse_catalog_size(value)

    text = str(value).lower().replace("×", "x")
    if "x" not in text:
        return None
    left, right = text.split("x", 1)
    numbers_left = [float(token) for token in __import__("re").findall(r"\d+\.?\d*", left)]
    numbers_right = [float(token) for token in __import__("re").findall(r"\d+\.?\d*", right)]
    if not numbers_left or not numbers_right:
        return None
    width = numbers_left[0] + (numbers_left[1] / 12.0 if len(numbers_left) > 1 else 0.0)
    length = numbers_right[0] + (numbers_right[1] / 12.0 if len(numbers_right) > 1 else 0.0)
    return tuple(sorted((width, length)))


def _first_available_image(patterns: list[str] | None = None) -> Path | None:
    patterns = patterns or ["*.jpg", "*.jpeg", "*.JPG", "*.png"]
    image_dir = ROOT / "images"
    for pattern in patterns:
        for candidate in sorted(image_dir.glob(pattern)):
            if candidate.exists():
                return candidate
    return None


def _warm_up_part2_indexes() -> None:
    sample_image = _first_available_image()
    if sample_image is None:
        raise FileNotFoundError("No images found in images/")
    # Trigger the real Part 2 pipeline so the indexes are created on disk.
    search_multimodal(sample_image, "modern neutral", top_k=1)


PARSER_TESTS = [
    ("8x10 beige traditional rug", {"dimensions": {"width": 8, "length": 10}, "color": ["beige"], "style": ["traditional"]}),
    ("runner 2x10 blue rug", {"dimensions": {"width": 2, "length": 10}, "shape": "runner", "color": ["blue"]}),
    ("round 6 ft modern rug", {"shape": "round", "dimensions": {"width": 6, "length": 6}, "style": ["modern"]}),
    ("large neutral rug", {"size_class": "large", "color": ["neutral"], "dimensions": None}),
    ("8 by 10 grey persian", {"dimensions": {"width": 8, "length": 10}, "color": ["grey"], "style": ["persian"]}),
    ("7.5x9.5 ivory transitional", {"dimensions": {"width": 7.5, "length": 9.5}, "color": ["ivory"]}),
    ("8x10", {"dimensions": {"width": 8, "length": 10}, "color": [], "style": []}),
    ("10x8 rug", {"dimensions": {"width": 8, "length": 10}}),
    ("gray bohemian rug", {"color": ["grey"], "style": ["bohemian"]}),
    ("boho multicolor runner", {"style": ["bohemian"], "color": ["multi"], "shape": "runner"}),
    ("small accent rug", {"size_class": "small", "dimensions": None}),
    ("oversized area rug", {"size_class": "xlarge"}),
    ("nice rug", {"dimensions": None, "color": [], "style": [], "size_class": None}),
    ("", None),
]


def test_parser() -> None:
    print("\n── PARSER TESTS ──")
    for query, expected in PARSER_TESTS:
        try:
            result = parse_query(query)
            if expected is None:
                assert result is None or result == {}, f"Expected None/empty for blank query, got {result}"
                log(f"parse_query('{query}')", "PASS", "empty query handled")
                continue

            assert result is not None, "parse_query returned None unexpectedly"

            for field, expected_val in expected.items():
                actual_val = result.get(field)
                if isinstance(expected_val, dict):
                    assert isinstance(actual_val, dict), f"Field {field}: expected dict, got {type(actual_val)}"
                    for key, value in expected_val.items():
                        assert abs(float(actual_val[key]) - float(value)) < 0.01, (
                            f"Field {field}.{key}: expected {value}, got {actual_val.get(key)}"
                        )
                elif isinstance(expected_val, list):
                    assert isinstance(actual_val, list), f"Field {field}: expected list, got {type(actual_val)}"
                    for item in expected_val:
                        assert item in actual_val, f"Field {field}: expected '{item}' in {actual_val}"
                else:
                    assert actual_val == expected_val, f"Field {field}: expected {expected_val}, got {actual_val}"

            log(f"parse_query('{query}')", "PASS", str(result))
        except AssertionError as exc:
            log(f"parse_query('{query}')", "FAIL", str(exc))
        except Exception:
            log(f"parse_query('{query}')", "FAIL", f"EXCEPTION: {traceback.format_exc()}")


def test_filter() -> None:
    print("\n── FILTER TESTS ──")

    try:
        parsed = parse_query("8x10 rug")
        results = filter_products(parsed, CATALOG)
        for product in results:
            size_variants = product.get("sizes", [])
            if not size_variants:
                continue
            compatible = False
            for size_variant in size_variants:
                parsed_size = _as_float_pair(size_variant)
                if parsed_size is None:
                    continue
                width, length = parsed_size
                if abs(width - 8) <= 1.5 and abs(length - 10) <= 1.5:
                    compatible = True
                    break
            assert compatible, f"Product {product['handle']} passed size filter incorrectly"
        log("filter: size 8x10", "PASS", f"{len(results)} products returned")
    except Exception as exc:
        log("filter: size 8x10", "FAIL", str(exc))

    try:
        parsed = parse_query("nice rug")
        results = filter_products(parsed, CATALOG)
        assert len(results) == len(CATALOG), f"Expected all {len(CATALOG)} products, got {len(results)}"
        log("filter: no filters", "PASS", "all products returned")
    except Exception as exc:
        log("filter: no filters", "FAIL", str(exc))

    try:
        parsed = parse_query("red rug")
        results = filter_products(parsed, CATALOG)
        assert len(results) > 0, "Color filter discarded all products — should be soft"
        log("filter: soft color", "PASS", f"{len(results)} products returned (soft filter)")
    except Exception as exc:
        log("filter: soft color", "FAIL", str(exc))

    try:
        parsed = parse_query("runner rug")
        results = filter_products(parsed, CATALOG)
        assert len(results) > 0, "Runner filter returned 0 products"
        log("filter: runner shape", "PASS", f"{len(results)} products")
    except Exception as exc:
        log("filter: runner shape", "FAIL", str(exc))


PART1_QUERIES = [
    "8x10 beige traditional rug",
    "runner 2x10 blue rug",
    "round 6 ft modern rug",
    "large neutral rug",
    "small bohemian accent rug",
    "9x12 navy geometric",
    "nice rug",
    "red",
    "",
]


def test_part1_search() -> None:
    print("\n── PART 1 END-TO-END SEARCH TESTS ──")
    for query in PART1_QUERIES:
        try:
            start = time.time()
            results = search_structured(query, CATALOG)
            elapsed = round(time.time() - start, 2)

            if query == "":
                assert results == [] or results is None, "Empty query should return [] not crash"
                log(f"search_structured('{query}')", "PASS", "empty query → []")
                continue

            assert isinstance(results, list), f"Expected list, got {type(results)}"
            assert len(results) <= 10, f"Returned {len(results)} results, expected ≤ 10"
            assert len(results) >= 1, "Returned 0 results for a valid query"

            for result in results:
                for field in ["rank", "handle", "title", "final_score", "metadata_score", "semantic_score"]:
                    assert field in result, f"Missing field '{field}' in result"

            scores = [item["final_score"] for item in results]
            assert scores == sorted(scores, reverse=True), f"Results not sorted by score: {scores}"
            for score in scores:
                assert 0.0 <= score <= 1.0, f"final_score out of range: {score}"

            log(
                f"search_structured('{query}')",
                "PASS",
                f"{len(results)} results in {elapsed}s | top: '{results[0]['title']}' ({results[0]['final_score']:.2f})",
            )
        except AssertionError as exc:
            log(f"search_structured('{query}')", "FAIL", str(exc))
        except Exception:
            log(f"search_structured('{query}')", "FAIL", f"EXCEPTION: {traceback.format_exc()}")


def test_embedder() -> None:
    print("\n── EMBEDDER TESTS ──")

    try:
        test_image = _first_available_image(["*.jpg", "*.jpeg", "*.JPG", "*.png"])
        assert test_image is not None, "No test images found in images/"
        embedding = embed_image(str(test_image))
        assert embedding is not None, "embed_image returned None for valid image"
        assert embedding.shape == (512,), f"Expected shape (512,), got {embedding.shape}"
        assert abs(np.linalg.norm(embedding) - 1.0) < 0.01, "Image embedding is not L2-normalized"
        log("embed_image: valid file", "PASS", f"shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f}")
    except Exception as exc:
        log("embed_image: valid file", "FAIL", str(exc))

    try:
        embedding = embed_image("https://thisurldoesnotexist.xyz/fake.jpg")
        assert embedding is None, f"Expected None for broken URL, got {embedding}"
        log("embed_image: broken URL", "PASS", "returned None gracefully")
    except Exception as exc:
        log("embed_image: broken URL", "FAIL", f"Should not raise — {str(exc)}")

    try:
        embedding = embed_text("modern neutral living room rug")
        assert embedding is not None, "embed_text returned None for valid text"
        assert embedding.shape == (512,), f"Expected shape (512,), got {embedding.shape}"
        assert abs(np.linalg.norm(embedding) - 1.0) < 0.01, "Text embedding is not L2-normalized"
        log("embed_text: valid text", "PASS", f"shape={embedding.shape}")
    except Exception as exc:
        log("embed_text: valid text", "FAIL", str(exc))

    try:
        embedding = embed_text("")
        assert embedding is None, f"Expected None for empty string, got {embedding}"
        log("embed_text: empty string", "PASS", "returned None")
    except Exception as exc:
        log("embed_text: empty string", "FAIL", str(exc))

    try:
        test_image = _first_available_image(["*.jpg", "*.jpeg", "*.JPG", "*.png"])
        assert test_image is not None, "No test images found in images/"
        image_embedding = embed_image(str(test_image))
        text_embedding = embed_text("room interior rug floor")
        assert image_embedding is not None and text_embedding is not None, "Embeddings could not be generated"
        similarity = float(np.dot(image_embedding, text_embedding))
        assert similarity > 0.10, f"Image-text similarity too low ({similarity:.3f}) — embeddings may not be in same space"
        log("embed: shared vector space", "PASS", f"cross-modal cosine sim = {similarity:.3f}")
    except Exception as exc:
        log("embed: shared vector space", "FAIL", str(exc))


def test_index() -> None:
    print("\n── INDEX TESTS ──")

    try:
        image_index, text_index, product_ids = load_index("index/")
    except Exception:
        try:
            _warm_up_part2_indexes()
            image_index, text_index, product_ids = load_index("index/")
        except Exception as exc:
            log("load_index", "FAIL", str(exc))
            return

    try:
        assert image_index is not None, "image_index is None"
        assert text_index is not None, "text_index is None"
        assert len(product_ids) > 0, "product_ids is empty"
        log("load_index", "PASS", f"{len(product_ids)} products indexed")
    except Exception as exc:
        log("load_index", "FAIL", str(exc))
        return

    try:
        test_image = _first_available_image(["*.jpg", "*.jpeg", "*.JPG", "*.png"])
        assert test_image is not None, "No test images found in images/"
        query_embedding = embed_image(str(test_image))
        assert query_embedding is not None, "Image query embedding is None"
        positions, scores = search_index(image_index, query_embedding, k=10)
        assert len(positions) > 0, "search_index returned empty positions"
        assert -1 not in positions, "search_index returned -1 positions (FAISS padding not filtered)"
        assert scores == sorted(scores, reverse=True), "Scores not sorted descending"
        assert all(0.0 <= score <= 1.0 for score in scores), f"Scores out of range: {scores}"
        log("search_index: image query", "PASS", f"top score={scores[0]:.3f}, positions={positions[:3]}")
    except Exception as exc:
        log("search_index: image query", "FAIL", str(exc))

    try:
        query_embedding = embed_text("traditional persian rug warm tones")
        assert query_embedding is not None, "Text query embedding is None"
        positions, scores = search_index(text_index, query_embedding, k=10)
        assert len(positions) > 0, "text search returned 0 results"
        assert -1 not in positions, "FAISS -1 padding not filtered"
        log("search_index: text query", "PASS", f"top score={scores[0]:.3f}")
    except Exception as exc:
        log("search_index: text query", "FAIL", str(exc))


def test_fusion() -> None:
    print("\n── FUSION TESTS ──")

    cases = [
        (0.8, None, "A", 0.8),
        (0.8, 0.6, "B", 0.72),
        (None, 0.7, "C", 0.7),
        (1.0, 1.0, "B", 1.0),
        (0.0, 0.0, "B", 0.0),
        (0.5, 0.5, "B", 0.5),
    ]

    for image_sim, text_sim, mode, expected in cases:
        try:
            result = compute_fusion_score(image_sim, text_sim, mode)
            assert abs(result - expected) < 0.01, f"mode={mode}, img={image_sim}, txt={text_sim}: expected {expected}, got {result}"
            log(f"fusion: mode={mode} img={image_sim} txt={text_sim}", "PASS", f"score={result:.3f}")
        except Exception as exc:
            log(f"fusion: mode={mode}", "FAIL", str(exc))

    try:
        compute_fusion_score(0.5, 0.5, "Z")
        log("fusion: invalid mode", "FAIL", "Should have raised ValueError")
    except ValueError:
        log("fusion: invalid mode", "PASS", "raised ValueError correctly")
    except Exception as exc:
        log("fusion: invalid mode", "FAIL", f"Wrong exception type: {exc}")


PART2_TESTS = [
    ("images/room1.jpg", "", "Mode A: image only"),
    ("images/room1.jpg", "modern neutral", "Mode B: image + vague text"),
    ("images/room2.jpg", "traditional Persian warm tones", "Mode B: image + specific style"),
    ("images/room1.jpg", "large rug for dining room", "Mode B: image + size intent"),
    ("images/room2.jpg", "runner 2x10 grey", "Mode B: image + structured text"),
    ("images/room1.jpg", "bohemian colorful", "Mode B: image + color+style"),
]


def test_part2_search() -> None:
    print("\n── PART 2 END-TO-END SEARCH TESTS ──")

    for room_image, text_query, label in PART2_TESTS:
        image_path = ROOT / room_image
        if not image_path.exists():
            log(f"search_multimodal: {label}", "FAIL", f"Image not found: {room_image}")
            continue

        try:
            start = time.time()
            results = search_multimodal(str(image_path), text_query, top_k=10)
            elapsed = round(time.time() - start, 2)

            assert isinstance(results, list), f"Expected list, got {type(results)}"
            assert 1 <= len(results) <= 10, f"Returned {len(results)} results"

            for result in results:
                for field in ["rank", "handle", "title", "fusion_score", "image_similarity", "mode"]:
                    assert field in result, f"Missing field '{field}'"
                assert 0.0 <= result["fusion_score"] <= 1.0, f"fusion_score out of range: {result['fusion_score']}"

            scores = [item["fusion_score"] for item in results]
            assert scores == sorted(scores, reverse=True), "Results not sorted"

            expected_mode = "A" if text_query == "" else "B"
            assert results[0]["mode"] == expected_mode, f"Expected mode {expected_mode}, got {results[0]['mode']}"

            log(
                f"search_multimodal: {label}",
                "PASS",
                f"{len(results)} results in {elapsed}s | top: '{results[0]['title']}' ({results[0]['fusion_score']:.2f})",
            )
        except AssertionError as exc:
            log(f"search_multimodal: {label}", "FAIL", str(exc))
        except Exception:
            log(f"search_multimodal: {label}", "FAIL", f"EXCEPTION: {traceback.format_exc()}")


def print_summary() -> None:
    print("\n" + "═" * 55)
    print("  PIPELINE TEST SUMMARY")
    print("═" * 55)

    passed = [record for record in RESULTS_LOG if record["status"] == "PASS"]
    failed = [record for record in RESULTS_LOG if record["status"] == "FAIL"]
    skipped = [record for record in RESULTS_LOG if record["status"] == "SKIP"]

    print(f"  ✅ PASSED  : {len(passed)}")
    print(f"  ❌ FAILED  : {len(failed)}")
    print(f"  ⏭️  SKIPPED : {len(skipped)}")
    print(f"  TOTAL     : {len(RESULTS_LOG)}")
    print("═" * 55)

    if failed:
        print("\n  FAILED TESTS:")
        for record in failed:
            print(f"  ❌ {record['test']}")
            print(f"     → {record['details']}")

    with open(ROOT / "test_results.json", "w", encoding="utf-8") as file_handle:
        json.dump(RESULTS_LOG, file_handle, indent=2)
    print("\n  Full log saved to test_results.json")

    sys.exit(0 if len(failed) == 0 else 1)


if __name__ == "__main__":
    print("═" * 55)
    print("  MULTIMODAL RUG SEARCH — PIPELINE TEST")
    print("═" * 55)

    test_parser()
    test_filter()
    test_part1_search()
    test_embedder()
    test_index()
    test_fusion()
    test_part2_search()

    print_summary()
