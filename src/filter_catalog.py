from __future__ import annotations

import re
from typing import Any

from .query_parser import SIZE_MAP

DIMENSION_PATTERN = re.compile(r"(\d+\.?\d*)\s*[xX×]\s*(\d+\.?\d*)")
NUMBER_PATTERN = re.compile(r"(\d+\.?\d*)")
NEUTRAL_COLORS = ["beige", "ivory", "cream", "taupe", "tan", "grey", "gray", "natural", "off-white"]


def _text_blob(product: dict[str, Any]) -> str:
    return str(product.get("searchable_text") or "").lower()


def _normalize_size(size: object) -> tuple[float, float] | None:
    if size is None:
        return None
    text = str(size).lower().replace("×", "x")
    match = DIMENSION_PATTERN.search(text)
    if not match:
        if "round" in text:
            numbers = NUMBER_PATTERN.findall(text)
            if numbers:
                diameter = float(numbers[0])
                if len(numbers) > 1 and any(symbol in text for symbol in ["'", '"', "ft"]):
                    diameter += float(numbers[1]) / 12.0
                return (diameter, diameter)

        if "x" not in text:
            return None

        left, right = text.split("x", 1)

        def _to_feet(segment: str) -> float | None:
            numbers = NUMBER_PATTERN.findall(segment)
            if not numbers:
                return None
            feet = float(numbers[0])
            inches = float(numbers[1]) if len(numbers) > 1 else 0.0
            if any(symbol in segment for symbol in ["'", '"', "ft"]):
                return feet + (inches / 12.0)
            return feet

        width = _to_feet(left)
        length = _to_feet(right)
        if width is None or length is None:
            return None
        return tuple(sorted((width, length)))

    width = float(match.group(1))
    length = float(match.group(2))
    return tuple(sorted((width, length)))


def _size_matches_variant(query_dimensions: dict[str, float], variant: tuple[float, float] | None) -> bool:
    if variant is None:
        return False
    query_width = float(query_dimensions["width"])
    query_length = float(query_dimensions["length"])
    variant_width, variant_length = variant
    return abs(query_width - variant_width) <= 1.5 and abs(query_length - variant_length) <= 1.5


def _size_matches_class(size_class: str, variant: tuple[float, float] | None) -> bool:
    if variant is None:
        return False
    area = variant[0] * variant[1]
    bounds = SIZE_MAP.get(size_class)
    if not bounds:
        return False
    return bounds["min_area"] <= area < bounds["max_area"]


def _product_variants(product: dict[str, Any]) -> list[tuple[float, float] | None]:
    variants: list[tuple[float, float] | None] = []
    for size in product.get("sizes") or []:
        variants.append(_normalize_size(size))
    if not variants and product.get("size"):
        variants.append(_normalize_size(product.get("size")))
    return variants


def _size_score(product: dict[str, Any], parsed_query: dict[str, Any]) -> float:
    dimensions = parsed_query.get("dimensions")
    size_class = parsed_query.get("size_class")
    shape = parsed_query.get("shape")
    variants = _product_variants(product)

    if not dimensions and not size_class and shape != "runner":
        return 0.0

    if dimensions:
        if shape in {"round", "square"}:
            if any(variant and abs(variant[0] - float(dimensions["width"])) <= 1.5 for variant in variants):
                return 1.0
            return 0.0
        if shape == "runner":
            if any(variant and abs(variant[0] - float(dimensions["width"])) <= 1.5 for variant in variants):
                return 1.0
            return 0.0
        if any(_size_matches_variant(dimensions, variant) for variant in variants):
            return 1.0
        return 0.0

    if size_class:
        if any(_size_matches_class(size_class, variant) for variant in variants):
            return 1.0
        return 0.0

    if shape == "runner":
        return 0.5 if any(variant and variant[0] <= 3.5 and variant[1] >= 6 for variant in variants) else 0.0

    return 0.0


def _color_score(product: dict[str, Any], parsed_query: dict[str, Any]) -> float:
    colors = parsed_query.get("color") or []
    if not colors:
        return 0.0
    blob = _text_blob(product)
    if "neutral" in colors:
        return 1.0 if any(color in blob for color in NEUTRAL_COLORS) else 0.0
    return 1.0 if any(color in blob for color in colors) else 0.0


def _style_score(product: dict[str, Any], parsed_query: dict[str, Any]) -> float:
    styles = parsed_query.get("style") or []
    if not styles:
        return 0.0
    blob = _text_blob(product)
    score = 0.0
    for style in styles:
        if style == "bohemian" and ("bohemian" in blob or "boho" in blob):
            score = max(score, 1.0)
        elif style == "modern":
            if "modern" in blob:
                score = max(score, 1.0)
            elif "contemporary" in blob:
                score = max(score, 0.5)
        elif style in blob:
            score = max(score, 1.0)
    return score


def metadata_score(product: dict[str, Any], parsed_query: dict[str, Any]) -> float:
    score = 0.0
    weight_total = 0.0

    if parsed_query.get("dimensions") or parsed_query.get("size_class") or parsed_query.get("shape") == "runner":
        score += _size_score(product, parsed_query) * 0.4
        weight_total += 0.4

    if parsed_query.get("color"):
        score += _color_score(product, parsed_query) * 0.3
        weight_total += 0.3

    if parsed_query.get("style"):
        score += _style_score(product, parsed_query) * 0.3
        weight_total += 0.3

    if weight_total == 0:
        return 0.5

    return score / weight_total


def filter_products(parsed_query: dict[str, Any], catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not catalog:
        raise ValueError("Catalog is empty")

    size_requested = bool(parsed_query.get("dimensions") or parsed_query.get("size_class") or parsed_query.get("shape") == "runner")
    matched_products: list[dict[str, Any]] = []
    size_match_found = False

    for product in catalog:
        current = dict(product)
        current["metadata_score"] = metadata_score(current, parsed_query)
        current["color_match"] = _color_score(current, parsed_query)
        current["style_match"] = _style_score(current, parsed_query)
        current["size_match"] = _size_score(current, parsed_query)
        current["matched_size"] = current.get("size") or (current.get("sizes") or [None])[0]

        if size_requested:
            if current["size_match"] > 0:
                matched_products.append(current)
                size_match_found = True
            elif not current.get("sizes"):
                matched_products.append(current)
        else:
            matched_products.append(current)

    if size_requested and not size_match_found:
        for product in catalog:
            fallback = dict(product)
            fallback["metadata_score"] = 0.5
            fallback["color_match"] = _color_score(fallback, parsed_query)
            fallback["style_match"] = _style_score(fallback, parsed_query)
            fallback["size_match"] = 0.0
            fallback["matched_size"] = fallback.get("size") or (fallback.get("sizes") or [None])[0]
            matched_products.append(fallback)
        return matched_products

    return matched_products


if __name__ == "__main__":
    from pathlib import Path
    import json

    from .preprocess import build_catalog
    from .query_parser import parse_query

    catalog_path = Path("data/products.csv")
    catalog = build_catalog(catalog_path) if catalog_path.exists() else []
    parsed = parse_query("8x10 beige traditional rug")
    results = filter_products(parsed, catalog)
    print(json.dumps(results[:3], indent=2))
