from __future__ import annotations

import math
import re
import unicodedata

COLORS = [
    "beige",
    "blue",
    "grey",
    "gray",
    "ivory",
    "red",
    "navy",
    "brown",
    "cream",
    "black",
    "white",
    "green",
    "gold",
    "orange",
    "pink",
    "purple",
    "rust",
    "teal",
    "charcoal",
    "taupe",
    "multi",
    "multicolor",
    "neutral",
    "tan",
    "yellow",
    "sage",
    "terracotta",
]

STYLES = [
    "traditional",
    "modern",
    "contemporary",
    "persian",
    "bohemian",
    "boho",
    "geometric",
    "abstract",
    "transitional",
    "oriental",
    "southwestern",
    "coastal",
    "farmhouse",
    "shag",
    "vintage",
    "moroccan",
    "tribal",
    "floral",
    "solid",
    "distressed",
]

SIZE_MAP = {
    "small": {"min_area": 0, "max_area": 24},
    "medium": {"min_area": 24, "max_area": 63},
    "large": {"min_area": 63, "max_area": 110},
    "xlarge": {"min_area": 110, "max_area": 9999},
    "runner": None,
}

SYNONYMS = {
    "big": "large",
    "huge": "xlarge",
    "tiny": "small",
    "grey": "gray",
    "boho": "bohemian",
    "circle": "round",
    "oriental": "persian",
    "multi-color": "multi",
    "multicolor": "multi",
    "multicolored": "multi",
    "off-white": "ivory",
    "natural": "beige",
}

DIMENSION_PATTERNS = [
    re.compile(r"(\d+\.?\d*)\s*[xX×]\s*(\d+\.?\d*)"),
    re.compile(r"(\d+\.?\d*)\s+by\s+(\d+\.?\d*)"),
    re.compile(r"(\d+\.?\d*)\s*ft?\s*[xX×]\s*(\d+\.?\d*)\s*ft?"),
    re.compile(r"(\d+\.?\d*)\s*'?\s*[xX×]\s*(\d+\.?\d*)\s*'?"),
]

COLOR_PRIORITY = {"navy": 0, "gray": 1, "grey": 1, "multi": 2, "multicolor": 2, "neutral": 3}
ADJECTIVE_PATTERN = re.compile(r"\b(?:dark|light|deep|pale|soft|rich|bright)\s+")


def _normalize_text(query: str) -> str:
    text = unicodedata.normalize("NFKD", query)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = text.replace("×", "x")
    for source, target in sorted(SYNONYMS.items(), key=lambda item: len(item[0]), reverse=True):
        text = re.sub(rf"\b{re.escape(source)}\b", target, text)
    text = ADJECTIVE_PATTERN.sub("", text)
    text = re.sub(r"[^a-z0-9\s.xby'/-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_dimensions(text: str) -> tuple[dict[str, float] | None, str]:
    working = text
    for pattern in DIMENSION_PATTERNS:
        match = pattern.search(working)
        if not match:
            continue
        width = float(match.group(1))
        length = float(match.group(2))
        if width >= 30 or length >= 30:
            continue
        width, length = sorted((width, length))
        return {"width": width, "length": length}, working

    round_match = re.search(r"\bround\b(?:\s+(\d+\.?\d*))?\s*ft?\b", working)
    if round_match:
        diameter = round_match.group(1)
        if diameter:
            value = float(diameter)
            if value < 30:
                return {"width": value, "length": value}, working
    round_match = re.search(r"\b(\d+\.?\d*)\s*ft?\s*round\b", working)
    if round_match:
        value = float(round_match.group(1))
        if value < 30:
            return {"width": value, "length": value}, working
    return None, working


def _contains_word(text: str, candidate: str) -> bool:
    return bool(re.search(rf"\b{re.escape(candidate)}\b", text))


def _extract_colors(text: str) -> list[str]:
    matches: list[str] = []
    for color in COLORS:
        if color in {"gray", "grey"} and (_contains_word(text, "gray") or _contains_word(text, "grey")):
            normalized = "grey"
        else:
            if not _contains_word(text, color):
                continue
            normalized = "multi" if color in {"multi", "multicolor"} else color

        if normalized not in matches:
            matches.append(normalized)

    if "neutral" in matches:
        return ["neutral"]

    if "navy" in matches and "blue" in matches:
        matches = [color for color in matches if color != "blue"]

    if "multi" in matches and "multicolor" in matches:
        matches = [color for color in matches if color != "multicolor"]

    return matches


def _extract_styles(text: str) -> list[str]:
    matches: list[str] = []
    for style in STYLES:
        if _contains_word(text, style):
            normalized = "bohemian" if style == "boho" else style
            if normalized not in matches:
                matches.append(normalized)
    return matches


def _extract_size_class(text: str, dimensions: dict[str, float] | None, shape: str | None) -> str | None:
    if dimensions is not None:
        return None
    if shape == "runner":
        return None
    for size_class in ["xlarge", "large", "medium", "small"]:
        if _contains_word(text, size_class):
            return size_class
    if _contains_word(text, "oversized"):
        return "xlarge"
    if _contains_word(text, "accent"):
        return "small"
    if _contains_word(text, "big"):
        return "large"
    return None


def parse_query(query: str) -> dict:
    if not query or not query.strip():
        return {}

    text = _normalize_text(query)
    dimensions, text = _extract_dimensions(text)

    shape = "rectangle"
    if _contains_word(text, "runner"):
        shape = "runner"
    elif _contains_word(text, "round") or _contains_word(text, "circular"):
        shape = "round"
    elif _contains_word(text, "square"):
        shape = "square"

    colors = _extract_colors(text)
    styles = _extract_styles(text)
    size_class = _extract_size_class(text, dimensions, shape)

    if dimensions is None and shape == "runner" and size_class is None:
        size_class = None

    return {
        "dimensions": dimensions,
        "shape": shape,
        "color": colors,
        "style": styles,
        "size_class": size_class,
    }


def _demo() -> None:
    examples = [
        "8x10 beige traditional rug",
        "round 6ft navy boho rug",
        "runner 2x10 ivory",
    ]
    for example in examples:
        print(example, "->", parse_query(example))


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Parse a rug search query.")
    parser.add_argument("query", nargs="*", help="Free-text query")
    args = parser.parse_args()

    if args.query:
        print(json.dumps(parse_query(" ".join(args.query)), indent=2))
    else:
        _demo()
