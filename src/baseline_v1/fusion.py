from __future__ import annotations

from typing import Any, Iterable

LOW_CONFIDENCE_THRESHOLD = 0.15


def compute_fusion_score(image_sim: float | None, text_sim: float | None, mode: str) -> float:
    if mode == "A":
        return float(image_sim or 0.0)
    if mode == "B":
        image_score = max(0.0, min(1.0, float(image_sim or 0.0)))
        text_score = max(0.0, min(1.0, float(text_sim or 0.0)))
        return 0.6 * image_score + 0.4 * text_score
    if mode == "C":
        return float(text_sim or 0.0)
    raise ValueError(f"Unsupported fusion mode: {mode}")


def determine_mode(image_present: bool, text_present: bool) -> str:
    if image_present and text_present:
        return "B"
    if image_present:
        return "A"
    if text_present:
        return "C"
    raise ValueError("At least one of room_image or text_query must be provided")


def merge_candidates(
    image_hits: list[tuple[str, float]],
    text_hits: list[tuple[str, float]],
    product_lookup: dict[str, dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    for handle, score in image_hits:
        if handle not in product_lookup:
            continue
        entry = merged.setdefault(
            handle,
            {
                **product_lookup[handle],
                "image_similarity": 0.0,
                "text_similarity": 0.0,
                "fusion_score": 0.0,
                "mode": mode,
            },
        )
        entry["image_similarity"] = max(float(entry["image_similarity"]), float(score))

    for handle, score in text_hits:
        if handle not in product_lookup:
            continue
        entry = merged.setdefault(
            handle,
            {
                **product_lookup[handle],
                "image_similarity": 0.0,
                "text_similarity": 0.0,
                "fusion_score": 0.0,
                "mode": mode,
            },
        )
        entry["text_similarity"] = max(float(entry["text_similarity"]), float(score))

    candidates: list[dict[str, Any]] = []
    for entry in merged.values():
        entry["fusion_score"] = compute_fusion_score(entry.get("image_similarity"), entry.get("text_similarity"), mode)
        entry["low_confidence"] = False
        candidates.append(entry)

    candidates.sort(key=lambda item: (float(item.get("fusion_score", 0.0)), float(item.get("price") or 0.0)), reverse=True)
    return candidates


def annotate_confidence(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return candidates
    low_confidence = max(float(candidate.get("fusion_score", 0.0)) for candidate in candidates) < LOW_CONFIDENCE_THRESHOLD
    for candidate in candidates:
        candidate["low_confidence"] = low_confidence
    return candidates


def rank_candidates(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: (float(item.get("fusion_score", 0.0)), float(item.get("price") or 0.0)), reverse=True)


if __name__ == "__main__":
    demo_lookup = {
        "a": {"handle": "a", "title": "Alpha", "price": 100.0, "image_url": None},
        "b": {"handle": "b", "title": "Beta", "price": 200.0, "image_url": None},
    }
    merged = merge_candidates([("a", 0.8)], [("a", 0.5), ("b", 0.9)], demo_lookup, "B")
    print(annotate_confidence(merged))
