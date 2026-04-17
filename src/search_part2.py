from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from .embedder import CLIPEmbedder, embed_product_catalog
from .explainer import explain_match
from .fusion import annotate_confidence, determine_mode, merge_candidates, rank_candidates
from .indexer import build_index_with_ids, load_index, save_index, search_index, should_rebuild_index
from .preprocess import build_catalog

INDEX_DIR = Path("index")
IMAGE_INDEX_PATH = INDEX_DIR / "image.index"
IMAGE_IDS_PATH = INDEX_DIR / "image_ids.npy"
IMAGE_METADATA_PATH = INDEX_DIR / "image_metadata.json"
TEXT_INDEX_PATH = INDEX_DIR / "text.index"
TEXT_IDS_PATH = INDEX_DIR / "text_ids.npy"
TEXT_METADATA_PATH = INDEX_DIR / "text_metadata.json"

ROOM_TYPES = ["living room", "bedroom", "dining room", "hallway", "office", "kids room"]
PALETTES = ["warm tones", "cool tones", "neutral tones", "dark tones", "bright colors"]
ROOM_STYLES = ["modern minimalist", "traditional classic", "bohemian eclectic", "rustic farmhouse", "coastal"]


def _safe_lower(text: str) -> str:
    return text.lower().strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()


def _classify_room_context(image_embedding: Any, embedder: CLIPEmbedder) -> dict[str, dict[str, Any]]:
    context: dict[str, dict[str, Any]] = {}
    if image_embedding is None:
        return context

    for label_type, labels in {
        "room_type": ROOM_TYPES,
        "palette": PALETTES,
        "style": ROOM_STYLES,
    }.items():
        label_embeddings = embedder.embed_texts(labels)
        scores: list[float] = []
        for label, embedding in zip(labels, label_embeddings, strict=False):
            if embedding is None:
                scores.append(-1.0)
            else:
                scores.append(float(image_embedding @ embedding))
        if not scores:
            continue
        best_index = max(range(len(scores)), key=scores.__getitem__)
        context[label_type] = {
            "label": labels[best_index],
            "score": scores[best_index],
        }
    return context


def _augment_query(text_query: str, room_context: dict[str, dict[str, Any]]) -> tuple[str, str]:
    base_query = text_query.strip()
    if not base_query:
        return base_query, ""

    additions: list[str] = []
    seen_tokens = set(base_query.lower().split())
    for key in ["palette", "room_type", "style"]:
        label_info = room_context.get(key)
        if not label_info:
            continue
        label = str(label_info.get("label") or "").strip()
        score = float(label_info.get("score") or 0.0)
        if score < 0.2 or not label:
            continue
        if any(token in seen_tokens for token in label.lower().split()):
            continue
        additions.append(label)

    augmented = " ".join(part for part in [base_query] + additions if part).strip()
    room_description = ", ".join(
        str(room_context[key]["label"])
        for key in ["room_type", "palette", "style"]
        if room_context.get(key) and float(room_context[key].get("score") or 0.0) >= 0.2
    )
    return augmented, room_description


def _build_product_lookup(catalog: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(product.get("handle")): product for product in catalog if product.get("handle")}


def _build_or_load_indexes(catalog: list[dict[str, Any]], embedder: CLIPEmbedder) -> tuple[Any, list[str], Any, list[str]]:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if (
        IMAGE_INDEX_PATH.exists()
        and IMAGE_IDS_PATH.exists()
        and IMAGE_METADATA_PATH.exists()
        and TEXT_INDEX_PATH.exists()
        and TEXT_IDS_PATH.exists()
        and TEXT_METADATA_PATH.exists()
        and not should_rebuild_index(IMAGE_METADATA_PATH, len(catalog))
        and not should_rebuild_index(TEXT_METADATA_PATH, len(catalog))
    ):
        image_index, image_ids, _ = load_index(IMAGE_INDEX_PATH, IMAGE_IDS_PATH, IMAGE_METADATA_PATH)
        text_index, text_ids, _ = load_index(TEXT_INDEX_PATH, TEXT_IDS_PATH, TEXT_METADATA_PATH)
        return image_index, image_ids, text_index, text_ids

    embedded_catalog = embed_product_catalog(catalog, embedder)
    image_embeddings = [product.get("image_embedding") for product in embedded_catalog]
    text_embeddings = [product.get("text_embedding") for product in embedded_catalog]
    product_ids = [str(product.get("handle")) for product in embedded_catalog]

    image_collection = build_index_with_ids(image_embeddings, product_ids)
    text_collection = build_index_with_ids(text_embeddings, product_ids)

    save_index(
        image_collection.index,
        IMAGE_INDEX_PATH,
        image_collection.product_ids,
        IMAGE_IDS_PATH,
        IMAGE_METADATA_PATH,
        product_count=len(catalog),
    )
    save_index(
        text_collection.index,
        TEXT_INDEX_PATH,
        text_collection.product_ids,
        TEXT_IDS_PATH,
        TEXT_METADATA_PATH,
        product_count=len(catalog),
    )

    return image_collection.index, image_collection.product_ids, text_collection.index, text_collection.product_ids


def search_multimodal(
    room_image: object | None,
    text_query: str = "",
    top_k: int = 10,
    csv_path: str | Path = "data/products.csv",
    include_explanations: bool = False,
) -> list[dict[str, Any]]:
    if room_image is None and not text_query.strip():
        raise ValueError("At least one of room_image or text_query must be provided")

    catalog = build_catalog(csv_path)
    if not catalog:
        raise ValueError("Catalog is empty")

    embedder = CLIPEmbedder()
    if not embedder.available:
        raise RuntimeError("CLIP dependencies are unavailable. Install transformers, torch, requests, and pillow.")

    room_vector = embedder.embed_image(room_image) if room_image is not None else None
    mode = determine_mode(room_vector is not None, bool(text_query.strip()))

    augmented_query = text_query.strip()
    room_description = ""
    if room_vector is not None and text_query.strip():
        room_context = _classify_room_context(room_vector, embedder)
        augmented_query, room_description = _augment_query(text_query, room_context)

    text_vector = embedder.embed_text(augmented_query) if augmented_query.strip() else None
    if room_vector is None and text_vector is None:
        raise ValueError("Unable to build a query embedding from the provided inputs")

    image_index, image_ids, text_index, text_ids = _build_or_load_indexes(catalog, embedder)
    product_lookup = _build_product_lookup(catalog)

    image_hits: list[tuple[str, float]] = []
    text_hits: list[tuple[str, float]] = []

    if room_vector is not None:
        image_positions, image_scores = search_index(image_index, room_vector, k=20)
        for position, score in zip(image_positions, image_scores, strict=False):
            if 0 <= position < len(image_ids):
                image_hits.append((image_ids[position], score))

    if text_vector is not None:
        text_positions, text_scores = search_index(text_index, text_vector, k=20)
        for position, score in zip(text_positions, text_scores, strict=False):
            if 0 <= position < len(text_ids):
                text_hits.append((text_ids[position], score))

    candidates = merge_candidates(image_hits, text_hits, product_lookup, mode)
    candidates = annotate_confidence(candidates)
    candidates = rank_candidates(candidates)

    if not candidates:
        return []

    if candidates and all(float(candidate.get("fusion_score", 0.0)) < 0.15 for candidate in candidates):
        limit = min(len(candidates), max(5, min(top_k, len(candidates))))
    else:
        limit = min(top_k, len(candidates))

    results: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates[:limit], start=1):
        result = {
            "rank": rank,
            "handle": candidate.get("handle"),
            "title": candidate.get("title"),
            "image_url": candidate.get("image_url") or None,
            "price": candidate.get("price"),
            "fusion_score": round(float(candidate.get("fusion_score", 0.0)), 4),
            "image_similarity": round(float(candidate.get("image_similarity", 0.0)), 4),
            "text_similarity": round(float(candidate.get("text_similarity", 0.0)), 4),
            "mode": mode,
            "low_confidence": bool(candidate.get("low_confidence", False)),
        }
        if include_explanations:
            result["explanation"] = explain_match(
                candidate,
                room_description,
                augmented_query or text_query,
            )
        results.append(result)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the multimodal rug search pipeline.")
    parser.add_argument("room_image", nargs="?", default="images/room1.jpeg", help="Path or URL to a room image")
    parser.add_argument("--text", default="modern neutral", help="Optional text query")
    parser.add_argument("--csv", default="data/products.csv", help="Path to catalog CSV")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    parser.add_argument("--explain", action="store_true", help="Attach one-sentence explanations")
    args = parser.parse_args()

    try:
        results = search_multimodal(args.room_image, args.text, top_k=args.top_k, csv_path=args.csv, include_explanations=args.explain)
        print(json.dumps(results, indent=2))
    except Exception as exc:
        print(f"Multimodal search failed: {exc}")
