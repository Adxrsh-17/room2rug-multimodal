from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency guard
    SentenceTransformer = None  # type: ignore[assignment]

from .filter_catalog import metadata_score

TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
MIN_SCORE_THRESHOLD = 0.15


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer | None:
    if SentenceTransformer is None:
        return None  # type: ignore[return-value]
    try:
        return SentenceTransformer(TEXT_MODEL_NAME)
    except Exception:
        return None  # type: ignore[return-value]


@lru_cache(maxsize=8)
def _cache_key(texts_key: tuple[str, ...]) -> tuple[tuple[str, ...], np.ndarray]:
    model = _get_model()
    embeddings = model.encode(list(texts_key), normalize_embeddings=True, show_progress_bar=False)
    return texts_key, np.asarray(embeddings, dtype="float32")


def _build_searchable_text(product: dict[str, Any]) -> str:
    title = str(product.get("title") or "")
    tags = str(product.get("tags") or "")
    body = str(product.get("body") or product.get("body_html") or "")
    body = body[:300]
    text = " ".join(part for part in [title, tags, body] if part)
    return text[:2000]


def _product_texts(catalog: list[dict[str, Any]]) -> list[str]:
    return [_build_searchable_text(product) for product in catalog]


def _clip_negative(values: np.ndarray) -> np.ndarray:
    clipped = values.copy()
    clipped[clipped < 0] = 0
    return clipped


def rank_products(parsed_query: dict[str, Any], catalog: list[dict[str, Any]], raw_query: str, top_k: int = 10) -> list[dict[str, Any]]:
    if not catalog:
        raise ValueError("Catalog is empty")

    model = _get_model()
    query_embedding = None
    if model is not None:
        query_embedding = model.encode([raw_query], normalize_embeddings=True, show_progress_bar=False)
        query_embedding = np.asarray(query_embedding, dtype="float32")
    else:
        print("Warning: sentence-transformers is unavailable; ranking with metadata only.")

    texts = tuple(_product_texts(catalog))
    similarities = np.zeros(len(catalog), dtype="float32")
    if model is not None and query_embedding is not None:
        _, embeddings = _cache_key(texts)
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        similarities = _clip_negative(similarities)

    ranked: list[dict[str, Any]] = []
    for index, product in enumerate(catalog):
        metadata = metadata_score(product, parsed_query)
        semantic = float(similarities[index])
        final_score = 0.5 * metadata + 0.5 * semantic
        ranked.append(
            {
                "rank": 0,
                "handle": product.get("handle"),
                "title": product.get("title"),
                "image_url": product.get("image_url") or None,
                "price": product.get("price"),
                "matched_size": product.get("size") or (product.get("sizes") or [None])[0],
                "final_score": round(float(final_score), 4),
                "metadata_score": round(float(metadata), 4),
                "semantic_score": round(float(semantic), 4),
            }
        )

    ranked.sort(key=lambda item: (item["final_score"], float(item.get("price") or 0)), reverse=True)

    filtered = [item for item in ranked if item["final_score"] >= MIN_SCORE_THRESHOLD]
    if len(filtered) < 5:
        print("Warning: fewer than 5 products met the minimum score threshold; returning available matches.")
        for index, item in enumerate(filtered[:top_k], start=1):
            item["rank"] = index
        return filtered[:top_k]

    for index, item in enumerate(filtered[:top_k], start=1):
        item["rank"] = index
    return filtered[:top_k]


def rank_results(parsed_query: dict[str, Any], catalog: list[dict[str, Any]], raw_query: str, top_k: int = 10) -> list[dict[str, Any]]:
    return rank_products(parsed_query, catalog, raw_query, top_k=top_k)


if __name__ == "__main__":
    from pathlib import Path
    import json

    from .preprocess import build_catalog
    from .query_parser import parse_query

    catalog_path = Path("data/products.csv")
    catalog = build_catalog(catalog_path) if catalog_path.exists() else []
    parsed = parse_query("beige traditional 8x10 rug")
    print(json.dumps(rank_products(parsed, catalog, "beige traditional 8x10 rug", top_k=3), indent=2))
