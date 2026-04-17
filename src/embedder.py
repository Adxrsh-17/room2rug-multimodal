from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageOps

try:
    import requests
    import torch
    from transformers import CLIPModel, CLIPProcessor
except Exception as exc:  # pragma: no cover - dependency import guard
    requests = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    CLIPModel = None  # type: ignore[assignment]
    CLIPProcessor = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


@dataclass(frozen=True)
class EmbeddingPair:
    image_embedding: np.ndarray | None
    text_embedding: np.ndarray | None


class CLIPEmbedder:
    def __init__(self, model_name: str = CLIP_MODEL_NAME) -> None:
        self.model_name = model_name
        self._processor = None
        self._model = None

    @property
    def available(self) -> bool:
        return _IMPORT_ERROR is None and CLIPModel is not None and CLIPProcessor is not None and torch is not None

    @property
    def processor(self):
        if not self.available:
            raise ImportError("transformers, torch, and requests are required for CLIP embeddings") from _IMPORT_ERROR
        if self._processor is None:
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
        return self._processor

    @property
    def model(self):
        if not self.available:
            raise ImportError("transformers, torch, and requests are required for CLIP embeddings") from _IMPORT_ERROR
        if self._model is None:
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._model.eval()
        return self._model

    def _normalize(self, tensor: "torch.Tensor") -> "torch.Tensor":
        return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        image = image.copy()
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image).convert("RGB")
        else:
            image = image.convert("RGB")
        image.thumbnail((1024, 1024))
        return image

    def _load_image(self, image_source: object) -> Image.Image:
        if isinstance(image_source, Image.Image):
            return self._prepare_image(image_source)

        if isinstance(image_source, (str, Path)):
            text_source = str(image_source)
            parsed = urlparse(text_source)
            if parsed.scheme in {"http", "https"}:
                if requests is None:
                    raise ImportError("requests is required to download remote images") from _IMPORT_ERROR
                for timeout in (10, 15):
                    try:
                        response = requests.get(text_source, timeout=timeout)
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content))
                        return self._prepare_image(image)
                    except Exception:
                        continue
                raise ValueError(f"Unable to download image from URL: {text_source}")

            image = Image.open(text_source)
            return self._prepare_image(image)

        raise TypeError("image_source must be a file path, URL, or PIL Image")

    def _l2_numpy(self, tensor: "torch.Tensor") -> np.ndarray:
        normalized = self._normalize(tensor).detach().cpu().numpy()
        return normalized.squeeze().astype(np.float32)

    def embed_image(self, image_source: object) -> np.ndarray | None:
        if not self.available:
            print("Warning: CLIP dependencies are unavailable; image embedding skipped.")
            return None

        try:
            image = self._load_image(image_source)
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            return self._l2_numpy(features)
        except Exception as exc:
            print(f"Warning: failed to embed image ({exc})")
            return None

    def embed_text(self, text: str) -> np.ndarray | None:
        if not self.available:
            print("Warning: CLIP dependencies are unavailable; text embedding skipped.")
            return None

        text = (text or "").strip()
        if not text:
            return None

        words = text.split()
        if len(words) > 60:
            text = " ".join(words[:60])
        if any(ord(char) > 127 for char in text):
            print("Warning: non-English text may embed less reliably with CLIP.")

        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)
            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
            return self._l2_numpy(features)
        except Exception as exc:
            print(f"Warning: failed to embed text ({exc})")
            return None

    def embed_texts(self, texts: Iterable[str]) -> list[np.ndarray | None]:
        return [self.embed_text(text) for text in texts]

    def embed_images(self, image_sources: Iterable[object]) -> list[np.ndarray | None]:
        return [self.embed_image(image_source) for image_source in image_sources]


def embed_product_text(product: dict[str, Any], embedder: CLIPEmbedder | None = None) -> np.ndarray | None:
    embedder = embedder or CLIPEmbedder()
    title = str(product.get("title") or "").strip()
    tags = str(product.get("tags") or "").strip()
    text = " ".join(part for part in [title, tags] if part).strip()
    if not text:
        return None
    return embedder.embed_text(text)


def embed_product_image(product: dict[str, Any], embedder: CLIPEmbedder | None = None) -> np.ndarray | None:
    embedder = embedder or CLIPEmbedder()
    image_url = str(product.get("image_url") or "").strip()
    if not image_url:
        return None
    return embedder.embed_image(image_url)


def embed_product_record(product: dict[str, Any], embedder: CLIPEmbedder | None = None) -> EmbeddingPair:
    embedder = embedder or CLIPEmbedder()
    return EmbeddingPair(
        image_embedding=embed_product_image(product, embedder),
        text_embedding=embed_product_text(product, embedder),
    )


def embed_product_catalog(catalog: list[dict[str, Any]], embedder: CLIPEmbedder | None = None) -> list[dict[str, Any]]:
    import time

    embedder = embedder or CLIPEmbedder()
    image_cache: dict[str, np.ndarray | None] = {}
    embedded_products: list[dict[str, Any]] = []

    for product in catalog:
        title = str(product.get("title") or "").strip()
        tags = str(product.get("tags") or "").strip()
        if not title and not tags:
            print(f"Warning: skipping product without title/tags: {product.get('handle')}")
            continue

        text_embedding = embedder.embed_text(" ".join(part for part in [title, tags] if part).strip())

        image_url = str(product.get("image_url") or "").strip()
        if image_url in image_cache:
            image_embedding = image_cache[image_url]
        else:
            if image_url.startswith("http") and text_embedding is not None:
                image_embedding = text_embedding
            else:
                image_embedding = embedder.embed_image(image_url) if image_url else None
                if image_embedding is None and text_embedding is not None:
                    image_embedding = text_embedding
            image_cache[image_url] = image_embedding
            if image_url and not image_url.startswith("http"):
                time.sleep(0.1)
        if text_embedding is None and image_embedding is not None:
            text_embedding = image_embedding

        embedded_products.append(
            {
                **product,
                "image_embedding": image_embedding,
                "text_embedding": text_embedding,
            }
        )

    return embedded_products


@lru_cache(maxsize=1)
def _shared_embedder() -> CLIPEmbedder:
    return CLIPEmbedder()


def embed_image(image_source: object) -> np.ndarray | None:
    return _shared_embedder().embed_image(image_source)


def embed_text(text: str) -> np.ndarray | None:
    return _shared_embedder().embed_text(text)


if __name__ == "__main__":
    import json

    demo = CLIPEmbedder()
    print("CLIP available:", demo.available)
    print("Empty text:", demo.embed_text(""))
    print("Broken image URL:", demo.embed_image("https://example.com/does-not-exist.jpg"))
