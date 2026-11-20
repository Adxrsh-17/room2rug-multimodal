from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import faiss
except Exception:  # pragma: no cover - optional dependency
    faiss = None  # type: ignore[assignment]

FAISS_DIMENSION = 512


@dataclass(frozen=True)
class IndexedCollection:
    index: Any
    product_ids: list[str]
    vectors: np.ndarray | None = None
    backend: str = "numpy"


class NumpyInnerProductIndex:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = np.asarray(vectors, dtype="float32")
        if self.vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        self.dimension = self.vectors.shape[1]

    def search(self, query: np.ndarray, k: int = 20) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(query, dtype="float32").reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise ValueError("Query dimension does not match index dimension")
        scores = query @ self.vectors.T
        scores = np.clip(scores, 0.0, 1.0)
        top_k = min(k, self.vectors.shape[0])
        if top_k == 0:
            return np.empty((1, 0), dtype=np.float32), np.empty((1, 0), dtype=np.int64)
        indices = np.argsort(-scores[0])[:top_k]
        return scores[:, indices], indices.reshape(1, -1)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    stacked = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return stacked / norms


def _stack_embeddings(embeddings: Iterable[np.ndarray | None]) -> np.ndarray:
    vectors = [np.asarray(vector, dtype="float32") for vector in embeddings if vector is not None]
    if not vectors:
        raise RuntimeError("No valid product embeddings found. Check image URLs.")
    stacked = np.vstack(vectors).astype("float32")
    return _normalize_vectors(stacked)


def build_index(embeddings: list[np.ndarray | None]) -> Any:
    valid = _stack_embeddings(embeddings)
    if faiss is not None:
        index = faiss.IndexFlatIP(valid.shape[1])
        index.add(valid)
        return index
    return NumpyInnerProductIndex(valid)


def build_index_with_ids(embeddings: list[np.ndarray | None], product_ids: list[str]) -> IndexedCollection:
    valid_vectors: list[np.ndarray] = []
    valid_ids: list[str] = []
    for product_id, vector in zip(product_ids, embeddings, strict=False):
        if vector is None:
            continue
        valid_vectors.append(np.asarray(vector, dtype="float32"))
        valid_ids.append(product_id)

    if not valid_vectors:
        raise RuntimeError("No valid product embeddings found. Check image URLs.")

    stacked = _normalize_vectors(np.vstack(valid_vectors).astype("float32"))
    if faiss is not None:
        index = faiss.IndexFlatIP(stacked.shape[1])
        index.add(stacked)
        return IndexedCollection(index=index, product_ids=valid_ids, vectors=stacked, backend="faiss")

    index = NumpyInnerProductIndex(stacked)
    return IndexedCollection(index=index, product_ids=valid_ids, vectors=stacked, backend="numpy")


def search_index(index: Any, query_embedding: np.ndarray, k: int = 20) -> tuple[list[int], list[float]]:
    if index is None:
        raise ValueError("Index has not been built")

    query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
    if hasattr(index, "search"):
        scores, positions = index.search(query, k)
        positions_array = np.asarray(positions).reshape(-1)
        scores_array = np.asarray(scores).reshape(-1)
    else:
        raise TypeError("Unsupported index type")

    valid_pairs = [
        (int(position), float(max(0.0, min(1.0, score))))
        for position, score in zip(positions_array, scores_array, strict=False)
        if int(position) != -1
    ]
    if not valid_pairs:
        return [], []
    positions_out, scores_out = zip(*valid_pairs, strict=False)
    return list(positions_out), list(scores_out)


def _index_vector_path(index_path: str | Path) -> Path:
    return Path(str(index_path) + ".npy")


def save_index(index: Any, index_path: str | Path, product_ids: list[str], ids_path: str | Path, metadata_path: str | Path, product_count: int) -> None:
    index_path = Path(index_path)
    ids_path = Path(ids_path)
    metadata_path = Path(metadata_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    vectors = None
    if hasattr(index, "vectors"):
        vectors = np.asarray(index.vectors, dtype="float32")
    elif isinstance(index, np.ndarray):
        vectors = np.asarray(index, dtype="float32")

    if faiss is not None and not isinstance(index, NumpyInnerProductIndex) and vectors is None:
        faiss.write_index(index, str(index_path))
    else:
        if vectors is None:
            raise ValueError("Fallback index requires vector data to save")
        np.save(_index_vector_path(index_path), vectors)

    np.save(ids_path, np.asarray(product_ids, dtype=object), allow_pickle=True)
    metadata = {"product_count": int(product_count), "backend": "faiss" if faiss is not None and vectors is None else "numpy", "built_at": __import__("datetime").datetime.utcnow().isoformat()}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _load_single_index(index_path: Path) -> Any:
    if faiss is not None and index_path.exists():
        return faiss.read_index(str(index_path))

    vector_path = _index_vector_path(index_path)
    if vector_path.exists():
        vectors = np.load(vector_path)
        return NumpyInnerProductIndex(vectors)

    if faiss is not None and vector_path.exists():
        # Defensive fallback if faiss is available but the explicit vector file exists.
        vectors = np.load(vector_path)
        return NumpyInnerProductIndex(vectors)

    raise FileNotFoundError(f"No saved index found for {index_path}")


def load_index(index_path: str | Path, ids_path: str | Path | None = None, metadata_path: str | Path | None = None):
    index_path = Path(index_path)

    if ids_path is None and metadata_path is None:
        image_index = _load_single_index(index_path / "image.index")
        text_index = _load_single_index(index_path / "text.index")
        product_ids_path = index_path / "product_ids.npy"
        if product_ids_path.exists():
            product_ids = np.load(product_ids_path, allow_pickle=True).tolist()
        else:
            image_ids = np.load(index_path / "image_ids.npy", allow_pickle=True).tolist()
            text_ids = np.load(index_path / "text_ids.npy", allow_pickle=True).tolist()
            product_ids = list(dict.fromkeys(list(image_ids) + list(text_ids)))
        return image_index, text_index, list(product_ids)

    if ids_path is None or metadata_path is None:
        raise ValueError("Both ids_path and metadata_path must be provided when loading a single index")

    index = _load_single_index(index_path)
    product_ids = np.load(ids_path, allow_pickle=True).tolist()
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return index, list(product_ids), metadata


def should_rebuild_index(metadata_path: str | Path, current_count: int) -> bool:
    path = Path(metadata_path)
    if not path.exists():
        return True
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return True
    return int(metadata.get("product_count", -1)) != int(current_count)


if __name__ == "__main__":
    demo_vectors = [np.array([1.0] * FAISS_DIMENSION, dtype=np.float32), None, np.array([0.5] * FAISS_DIMENSION, dtype=np.float32)]
    demo_ids = ["a", "b", "c"]
    try:
        collection = build_index_with_ids(demo_vectors, demo_ids)
        positions, scores = search_index(collection.index, demo_vectors[0], k=2)
        print("positions:", positions)
        print("scores:", scores)
    except Exception as exc:
        print(f"Index demo failed: {exc}")
