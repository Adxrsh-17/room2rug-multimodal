import numpy as np
from PIL import Image

from src.data.dedup import (
    compute_dhash,
    deduplicate_embeddings,
    deduplicate_images_by_hash,
    hamming_distance,
)


def test_dhash_identical_images():
    arr = np.random.RandomState(42).randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img1 = Image.fromarray(arr)
    img2 = Image.fromarray(arr)
    h1 = compute_dhash(img1)
    h2 = compute_dhash(img2)
    assert hamming_distance(h1, h2) == 0


def test_deduplicate_images_by_hash():
    # Use distinct seeded noise to ensure realistic entropy
    arr1 = np.random.RandomState(100).randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img1 = Image.fromarray(arr1)
    img2 = Image.fromarray(arr1)  # exact duplicate

    arr3 = np.random.RandomState(999).randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img3 = Image.fromarray(arr3)  # completely distinct image

    items = [("img_1", img1), ("img_2", img2), ("img_3", img3)]
    kept, dropped = deduplicate_images_by_hash(items, max_hamming_distance=4)

    kept_ids = [item[0] for item in kept]
    assert "img_1" in kept_ids
    assert "img_2" in dropped
    assert "img_3" in kept_ids


def test_deduplicate_embeddings():
    ids = ["vec_1", "vec_2", "vec_3"]
    # vec_1 and vec_2 are nearly identical
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.999, 0.01, 0.0], dtype=np.float32)
    v2 /= np.linalg.norm(v2)
    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    emb = np.vstack([v1, v2, v3])
    kept_ids, dropped_ids = deduplicate_embeddings(ids, emb, similarity_threshold=0.98)

    assert "vec_1" in kept_ids
    assert "vec_2" in dropped_ids
    assert "vec_3" in kept_ids
