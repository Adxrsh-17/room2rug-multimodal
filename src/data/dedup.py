from __future__ import annotations

import numpy as np
from PIL import Image


def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Compute a 64-bit difference hash (dHash) for an image."""
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized)
    diff = pixels[:, 1:] > pixels[:, :-1]
    val = 0
    for i, bit in enumerate(diff.flatten()):
        if bit:
            val |= 1 << i
    return val


def hamming_distance(hash1: int, hash2: int) -> int:
    """Compute the Hamming distance between two integer hashes."""
    return bin(hash1 ^ hash2).count("1")


def deduplicate_images_by_hash(
    images: list[tuple[str, Image.Image]],
    max_hamming_distance: int = 4,
) -> tuple[list[tuple[str, Image.Image]], list[str]]:
    """
    Deduplicate a list of (image_id, PIL.Image) items using perceptual hashing.
    Returns (kept_items, dropped_ids).
    """
    kept: list[tuple[str, Image.Image]] = []
    kept_hashes: list[int] = []
    dropped: list[str] = []

    for image_id, img in images:
        h = compute_dhash(img)
        is_duplicate = False
        for kh in kept_hashes:
            if hamming_distance(h, kh) <= max_hamming_distance:
                is_duplicate = True
                dropped.append(image_id)
                break
        if not is_duplicate:
            kept.append((image_id, img))
            kept_hashes.append(h)

    return kept, dropped


def deduplicate_embeddings(
    ids: list[str],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.98,
) -> tuple[list[str], list[str]]:
    """
    Deduplicate items using pairwise cosine similarity on unit-normalized embeddings.
    Returns (kept_ids, dropped_ids).
    """
    if len(ids) == 0:
        return [], []

    kept_indices: list[int] = []
    dropped_ids: list[str] = []

    for i in range(len(ids)):
        vec_i = embeddings[i]
        is_dup = False
        for k_idx in kept_indices:
            sim = float(np.dot(vec_i, embeddings[k_idx]))
            if sim >= similarity_threshold:
                is_dup = True
                dropped_ids.append(ids[i])
                break
        if not is_dup:
            kept_indices.append(i)

    kept_ids = [ids[idx] for idx in kept_indices]
    return kept_ids, dropped_ids
