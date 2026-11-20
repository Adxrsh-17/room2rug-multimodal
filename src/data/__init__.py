"""Data pipelines, deduplication, and group-aware splitting."""

from .dedup import compute_dhash, deduplicate_embeddings, deduplicate_images_by_hash, hamming_distance
from .splits import create_group_splits, verify_no_leakage

__all__ = [
    "create_group_splits",
    "verify_no_leakage",
    "compute_dhash",
    "hamming_distance",
    "deduplicate_images_by_hash",
    "deduplicate_embeddings",
]
