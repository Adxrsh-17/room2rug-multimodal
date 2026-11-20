"""Artifact integrity and lineage verification script."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def verify_lineage(manifest_path: str = "lineage.json") -> bool:
    """Verifies that all tracked dataset and model files match their committed SHA256 hashes."""
    if not os.path.exists(manifest_path):
        logger.error("Lineage manifest %s not found!", manifest_path)
        return False

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    artifacts = manifest.get("artifacts", {})
    all_valid = True

    logger.info("Verifying %d tracked artifacts against lineage manifest...", len(artifacts))

    for rel_path, meta in artifacts.items():
        if not os.path.exists(rel_path):
            logger.error("MISSING ARTIFACT: %s does not exist on disk!", rel_path)
            all_valid = False
            continue

        with open(rel_path, "rb") as f:
            content = f.read()

        # Normalize line endings for text artifacts to ensure cross-platform determinism
        if rel_path.endswith((".json", ".csv", ".txt", ".yaml", ".yml")):
            content = content.replace(b"\r\n", b"\n")

        computed_sha256 = hashlib.sha256(content).hexdigest()
        expected_sha256 = meta.get("sha256")
        expected_size = meta.get("size_bytes")
        actual_size = len(content)

        if computed_sha256 != expected_sha256:
            logger.error(
                "CHECKSUM MISMATCH for %s:\n  Expected: %s\n  Computed: %s",
                rel_path,
                expected_sha256,
                computed_sha256,
            )
            all_valid = False
        elif actual_size != expected_size:
            logger.error(
                "FILE SIZE MISMATCH for %s: Expected %d bytes, got %d bytes",
                rel_path,
                expected_size,
                actual_size,
            )
            all_valid = False
        else:
            logger.info("VERIFIED: %s (SHA256: %s...)", rel_path, computed_sha256[:12])

    if all_valid:
        logger.info("All artifacts successfully verified against lineage manifest.")
    else:
        logger.error("Artifact lineage verification failed!")

    return all_valid


if __name__ == "__main__":
    success = verify_lineage()
    sys.exit(0 if success else 1)
