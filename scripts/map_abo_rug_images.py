"""
Maps genuine Amazon Berkeley Objects (ABO) rug image URLs to products.csv
using exact 'RUG' product_type filtering and CLIP text semantic matching.
"""

from __future__ import annotations

import csv
import gzip
import json
import time
from pathlib import Path
import urllib.request
import numpy as np

from src.preprocess import build_catalog
from src.embedder import CLIPEmbedder

CACHE_DIR = Path("scratch/abo_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_with_retry(url: str, dest: Path, max_retries: int = 5) -> None:
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"Using cached: {dest.name}")
        return
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading {url} (attempt {attempt}/{max_retries})...")
            urllib.request.urlretrieve(url, dest)
            print(f"Saved to {dest.name} ({dest.stat().st_size} bytes)")
            return
        except Exception as e:
            print(f"Download error on attempt {attempt}: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts.")


def get_genuine_abo_rugs() -> list[dict]:
    # 1. Download images.csv.gz
    img_csv_path = CACHE_DIR / "images.csv.gz"
    download_with_retry("https://amazon-berkeley-objects.s3.amazonaws.com/images/metadata/images.csv.gz", img_csv_path)

    print("Reading image ID -> path map...")
    with gzip.open(img_csv_path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        image_map = {row["image_id"]: row["path"] for row in reader}

    genuine_abo_rugs = []
    listing_idx = 0

    while len(genuine_abo_rugs) < 250:
        list_url = f"https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_{listing_idx}.json.gz"
        list_path = CACHE_DIR / f"listings_{listing_idx}.json.gz"
        try:
            download_with_retry(list_url, list_path)
        except Exception:
            break

        with gzip.open(list_path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                except Exception:
                    continue

                pts = item.get("product_type", [])
                is_rug = False
                if isinstance(pts, list):
                    is_rug = any(isinstance(p, dict) and p.get("value") == "RUG" for p in pts)
                elif isinstance(pts, str):
                    is_rug = pts.upper() == "RUG"

                if is_rug:
                    img_id = item.get("main_image_id")
                    if img_id and img_id in image_map:
                        item_names = item.get("item_name", [])
                        name = item_names[0].get("value", "") if isinstance(item_names, list) and item_names else str(item_names)
                        bullets = item.get("bullet_point", [])
                        bullet_text = " ".join(b.get("value", "") for b in bullets if isinstance(b, dict)) if isinstance(bullets, list) else str(bullets)

                        if name:
                            genuine_abo_rugs.append({
                                "item_id": item.get("item_id"),
                                "item_name": name,
                                "bullet_text": bullet_text,
                                "product_type": "RUG",
                                "main_image_id": img_id,
                                "image_url": f"https://amazon-berkeley-objects.s3.amazonaws.com/images/small/{image_map[img_id]}",
                                "search_text": f"{name} {bullet_text}".strip()
                            })
        listing_idx += 1

    print(f"Total genuine RUG items found in ABO: {len(genuine_abo_rugs)}")
    return genuine_abo_rugs


def update_catalog_with_semantic_abo_images(csv_path: Path = Path("data/products.csv")) -> None:
    catalog = build_catalog(csv_path)
    abo_rugs = get_genuine_abo_rugs()

    if len(abo_rugs) < len(catalog):
        print(f"Warning: ABO rugs pool ({len(abo_rugs)}) is smaller than catalog ({len(catalog)})")

    embedder = CLIPEmbedder()
    print("Embedding catalog items...")
    cat_texts = [f"{p.get('title', '')} {p.get('tags', '')}".strip() for p in catalog]
    cat_embeddings = [embedder.embed_text(t) for t in cat_texts]

    print("Embedding ABO genuine rugs...")
    abo_embeddings = [embedder.embed_text(r["search_text"]) for r in abo_rugs]

    assigned_urls: dict[str, str] = {}
    available_abo_indices = set(range(len(abo_rugs)))

    for cat_idx, (p, cat_emb) in enumerate(zip(catalog, cat_embeddings)):
        best_abo_idx = None
        best_sim = -1.0

        for abo_idx in available_abo_indices:
            abo_emb = abo_embeddings[abo_idx]
            if cat_emb is not None and abo_emb is not None:
                sim = float(np.dot(cat_emb, abo_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_abo_idx = abo_idx

        if best_abo_idx is not None:
            assigned_urls[p["handle"]] = abo_rugs[best_abo_idx]["image_url"]
            available_abo_indices.remove(best_abo_idx)
        else:
            assigned_urls[p["handle"]] = abo_rugs[cat_idx % len(abo_rugs)]["image_url"]

    # Write back to products.csv
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        fieldnames = reader[0].keys()

    for row in reader:
        h = row.get("Handle", "").strip()
        if h in assigned_urls:
            row["Image Src"] = assigned_urls[h]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reader)

    print(f"Successfully updated Image Src in {csv_path} for all {len(catalog)} products.")


if __name__ == "__main__":
    update_catalog_with_semantic_abo_images()
