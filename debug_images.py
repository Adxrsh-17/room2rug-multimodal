"""Debug: simulate exactly what app.py does for the palace rug images."""
import requests
from io import BytesIO
from pathlib import Path
import sys

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.search_part1 import search_part1

# Run same search as the UI
results = search_part1("8x10 beige traditional rug",
                       csv_path=ROOT / "data" / "products.csv", top_k=6)

print(f"Got {len(results)} results\n")
for r in results:
    title = r.get("title", "?")
    url   = r.get("image_url") or ""
    print(f"#{r.get('rank')} {title[:50]}")
    print(f"   url: {repr(url[:80])}")

    if not url:
        print("   SKIP: no url")
        continue

    # Fix protocol-relative URLs  
    fetch_url = ("https:" + url) if url.startswith("//") else url

    try:
        resp = requests.get(fetch_url, timeout=10,
                            headers={"User-Agent": "Mozilla/5.0 (compatible)"})
        print(f"   HTTP {resp.status_code}, {len(resp.content)} bytes, "
              f"Content-Type: {resp.headers.get('Content-Type','?')}")

        # Try opening with PIL to confirm it's a valid image
        from PIL import Image
        img = Image.open(BytesIO(resp.content))
        print(f"   PIL: {img.format} {img.size} {img.mode} -> OK")
    except Exception as e:
        print(f"   ERROR: {e}")
    print()
