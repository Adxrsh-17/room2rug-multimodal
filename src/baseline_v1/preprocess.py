from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

DIMENSION_PATTERN = re.compile(r"(\d+\.?\d*)\s*[xX×]\s*(\d+\.?\d*)")


def strip_html(html: str | None) -> str:
    if not html:
        return ""
    return BeautifulSoup(str(html), "html.parser").get_text(" ", strip=True)


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return " ".join(text.split())


def _first_non_empty(*values: object) -> str:
    for value in values:
        text = _clean_text(value)
        if text:
            return text
    return ""


def _normalize_size_label(value: object) -> str:
    text = _clean_text(value)
    return text.replace("×", "x").replace("X", "x")


def _extract_price(group: pd.DataFrame) -> float | None:
    prices = pd.to_numeric(group.get("Variant Price"), errors="coerce").dropna()
    if prices.empty:
        return None
    return float(prices.iloc[0])


def _extract_primary_row(group: pd.DataFrame) -> pd.Series:
    if "Image Position" in group.columns:
        image_rows = group[pd.to_numeric(group["Image Position"], errors="coerce").fillna(0).eq(1)]
        if not image_rows.empty:
            return image_rows.iloc[0]
    return group.iloc[0]


def _collect_sizes(group: pd.DataFrame) -> list[str]:
    size_values: list[str] = []
    for column in ["Option1 Value", "Option2 Value", "Option3 Value", "Variant Title", "Size"]:
        if column not in group.columns:
            continue
        for value in group[column].tolist():
            text = _normalize_size_label(value)
            if text and text not in size_values:
                if DIMENSION_PATTERN.search(text) or any(keyword in text.lower() for keyword in ["runner", "round", "square", "x"]):
                    size_values.append(text)
    return size_values


def build_catalog(csv_path: str | Path) -> list[dict[str, Any]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Catalog CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Catalog CSV is empty")

    df.columns = [column.strip() for column in df.columns]
    if "Handle" not in df.columns:
        raise ValueError("Catalog CSV must include a Handle column")

    records: list[dict[str, Any]] = []
    for handle, group in df.groupby("Handle", dropna=False):
        group = group.copy()
        primary_row = _extract_primary_row(group)
        title = _first_non_empty(primary_row.get("Title"))
        body_html = _clean_text(primary_row.get("Body (HTML)"))
        body = strip_html(body_html)
        tags = _first_non_empty(primary_row.get("Tags"))
        image_url = _first_non_empty(primary_row.get("Image Src")) or None
        if image_url and image_url.startswith("//"):
            image_url = f"https:{image_url}"
        sizes = _collect_sizes(group)
        size = sizes[0] if sizes else _first_non_empty(primary_row.get("Option1 Value"), primary_row.get("Variant Title"))
        price = _extract_price(group)
        style = _first_non_empty(primary_row.get("Type"), tags)
        color = _first_non_empty(primary_row.get("Option1 Value"), primary_row.get("Option2 Value"), primary_row.get("Option3 Value"))

        searchable_text = " ".join(
            part.lower()
            for part in [title, tags, body]
            if part
        ).strip()

        records.append(
            {
                "handle": _first_non_empty(handle),
                "title": title,
                "body_html": body,
                "body": body,
                "tags": tags,
                "image_url": image_url,
                "size": size,
                "sizes": sizes,
                "price": price,
                "style": style,
                "color": color,
                "searchable_text": searchable_text,
            }
        )

    return records


def load_catalog(csv_path: str | Path) -> list[dict[str, Any]]:
    return build_catalog(csv_path)


def save_catalog_json(records: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_json(output_path, orient="records", indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a clean rug catalog from the Shopify export.")
    parser.add_argument("csv_path", help="Path to the Shopify CSV export")
    parser.add_argument("--output", default="data/products.json", help="Destination JSON file")
    args = parser.parse_args()

    catalog = build_catalog(args.csv_path)
    save_catalog_json(catalog, args.output)
    print(f"Saved {len(catalog)} product records to {args.output}")
