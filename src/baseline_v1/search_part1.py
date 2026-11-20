from __future__ import annotations

from pathlib import Path
from typing import Any

from .filter_catalog import filter_products
from .preprocess import build_catalog
from .query_parser import parse_query
from .ranker import rank_products


def search_part1(query: str, csv_path: str | Path = "data/products.csv", top_k: int = 10) -> list[dict[str, Any]]:
    if not query or not query.strip():
        print("Please enter a search query")
        return []

    catalog = build_catalog(csv_path)
    if not catalog:
        raise ValueError("Catalog is empty")

    parsed_query = parse_query(query)
    filtered_catalog = filter_products(parsed_query, catalog)
    if not filtered_catalog:
        filtered_catalog = catalog

    results = rank_products(parsed_query, filtered_catalog, query, top_k=top_k)
    return results


def search_structured(query: str, catalog: list[dict[str, Any]] | str | Path = "data/products.csv", top_k: int = 10) -> list[dict[str, Any]]:
    if isinstance(catalog, (str, Path)):
        return search_part1(query, catalog, top_k=top_k)

    if not query or not query.strip():
        return []

    parsed_query = parse_query(query)
    filtered_catalog = filter_products(parsed_query, catalog)
    if not filtered_catalog:
        filtered_catalog = catalog
    return rank_products(parsed_query, filtered_catalog, query, top_k=top_k)


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Structured rug search pipeline (Part 1).")
    parser.add_argument("query", nargs="*", help="Free-text query")
    parser.add_argument("--csv", default="data/products.csv", help="Path to catalog CSV")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    args = parser.parse_args()

    query = " ".join(args.query).strip()
    results = search_part1(query, args.csv, top_k=args.top_k)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
