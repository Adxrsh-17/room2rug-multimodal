from __future__ import annotations

"""Unified search entry point — exposes both Part 1 and Part 2 pipelines."""

from .search_part1 import search_part1, search_structured
from .search_part2 import search_multimodal

__all__ = ["search_part1", "search_structured", "search_multimodal"]


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Multimodal Rug Search — run Part 1 (text) or Part 2 (image+text)."
    )
    subparsers = parser.add_subparsers(dest="command")

    # Part 1 sub-command
    p1 = subparsers.add_parser("text", help="Part 1: structured text search")
    p1.add_argument("query", nargs="*", help="Free-text query")
    p1.add_argument("--csv", default="data/products.csv", help="Path to catalog CSV")
    p1.add_argument("--top-k", type=int, default=10, help="Number of results")

    # Part 2 sub-command
    p2 = subparsers.add_parser("image", help="Part 2: multimodal image+text search")
    p2.add_argument("room_image", nargs="?", default=None, help="Path or URL to a room image")
    p2.add_argument("--text", default="", help="Optional text query")
    p2.add_argument("--csv", default="data/products.csv", help="Path to catalog CSV")
    p2.add_argument("--top-k", type=int, default=10, help="Number of results")
    p2.add_argument("--explain", action="store_true", help="Attach match explanations")

    args = parser.parse_args()

    if args.command == "text":
        query = " ".join(args.query).strip()
        results = search_part1(query, args.csv, top_k=args.top_k)
        print(json.dumps(results, indent=2))

    elif args.command == "image":
        try:
            results = search_multimodal(
                args.room_image, args.text,
                top_k=args.top_k, csv_path=args.csv,
                include_explanations=args.explain,
            )
            print(json.dumps(results, indent=2))
        except Exception as exc:
            print(f"Multimodal search failed: {exc}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
