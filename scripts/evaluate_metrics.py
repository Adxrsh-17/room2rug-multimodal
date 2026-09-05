"""
Evaluation & Benchmarking Suite for room2rug-multimodal
Calculates:
1. Information Retrieval Metrics (Recall@K, Precision@K, MRR@10, NDCG@5, NDCG@10, mAP@10)
2. Modality Ablation Study (Metadata-Only vs Text-Only vs Visual CLIP vs Late Fusion)
3. Intent Parsing & Entity Extraction Accuracy
4. System Latency (P50, P90, P95, P99) and Throughput (QPS) Profiling
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.embedder import CLIPEmbedder
from src.filter_catalog import filter_products
from src.fusion import compute_fusion_score
from src.indexer import load_index, search_index
from src.preprocess import build_catalog
from src.query_parser import parse_query
from src.ranker import rank_products
from src.search_part1 import search_part1
from src.search_part2 import search_multimodal

# ---------------------------------------------------------------------------
# Ground Truth Benchmark Dataset
# ---------------------------------------------------------------------------
# Graded relevance: 2 = Highly Relevant, 1 = Moderately Relevant, 0 = Irrelevant
EVAL_BENCHMARK = [
    {
        "query_id": "q01",
        "query_type": "structured_text",
        "text": "8x10 beige traditional rug",
        "image": None,
        "criteria": {"size": "8x10", "colors": ["beige", "brown", "tan", "cream"], "styles": ["traditional", "oriental", "persian"]},
    },
    {
        "query_id": "q02",
        "query_type": "structured_text",
        "text": "runner 2x10 blue rug",
        "image": None,
        "criteria": {"shape": "runner", "colors": ["blue", "navy", "teal"], "styles": []},
    },
    {
        "query_id": "q03",
        "query_type": "structured_text",
        "text": "round 6 ft modern rug",
        "image": None,
        "criteria": {"shape": "round", "styles": ["modern", "contemporary", "geometric"]},
    },
    {
        "query_id": "q04",
        "query_type": "structured_text",
        "text": "large neutral rug",
        "image": None,
        "criteria": {"size_class": "large", "colors": ["beige", "ivory", "grey", "cream", "tan", "neutral"]},
    },
    {
        "query_id": "q05",
        "query_type": "structured_text",
        "text": "9x12 navy geometric",
        "image": None,
        "criteria": {"size": "9x12", "colors": ["navy", "blue"], "styles": ["geometric", "lattice", "contemporary"]},
    },
    {
        "query_id": "q06",
        "query_type": "structured_text",
        "text": "small bohemian accent rug",
        "image": None,
        "criteria": {"size_class": "small", "styles": ["bohemian", "boho", "oriental", "persian", "multi"]},
    },
    {
        "query_id": "q07",
        "query_type": "structured_text",
        "text": "8 by 10 grey persian",
        "image": None,
        "criteria": {"size": "8x10", "colors": ["grey", "gray", "charcoal"], "styles": ["persian", "oriental", "traditional"]},
    },
    {
        "query_id": "q08",
        "query_type": "structured_text",
        "text": "7.5x9.5 ivory transitional",
        "image": None,
        "criteria": {"size": "8x11", "colors": ["ivory", "cream", "white", "beige"], "styles": ["transitional", "traditional", "contemporary"]},
    },
    {
        "query_id": "q09",
        "query_type": "structured_text",
        "text": "red geometric outdoor rug",
        "image": None,
        "criteria": {"colors": ["red", "rust"], "styles": ["geometric", "lattice", "contemporary"]},
    },
    {
        "query_id": "q10",
        "query_type": "structured_text",
        "text": "boho multicolor runner",
        "image": None,
        "criteria": {"shape": "runner", "colors": ["multi", "multicolor", "blue", "beige"], "styles": ["bohemian", "boho", "oriental"]},
    },
    {
        "query_id": "q11",
        "query_type": "visual_image_only",
        "text": "",
        "image": "images/room1.jpeg",
        "criteria": {"colors": ["beige", "grey", "neutral", "brown", "cream"], "styles": ["modern", "transitional", "contemporary", "oriental"]},
    },
    {
        "query_id": "q12",
        "query_type": "visual_image_only",
        "text": "",
        "image": "images/room3.jpeg",
        "criteria": {"colors": ["neutral", "grey", "beige", "ivory"], "styles": ["modern", "minimalist", "contemporary"]},
    },
    {
        "query_id": "q13",
        "query_type": "multimodal_late_fusion",
        "text": "warm traditional rug",
        "image": "images/room1.jpeg",
        "criteria": {"colors": ["beige", "brown", "tan", "red", "warm"], "styles": ["traditional", "oriental", "persian"]},
    },
    {
        "query_id": "q14",
        "query_type": "multimodal_late_fusion",
        "text": "modern neutral living room",
        "image": "images/room1.jpeg",
        "criteria": {"colors": ["beige", "grey", "ivory", "neutral"], "styles": ["modern", "contemporary", "transitional"]},
    },
    {
        "query_id": "q15",
        "query_type": "multimodal_late_fusion",
        "text": "blue geometric runner",
        "image": "images/room3.jpeg",
        "criteria": {"shape": "runner", "colors": ["blue", "navy", "teal"], "styles": ["geometric", "modern", "contemporary"]},
    },
]

PARSER_EVAL_CASES = [
    {"input": "8x10 beige traditional rug", "expected": {"dims": (8.0, 10.0), "shape": "rectangle", "color": ["beige"], "style": ["traditional"], "size_class": None}},
    {"input": "runner 2x10 blue rug", "expected": {"dims": (2.0, 10.0), "shape": "runner", "color": ["blue"], "style": [], "size_class": None}},
    {"input": "round 6 ft modern rug", "expected": {"dims": (6.0, 6.0), "shape": "round", "color": [], "style": ["modern"], "size_class": None}},
    {"input": "large neutral rug", "expected": {"dims": None, "shape": "rectangle", "color": ["neutral"], "style": [], "size_class": "large"}},
    {"input": "8 by 10 grey persian", "expected": {"dims": (8.0, 10.0), "shape": "rectangle", "color": ["grey"], "style": ["persian"], "size_class": None}},
    {"input": "7.5x9.5 ivory transitional", "expected": {"dims": (7.5, 9.5), "shape": "rectangle", "color": ["ivory"], "style": ["transitional"], "size_class": None}},
    {"input": "8x10", "expected": {"dims": (8.0, 10.0), "shape": "rectangle", "color": [], "style": [], "size_class": None}},
    {"input": "10x8 rug", "expected": {"dims": (8.0, 10.0), "shape": "rectangle", "color": [], "style": [], "size_class": None}},
    {"input": "gray bohemian rug", "expected": {"dims": None, "shape": "rectangle", "color": ["grey"], "style": ["bohemian"], "size_class": None}},
    {"input": "boho multicolor runner", "expected": {"dims": None, "shape": "runner", "color": ["multi"], "style": ["bohemian"], "size_class": None}},
    {"input": "small accent rug", "expected": {"dims": None, "shape": "rectangle", "color": [], "style": [], "size_class": "small"}},
    {"input": "oversized area rug", "expected": {"dims": None, "shape": "rectangle", "color": [], "style": [], "size_class": "xlarge"}},
    {"input": "circular 5x5 wool rug", "expected": {"dims": (5.0, 5.0), "shape": "round", "color": [], "style": [], "size_class": None}},
    {"input": "12x15 huge navy classic", "expected": {"dims": (12.0, 15.0), "shape": "rectangle", "color": ["navy"], "style": ["traditional"], "size_class": "xlarge"}},
    {"input": "4x6 medium green farmhouse rug", "expected": {"dims": (4.0, 6.0), "shape": "rectangle", "color": ["green"], "style": ["farmhouse"], "size_class": "medium"}},
]


def score_ground_truth(product: dict[str, Any], criteria: dict[str, Any]) -> int:
    """Calculates graded relevance (0, 1, or 2) of a product given evaluation criteria."""
    searchable = (product.get("searchable_text") or "").lower()
    title = (product.get("title") or "").lower()
    tags = (product.get("tags") or "").lower()
    full_text = f"{searchable} {title} {tags}"

    matches_color = False
    req_colors = criteria.get("colors") or []
    if not req_colors:
        matches_color = True
    else:
        for c in req_colors:
            if c in full_text:
                matches_color = True
                break

    matches_style = False
    req_styles = criteria.get("styles") or []
    if not req_styles:
        matches_style = True
    else:
        for s in req_styles:
            if s in full_text:
                matches_style = True
                break

    matches_shape = False
    req_shape = criteria.get("shape")
    if not req_shape:
        matches_shape = True
    else:
        if req_shape == "runner" and ("runner" in title or "runner" in tags or "runner" in full_text):
            matches_shape = True
        elif req_shape == "round" and ("round" in title or "round" in tags or "round" in full_text):
            matches_shape = True
        elif req_shape == "rectangle":
            matches_shape = True

    matches_size = False
    req_size = criteria.get("size")
    req_size_class = criteria.get("size_class")
    if not req_size and not req_size_class:
        matches_size = True
    else:
        if req_size:
            dim_key = req_size.replace("x", " ")
            if req_size in full_text or dim_key in full_text or any(req_size in str(s) for s in product.get("sizes", [])):
                matches_size = True
        if req_size_class:
            if req_size_class == "small" and any(any(k in s for k in ["scatter", "2x", "3x", "4x"]) for s in product.get("sizes", [])):
                matches_size = True
            elif req_size_class == "large" and any(any(k in s for k in ["8x", "9x", "10x"]) for s in product.get("sizes", [])):
                matches_size = True
            elif req_size_class == "xlarge" and any(any(k in s for k in ["10x", "12x", "13x", "15x"]) for s in product.get("sizes", [])):
                matches_size = True

    score = 0
    passed = sum([matches_color, matches_style, matches_shape, matches_size])
    total = len([k for k in ["colors", "styles", "shape", "size"] if criteria.get(k) or (k == "size" and criteria.get("size_class"))])
    if total == 0:
        return 1

    if passed == total:
        return 2  # Highly relevant
    elif passed >= total / 2:
        return 1  # Moderately relevant
    return 0


# ---------------------------------------------------------------------------
# IR Metric Calculation Functions
# ---------------------------------------------------------------------------
def compute_dcg(relevances: list[int], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg


def compute_ndcg(relevances: list[int], ideal_relevances: list[int], k: int) -> float:
    dcg = compute_dcg(relevances, k)
    idcg = compute_dcg(ideal_relevances, k)
    if idcg == 0.0:
        return 1.0 if dcg == 0.0 else 0.0
    return dcg / idcg


def compute_precision_recall_mrr_map(retrieved_handles: list[str], ground_truth_map: dict[str, int], k_list: list[int] = [1, 5, 10]) -> dict[str, float]:
    relevances = [ground_truth_map.get(h, 0) for h in retrieved_handles]
    total_relevant = sum(1 for v in ground_truth_map.values() if v > 0)
    if total_relevant == 0:
        return {f"precision@{k}": 1.0 for k in k_list} | {f"recall@{k}": 1.0 for k in k_list} | {"mrr@10": 1.0, "map@10": 1.0}

    metrics = {}
    for k in k_list:
        sub_rel = relevances[:k]
        rel_count = sum(1 for r in sub_rel if r > 0)
        metrics[f"precision@{k}"] = rel_count / k
        metrics[f"recall@{k}"] = rel_count / total_relevant

    # MRR@10
    mrr = 0.0
    for i, r in enumerate(relevances[:10]):
        if r > 0:
            mrr = 1.0 / (i + 1)
            break
    metrics["mrr@10"] = mrr

    # mAP@10
    cum_hits = 0
    prec_sum = 0.0
    for i, r in enumerate(relevances[:10]):
        if r > 0:
            cum_hits += 1
            prec_sum += cum_hits / (i + 1)
    metrics["map@10"] = prec_sum / min(total_relevant, 10) if total_relevant > 0 else 0.0

    # NDCG@5 and NDCG@10
    all_ideal = sorted(ground_truth_map.values(), reverse=True)
    metrics["ndcg@5"] = compute_ndcg(relevances, all_ideal, 5)
    metrics["ndcg@10"] = compute_ndcg(relevances, all_ideal, 10)

    return metrics


# ---------------------------------------------------------------------------
# Main Evaluation Workflow
# ---------------------------------------------------------------------------
def run_evaluation() -> dict[str, Any]:
    print("=" * 60)
    print(">> RUNNING QUANTITATIVE BENCHMARK EVALUATION")
    print("=" * 60)

    catalog = build_catalog("data/products.csv")
    embedder = CLIPEmbedder()

    # 1. Intent Parsing Accuracy
    print("\n[1/4] Evaluating Query Intent Parsing & Extraction...")
    correct_dims = 0
    correct_shapes = 0
    correct_colors = 0
    correct_styles = 0

    for case in PARSER_EVAL_CASES:
        inp = case["input"]
        exp = case["expected"]
        parsed = parse_query(inp)

        # Dimension match
        actual_dims = (parsed["dimensions"]["width"], parsed["dimensions"]["length"]) if parsed["dimensions"] else None
        if actual_dims == exp["dims"]:
            correct_dims += 1

        if parsed["shape"] == exp["shape"]:
            correct_shapes += 1

        if sorted(parsed["color"]) == sorted(exp["color"]):
            correct_colors += 1

        if set(exp["style"]).issubset(set(parsed["style"])):
            correct_styles += 1

    n_cases = len(PARSER_EVAL_CASES)
    parser_accuracy = {
        "dimension_extraction_acc": correct_dims / n_cases,
        "shape_classification_acc": correct_shapes / n_cases,
        "color_extraction_acc": correct_colors / n_cases,
        "style_extraction_acc": correct_styles / n_cases,
        "overall_parser_f1": statistics.mean([correct_dims / n_cases, correct_shapes / n_cases, correct_colors / n_cases, correct_styles / n_cases]),
    }

    print(f"  * Dimension Extraction Accuracy: {parser_accuracy['dimension_extraction_acc']*100:.1f}%")
    print(f"  * Shape Classification Accuracy:  {parser_accuracy['shape_classification_acc']*100:.1f}%")
    print(f"  * Color Extraction Accuracy:     {parser_accuracy['color_extraction_acc']*100:.1f}%")
    print(f"  * Style Extraction Accuracy:     {parser_accuracy['style_extraction_acc']*100:.1f}%")
    print(f"  * Overall Parsing Intent F1:     {parser_accuracy['overall_parser_f1']*100:.1f}%")

    # 2. Information Retrieval Benchmarks & Modality Ablation Study
    print("\n[2/4] Evaluating IR Metrics & Modality Ablation...")
    ablation_metrics = {
        "metadata_only": [],
        "dense_text_only": [],
        "visual_clip_only": [],
        "multimodal_hybrid": [],
    }

    for item in EVAL_BENCHMARK:
        query_text = item["text"]
        image_path = item["image"]
        criteria = item["criteria"]

        gt_map = {p["handle"]: score_ground_truth(p, criteria) for p in catalog}

        # Config A: Metadata Only (filter_products score)
        parsed = parse_query(query_text) if query_text else {"dimensions": None, "shape": "rectangle", "color": [], "style": [], "size_class": None}
        filtered = filter_products(parsed, catalog)
        meta_ranked = sorted(filtered, key=lambda x: x.get("metadata_score", 0.0), reverse=True)
        meta_handles = [p["handle"] for p in meta_ranked]
        ablation_metrics["metadata_only"].append(compute_precision_recall_mrr_map(meta_handles, gt_map))

        # Config B: Dense Text Search (Part 1 Search)
        if query_text:
            text_results = search_part1(query_text, top_k=10)
            text_handles = [p["handle"] for p in text_results]
        else:
            text_handles = meta_handles[:10]
        ablation_metrics["dense_text_only"].append(compute_precision_recall_mrr_map(text_handles, gt_map))

        # Config C: Visual CLIP Only (Mode A)
        vis_img = image_path or "images/room1.jpeg"
        vis_results = search_multimodal(vis_img, text_query="", top_k=10, include_explanations=False)
        vis_handles = [p["handle"] for p in vis_results]
        ablation_metrics["visual_clip_only"].append(compute_precision_recall_mrr_map(vis_handles, gt_map))

        # Config D: Hybrid Multimodal Late Fusion (Proposed Part 2 Mode B or Part 1 full)
        if image_path and query_text:
            hybrid_results = search_multimodal(image_path, text_query=query_text, top_k=10, include_explanations=False)
        elif image_path:
            hybrid_results = search_multimodal(image_path, text_query="", top_k=10, include_explanations=False)
        else:
            hybrid_results = search_part1(query_text, top_k=10)
        hybrid_handles = [p["handle"] for p in hybrid_results]
        ablation_metrics["multimodal_hybrid"].append(compute_precision_recall_mrr_map(hybrid_handles, gt_map))

    # Aggregate Ablation
    aggregated_ablation = {}
    for config, metric_list in ablation_metrics.items():
        keys = metric_list[0].keys()
        aggregated_ablation[config] = {k: statistics.mean([m[k] for m in metric_list]) for k in keys}

    print("\n--- ABLATION RESULTS (Averaged across Benchmark Queries) ---")
    for cfg, vals in aggregated_ablation.items():
        print(f"[{cfg.upper()}] NDCG@10: {vals['ndcg@10']:.3f} | MRR@10: {vals['mrr@10']:.3f} | Recall@5: {vals['recall@5']:.3f} | Prec@5: {vals['precision@5']:.3f} | mAP@10: {vals['map@10']:.3f}")

    # 3. System Latency Profiling
    print("\n[3/4] Benchmarking Latency & Computational Throughput...")
    latency_records = {
        "intent_parsing_ms": [],
        "metadata_filtering_ms": [],
        "text_search_part1_ms": [],
        "clip_text_embed_ms": [],
        "faiss_vector_search_ms": [],
        "multimodal_search_part2_ms": [],
    }

    # Warmup
    search_part1("8x10 beige traditional rug", top_k=5)
    search_multimodal("images/room1.jpeg", "modern neutral", top_k=5, include_explanations=False)

    sample_queries = [
        "8x10 beige traditional rug",
        "runner 2x10 blue rug",
        "round 6 ft modern rug",
        "large neutral rug",
        "9x12 navy geometric",
    ]

    for _ in range(30):
        # Intent parsing
        for q in sample_queries:
            t0 = time.perf_counter()
            p = parse_query(q)
            latency_records["intent_parsing_ms"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            f = filter_products(p, catalog)
            latency_records["metadata_filtering_ms"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            r = rank_products(p, f, q, top_k=10)
            latency_records["text_search_part1_ms"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            te = embedder.embed_text(q)
            latency_records["clip_text_embed_ms"].append((time.perf_counter() - t0) * 1000)

    # Multimodal latency profiling
    image_index, text_index, product_ids = load_index(Path("index"))
    for _ in range(5):
        t0 = time.perf_counter()
        _ = search_index(image_index, te, k=10)
        latency_records["faiss_vector_search_ms"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        _ = search_multimodal("images/room1.jpeg", "modern neutral", top_k=10, include_explanations=False)
        latency_records["multimodal_search_part2_ms"].append((time.perf_counter() - t0) * 1000)

    latency_summary = {}
    for comp, times in latency_records.items():
        arr = np.array(times)
        latency_summary[comp] = {
            "mean_ms": float(np.mean(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
        }

    throughput_part1 = 1000.0 / latency_summary["text_search_part1_ms"]["mean_ms"]

    print("\n--- LATENCY BENCHMARK SUMMARY ---")
    for comp, s in latency_summary.items():
        print(f"  * {comp:<28}: Mean={s['mean_ms']:6.2f}ms | P50={s['p50_ms']:6.2f}ms | P95={s['p95_ms']:6.2f}ms | P99={s['p99_ms']:6.2f}ms")
    print(f"  * Part 1 Search Throughput: {throughput_part1:.1f} Queries/Sec")

    # 4. Save Final Evaluation Report
    final_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "catalog_size": len(catalog),
        "parser_accuracy": parser_accuracy,
        "information_retrieval_benchmarks": aggregated_ablation["multimodal_hybrid"],
        "ablation_comparison": aggregated_ablation,
        "latency_summary": latency_summary,
        "throughput_qps": {
            "part1_structured_search": throughput_part1,
            "faiss_vector_retrieval": 1000.0 / latency_summary["faiss_vector_search_ms"]["mean_ms"],
        },
    }

    out_file = Path("data/eval_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"\n[4/4] [SUCCESS] Evaluation complete! Full results saved to {out_file}")
    return final_report


if __name__ == "__main__":
    run_evaluation()
