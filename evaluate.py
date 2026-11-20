"""Unified Evaluation CLI for room2rug retrieval and ranking systems.

Runs all baselines against the human gold test set with:
- 1,000-sample bootstrap 95% Confidence Intervals
- Paired permutation tests vs baseline_v1
- Slice breakdown by scene category
- Inter-rater agreement (Cohen's Kappa)
- VLM-as-judge agreement & correlation analysis

Outputs results to results/baseline_leaderboard.json and results/baseline_leaderboard.md.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.eval.agreement import agreement_report
from src.eval.base import BaseRanker, CandidateRug, Query
from src.eval.baselines import (
    AttributeFilterBaseline,
    BaselineV1Wrapper,
    BM25Baseline,
    CLIPBaseline,
    MiniLMBaseline,
    SigLIPBaseline,
)
from src.eval.benchmark_data import load_test_data
from src.eval.metrics import (
    compute_bootstrap_ci,
    evaluate_rankings,
    evaluate_slices,
    paired_permutation_test,
)
from src.eval.vlm_judge import OfflineVLMJudge, validate_vlm_against_gold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_evaluation(
    queries: list[Query],
    candidates: list[CandidateRug],
    gold_labels: dict[str, dict[str, float]],
    query_slices: dict[str, str],
    gold_raw: dict[str, Any],
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    rankers: list[BaseRanker] = [
        AttributeFilterBaseline(),
        BM25Baseline(),
        MiniLMBaseline(),
        CLIPBaseline(),
        BaselineV1Wrapper(),
        SigLIPBaseline(),
    ]

    all_rankings: dict[str, dict[str, list[str]]] = {}
    system_results: dict[str, dict[str, Any]] = {}

    logger.info("Running retrieval across %d baselines for %d queries...", len(rankers), len(queries))

    for ranker in rankers:
        sys_name = ranker.name
        sys_rankings: dict[str, list[str]] = {}

        for q in queries:
            ranked = ranker.rank(q, candidates, top_k=10)
            sys_rankings[q.query_id] = [r.rug_id for r in ranked]

        all_rankings[sys_name] = sys_rankings
        eval_metrics = evaluate_rankings(sys_rankings, gold_labels)
        system_results[sys_name] = eval_metrics

    # 1. Headline metrics with bootstrap CIs
    headline_metrics = ["ndcg@5", "ndcg@10", "recall@5", "recall@10", "mrr@10", "map@10"]
    leaderboard: list[dict[str, Any]] = []

    baseline_v1_metrics = system_results.get("baseline_v1", {}).get("per_query", {})

    for ranker in rankers:
        s_name = ranker.name
        s_data = system_results[s_name]
        per_query = s_data["per_query"]

        row: dict[str, Any] = {"system": s_name}
        for m in headline_metrics:
            vals = per_query.get(m, [])
            mean, ci_low, ci_high = compute_bootstrap_ci(vals, n_bootstraps=n_bootstraps, seed=seed)
            row[m] = {
                "mean": round(mean, 4),
                "ci_95": [round(ci_low, 4), round(ci_high, 4)],
            }

        # Paired permutation significance test against baseline_v1 on NDCG@5
        if s_name != "baseline_v1" and "ndcg@5" in baseline_v1_metrics:
            p_val = paired_permutation_test(
                per_query["ndcg@5"],
                baseline_v1_metrics["ndcg@5"],
                seed=seed,
            )
            row["p_val_vs_v1_ndcg@5"] = round(p_val, 4)
        else:
            row["p_val_vs_v1_ndcg@5"] = 1.0

        leaderboard.append(row)

    # 2. Slice analysis for top models on NDCG@5
    slice_reports: dict[str, Any] = {}
    for s_name in ["baseline_v1", "clip_zero_shot", "siglip"]:
        slice_reports[s_name] = evaluate_slices(
            system_rankings=all_rankings[s_name],
            gold_labels=gold_labels,
            query_slices=query_slices,
            target_metric="ndcg@5",
        )

    # 3. Inter-rater agreement
    annotations = gold_raw.get("annotations", [])
    primary_map = {
        (a["query_id"], a["rug_id"]): a["relevance"]
        for a in annotations
        if a.get("annotator_id") == "primary_evaluator"
    }
    secondary_map = {
        (a["query_id"], a["rug_id"]): a["relevance"]
        for a in annotations
        if a.get("annotator_id") == "secondary_evaluator"
    }

    common_pairs = [p for p in primary_map if p in secondary_map]
    if common_pairs:
        r1_vals = [primary_map[p] for p in common_pairs]
        r2_vals = [secondary_map[p] for p in common_pairs]
        agreement_rep = agreement_report(
            rater1=r1_vals,
            rater2=r2_vals,
            rater1_name="primary_evaluator",
            rater2_name="secondary_evaluator",
        )
    else:
        agreement_rep = {"n_pairs": 0}

    # 4. VLM-as-Judge validation against gold labels
    cand_by_id = {c.rug_id: c for c in candidates}
    vlm_judge = OfflineVLMJudge(seed=seed)
    vlm_labels: dict[tuple[str, str], int] = {}

    for q in queries:
        for rug_id in gold_labels.get(q.query_id, {}):
            rug = cand_by_id.get(rug_id)
            if rug:
                j_res = vlm_judge.judge_pair(
                    query_id=q.query_id,
                    room_image_path=q.room_image_path,
                    scene_type=q.scene_type,
                    query_text=q.text_query,
                    rug_id=rug.rug_id,
                    rug_title=rug.title,
                    rug_color=rug.color,
                    rug_style=rug.style,
                    rug_tags=rug.tags,
                )
                vlm_labels[(q.query_id, rug_id)] = j_res.rating

    gold_tuple_map = {(qid, rid): int(score) for qid, rugs in gold_labels.items() for rid, score in rugs.items()}
    vlm_validation = validate_vlm_against_gold(vlm_labels, gold_tuple_map)

    return {
        "n_queries": len(queries),
        "n_candidates": len(candidates),
        "leaderboard": leaderboard,
        "slice_reports": slice_reports,
        "inter_rater_agreement": agreement_rep,
        "vlm_judge_validation": vlm_validation,
    }


def format_markdown_report(results: dict[str, Any]) -> str:
    md = []
    md.append("# Baseline Leaderboard & Statistical Evaluation\n")
    md.append(
        f"**Evaluation Queries**: {results['n_queries']} | **Candidate Catalog Size**: {results['n_candidates']}\n"
    )
    md.append(
        "All metrics evaluated strictly on human gold labels with **95% Bootstrap Confidence Intervals** (1,000 resamples).\n\n"
    )

    # Table 1: Leaderboard
    md.append("### Table 1: Baseline Retrieval Leaderboard\n")
    md.append(
        "| System | NDCG@5 (95% CI) | NDCG@10 (95% CI) | Recall@5 (95% CI) | Recall@10 (95% CI) | MRR@10 | MAP@10 | p-val vs v1 |\n"
    )
    md.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

    for r in results["leaderboard"]:
        sys = r["system"]
        n5 = f"{r['ndcg@5']['mean']:.3f} [{r['ndcg@5']['ci_95'][0]:.3f}, {r['ndcg@5']['ci_95'][1]:.3f}]"
        n10 = f"{r['ndcg@10']['mean']:.3f} [{r['ndcg@10']['ci_95'][0]:.3f}, {r['ndcg@10']['ci_95'][1]:.3f}]"
        r5 = f"{r['recall@5']['mean']:.3f} [{r['recall@5']['ci_95'][0]:.3f}, {r['recall@5']['ci_95'][1]:.3f}]"
        r10 = f"{r['recall@10']['mean']:.3f} [{r['recall@10']['ci_95'][0]:.3f}, {r['recall@10']['ci_95'][1]:.3f}]"
        mrr = f"{r['mrr@10']['mean']:.3f}"
        map10 = f"{r['map@10']['mean']:.3f}"
        pval = f"{r['p_val_vs_v1_ndcg@5']:.4f}" if sys != "baseline_v1" else "-"
        md.append(f"| **{sys}** | {n5} | {n10} | {r5} | {r10} | {mrr} | {map10} | {pval} |\n")

    # Table 2: Slice Analysis
    md.append("\n### Table 2: Slice Analysis (NDCG@5 by Scene Category)\n")
    md.append("| Scene Class | baseline_v1 (95% CI) | clip_zero_shot (95% CI) | siglip (95% CI) |\n")
    md.append("|---|:---:|:---:|:---:|\n")

    slices = results.get("slice_reports", {})
    all_scenes = sorted(list(slices.get("baseline_v1", {}).keys()))
    for sc in all_scenes:
        v1_s = slices.get("baseline_v1", {}).get(sc, {})
        clip_s = slices.get("clip_zero_shot", {}).get(sc, {})
        siglip_s = slices.get("siglip", {}).get(sc, {})

        v1_str = f"{v1_s.get('mean', 0.0):.3f} [{v1_s.get('ci_95', [0, 0])[0]:.3f}, {v1_s.get('ci_95', [0, 0])[1]:.3f}]"
        clip_str = f"{clip_s.get('mean', 0.0):.3f} [{clip_s.get('ci_95', [0, 0])[0]:.3f}, {clip_s.get('ci_95', [0, 0])[1]:.3f}]"
        sig_str = f"{siglip_s.get('mean', 0.0):.3f} [{siglip_s.get('ci_95', [0, 0])[0]:.3f}, {siglip_s.get('ci_95', [0, 0])[1]:.3f}]"
        md.append(f"| `{sc}` | {v1_str} | {clip_str} | {sig_str} |\n")

    # Table 3: Agreement
    md.append("\n### Table 3: Human Inter-Rater Reliability\n")
    ira = results.get("inter_rater_agreement", {})
    md.append(f"- **Evaluated Pairs**: {ira.get('n_pairs')}\n")
    md.append(f"- **Exact Agreement**: {ira.get('exact_agreement', 0.0):.1%}\n")
    md.append(f"- **Adjacent Agreement (|r1 - r2| <= 1)**: {ira.get('adjacent_agreement', 0.0):.1%}\n")
    md.append(f"- **Cohen's Kappa (Quadratic)**: **`{ira.get('cohens_kappa_quadratic', 0.0):.4f}`**\n")
    md.append(f"- **Cohen's Kappa (Unweighted)**: `{ira.get('cohens_kappa_unweighted', 0.0):.4f}`\n\n")

    # Table 4: VLM Judge Validation
    md.append("### Table 4: VLM-as-Judge Validation Against Human Gold\n")
    vlm = results.get("vlm_judge_validation", {})
    md.append(f"- **Evaluated Pairs**: {vlm.get('n_pairs')}\n")
    md.append(f"- **Exact Agreement**: {vlm.get('exact_agreement', 0.0):.1%}\n")
    md.append(f"- **Cohen's Kappa (Quadratic)**: `{vlm.get('cohens_kappa_quadratic', 0.0):.4f}`\n")
    md.append(
        f"- **Spearman Rank Correlation (rho)**: **`{vlm.get('spearman_rho', 0.0):.4f}`** (p = {vlm.get('spearman_pvalue', 1.0):.2e})\n"
    )
    md.append(
        f"- **Severe False Positives (Gold=0, VLM=2)**: {vlm.get('severe_disagreements', {}).get('vlm_overoptimistic (gold=0, vlm=2)', 0)}\n"
    )
    md.append(
        f"- **Severe False Negatives (Gold=2, VLM=0)**: {vlm.get('severe_disagreements', {}).get('vlm_overpessimistic (gold=2, vlm=0)', 0)}\n"
    )
    md.append("> [!NOTE]\n")
    md.append(
        "> VLM Judge labels are used strictly as a weak supervision signal for model training in Phase 3. The test set remains 100% human-verified gold.\n"
    )

    return "".join(md)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Room2Rug retrieval systems on human gold test set.")
    parser.add_argument("--queries", default="data/test_queries.json", help="Path to test queries")
    parser.add_argument("--gold", default="data/gold_labels.json", help="Path to gold labels")
    parser.add_argument("--products", default="data/products.json", help="Path to catalog products")
    parser.add_argument("--output-json", default="results/baseline_leaderboard.json", help="Output JSON path")
    parser.add_argument("--output-md", default="results/baseline_leaderboard.md", help="Output Markdown path")
    parser.add_argument("--bootstraps", type=int, default=1000, help="Number of bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    queries, candidates, gold_labels, query_slices, gold_raw = load_test_data(
        queries_path=args.queries,
        gold_path=args.gold,
        products_path=args.products,
    )

    results = run_evaluation(
        queries=queries,
        candidates=candidates,
        gold_labels=gold_labels,
        query_slices=query_slices,
        gold_raw=gold_raw,
        n_bootstraps=args.bootstraps,
        seed=args.seed,
    )

    # Save output files
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md_content = format_markdown_report(results)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n" + md_content)
    logger.info("Evaluation complete! Results saved to %s and %s", args.output_json, args.output_md)


if __name__ == "__main__":
    main()
