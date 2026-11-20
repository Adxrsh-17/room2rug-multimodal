"""CI Quality Gate: Automated Pull-Request validation script for zero-leakage, metric regressions, and latency budgets."""

from __future__ import annotations

import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from scripts.check_lineage import verify_lineage
from src.data.splits import create_group_splits, verify_no_leakage
from src.eval.benchmark_data import load_test_data
from src.eval.metrics import evaluate_rankings
from src.rerank.lambdamart import LambdaMARTRanker
from src.service.pipeline import ProductionSearchPipeline
from src.service.schemas import SearchRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CIQualityGate")

# Thresholds
MIN_ALLOWED_NDCG5 = 0.58
MAX_ALLOWED_RETRIEVAL_LATENCY_MS = 500.0


def check_leakage_gate() -> bool:
    """Gate 1: Verifies zero group overlap between train, val, and test splits."""
    logger.info("--- Quality Gate 1: Split Leakage Verification ---")
    with open("data/test_queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    records = []
    for q in queries:
        for rid in q.get("candidate_rug_ids", []):
            records.append({"rug_id": rid, "room_id": q["query_id"]})
    df = pd.DataFrame(records)
    train_df, val_df, test_df = create_group_splits(df, rug_col="rug_id", room_col="room_id", random_state=42)
    report = verify_no_leakage(train_df, val_df, test_df, rug_col="rug_id", room_col="room_id")
    if report["has_leakage"]:
        logger.error("GATE FAILED: Data leakage detected between splits!\n%s", report)
        return False
    logger.info("GATE PASSED: Zero data leakage confirmed (0 group overlap).")
    return True


def check_ndcg_regression_gate() -> bool:
    """Gate 2: Evaluates ranker on human gold test set; blocks PRs that regress NDCG@5."""
    logger.info("--- Quality Gate 2: NDCG@5 Regression Guard ---")
    if not os.path.exists("models/lambdamart_reranker.txt"):
        logger.error("GATE FAILED: LambdaMART model checkpoint missing!")
        return False

    queries, candidates, gold_labels, _, _ = load_test_data(
        queries_path="data/test_queries.json",
        gold_path="data/gold_labels.json",
        products_path="data/products.json",
    )
    catalog_map = {c.rug_id: c for c in candidates}
    ranker = LambdaMARTRanker(model_path="models/lambdamart_reranker.txt")

    query_rankings = {}
    for q in queries:
        cand_ids = q.metadata.get("candidate_rug_ids", []) if q.metadata else []
        cand_objs = [catalog_map[rid] for rid in cand_ids if rid in catalog_map]
        ranked = ranker.rank(q, cand_objs, top_k=10)
        query_rankings[q.query_id] = [r.rug_id for r in ranked]

    eval_res = evaluate_rankings(query_rankings, gold_labels, k_values=[5, 10])
    ndcg5 = eval_res.get("means", {}).get("ndcg@5", 0.0)

    logger.info("Evaluated NDCG@5 on Gold Test Set: %.4f (Threshold: >= %.4f)", ndcg5, MIN_ALLOWED_NDCG5)

    if ndcg5 < MIN_ALLOWED_NDCG5:
        logger.error("GATE FAILED: NDCG@5 (%.4f) fell below acceptable threshold (%.4f)!", ndcg5, MIN_ALLOWED_NDCG5)
        return False

    logger.info("GATE PASSED: NDCG@5 satisfies quality threshold.")
    return True


def check_latency_budget_gate() -> bool:
    """Gate 3: Verifies single-query serving latency meets production SLA."""
    logger.info("--- Quality Gate 3: Serving Latency Budget ---")
    pipeline = ProductionSearchPipeline()

    test_req = SearchRequest(
        text_query="8x10 warm beige living room rug",
        top_k=10,
        mode="two_stage",
        explain=False,
    )

    # Warmup
    pipeline.search(test_req)

    # Measurement
    t0 = time.perf_counter()
    res = pipeline.search(test_req)
    total_ms = (time.perf_counter() - t0) * 1000.0
    retrieval_ms = res.latency_breakdown_ms.get("retrieval_ms", total_ms)

    logger.info("Measured Retrieval Latency: %.2f ms (Budget: <= %.2f ms)", retrieval_ms, MAX_ALLOWED_RETRIEVAL_LATENCY_MS)

    if retrieval_ms > MAX_ALLOWED_RETRIEVAL_LATENCY_MS:
        logger.error("GATE FAILED: Retrieval latency (%.2f ms) exceeded budget (%.2f ms)!", retrieval_ms, MAX_ALLOWED_RETRIEVAL_LATENCY_MS)
        return False

    logger.info("GATE PASSED: Serving latency within budget.")
    return True


def run_all_quality_gates() -> bool:
    """Runs all 4 automated CI quality gates."""
    logger.info("==================================================")
    logger.info("RUNNING AUTOMATED CI QUALITY GATES")
    logger.info("==================================================")

    # 1. Lineage check
    if not verify_lineage():
        logger.error("CI GATE FAILURE: Artifact lineage validation failed.")
        return False

    # 2. Leakage check
    if not check_leakage_gate():
        logger.error("CI GATE FAILURE: Split leakage validation failed.")
        return False

    # 3. NDCG regression check
    if not check_ndcg_regression_gate():
        logger.error("CI GATE FAILURE: Metric regression detected.")
        return False

    # 4. Latency budget check
    if not check_latency_budget_gate():
        logger.error("CI GATE FAILURE: Latency budget exceeded.")
        return False

    logger.info("==================================================")
    logger.info("ALL QUALITY GATES PASSED (Lineage, Leakage, NDCG, Latency)")
    logger.info("==================================================")
    return True


if __name__ == "__main__":
    passed = run_all_quality_gates()
    sys.exit(0 if passed else 1)
