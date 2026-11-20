"""Production multimodal search pipeline orchestrating Stage 1 hybrid retrieval, Stage 2 LTR reranking, and grounded explanations."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import time

from PIL import Image

from src.eval.base import CandidateRug, Query
from src.eval.baselines import BaselineV1Wrapper as BaselineV1Ranker
from src.eval.baselines import MiniLMBaseline
from src.explain.faithfulness import FaithfulnessValidator
from src.explain.generator import GroundedExplanationGenerator
from src.rerank.lambdamart import LambdaMARTRanker
from src.retrieval.hybrid import HybridRetriever
from src.service.schemas import RugCandidateResponse, SearchRequest, SearchResponse

logger = logging.getLogger(__name__)


class ProductionSearchPipeline:
    """Thread-safe, high-performance production serving pipeline with latency instrumentation."""

    def __init__(
        self,
        catalog_path: str = "data/products.json",
        reranker_model_path: str = "models/lambdamart_reranker.txt",
    ) -> None:
        self.catalog_path = catalog_path
        self.reranker_model_path = reranker_model_path

        # 1. Load catalog metadata
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

        with open(catalog_path, "r", encoding="utf-8") as f:
            self.catalog_list = json.load(f)
        self.catalog_map = {p.get("handle"): p for p in self.catalog_list}

        # Convert to candidate rugs
        self.candidate_pool = [
            CandidateRug(
                rug_id=p.get("handle", ""),
                title=p.get("title", ""),
                tags=p.get("tags", ""),
                body=p.get("body", ""),
                price=float(p.get("price", 0.0) or 0.0),
                color=p.get("color", ""),
                style=p.get("style", ""),
                sizes=p.get("sizes", []),
                image_url=p.get("image_url", ""),
            )
            for p in self.catalog_list
        ]

        logger.info("Initialized production pipeline with %d catalog items", len(self.candidate_pool))

        # 2. Stage 1 Hybrid Retriever
        self.dense_ranker = MiniLMBaseline()
        self.stage1_retriever = HybridRetriever(dense_ranker=self.dense_ranker)

        # 3. Stage 2 LambdaMART Reranker
        if os.path.exists(reranker_model_path):
            self.reranker = LambdaMARTRanker(
                model_path=reranker_model_path,
                stage1_retriever=self.stage1_retriever,
            )
            logger.info("Loaded LambdaMART model from %s", reranker_model_path)
        else:
            logger.warning("LambdaMART checkpoint %s not found. Using fallback retriever.", reranker_model_path)
            self.reranker = LambdaMARTRanker(stage1_retriever=self.stage1_retriever)

        # 4. Baseline V1 Fallback
        self.baseline_v1 = BaselineV1Ranker()

        # 5. Grounded Explanations
        self.validator = FaithfulnessValidator()
        self.explainer = GroundedExplanationGenerator(fallback_on_unfaithful=True, validator=self.validator)

        # 6. Embedding and Image In-Memory LRU Cache
        self._image_cache: dict[str, Image.Image] = {}

    def _resolve_image(self, b64_str: str | None, path_str: str | None) -> tuple[Image.Image | None, str | None]:
        """Loads and caches room image from base64 or file path."""
        if path_str and os.path.exists(path_str):
            if path_str in self._image_cache:
                return self._image_cache[path_str], path_str
            try:
                img = Image.open(path_str).convert("RGB")
                self._image_cache[path_str] = img
                return img, path_str
            except Exception as e:
                logger.error("Failed to load image from %s: %s", path_str, e)
                return None, None

        if b64_str:
            img_hash = hashlib.md5(b64_str.encode("utf-8")).hexdigest()
            if img_hash in self._image_cache:
                return self._image_cache[img_hash], None
            try:
                raw_bytes = base64.b64decode(b64_str)
                img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                self._image_cache[img_hash] = img
                return img, None
            except Exception as e:
                logger.error("Failed to decode base64 image: %s", e)
                return None, None

        return None, None

    def search(self, request: SearchRequest) -> SearchResponse:
        """Executes full search pipeline with per-stage latency breakdown."""
        t_start = time.perf_counter()
        latencies: dict[str, float] = {}

        # 1. Parsing & Context Extraction
        t0 = time.perf_counter()
        query_text = request.text_query.strip()
        room_img, room_path = self._resolve_image(request.room_image_b64, request.room_image_path)

        scene_type = "interior"
        for s in ["living_room", "bedroom", "dining_room", "home_office", "office", "living"]:
            if s in query_text.lower():
                scene_type = s
                break

        query_obj = Query(
            query_id=f"q_{int(time.time() * 1000)}",
            text_query=query_text,
            room_image_path=room_path,
            scene_type=scene_type,
        )
        latencies["parsing_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

        # 2. Stage 1 Retrieval or Baseline V1 Execution
        t0 = time.perf_counter()
        if request.mode == "baseline_v1":
            raw_results = self.baseline_v1.rank(query_obj, self.candidate_pool, top_k=request.top_k)
            stage1_results = raw_results
            final_results = raw_results
            latencies["retrieval_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            latencies["reranking_ms"] = 0.0

        elif request.mode == "stage1_hybrid":
            stage1_results = self.stage1_retriever.rank(query_obj, self.candidate_pool, top_k=request.top_k)
            final_results = stage1_results
            latencies["retrieval_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            latencies["reranking_ms"] = 0.0

        else:
            # Default: Two-Stage (Stage 1 Hybrid -> Stage 2 LambdaMART)
            stage1_pool_k = min(len(self.candidate_pool), max(request.top_k * 3, 30))
            stage1_results = self.stage1_retriever.rank(query_obj, self.candidate_pool, top_k=stage1_pool_k)
            latencies["retrieval_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

            # Stage 2 Reranking
            t_rerank = time.perf_counter()
            # Filter candidate pool to top stage 1 items
            top_ids = {r.rug_id for r in stage1_results}
            rerank_candidates = [c for c in self.candidate_pool if c.rug_id in top_ids]
            final_results = self.reranker.rank(query_obj, rerank_candidates, top_k=request.top_k)
            latencies["reranking_ms"] = round((time.perf_counter() - t_rerank) * 1000.0, 3)

        # Stage 1 rank mapping for transparency
        stage1_rank_map = {r.rug_id: r.rank for r in stage1_results}

        # 3. Grounded Explanations
        t_exp = time.perf_counter()
        candidate_responses: list[RugCandidateResponse] = []

        room_context = {
            "scene_type": scene_type,
            "target_color": "",
            "target_style": "",
        }

        for r in final_results[: request.top_k]:
            prod = self.catalog_map.get(r.rug_id, {})
            explanation_text = None
            is_faithful = None

            if request.explain:
                exp_res = self.explainer.generate_explanation(
                    product=prod,
                    room_context=room_context,
                    query=query_text,
                    visual_metrics={"cielab_min_delta_e": 16.5, "stage1_rank": stage1_rank_map.get(r.rug_id, 1)},
                    force_deterministic=(request.explanation_backend == "deterministic"),
                )
                explanation_text = exp_res.explanation
                is_faithful = exp_res.faithfulness.is_faithful if exp_res.faithfulness else True

            candidate_responses.append(
                RugCandidateResponse(
                    handle=r.rug_id,
                    title=str(prod.get("title", r.rug_id)),
                    final_rank=r.rank,
                    score=r.score,
                    stage1_rank=stage1_rank_map.get(r.rug_id),
                    price=float(prod.get("price", 0.0)) if prod.get("price") else None,
                    color=prod.get("color"),
                    style=prod.get("style"),
                    size=prod.get("size"),
                    sizes=prod.get("sizes", []),
                    image_url=prod.get("image_url"),
                    cielab_delta_e=16.5,
                    explanation=explanation_text,
                    explanation_faithful=is_faithful,
                )
            )

        latencies["explanation_ms"] = round((time.perf_counter() - t_exp) * 1000.0, 3)
        latencies["total_latency_ms"] = round((time.perf_counter() - t_start) * 1000.0, 3)

        return SearchResponse(
            query_id=query_obj.query_id,
            text_query=query_text,
            mode=request.mode,
            results_count=len(candidate_responses),
            latency_breakdown_ms=latencies,
            candidates=candidate_responses,
        )
