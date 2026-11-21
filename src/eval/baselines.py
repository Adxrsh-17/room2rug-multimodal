"""Baseline rankers conforming to the common BaseRanker interface.

Implements:
1. AttributeFilterBaseline (Metadata rule/regex filter only)
2. BM25Baseline (BM25 sparse text retriever)
3. MiniLMBaseline (SentenceTransformer all-MiniLM-L6-v2 dense text retriever)
4. CLIPBaseline (CLIP ViT-B/32 zero-shot multimodal retriever)
5. BaselineV1Wrapper (Legacy v1 heuristic fusion: 0.6 image + 0.4 text/metadata)
6. SigLIPBaseline (Modern open vision-language encoder: google/siglip-base-patch16-224)
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
from PIL import Image

from src.eval.base import BaseRanker, CandidateRug, Query, RankResult

logger = logging.getLogger(__name__)

# Process-level singleton caches to prevent redundant model instantiation & downloading
_SHARED_MODELS: dict[str, Any] = {}
_FAILED_MODELS: set[str] = set()


def _extract_tensor(features: Any) -> Any:
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
        return features.last_hidden_state
    if isinstance(features, (tuple, list)):
        return features[0]
    return features


# -------------------------------------------------------------------------
# 1. Attribute Filter Baseline
# -------------------------------------------------------------------------

COLOR_KEYWORDS = {
    "beige",
    "blue",
    "red",
    "grey",
    "gray",
    "white",
    "black",
    "green",
    "brown",
    "yellow",
    "ivory",
    "cream",
    "navy",
    "terracotta",
    "rust",
    "gold",
    "pink",
}

STYLE_KEYWORDS = {
    "modern",
    "contemporary",
    "traditional",
    "vintage",
    "boho",
    "bohemian",
    "persian",
    "oriental",
    "geometric",
    "abstract",
    "minimalist",
    "scandinavian",
    "moroccan",
}

SIZE_PATTERN = re.compile(r"(\d+)\s*(?:x|by|\*)\s*(\d+)", re.IGNORECASE)


class AttributeFilterBaseline(BaseRanker):
    """Scores strictly on metadata/attribute text overlaps (color, style, size)."""

    name = "attribute_filter"

    def rank(
        self,
        query: Query,
        candidates: list[CandidateRug],
        top_k: int = 10,
    ) -> list[RankResult]:
        query_text = (query.text_query + " " + query.scene_type).lower()

        # Extract target attributes from query
        target_colors = {c for c in COLOR_KEYWORDS if c in query_text}
        target_styles = {s for s in STYLE_KEYWORDS if s in query_text}
        size_match = SIZE_PATTERN.search(query_text)
        target_size_str = f"{size_match.group(1)}x{size_match.group(2)}" if size_match else None

        results: list[RankResult] = []
        for rug in candidates:
            score = 0.0
            rug_text = f"{rug.title} {rug.color} {rug.style} {rug.tags} {rug.pattern} {rug.searchable_text}".lower()

            # Color overlap
            if target_colors:
                matched_colors = sum(1 for c in target_colors if c in rug_text or c in rug.color.lower())
                score += (matched_colors / len(target_colors)) * 0.4
            else:
                score += 0.2  # neutral prior

            # Style overlap
            if target_styles:
                matched_styles = sum(1 for s in target_styles if s in rug_text or s in rug.style.lower())
                score += (matched_styles / len(target_styles)) * 0.4
            else:
                score += 0.2

            # Size match
            if target_size_str:
                rug_sizes = [s.lower().replace(" ", "") for s in rug.sizes]
                if target_size_str in rug_sizes or target_size_str in rug.size.lower():
                    score += 0.2
            else:
                score += 0.1

            # Small tiebreaker from title match
            words = [w for w in query_text.split() if len(w) > 3]
            if words:
                overlap = sum(1 for w in words if w in rug_text)
                score += (overlap / len(words)) * 0.1

            results.append(
                RankResult(
                    rug_id=rug.rug_id,
                    score=float(round(score, 4)),
                    rank=0,
                    system_name=self.name,
                    explanation=f"Attribute overlap score: {score:.3f}",
                )
            )

        # Sort descending by score
        results.sort(key=lambda r: r.score, reverse=True)
        for idx, res in enumerate(results[:top_k], start=1):
            res.rank = idx
        return results[:top_k]


# -------------------------------------------------------------------------
# 2. BM25 Baseline
# -------------------------------------------------------------------------


class BM25Baseline(BaseRanker):
    """Sparse text search using BM25Okapi on catalog metadata & descriptions."""

    name = "bm25"

    def __init__(self) -> None:
        try:
            from rank_bm25 import BM25Okapi

            self._bm25_class = BM25Okapi
        except ImportError:
            self._bm25_class = None

    def _tokenize(self, text: str) -> list[str]:
        return [w.lower() for w in re.findall(r"\b\w{2,}\b", text)]

    def rank(
        self,
        query: Query,
        candidates: list[CandidateRug],
        top_k: int = 10,
    ) -> list[RankResult]:
        if not candidates:
            return []

        query_tokens = self._tokenize(f"{query.text_query} {query.scene_type}")
        if not query_tokens:
            query_tokens = ["rug", "carpet"]

        corpus_tokens = [
            self._tokenize(f"{r.title} {r.color} {r.style} {r.pattern} {r.tags} {r.body}") for r in candidates
        ]

        if self._bm25_class is not None:
            bm25 = self._bm25_class(corpus_tokens)
            scores = bm25.get_scores(query_tokens)
        else:
            # Fallback simple TF-IDF scoring if rank_bm25 missing
            scores = []
            q_set = set(query_tokens)
            for doc in corpus_tokens:
                d_set = set(doc)
                score = len(q_set.intersection(d_set)) / (len(q_set) + 1.0)
                scores.append(score)

        # Normalize scores to [0, 1]
        max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0
        results: list[RankResult] = []
        for rug, sc in zip(candidates, scores):
            norm_score = float(sc / max_score) if max_score > 0 else 0.0
            results.append(
                RankResult(
                    rug_id=rug.rug_id,
                    score=round(norm_score, 4),
                    rank=0,
                    system_name=self.name,
                    explanation=f"BM25 score: {sc:.3f}",
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        for idx, res in enumerate(results[:top_k], start=1):
            res.rank = idx
        return results[:top_k]


# -------------------------------------------------------------------------
# 3. MiniLM Dense Text Baseline
# -------------------------------------------------------------------------


class MiniLMBaseline(BaseRanker):
    """Dense semantic text retrieval using sentence-transformers all-MiniLM-L6-v2."""

    name = "minilm_dense"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = _SHARED_MODELS.get(model_name)

    def _get_model(self):
        if self._model is not None:
            return self._model
        if self.model_name in _SHARED_MODELS:
            self._model = _SHARED_MODELS[self.model_name]
            return self._model
        if self.model_name in _FAILED_MODELS:
            return None
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.model_name)
            _SHARED_MODELS[self.model_name] = model
            self._model = model
            return model
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer '{self.model_name}': {e}")
            _FAILED_MODELS.add(self.model_name)
            self._model = None
            return None

    def rank(
        self,
        query: Query,
        candidates: list[CandidateRug],
        top_k: int = 10,
    ) -> list[RankResult]:
        model = self._get_model()
        q_text = f"{query.text_query} {query.scene_type}".strip() or "area rug for residential room"

        if model is None:
            # Fallback to token overlap
            return BM25Baseline().rank(query, candidates, top_k=top_k)

        cand_texts = [f"{r.title}. {r.color} {r.style} {r.pattern}. {r.tags}. {r.body[:200]}" for r in candidates]

        q_emb = model.encode([q_text], normalize_embeddings=True, show_progress_bar=False)[0]
        cand_embs = model.encode(cand_texts, normalize_embeddings=True, show_progress_bar=False)

        sims = np.dot(cand_embs, q_emb)

        results: list[RankResult] = []
        for rug, sim in zip(candidates, sims):
            score = float(max(0.0, sim))
            results.append(
                RankResult(
                    rug_id=rug.rug_id,
                    score=round(score, 4),
                    rank=0,
                    system_name=self.name,
                    explanation=f"MiniLM cosine similarity: {sim:.4f}",
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        for idx, res in enumerate(results[:top_k], start=1):
            res.rank = idx
        return results[:top_k]


# -------------------------------------------------------------------------
# 4. CLIP Zero-Shot Baseline
# -------------------------------------------------------------------------


class CLIPBaseline(BaseRanker):
    """Zero-shot multimodal retrieval using CLIP ViT-B/32."""

    name = "clip_zero_shot"

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self.model_name = model_name
        cached = _SHARED_MODELS.get(model_name)
        if cached is not None:
            self._model, self._processor = cached
        else:
            self._model = None
            self._processor = None
        self._fallback_minilm: MiniLMBaseline | None = None

    def _load_model(self):
        if self._model is not None and self._processor is not None:
            return
        if self.model_name in _SHARED_MODELS:
            self._model, self._processor = _SHARED_MODELS[self.model_name]
            return
        if self.model_name in _FAILED_MODELS:
            self._model, self._processor = None, None
            return
        try:
            from transformers import CLIPModel, CLIPProcessor

            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._model.eval()
            _SHARED_MODELS[self.model_name] = (self._model, self._processor)
        except Exception as e:
            logger.warning(f"Could not load CLIP model '{self.model_name}': {e}")
            _FAILED_MODELS.add(self.model_name)
            self._model = None
            self._processor = None

    def rank(
        self,
        query: Query,
        candidates: list[CandidateRug],
        top_k: int = 10,
    ) -> list[RankResult]:
        self._load_model()
        if self._model is None or self._processor is None:
            # Fallback to MiniLM
            if self._fallback_minilm is None:
                self._fallback_minilm = MiniLMBaseline()
            return self._fallback_minilm.rank(query, candidates, top_k=top_k)

        import torch

        # 1. Encode query
        with torch.no_grad():
            query_image = None
            if query.room_image_path:
                try:
                    img = Image.open(query.room_image_path).convert("RGB")
                    inputs = self._processor(images=img, return_tensors="pt")
                    q_img_features = _extract_tensor(self._model.get_image_features(**inputs))
                    q_img_features = q_img_features / q_img_features.norm(dim=-1, keepdim=True)
                    query_image = q_img_features.cpu().numpy()[0]
                except Exception as e:
                    logger.debug(f"Could not open room image {query.room_image_path}: {e}")

            q_text = query.text_query or f"a photo of an area rug in a {query.scene_type}"
            text_inputs = self._processor(text=[q_text], return_tensors="pt", padding=True, truncation=True)
            q_txt_features = _extract_tensor(self._model.get_text_features(**text_inputs))
            q_txt_features = q_txt_features / q_txt_features.norm(dim=-1, keepdim=True)
            query_text_emb = q_txt_features.cpu().numpy()[0]

            if query_image is not None:
                # Late blend query representation
                query_vec = 0.6 * query_image + 0.4 * query_text_emb
                query_vec = query_vec / np.linalg.norm(query_vec)
            else:
                query_vec = query_text_emb

            # 2. Encode candidate representations (cached across queries)
            cand_key = tuple(r.rug_id for r in candidates)
            if not hasattr(self, "_cand_cache") or getattr(self, "_cand_cache_key", None) != cand_key:
                cand_prompts = [
                    f"a rug: {r.title}, {r.color} color, {r.style} style, {r.pattern} pattern" for r in candidates
                ]
                cand_inputs = self._processor(text=cand_prompts, return_tensors="pt", padding=True, truncation=True)
                cand_features = _extract_tensor(self._model.get_text_features(**cand_inputs))
                cand_features = cand_features / cand_features.norm(dim=-1, keepdim=True)
                self._cand_cache = cand_features.cpu().numpy()
                self._cand_cache_key = cand_key
            cand_vecs = self._cand_cache

            sims = np.dot(cand_vecs, query_vec)

        results: list[RankResult] = []
        for rug, sim in zip(candidates, sims):
            score = float(max(0.0, sim))
            results.append(
                RankResult(
                    rug_id=rug.rug_id,
                    score=round(score, 4),
                    rank=0,
                    system_name=self.name,
                    explanation=f"CLIP similarity: {sim:.4f}",
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        for idx, res in enumerate(results[:top_k], start=1):
            res.rank = idx
        return results[:top_k]


# -------------------------------------------------------------------------
# 5. Baseline v1 Wrapper (Legacy pipeline)
# -------------------------------------------------------------------------


class BaselineV1Wrapper(BaseRanker):
    """Replicates legacy v1 heuristic fusion: 0.6 image similarity + 0.4 text/attribute score."""

    name = "baseline_v1"

    def __init__(self) -> None:
        self.attr_baseline = AttributeFilterBaseline()
        self.minilm_baseline = MiniLMBaseline()
        self.clip_baseline = CLIPBaseline()

    def rank(
        self,
        query: Query,
        candidates: list[CandidateRug],
        top_k: int = 10,
    ) -> list[RankResult]:
        # Legacy Mode determination:
        has_image = bool(query.room_image_path)
        has_text = bool(query.text_query.strip())

        attr_ranks = {r.rug_id: r.score for r in self.attr_baseline.rank(query, candidates, top_k=len(candidates))}
        minilm_ranks = {r.rug_id: r.score for r in self.minilm_baseline.rank(query, candidates, top_k=len(candidates))}
        clip_ranks = {r.rug_id: r.score for r in self.clip_baseline.rank(query, candidates, top_k=len(candidates))}

        results: list[RankResult] = []
        for rug in candidates:
            rid = rug.rug_id
            attr_sc = attr_ranks.get(rid, 0.0)
            text_sc = 0.5 * attr_sc + 0.5 * minilm_ranks.get(rid, 0.0)
            img_sc = clip_ranks.get(rid, 0.0)

            if has_image and has_text:
                score = 0.6 * img_sc + 0.4 * text_sc
            elif has_image:
                score = img_sc
            else:
                score = text_sc

            results.append(
                RankResult(
                    rug_id=rid,
                    score=round(float(score), 4),
                    rank=0,
                    system_name=self.name,
                    explanation=f"v1 heuristic fusion (img={img_sc:.2f}, txt={text_sc:.2f})",
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        for idx, res in enumerate(results[:top_k], start=1):
            res.rank = idx
        return results[:top_k]


# -------------------------------------------------------------------------
# 6. SigLIP Baseline (Strong open encoder)
# -------------------------------------------------------------------------


class SigLIPBaseline(BaseRanker):
    """Modern open vision-language encoder using google/siglip-base-patch16-224."""

    name = "siglip"

    def __init__(self, model_name: str = "google/siglip-base-patch16-224") -> None:
        self.model_name = model_name
        cached = _SHARED_MODELS.get(model_name)
        if cached is not None:
            self._model, self._processor = cached
        else:
            self._model = None
            self._processor = None
        self._fallback_clip: CLIPBaseline | None = None

    def _load_model(self):
        if self._model is not None and self._processor is not None:
            return
        if self.model_name in _SHARED_MODELS:
            self._model, self._processor = _SHARED_MODELS[self.model_name]
            return
        if self.model_name in _FAILED_MODELS:
            self._model, self._processor = None, None
            return
        try:
            from transformers import AutoModel, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            _SHARED_MODELS[self.model_name] = (self._model, self._processor)
        except Exception as e:
            logger.warning(f"Could not load SigLIP model '{self.model_name}': {e}")
            _FAILED_MODELS.add(self.model_name)
            self._model = None
            self._processor = None

    def rank(
        self,
        query: Query,
        candidates: list[CandidateRug],
        top_k: int = 10,
    ) -> list[RankResult]:
        self._load_model()
        if self._model is None or self._processor is None:
            # Fallback to CLIP if SigLIP weights cannot be downloaded
            if self._fallback_clip is None:
                self._fallback_clip = CLIPBaseline()
            return self._fallback_clip.rank(query, candidates, top_k=top_k)

        import torch

        with torch.no_grad():
            q_text = query.text_query or f"an area rug styled for a {query.scene_type}"
            query_image = None
            if query.room_image_path:
                try:
                    img = Image.open(query.room_image_path).convert("RGB")
                    img_inputs = self._processor(images=img, return_tensors="pt")
                    q_img_features = _extract_tensor(self._model.get_image_features(**img_inputs))
                    q_img_features = q_img_features / q_img_features.norm(dim=-1, keepdim=True)
                    query_image = q_img_features.cpu().numpy()[0]
                except Exception as e:
                    logger.debug(f"Could not load room image for SigLIP: {e}")

            txt_inputs = self._processor(text=[q_text], return_tensors="pt", padding=True, truncation=True)
            q_txt_features = _extract_tensor(self._model.get_text_features(**txt_inputs))
            q_txt_features = q_txt_features / q_txt_features.norm(dim=-1, keepdim=True)
            query_txt_vec = q_txt_features.cpu().numpy()[0]

            if query_image is not None:
                query_vec = 0.5 * query_image + 0.5 * query_txt_vec
                query_vec = query_vec / np.linalg.norm(query_vec)
            else:
                query_vec = query_txt_vec

            cand_key = tuple(r.rug_id for r in candidates)
            if not hasattr(self, "_cand_cache") or getattr(self, "_cand_cache_key", None) != cand_key:
                cand_prompts = [
                    f"{r.title}. Colors: {r.color}. Style: {r.style}. Pattern: {r.pattern}" for r in candidates
                ]
                c_inputs = self._processor(text=cand_prompts, return_tensors="pt", padding=True, truncation=True)
                cand_features = _extract_tensor(self._model.get_text_features(**c_inputs))
                cand_features = cand_features / cand_features.norm(dim=-1, keepdim=True)
                self._cand_cache = cand_features.cpu().numpy()
                self._cand_cache_key = cand_key
            cand_vecs = self._cand_cache

            sims = np.dot(cand_vecs, query_vec)

        results: list[RankResult] = []
        for rug, sim in zip(candidates, sims):
            score = float(max(0.0, sim))
            results.append(
                RankResult(
                    rug_id=rug.rug_id,
                    score=round(score, 4),
                    rank=0,
                    system_name=self.name,
                    explanation=f"SigLIP similarity: {sim:.4f}",
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        for idx, res in enumerate(results[:top_k], start=1):
            res.rank = idx
        return results[:top_k]
