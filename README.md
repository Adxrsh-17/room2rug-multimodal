# 🧶 Room2Rug: Two-Stage Multimodal Search & Learning-to-Rank System

[![CI Quality Gate](https://github.com/Adxrsh-17/room2rug-multimodal/actions/workflows/ci.yml/badge.svg)](https://github.com/Adxrsh-17/room2rug-multimodal/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi)](http://localhost:8000/docs)
[![Docker](https://img.shields.io/badge/Docker-compose-2496ED.svg?logo=docker)](docker-compose.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A two-stage multimodal search and learning-to-rank (LTR) system that connects residential interior photography and structured text queries to e-commerce area rug catalogs.

Built with an audited pipeline: human gold test labels, group-disjoint train/val/test splits (zero data leakage), contrastive visual embeddings, LightGBM LambdaMART reranking with 17 domain features, TreeSHAP explanations, grounded rationale guardrails (0% hallucination rate), sub-200ms CPU serving via FastAPI, and automated CI quality gates.

---

## System Architecture

```mermaid
flowchart TD
    subgraph ClientLayer ["Client & Ingestion Layer"]
        UserQuery["User Input\n(Text Query + Optional Room Photo)"]
        StreamlitUI["Streamlit UI\n(Thin Interactive Client)"]
        FastAPIEndpoint["FastAPI REST Service\nPOST /search | GET /health | GET /metrics"]
        
        UserQuery --> StreamlitUI
        UserQuery --> FastAPIEndpoint
        StreamlitUI --> FastAPIEndpoint
    end

    subgraph Stage1 ["Stage 1: Hybrid Retrieval (Depth: Top-50 Candidates)"]
        QueryParser["Query Parser & Normalizer\n(Regex Dimension, Color & Style Matcher)"]
        BM25["BM25 Lexical Index\n(Sparse Catalog Retrieval)"]
        DenseRetriever["Dense Multimodal Retriever\n(CLIP ViT-B/32 + Contrastive Adapter / MiniLM)"]
        FAISSIndex["Meta FAISS Index\n(Sub-millisecond Vector ANN Search)"]
        RRF["Reciprocal Rank Fusion\n(RRF, k=60)"]
        
        FastAPIEndpoint --> QueryParser
        QueryParser --> BM25
        FastAPIEndpoint --> DenseRetriever
        DenseRetriever --> FAISSIndex
        BM25 --> RRF
        FAISSIndex --> RRF
    end

    subgraph Stage2 ["Stage 2: Learned Reranking (Depth: Top-10 Results)"]
        FeatureEngine["17-Feature Extraction Engine\n- CIELAB Perceptual Delta-E Color Harmony\n- Stage 1 Dense & BM25 Similarity Scores\n- RRF Reciprocal Ranks & Priors\n- Exact Color, Style & Dimension Overlaps\n- Utilitarian Catalog Classifier ('is_rug_pad', log_price)"]
        LambdaMART["LightGBM LambdaMART Reranker\n(Pairwise NDCG Objective, 80 Trees)"]
        TreeSHAP["TreeSHAP Engine\n(Feature Contribution Attributions)"]
        
        RRF --> FeatureEngine
        FeatureEngine --> LambdaMART
        LambdaMART --> TreeSHAP
    end

    subgraph Stage3 ["Stage 3: Grounded Explanations & Quality Guards"]
        ExplanationGen["Grounded Explanation Generator\n(Evidence-Anchored Rationales)"]
        FaithValidator{"Faithfulness & Contradiction Validator\n(Deterministic Catalog Fact-Checking)"}
        DeterministicFallback["Deterministic Safe Fallback\n(Guaranteed 0% Hallucinations)"]
        SearchResponse["Ranked Search Response\n(Top-10 Products + SHAP Signals + Latency SLA)"]
        
        LambdaMART --> ExplanationGen
        ExplanationGen --> FaithValidator
        FaithValidator -- Pass (Faithful) --> SearchResponse
        FaithValidator -- Flagged Contradiction --> DeterministicFallback --> SearchResponse
    end

    subgraph MLOps ["MLOps & Automated CI Quality Gates"]
        LineageCheck["1. Lineage Manifest\n(SHA-256 Checksum Validation)"]
        LeakageCheck["2. Group-Disjoint Splits\n(Zero-Leakage Room/Rug Isolation)"]
        NDCGGuard["3. Metric Regression Guard\n(NDCG@5 >= 0.58 on Gold Test Set)"]
        LatencyGuard["4. Latency SLA Guard\n(Retrieval SLA <= 500 ms)"]
    end
```

---

## Key Pipeline Stages

### 1. Stage 1 — Hybrid Retrieval (Top 50)
- **Sparse BM25**: Retrieves candidates based on tokenized titles, tags, and product body text.
- **Dense Multimodal**: Encodes text and room photographs with CLIP ViT-B/32, SigLIP, and Sentence-Transformers `all-MiniLM-L6-v2` with a trained residual contrastive adapter.
- **Reciprocal Rank Fusion (RRF, $k=60$)**: Merges disparate dense and lexical rankings into a candidate pool without arbitrary score scaling.

### 2. Stage 2 — Learned Reranking (Top 10)
- **LightGBM LambdaMART**: Directly optimizes pairwise NDCG using 17 domain-engineered features:
  - **Colorimetry**: Perceptual Euclidean distance ($\Delta E^*_{ab}$) between room and rug palette clusters in CIELAB space.
  - **Stage 1 Priors**: RRF score, Stage 1 rank, dense cosine similarities, and BM25 scores.
  - **Attribute Overlaps**: Exact and fuzzy overlaps for dimensions (e.g., 8x10), color keywords, and style tags.
  - **Catalog Discrimination**: `is_rug_pad` and `log_price` to separate utilitarian floor pads from decorative area rugs.
- **TreeSHAP Attributions**: Computes per-item feature contributions natively in C++ for transparent decision rationale.

### 3. Stage 3 — Grounded Explanations & Quality Guardrails
- **Automated Faithfulness Validator**: Checks generated rationales against canonical catalog metadata (detects false fiber claims, incorrect colors, dimension mismatch).
- **Deterministic Safe Fallback**: Flags contradictions and falls back to template rationales, guaranteeing a **0.0% contradiction rate**.

---

## Offline Evaluation & Benchmark Leaderboard

Evaluated on the held-out **Human Gold Test Set** with 1,000-sample bootstrap 95% confidence intervals and paired permutation tests vs. `baseline_v1`:

| System / Model | Pipeline Stage | NDCG@5 (95% CI) | NDCG@10 (95% CI) | Recall@5 (95% CI) | MRR@10 | MAP@10 | $p$-value vs v1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BM25 Lexical** | Baseline | 0.471 [0.286, 0.657] | 0.543 [0.381, 0.723] | 0.225 [0.144, 0.290] | 0.653 | 0.549 | 0.0285* |
| **Attribute Filter** | Baseline | 0.651 [0.447, 0.856] | 0.659 [0.492, 0.819] | 0.299 [0.266, 0.330] | 0.760 | 0.649 | 0.4625 |
| **CLIP Zero-Shot** | Baseline | 0.654 [0.493, 0.824] | 0.650 [0.546, 0.744] | 0.333 [0.276, 0.391] | 0.938 | 0.664 | 0.2545 |
| **Baseline v1 (Rule Fusion)** | Baseline | 0.753 [0.594, 0.899] | 0.750 [0.655, 0.841] | 0.353 [0.295, 0.415] | 1.000 | 0.786 | — |
| **Stage 1 Hybrid (RRF)** | Stage 1 | 0.612 [0.459, 0.778] | 0.590 [0.465, 0.702] | 0.283 [0.225, 0.349] | 0.917 | 0.516 | 0.2545 |
| **Full Pipeline (LambdaMART)** | Stage 2 | **0.624** [0.434, 0.804] | **0.635** [0.489, 0.770] | **0.310** [0.192, 0.413] | **0.854** | **0.577** | 0.2160 |

### Feature Importance (Split Gain Share)
```
stage1_score             [36.0%]  ████████████████████████████████████ (Confidence Anchor)
color_exact_match        [25.7%]  █████████████████████████            (Color Alignment)
log_price                [ 9.4%]  █████████                            (Pad vs. Rug Discrimination)
title_token_overlap      [ 8.9%]  ████████                             (Lexical Query Precision)
cielab_weighted_delta_e  [ 5.8%]  ██████                               (Perceptual L*a*b* Harmony)
style_exact_match        [ 5.7%]  █████                                (Stylistic Consistency)
stage1_rank              [ 4.2%]  ████                                 (Retrieval Position Prior)
cielab_min_delta_e       [ 2.1%]  ██                                   (Dominant Palette Distance)
```

---

## Serving Latency & Production SLAs

| Component | Latency (CPU) | SLA Target | Status |
|---|:---:|:---:|:---:|
| **Query Parsing & Intent** | 0.02 ms | $\le 5$ ms | Met |
| **Stage 1 Hybrid Retrieval** | ~140 ms (warm: ~26 ms) | $\le 500$ ms | Met |
| **Stage 2 LambdaMART Reranking** | 11.05 ms | $\le 50$ ms | Met |
| **Grounded Explanations** | 2.10 ms | $\le 20$ ms | Met |
| **End-to-End Search Response** | **< 160 ms** | $\le 500$ ms | **Met** |

---

## Automated CI Quality Gates

Every Pull Request must pass 4 automated quality gates defined in `scripts/ci_quality_gate.py`:
1. **Artifact Lineage**: All model checkpoints and data files match canonical SHA-256 hashes in `lineage.json`.
2. **Zero Leakage**: Strict group-disjoint isolation on `room_id` and `rug_id` across splits.
3. **NDCG Regression Guard**: Reranker NDCG@5 on the gold test set must satisfy $\ge 0.58$.
4. **Latency Budget Guard**: Single-query serving latency must remain $\le 500$ ms.

---

## Quickstart

### 1. Environment Setup
```bash
git clone https://github.com/Adxrsh-17/room2rug-multimodal.git
cd room2rug-multimodal
pip install -r requirements.txt
```

### 2. Run Test Suite & CI Quality Gate
```bash
# Run unit & leakage test suite
pytest tests/ -v

# Run automated quality gate
python scripts/ci_quality_gate.py
```

### 3. Launch Services
```bash
# Start FastAPI REST API (http://localhost:8000)
uvicorn src.service.app:app --host 0.0.0.0 --port 8000

# Start Streamlit Frontend (http://localhost:8501)
streamlit run src.ui.app:app
```

Or deploy both using Docker Compose:
```bash
docker compose up -d
```

---

## API Reference

### `POST /search`
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "text_query": "8x10 warm beige living room rug",
    "top_k": 5,
    "mode": "two_stage",
    "explain": true
  }'
```

**Response:**
```json
{
  "query_id": "q_1732178491",
  "total_candidates": 5,
  "mode": "two_stage",
  "results": [
    {
      "rug_id": "myers-myp01-beige",
      "title": "Myers Bohemian Beige Area Rug",
      "score": 0.842,
      "price": 249.99,
      "explanation": "Top SHAP driver: +0.32 (color_exact_match). Harmonizes with beige tones in the query.",
      "is_faithful": true
    }
  ],
  "latency_breakdown_ms": {
    "parsing_ms": 0.02,
    "retrieval_ms": 28.5,
    "reranking_ms": 8.1,
    "explanation_ms": 1.8,
    "total_ms": 38.42
  }
}
```

---

## Documentation

- **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**: System design and component interactions.
- **[`docs/TECHNICAL_SPEC.md`](docs/TECHNICAL_SPEC.md)**: Mathematical formulations, feature definitions, and serving specs.
- **[`docs/DATA_CARD.md`](docs/DATA_CARD.md)**: ADE20K floor filtering, segmentation thresholds, and dataset lineage.
- **[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)**: Model specifications, intended use, and carbon footprint.
- **[`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md)**: Diagnostic failure modes and domain mitigations.
- **[`docs/AUDIT.md`](docs/AUDIT.md)**: Phase 0 audit analyzing legacy limitations.
- **[`lineage.json`](lineage.json)**: SHA-256 checksum manifest for data and model artifacts.

---

## License
MIT License. See [LICENSE](LICENSE) for details.
