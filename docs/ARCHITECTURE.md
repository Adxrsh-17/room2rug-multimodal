# System Architecture Specification

## Overview

`room2rug-multimodal` is a production-grade, two-stage search and learning-to-rank system designed to match residential interior photography and structured text queries to e-commerce area rug catalogs.

The architecture solves three core challenges in home decor discovery:
1. **Multimodal Intent Gap**: Text queries express dimensions and styles ("8x10 modern"), while room photos provide spatial layout and color palette.
2. **Chromatic Undertone Discordance**: Standard vision-language models often confuse luminance with color harmony, matching warm wood floors with cool blue rugs.
3. **Hallucination in Automated Recommendations**: Standard LLM rationale generation often hallucinates materials, colors, or sizing not present in the catalog item.

---

## Architectural Diagram

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

## Detailed Pipeline Stages

### Stage 1: Hybrid Retrieval (Depth: Top 50)
- **Goal**: High-recall candidate generation combining lexical precision with multimodal semantic search.
- **Sparse Path**: BM25 lexical scoring across cleaned product titles, descriptions, and tags.
- **Dense Path**:
  - Image Encoder: CLIP ViT-B/32 or SigLIP base patch16 with contrastive residual projection adapter.
  - Text Encoder: Sentence-Transformers `all-MiniLM-L6-v2` with candidate embedding in-memory caching.
  - Indexing: Meta FAISS inner product (`IndexFlatIP` for catalog scale, HNSW for enterprise scaling).
- **Fusion**: Reciprocal Rank Fusion (RRF, $k=60$):
  $$\text{RRF}(d) = \sum_{m \in M} \frac{w_m}{60 + r_m(d)}$$

### Stage 2: Learned Reranking (LTR, Depth: Top 10)
- **Model**: LightGBM LambdaMART ranker optimizing pairwise NDCG.
- **17 Ranking Features**:
  1. Dense Similarities: `clip_sim`, `siglip_sim`, `minilm_sim`.
  2. Stage 1 Signals: `stage1_score`, `stage1_rrf`, `stage1_rank`, `bm25_score`.
  3. Perceptual Colorimetry: `cielab_min_delta_e`, `cielab_weighted_delta_e` (Euclidean distance across dominant $L^*a^*b^*$ palette clusters).
  4. Attribute Overlaps: `color_exact_match`, `color_overlap_score`, `style_exact_match`, `style_overlap_score`, `size_exact_match`.
  5. Catalog Hygiene & Lexical Overlap: `is_rug_pad`, `log_price`, `title_token_overlap`.
- **Explainability**: TreeSHAP feature contributions computed via native C++ booster routines.

### Stage 3: Grounded Explanations & Faithfulness Guard
- **Engine**: Template-grounded and LLM-constrained rationale generator.
- **Faithfulness Validator**: Verifies every claim against canonical catalog records:
  - Color contradiction checks.
  - Material and fiber verification.
  - Dimension and shape conformity.
  - Rug pad type enforcement.
- If any contradiction is detected, the pipeline automatically falls back to deterministic template generation, ensuring a **0% hallucination rate**.

### Stage 4: Production Serving & MLOps
- **Serving Layer**: FastAPI asynchronous microservice exposing `/search`, `/health`, and `/metrics`.
- **UI Client**: Streamlit thin client for interactive room photo upload, query entry, and visual inspection.
- **Automated CI Quality Gate**: Pull request validation script (`scripts/ci_quality_gate.py`):
  1. Artifact Lineage: SHA-256 integrity verification (`lineage.json`).
  2. Data Leakage: Group-disjoint room/rug split verification.
  3. Metric Regression: NDCG@5 $\ge 0.58$ on human gold annotations.
  4. Latency SLA: Single-query retrieval $\le 500$ ms.
