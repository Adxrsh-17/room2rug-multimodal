# Technical Specification: Two-Stage Multimodal Search & Reranking System

## 1. System Overview

`room2rug-multimodal` provides real-time search and recommendation for e-commerce rug discovery. The pipeline processes multimodal queries (free-text specifications combined with residential interior photography) and returns ranked catalog items with grounded explanations.

### Key Capabilities
- **Multimodal Intent Parsing**: Jointly resolves dimensional constraints (e.g., "8x10", "runner"), style cues ("boho", "traditional"), and chromatic context extracted from room photography.
- **Two-Stage Cascaded Retrieval**: Fuses sparse lexical search (BM25) and dense embeddings (CLIP ViT-B/32, SigLIP, MiniLM with contrastive residual adapter) via Reciprocal Rank Fusion ($k=60$), followed by a LightGBM LambdaMART reranker.
- **Colorimetric Harmony**: Computes perceptual color distance ($\Delta E^*_{ab}$) between room palette clusters and rug dominant colors in CIELAB color space.
- **Grounded Explanations with Quality Guardrails**: Verifies generated rationales against catalog facts, guaranteeing zero attribute contradictions.

---

## 2. Ingestion & Preprocessing

The product catalog is derived from Amazon Berkeley Objects (ABO) and Shopify exports ($N=68$ unique products across multiple size variants):
1. **Deduplication**: Image-position filtering selects primary product images and removes redundant variant assets.
2. **Text Normalization**: Strips HTML tags, unifies dimensions into standard notation (e.g., `8 × 10` $\to$ `8x10`), and aggregates titles, tags, and descriptions into a structured search document.
3. **Palette Extraction**: Computes dominant color clusters and cluster weights in CIELAB ($L^*a^*b^*$) space using $k$-means ($k=3$).

---

## 3. Stage 1: Hybrid Retrieval

### 3.1 Sparse Retrieval (BM25)
- Standard BM25Okapi scoring across tokenized query strings and catalog documents ($k_1=1.5, b=0.75$).

### 3.2 Dense Multimodal Retrieval
- **Text Encoder**: Sentence-Transformers `all-MiniLM-L6-v2` with candidate embedding caching in memory.
- **Vision-Language Encoders**: OpenAI CLIP ViT-B/32 and Google SigLIP (`siglip-base-patch16-224`).
- **Contrastive Projection Adapter**: A residual projection head trained via InfoNCE loss with hard negatives to align room floor representations with product catalog vectors.
- **Vector Search**: Meta FAISS inner product index (`IndexFlatIP`).

### 3.3 Reciprocal Rank Fusion (RRF)
Combines candidate rankings from sparse and dense sub-systems into an unified Top-50 candidate pool:
$$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{w_m}{k_{\text{rrf}} + \text{rank}_m(d)}$$
where $k_{\text{rrf}} = 60$, with default weights $w_{\text{dense}} = 0.50$, $w_{\text{bm25}} = 0.25$, and $w_{\text{attr}} = 0.25$.

---

## 4. Stage 2: Learned Reranking (LambdaMART)

Candidates from Stage 1 are reranked using a LightGBM LambdaMART model (`models/lambdamart_reranker.txt`).

### 4.1 Feature Set (17 Signals)
| Feature Name | Category | Description | Split Gain Share |
|---|---|---|:---:|
| `stage1_score` | Stage 1 Prior | Normalized retrieval score from Stage 1 | 36.0% |
| `color_exact_match` | Attribute Match | Exact query color token overlap with product metadata | 25.7% |
| `log_price` | Price / Category | Natural logarithm of product price ($\ln(1 + \text{price})$) | 9.4% |
| `title_token_overlap` | Lexical Overlap | Jaccard token overlap between query and product title | 8.9% |
| `cielab_weighted_delta_e` | Colorimetry | Weight-averaged $\Delta E^*_{ab}$ distance between room and rug palettes | 5.8% |
| `style_exact_match` | Attribute Match | Exact query style token overlap with product metadata | 5.7% |
| `stage1_rank` | Stage 1 Prior | Candidate rank position from Stage 1 retrieval | 4.2% |
| `cielab_min_delta_e` | Colorimetry | Minimum pairwise $\Delta E^*_{ab}$ across palette clusters | 2.1% |
| `stage1_rrf` | Stage 1 Prior | Raw RRF fusion score ($1 / (60 + \text{rank})$) | 1.8% |
| `color_overlap_score` | Attribute Match | Normalized recall of query colors in product metadata | 0.4% |
| `clip_sim` | Dense Similarity | Cosine similarity between query and product in CLIP space | < 0.1% |
| `siglip_sim` | Dense Similarity | Cosine similarity between query and product in SigLIP space | < 0.1% |
| `minilm_sim` | Dense Similarity | Cosine similarity between query text and product text in MiniLM space | < 0.1% |
| `bm25_score` | Sparse Similarity | Normalized BM25 lexical similarity score | < 0.1% |
| `style_overlap_score` | Attribute Match | Normalized recall of query styles in product metadata | < 0.1% |
| `size_exact_match` | Attribute Match | Binary match on query dimensions (e.g., 8x10) against product sizes | < 0.1% |
| `is_rug_pad` | Guardrail | Binary indicator detecting utilitarian rug pads vs. area rugs | < 0.1% |

### 4.2 Model Hyperparameters
- Objective: `lambdarank`
- Metric: `ndcg` evaluated at $k \in \{1, 3, 5, 10\}$
- Number of estimators: 80
- Learning rate: 0.05
- Number of leaves: 15
- Min child samples: 3

### 4.3 Feature Attributions (TreeSHAP)
Individual ranking decisions are explained via TreeSHAP feature contributions computed using LightGBM's native C++ implementation (`booster.predict(X, pred_contrib=True)`).

---

## 5. Stage 3: Grounded Explanations & Faithfulness Guard

To prevent generative hallucinations in consumer-facing recommendations, the explanation subsystem implements strict guardrails:
1. **Grounded Generation**: Produces rationales structured around verified attributes (color compatibility, dimension alignment, material properties).
2. **Faithfulness Validator**: Verifies explanation text against the canonical catalog record:
   - Validates stated materials (e.g., wool, jute, polypropylene).
   - Validates stated primary colors and dimensions.
   - Detects utilitarian product confusion (rug pad vs. decorative rug).
3. **Deterministic Fallback**: If an unfaithful claim is flagged, the pipeline automatically substitutes a deterministic template rationale, ensuring a **0% contradiction rate**.

---

## 6. Offline Evaluation & Benchmark Results

Evaluated on the held-out **Human Gold Test Set** ($N=8$ test queries, 68 catalog items) with 1,000-sample bootstrap 95% confidence intervals and paired permutation tests vs. `baseline_v1`:

| System / Model | Pipeline Stage | NDCG@5 (95% CI) | NDCG@10 (95% CI) | Recall@5 (95% CI) | MRR@10 | MAP@10 | $p$-value vs v1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BM25 Lexical** | Baseline | 0.471 [0.286, 0.657] | 0.543 [0.381, 0.723] | 0.225 [0.144, 0.290] | 0.653 | 0.549 | 0.0285* |
| **Attribute Filter** | Baseline | 0.651 [0.447, 0.856] | 0.659 [0.492, 0.819] | 0.299 [0.266, 0.330] | 0.760 | 0.649 | 0.4625 |
| **CLIP Zero-Shot** | Baseline | 0.654 [0.493, 0.824] | 0.650 [0.546, 0.744] | 0.333 [0.276, 0.391] | 0.938 | 0.664 | 0.2545 |
| **Baseline v1 (Rule Fusion)** | Baseline | 0.753 [0.594, 0.899] | 0.750 [0.655, 0.841] | 0.353 [0.295, 0.415] | 1.000 | 0.786 | — |
| **Stage 1 Hybrid (RRF)** | Stage 1 | 0.612 [0.459, 0.778] | 0.590 [0.465, 0.702] | 0.283 [0.225, 0.349] | 0.917 | 0.516 | 0.2545 |
| **Full Pipeline (LambdaMART)** | Stage 2 | **0.624** [0.434, 0.804] | **0.635** [0.489, 0.770] | **0.310** [0.192, 0.413] | **0.854** | **0.577** | 0.2160 |

---

## 7. Production Serving & Latency Budget

### Latency Budget (Target: $\le 500$ ms SLA)
- **Query Parsing**: 0.02 ms
- **Stage 1 Retrieval**: ~140 ms (median on CPU; sub-30ms with warm in-memory cache)
- **Stage 2 Reranking**: ~11 ms
- **Grounded Explanations**: ~2 ms
- **Total End-to-End Latency**: **< 160 ms** (P95 < 250 ms)

### REST API Endpoints
- `POST /search`: Accepts `SearchRequest` (text query, optional base64 room image, top_k, mode, explain flag).
- `GET /health`: Healthcheck endpoint reporting model and index status.
- `GET /metrics`: Service telemetry and latency metrics.
