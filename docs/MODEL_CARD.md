# Model Card: Room2Rug Two-Stage Multimodal Retrieval & Reranking System

## Model Details

- **Model Name**: `room2rug-multimodal-twostage`
- **Version**: `0.2.0`
- **Model Type**: Two-stage cascaded information retrieval and Learning-to-Rank (LTR) system:
  - **Stage 1 (Hybrid Deep Retrieval)**: Multi-encoder fusion combining dense vision-language representations (OpenAI CLIP ViT-B/32, SigLIP, Sentence-Transformers `all-MiniLM-L6-v2`) with sparse BM25 lexical search via **Reciprocal Rank Fusion (RRF, $k=60$)**.
  - **Stage 2 (Learned Reranking)**: **LightGBM LambdaMART** ranker trained on pairwise NDCG loss using 17 domain-engineered features (CIELAB color harmony, dense cosine similarities, metadata compatibility, price signals, and catalog integrity indicators).
  - **Stage 3 (Grounded Explanations)**: Constraint-bounded prompt generation with an automated **Faithfulness & Contradiction Validator** and deterministic template fallback.
- **Developers**: Adarsh Pradeep
- **License**: MIT
- **Release Date**: November 2026

---

## Intended Use & Domain Boundaries

### Primary Intended Uses
- Finding interior area rugs that harmonize with room photographs (living room, bedroom, dining room, home office) based on visual color palette, style cues, and user search constraints.
- Real-time search serving via REST API (`/search`, `/health`, `/metrics`) and interactive Streamlit UI.
- Explainable recommendation providing grounded rationales tied directly to verified catalog attributes.

### Out-of-Scope Uses
- General interior furniture recommendation (sofas, lighting, tables) not supported by the rug-specific feature pipeline.
- Outdoor architectural or commercial real estate analysis.
- Unconstrained conversational chat: The model generates structured search results and grounded rationales, not open-domain conversational dialogue.

---

## Training Data & Group-Aware Split Architecture

- **Catalog Corpus**: Amazon Berkeley Objects (ABO) curated rug catalog ($N=68$ products with complete metadata: title, description, sizes, prices, primary color, style, fiber type, and weave).
- **Scene Corpus**: ADE20K residential interiors filtered to scenes with $\ge 10\%$ floor visibility across `living_room`, `bedroom`, `dining_room`, and `home_office`.
- **Group-Aware Splitting Strategy**:
  - The dataset is partitioned into **Train (70%)**, **Validation (15%)**, and **Test (15%)**.
  - **Strict Zero-Leakage Guarantee**: Splits are group-disjoint on both `rug_id` (product handle) and `room_id` (scene photo). No product or room image appearing in training ever appears in the test set.
  - Verified by unit tests in [`tests/test_leakage.py`](../tests/test_leakage.py).
- **Separation of Weak and Gold Labels**:
  - *Training Split*: Annotated using weak rule-based compatibility heuristics and an offline **VLM-as-Judge** (Spearman $\rho = 0.4168$, $p < 10^{-6}$).
  - *Test Split*: **100% Human Gold Annotations** (graded 0/1/2) collected using our Streamlit Labeling Studio with multi-rater agreement (Cohen's quadratic $\kappa = 0.7040$).

---

## Quantitative Evaluation on Human Gold Test Set

All headline numbers evaluated on the held-out **Human Gold Test Set** using **1,000-sample Bootstrap 95% Confidence Intervals** and paired permutation tests vs. `baseline_v1`:

| System / Model | Pipeline Stage | NDCG@5 (95% CI) | NDCG@10 (95% CI) | Recall@5 (95% CI) | MRR@10 | MAP@10 | $p$-value vs v1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BM25 Lexical** | Baseline | 0.471 [0.286, 0.657] | 0.543 [0.381, 0.723] | 0.225 [0.144, 0.290] | 0.653 | 0.549 | 0.0285* |
| **Attribute Filter** | Baseline | 0.651 [0.447, 0.856] | 0.659 [0.492, 0.819] | 0.299 [0.266, 0.330] | 0.760 | 0.649 | 0.4625 |
| **CLIP Zero-Shot** | Baseline | 0.654 [0.493, 0.824] | 0.650 [0.546, 0.744] | 0.333 [0.276, 0.391] | 0.938 | 0.664 | 0.2545 |
| **Baseline v1 (Rule Fusion)** | Baseline | 0.753 [0.594, 0.899] | 0.750 [0.655, 0.841] | 0.353 [0.295, 0.415] | 1.000 | 0.786 | — |
| **Stage 1 Hybrid (RRF)** | Stage 1 | 0.612 [0.459, 0.778] | 0.590 [0.465, 0.702] | 0.283 [0.225, 0.349] | 0.917 | 0.516 | 0.2545 |
| **Full Pipeline (LambdaMART)** | Stage 2 | **0.624** [0.434, 0.804] | **0.635** [0.489, 0.770] | **0.310** [0.192, 0.413] | **0.854** | **0.577** | 0.2160 |

---

## Known Failure Modes & Diagnostic Mitigations

Comprehensive error analysis is documented in [`docs/ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md):

1. **Palette Undertone Discordance**: Dense vision embeddings occasionally group warm-yellow wood floors with cool-blue rugs due to similar average luminance. *Mitigation*: CIELAB Euclidean color distance ($\Delta E^*_{ab}$) penalizes chromatic Euclidean distance directly.
2. **Pattern Competition in Living Rooms**: High visual complexity in room photos conflicts with intricate traditional medallions. *Mitigation*: LambdaMART learns split gains prioritizing low-complexity abstract rugs for busy rooms.
3. **Utilitarian Confusion (`is_rug_pad`)**: Raw dense models rank plain beige floor pads highly for neutral living rooms. *Mitigation*: The `is_rug_pad` indicator feature receives high split gain, suppressing non-decorative pads unless explicitly requested.

---

## Serving Latency & Performance Profile

Measured on standard CPU laptop hardware:

| Component Stage | Median Latency | Optimization Mechanism |
|---|:---:|---|
| **Query Parsing & Tokenization** | 0.02 ms | Compiled regex & dictionary lookup |
| **Stage 1 Hybrid Retrieval** | 144.25 ms | Precomputed embeddings & FAISS FlatIP |
| **Stage 2 LambdaMART Reranking** | 11.05 ms | Vectorized feature extraction & native C++ TreeSHAP |
| **Grounded Explanations** | 2.10 ms | Deterministic template fallback |
| **FAISS ANN Search (N=100k synthetic)** | 1.68 ms | Graph HNSW index ($M=32$, $efSearch=64$) |

---

## Environmental & Compute Footprint

- **Hardware**: Standard laptop CPU (Intel Core i7 / AMD Ryzen) + Free Google Colab T4 GPU for training experiments.
- **Total Compute Time**: ~1.5 hours across all phases (data processing, contrastive adapter training, LightGBM fitting, benchmarking).
- **Estimated Energy Consumption**: $< 0.45\text{ kWh}$.
- **Estimated Carbon Emissions**: $< 0.16\text{ kg CO}_2\text{eq}$ (assuming global average grid intensity of $350\text{ g CO}_2/\text{kWh}$).
