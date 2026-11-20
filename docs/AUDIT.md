# Technical Audit & Architectural Review: room2rug-multimodal v1

**Date:** November 2026  
**Auditor:** Senior ML Engineer  
**Repository:** `room2rug-multimodal` (https://github.com/Adxrsh-17/room2rug-multimodal)  
**Target:** MLE / AI Engineer Interview Defense  

---

## 1. Executive Summary

This audit assesses the initial version (`v1`) of `room2rug-multimodal`. While the codebase demonstrates a clean, functional proof-of-concept integrating regular expression parsing, dense text similarity, CLIP vision-language retrieval, and a Streamlit UI, **it cannot currently withstand technical scrutiny in a senior Machine Learning Engineer or AI Engineer interview**.

The primary blockers are:
1. **Critical Evaluation Circularity**: The evaluation benchmark labels (0/1/2 graded relevance) are computed using the identical substring and token matching heuristics implemented in the retrieval filter. The resulting metrics ($\text{Precision@1}=1.0$, $\text{Precision@5}=1.0$, $\text{MRR@10}=1.0$) are self-fulfilling artifacts rather than genuine measures of retrieval quality.
2. **Catalog Scale**: The catalog consists of only 68 unique product handles. Ranking metrics (NDCG@10, Recall@K) on 68 items suffer from extreme ceiling effects and lack statistical significance.
3. **Pure Zero-Shot System**: No models have been trained or adapted. Neither contrastive heads, adapters, nor learned rerankers have been applied to bridge the interior-design domain gap.
4. **Latency Bottlenecks**: End-to-end multimodal recommendations average over 6.8 seconds on CPU under cold/uncached runs due to image fetching, unbatched PyTorch inference, and unoptimized embeddings.
5. **Lack of Separation & Group Splitting**: No formal train/validation/test splits exist, let alone group-aware partitions preventing rug handle and room image leakage.

---

## 2. Evaluation Audit: The Circularity Problem

### 2.1 The Root Cause in `scripts/evaluate_metrics.py`
In `scripts/evaluate_metrics.py`, line 173 defines `score_ground_truth(product, criteria)`:

```python
def score_ground_truth(product: dict[str, Any], criteria: dict[str, Any]) -> int:
    searchable = (product.get("searchable_text") or "").lower()
    title = (product.get("title") or "").lower()
    tags = (product.get("tags") or "").lower()
    full_text = f"{searchable} {title} {tags}"
    ...
    # Checks if criteria colors, styles, shapes, or sizes exist as substrings in full_text
    passed = sum([matches_color, matches_style, matches_shape, matches_size])
    if passed == total:
        return 2  # Highly relevant
    elif passed >= total / 2:
        return 1  # Moderately relevant
    return 0
```

Now consider `src/filter_catalog.py`, which implements `metadata_score()` used by the ranker:
- `_color_score`: Checks whether requested colors exist as substrings in `searchable_text`.
- `_style_score`: Checks whether requested styles exist as substrings in `searchable_text`.
- `_size_score`: Checks numeric tolerance on dimension strings.

### 2.2 Why the Metrics are Artificially Inflated
1. **Self-Referential Relevance**: The benchmark queries define criteria (e.g. `"8x10 beige traditional rug"` $\rightarrow$ `colors: ["beige", "brown", "tan", "cream"]`, `styles: ["traditional", "oriental", "persian"]`). When the filter and ranker run, they score candidates based on whether these exact substrings appear in `title` or `tags`. The evaluation harness then declares a product "ground-truth relevant" if those exact same substrings appear in `title` or `tags`.
2. **Visual Query Bias**: For visual queries like `q11` (`room1.jpeg`), the "ground truth" criteria were manually defined as `colors: ["beige", "grey", "neutral", "brown", "cream"]` and `styles: ["modern", "transitional", "contemporary", "oriental"]`. Rather than measuring whether human interior designers find the rug aesthetically harmonious with the room, the harness simply evaluated whether the retrieved rug's text metadata contained those arbitrary tokens.
3. **Conclusion**: The headline metrics in v1 ($\text{Precision@5}=1.0$, $\text{MRR@10}=1.0$, $\text{mAP@10}=0.979$) must be categorized as **rule-consistency verification on a toy set**, not empirical retrieval performance.

---

## 3. Catalog Scale & Representation Audit

- **Catalog Size**: 68 unique products derived from a Shopify CSV export, with images mapped to the Amazon Berkeley Objects (ABO) dataset via `scripts/map_abo_rug_images.py`.
- **Retrieval Implications**: In a 68-item space, retrieving Top-10 covers $14.7\%$ of the entire universe. Benchmarks such as FAISS ANN index throughput ($4,545\text{ QPS}$) on 68 vectors are essentially testing in-memory L1 cache matrix multiplication of a $1 \times 512$ by $512 \times 68$ matrix, not approximate nearest neighbor indexing.
- **Metadata Sparsity**: Missing fields (e.g., price variants, material composition, weave type) were inconsistently handled.
- **Requirement for Phase 1**: Scale the catalog to the full genuine ABO rug subset ($>1,000$ items) or a dedicated e-commerce subset, deduplicate near-identical listings with perceptual hashing, and enforce strict metadata schemas.

---

## 4. Model Architecture & Zero-Shot Constraints

The v1 pipeline uses strictly off-the-shelf, frozen models:
1. **Text Retriever**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional). High speed, but lacks fine-tuning on interior design terminology, query phrasing, or catalog synonymy.
2. **Vision-Language Retriever**: `openai/clip-vit-base-patch32` (512-dimensional). The patch-32 variant is fast but has coarse spatial resolution. Crucially, CLIP was pre-trained on web image-caption pairs—it has never been trained on the task of matching a **3D room setting** to an **isolated 2D overhead rug photograph**.
3. **Late Fusion Heuristics**: Fixed manual weights ($0.6 \times \text{image} + 0.4 \times \text{text}$) in Mode B were chosen by intuition without validation tuning.
4. **Ranking Layer**: No second-stage learned reranker (LambdaMART / GBDT). Signals like price, color palette harmony (in CIELAB space), and spatial size ratio are either ignored or crudely weighted.

---

## 5. Latency Breakdown & Computational Profile

Benchmarking under CPU conditions revealed the following latency distribution:

| Operation | Typical Latency (ms) | Bottleneck Source |
|---|---|---|
| **Query Intent Parsing** | $0.43\text{ ms}$ | Regex evaluation (negligible) |
| **Metadata Filtering** | $1.12\text{ ms}$ | In-memory Python loop over 68 dicts |
| **MiniLM Text Embedding** | $21.5\text{ ms}$ | PyTorch CPU text forward pass |
| **FAISS Flat Search (68 vectors)** | $0.22\text{ ms}$ | In-memory dot product |
| **CLIP Vision Embedding** | $145\text{ ms}$ | ViT-B/32 vision transformer forward pass |
| **Network Image Retrieval** | $1,500 - 5,000\text{ ms}$ | Uncached HTTP requests to external S3 URLs |
| **End-to-End Multimodal Request** | **$6.8\text{ s}$ (cold)** | Sequential downloads + unbatched CPU inference |

### Optimization Targets for Phase 6:
- Local image caching and persistent disk storage.
- ONNX Runtime / Int8 quantization for transformer backbones.
- Asynchronous FastAPI service decoupling embedding computation from client rendering.

---

## 6. Test Coverage & Engineering Deficits

1. **Test Runner**: `tests/test_pipeline.py` executes imperatively with custom log arrays (`RESULTS_LOG`) and `sys.exit()`, rather than standard `pytest` fixtures, parameterized tests, and assertions.
2. **Hardcoded Constants**: Thresholds (`0.15`), size tolerance (`1.5`), fusion weights (`0.6, 0.4`), and model names are hardcoded inside library modules rather than managed through configuration files.
3. **No CI / CD**: The repository had no continuous integration workflows, pre-commit linting, or formatting standards.
4. **Experiment Tracking**: No tracking framework (e.g., MLflow) was configured; evaluation runs dumped unstructured JSON directly into `data/eval_results.json`.

---

## 7. Action Plan & Architectural Remediation

To elevate this repository to a defensible MLE standard, we execute the following roadmap:

- **Phase 0 (Foundations)**: Freeze v1 as `baseline_v1`. Add configuration management (`configs/`), deterministic seeding (`src/utils/seed.py`), local MLflow tracking (`src/utils/tracking.py`), modern `pyproject.toml`, pinned requirements, Makefile, ruff linting, and GitHub Actions CI.
- **Phase 1 (Data at Scale)**: Ingest the full ABO rug catalog ($>1,000$ products). Deduplicate with perceptual hashing (`pHash`) and CLIP embeddings. Source a licensed room image corpus. Establish strict **group-aware train/val/test splits** preventing rug product and room image leakage.
- **Phase 2 (Gold Evaluation Harness)**: Build an interactive Streamlit labeling interface. Collect a human gold-labeled test set with written rubrics. Implement an offline VLM-as-judge solely for weak training signal. Compute bootstrap 95% confidence intervals and slice metrics.
- **Phase 3 (Retrieval Fine-Tuning)**: Contrastive fine-tuning (InfoNCE) with hard-negative mining on lightweight projection heads/LoRA. Hybrid dense + sparse (BM25) reciprocal rank fusion (RRF).
- **Phase 4 (Learned Reranker)**: Two-stage retrieval + LightGBM LambdaMART ranking with palette distance (CIELAB k-means), style match, and price features. Error analysis and SHAP feature importances.
- **Phase 5 (Grounded Explanations)**: Faithfulness-verified attribution against catalog facts.
- **Phase 6 (Serving & Performance)**: Production FastAPI microservice, ONNX/quantization profiling, FAISS IVF/HNSW index benchmarks at synthetic scale, and Locust load testing.
- **Phase 7 (MLOps & Quality Gates)**: Data versioning, model card, embedding drift detection (PSI/MMD), and CI regression gates.
- **Phase 8 (Interview Packaging)**: Honest README with verified numbers from `results/`, technical blog post, interview Q&A guide, and grounded resume bullets.
