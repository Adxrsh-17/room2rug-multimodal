# Technical Write-Up: Multimodal Rug Search & Recommendation System

## Problem Statement

Build a two-part search and recommendation system for a rug e-commerce catalog:

- **Part 1 — Structured Text Search**: Accept free-text queries like "8x10 beige traditional rug" and return the most relevant products by combining metadata filtering with semantic similarity.
- **Part 2 — Multimodal Recommendation**: Accept a room photograph (optionally combined with a text query) and recommend rugs that visually fit the room's palette, style, and spatial context.

The catalog is a Shopify CSV export containing 68 unique products across multiple size variants.

---

## Data Preprocessing (`src/preprocess.py`)

The raw Shopify CSV has one row per product variant (size) and multiple rows per image. Preprocessing groups rows by `Handle` and:

1. **Selects the primary image row** (`Image Position == 1`) to avoid duplicate images.
2. **Strips HTML** from the `Body (HTML)` column using BeautifulSoup to get clean, searchable text.
3. **Collects all size variants** from `Option1 Value`, `Option2 Value`, `Variant Title`, etc. and normalizes dimension strings (e.g., `"8 × 10"` → `"8x10"`).
4. **Builds a single product dict** per handle with fields: `handle`, `title`, `body`, `tags`, `image_url`, `size`, `sizes`, `price`, `style`, `color`, `searchable_text`.

The result is 68 clean product records, each with a unified `searchable_text` blob for ranking.

---

## Part 1 — Structured Text Search

### Query Parser (`src/query_parser.py`)

A hybrid regex + rule-based parser extracts structured intent from free-text queries. No NLP library is required.

**Dimensions** are extracted using a set of regex patterns:
```python
r"(\d+\.?\d*)\s*[xX×]\s*(\d+\.?\d*)"  # classic: "8x10", "5 X 8"

r"(\d+\.?\d*)\s+by\s+(\d+\.?\d*)"  # natural: "8 by 10"
r"(\d+\.?\d*)\s*ft?\s*[xX×]\s*..."  # with units
```
Dimension values ≥ 30 are rejected (likely inches, not feet). Width and length are always sorted so `"10x8"` is treated identically to `"8x10"`.

**Shape** is determined by keyword presence: `runner`, `round`/`circular`, `square` → rectangle default.

**Color** is matched against a static list of ~27 rug colors (beige, blue, grey, ivory, navy, etc.). Synonyms like `gray→grey`, `boho→bohemian`, `off-white→ivory` are normalized. `neutral` supersedes all specific colors.

**Style** is matched against 19 style keywords: traditional, modern, contemporary, Persian, bohemian, geometric, abstract, transitional, etc.

**Size class** (`small`, `medium`, `large`, `xlarge`) is mapped to area ranges and used when no exact dimensions are given.

**Example outputs:**
```
"8x10 beige traditional rug"  →  {dimensions: {8,10}, color: [beige], style: [traditional]}
"runner 2x10 blue rug"        →  {shape: runner, dimensions: {2,10}, color: [blue]}
"large neutral rug"           →  {size_class: large, color: [neutral]}
```

### Filtering (`src/filter_catalog.py`)

Soft filters preserve recall while prioritizing precision:

- **Size filter**: checks all size variants using `±1.5 ft` tolerance on each dimension. Falls back to full catalog if no products match.
- **Color filter**: substring match against `searchable_text` (title + tags + body).
- **Style filter**: substring match with synonym expansion (`boho → bohemian`, `modern → contemporary`).

Each filter produces a `metadata_score` (0.0–1.0) using weighted components: size (40%), color (30%), style (30%).

### Ranking (`src/ranker.py`)

Uses `SentenceTransformers` (`all-MiniLM-L6-v2`) to embed both the raw query and each product's text blob (title + tags + first 300 chars of description). Final score:

```
final_score = 0.5 × metadata_score + 0.5 × semantic_cosine_similarity
```

Products below a minimum threshold (0.15) are filtered unless fewer than 5 results remain. Results are returned in descending score order, with price as a tiebreaker.

**Weight justification**: Equal weighting balances structured intent precision (size/color/style) with open-ended semantic understanding. This prevents a query like "blue transitional" from only returning size-matched products and vice versa.

---

## Part 2 — Multimodal Image-Based Recommendation

### CLIP Embedding (`src/embedder.py`)

Uses `openai/clip-vit-base-patch32` via HuggingFace Transformers. CLIP is the right choice here because it maps both images and text into the **same 512-dimensional vector space**, enabling direct cross-modal cosine similarity without a separate visual encoder.

```python
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
```

- `embed_image(source)` — accepts file paths, URLs, or `PIL.Image` objects. Downloads remote images with retry, composites RGBA transparencies onto white, resizes to max 1024px, and returns an L2-normalized 512-d vector.
- `embed_text(text)` — truncates to 60 words (CLIP's 77-token limit), returns L2-normalized 512-d vector.
- All vectors are L2-normalized so inner product equals cosine similarity.

### FAISS Index (`src/indexer.py`)

For each product, the catalog embeds:
1. **Image embedding** from `image_url` (product photo)
2. **Text embedding** from `title + tags`

Both are stored in separate `faiss.IndexFlatIP` (inner product) indexes — one for images, one for text. A fallback `NumpyInnerProductIndex` is used when FAISS is unavailable (e.g., on Windows without a Conda environment).

Indexes are persisted to `index/` and reloaded if the product count hasn't changed, making repeated queries fast.

### Fusion Scoring (`src/fusion.py`)

Three query modes depending on what inputs are provided:

| Mode | Inputs | Score Formula |
|------|--------|---------------|
| A | Room image only | `1.0 × image_similarity` |
| B | Image + text query | `0.6 × image_sim + 0.4 × text_sim` |
| C | Text query only | `1.0 × text_similarity` |

**Weight justification (Mode B)**: The room photograph carries the dominant signal — it captures color palette, room style (modern/traditional/rustic), spatial scale, and lighting context simultaneously. The text query refines intent (specific size or style preference) but is secondary. Hence image gets 0.6 and text gets 0.4.

### Room Context Classification (`src/search_part2.py`)

When both image and text are provided, the room image embedding is compared against CLIP text embeddings of curated label sets:

- **Room type**: living room, bedroom, dining room, hallway, office, kids room
- **Color palette**: warm tones, cool tones, neutral tones, dark tones, bright colors
- **Room style**: modern minimalist, traditional classic, bohemian eclectic, rustic farmhouse, coastal

The top-matching label per category is appended to the text query if it's not already present and its confidence ≥ 0.2. This query augmentation bridges the gap between the image's visual signal and the text index.

---

## Architecture

```
[Text Query] ──► Query Parser ──► Metadata Filter ──►
                                                       ├──► Fusion Score ──► Ranked Results
[Room Image] ──► CLIP Encoder ──► FAISS Search ─────►
[Optional Text]► CLIP Text Enc ──► Cosine Sim ──────►
```

See `architecture.png` for the full diagram.

---

## Evaluation & Benchmarks

The system was evaluated using an end-to-end benchmark suite ([`scripts/evaluate_metrics.py`](../scripts/evaluate_metrics.py)) across 48 automated pipeline tests, multi-intent search queries, and modality ablation settings:

### Information Retrieval (IR) Benchmarks
- **Precision@1**: `1.000` (100%)
- **Precision@5**: `1.000` (100%)
- **Precision@10**: `0.980` (98.0%)
- **MRR@10**: `1.000` (First relevant candidate placed at top rank)
- **mAP@10**: `0.979`
- **NDCG@10**: `0.664`

### Modality Ablation Study
| Configuration | NDCG@10 | MRR@10 | Precision@5 | mAP@10 | Lift (NDCG@10) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Metadata Filter Only** | `0.616` | `1.000` | `1.000` | `0.973` | Baseline |
| **Dense Text Only (`all-MiniLM-L6-v2`)** | `0.635` | `1.000` | `1.000` | `1.000` | +3.08% |
| **Visual CLIP Only (`ViT-B/32`)** | `0.587` | `1.000` | `0.907` | `0.871` | -4.71% |
| **Hybrid Late Fusion (Proposed)** | **`0.664`** | **`1.000`** | **`1.000`** | **`0.979`** | **+13.12% vs Visual** |

### Intent Extraction Precision
- **Dimensions**: `100.0%`
- **Shapes**: `100.0%`
- **Colors**: `100.0%`
- **Styles**: `93.3%`
- **Macro F1**: `98.3%`

### Latency Profile
- **Intent Parsing**: `0.43 ms` (P95 `0.51 ms`)
- **Part 1 Search (E2E)**: `21.47 ms` (46.6 QPS)
- **FAISS Vector Index Retrieval**: `0.22 ms` (4,602+ QPS)

---

## Limitations & Future Work

1. **Image embedding quality**: Product images in the catalog are sourced from the **Amazon Berkeley Objects (ABO)** dataset hosted on public Amazon S3 (`https://amazon-berkeley-objects.s3.amazonaws.com/images/small/...`), enabling 100% real image downloads (68 real images, 0 text-fallback, 0 missing URLs). Visual-similarity scores are calculated directly from pixel-level CLIP features across all products. A domain fine-tuned CLIP model on room-rug pairs would further improve recommendation accuracy.
2. **Size normalization**: Fractional sizes like `"7'9" × 10'2"` are not fully handled; a dedicated parser would improve coverage.
3. **Explainability**: The current LLM-based explainer falls back to a template sentence. Integration with a small generative model (e.g., Gemma-2B) would produce more specific explanations.
4. **Vector Database Scaling**: Current indexes use flat FAISS inner-product arrays with serialized `.npy` storage. For massive catalogs (>100k items), migrating to a distributed vector database (e.g., Pinecone, Qdrant, or Milvus) with HNSW indexing will enable sub-millisecond retrieval at scale.

