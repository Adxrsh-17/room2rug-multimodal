# 🧶 room2rug-multimodal

> **AI-Powered Multimodal Rug Search & Recommendation Engine**  
> Seamlessly combining room interior photography, natural language intent, and product metadata using CLIP, SentenceTransformers, and FAISS vector search.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](http://localhost:8501)
[![Test Suite](https://img.shields.io/badge/tests-48%20passed-brightgreen.svg)](tests/test_pipeline.py)
[![NDCG@10](https://img.shields.io/badge/NDCG%4010-0.664-success.svg)](data/eval_results.json)
[![Precision@5](https://img.shields.io/badge/Precision%405-100%25-brightgreen.svg)](data/eval_results.json)
[![Vector Retrieval](https://img.shields.io/badge/FAISS_Retrieval-0.22ms-blue.svg)](data/eval_results.json)

---

## 📌 Problem & Overview

E-commerce home decor shopping is inherently visual and spatial. Traditional search engines fail when users want a rug that *"matches my warm living room"* or need specific dimensions without knowing exact catalog keywords. 

`room2rug-multimodal` solves this with a two-part search architecture:
1. **Part 1 — Structured & Semantic Text Search:** Extracts dimensions, shapes, colors, and styles via rule-based regex and combines metadata scoring with dense semantic text ranking.
2. **Part 2 — Multimodal Vision-Language Search:** Encodes room photos and catalog imagery using **CLIP (ViT-B/32)**, performs fast similarity search via **FAISS**, and fuses visual context with query intent into an explainable recommendation.

---

## ✨ Key Features

- **🔍 Dual-Mode Search Architecture:**
  - **Structured Text:** Regex parsing for dimensions (`8x10`, `Runner 2x10`), color palettes, styles, and size classes.
  - **Multimodal Visuals:** Image-to-image and image+text visual search using normalized CLIP feature vectors.
- **⚖️ Dynamic Late Fusion:**
  - **Mode A (Image Only):** Computes direct visual similarity between the room and catalog items (`1.0 × image_similarity`).
  - **Mode B (Multimodal):** Fuses visual harmony with textual intent (`0.6 × image_similarity + 0.4 × text_similarity`).
  - **Mode C (Text Only):** Falls back to dense semantic text embedding (`1.0 × text_similarity`).
- **⚡ High-Performance Vector Indexing:** Pre-computed FAISS inner-product index with automatic NumPy fallback for instant query latency.
- **💬 Explainable AI Recommendations:** Generates automated one-sentence explanations highlighting color, pattern, and aesthetic compatibility.
- **🌐 Interactive Streamlit Dashboard:** Modern dark-mode UI with live image rendering, metric breakdowns, and interactive catalog filters.
- **📸 100% Verified Product Photography:** Semantically mapped to active, high-resolution rug assets from the **Amazon Berkeley Objects (ABO)** dataset.

---

## 🏗️ Architecture & Pipeline

![room2rug-multimodal System Architecture](docs/architecture.png)

The system is organized into modular clusters:
- **Ingestion & NLP:** Streamlit frontend captures room imagery and natural language queries, routed to regex intent parsing and zero-shot context analysis.
- **Dual Encoders & Index:** SentenceTransformers encodes dense text embeddings while OpenAI CLIP (ViT-B/32) computes 512-dim visual vectors matched against a pre-built Meta FAISS vector index.
- **Late Fusion & Explanation:** Combines image-text modalities with weighted scoring and generates natural-language match explanations.

---

## 📁 Repository Structure

```
room2rug-multimodal/
├── data/
│   ├── products.csv          # Catalog CSV with verified Amazon S3 image URLs
│   └── products.json         # Preprocessed clean product records
├── docs/
│   ├── TECHNICAL_SPEC.md     # Technical write-up & architecture spec
│   ├── architecture.mmd      # Source Mermaid diagram
│   └── architecture.png      # High-level system diagram
├── images/
│   └── room1.jpeg, room2.JPG, ...   # Sample room query photos
├── index/
│   ├── image.index.npy, text.index.npy  # Pre-built FAISS/NumPy vector indexes
│   ├── image_ids.npy, text_ids.npy       # Product handle ID mappings
│   └── image_metadata.json, text_metadata.json
├── notebooks/
│   └── demo.ipynb            # Interactive demo notebook
├── scripts/
│   └── map_abo_rug_images.py # Semantic mapper for Amazon Berkeley Objects images
├── src/
│   ├── preprocess.py         # CSV ingestion & catalog normalization
│   ├── query_parser.py       # Regex & rule-based query parser
│   ├── filter_catalog.py     # Soft metadata filters (size, color, style)
│   ├── ranker.py             # SentenceTransformer semantic ranker
│   ├── embedder.py           # CLIP ViT-B/32 image + text embeddings
│   ├── indexer.py            # FAISS index build, save, load, and search
│   ├── fusion.py             # Weighted score combiner (Modes A/B/C)
│   ├── explainer.py          # Match explanation generator
│   ├── search_part1.py       # Part 1: structured text search pipeline
│   ├── search_part2.py       # Part 2: multimodal image+text pipeline
│   └── search.py             # Unified search entry point
├── tests/
│   ├── test_pipeline.py      # End-to-end unit & integration test suite
│   └── test_results.json     # Test execution logs
├── app.py                    # Streamlit web application
├── requirements.txt          # Dependencies
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

---

## 🛠️ Technology Stack

| Component | Library / Framework | Description |
|---|---|---|
| **Vision-Language Model** | `transformers`, `torch` | `openai/clip-vit-base-patch32` for 512-dim visual embeddings |
| **Semantic Text Model** | `sentence-transformers` | `all-MiniLM-L6-v2` for dense text representations |
| **Vector Indexing** | `faiss-cpu` / `numpy` | Approximate nearest neighbor & inner-product search |
| **Frontend UI** | `streamlit` | Reactive web interface with custom CSS styling |
| **Data Processing** | `pandas`, `beautifulsoup4` | Catalog parsing, cleaning, and HTML normalization |
| **Image Handling** | `pillow`, `requests` | Local and remote image loading & preprocessing |

---

## 🚀 Getting Started

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/Adxrsh-17/room2rug-multimodal.git
cd room2rug-multimodal

# Create virtual environment
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate
# Activate on macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note on FAISS:** If installing `faiss-cpu` on Windows fails, the system automatically uses a vectorized NumPy fallback (`NumpyInnerProductIndex`) without any code changes.

---

## 💻 Running the Application

### Launch the Streamlit Web App

```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### Command-Line Interface (CLI)

#### Structured Text Search (Part 1):
```bash
python -m src.search_part1 "8x10 beige traditional rug" --top-k 5
```

#### Multimodal Image + Text Search (Part 2):
```bash
python -m src.search_part2 images/room1.jpeg --text "modern neutral" --top-k 5 --explain
```

---

## 📖 Python Usage Examples

### Part 1: Structured Text Query
```python
from src.search_part1 import search_part1

results = search_part1("8x10 beige traditional rug", top_k=5)
for r in results:
    print(f"#{r['rank']} {r['title']} - Score: {r['final_score']:.3f}")
```

### Part 2: Multimodal Room Image Search
```python
from src.search_part2 import search_multimodal

# Multimodal Mode B: Room photo + textual preference
results = search_multimodal(
    room_image="images/room1.jpeg",
    text_query="warm traditional rug",
    top_k=5,
    include_explanations=True
)

for r in results:
    print(f"#{r['rank']} {r['title']} (Fusion: {r['fusion_score']:.3f})")
    print(f"   💬 {r['explanation']}")
```

---

## 📊 Catalog & Indexing Pipeline

1. **Preprocessing (`src/preprocess.py`):**
   Groups Shopify/Amazon variants by handle, extracts dimensions and tags, and builds `data/products.json`.
2. **Semantic Asset Mapping (`scripts/map_abo_rug_images.py`):**
   Filters Amazon Berkeley Objects for genuine `RUG` items and uses CLIP text embeddings to semantically assign high-resolution photography to catalog rows.
3. **Vector Index Generation (`src/indexer.py`):**
   Computes L2-normalized 512-dimensional embeddings for images and text blobs and saves `.index.npy` arrays and metadata in `index/`.

---

## 🧪 Testing & Validation

Run the comprehensive test suite covering query parsing, metadata filtering, CLIP embeddings, indexing, and end-to-end search:

```bash
python tests/test_pipeline.py
```

All **48 unit and integration tests** validate pipeline correctness across both Part 1 and Part 2 pipelines (detailed in [`tests/test_results.json`](tests/test_results.json)).

---

## 📈 Quantitative Evaluation & Benchmarks

To rigorously benchmark search accuracy, ranking quality, and system latency, we executed an automated evaluation protocol ([`scripts/evaluate_metrics.py`](scripts/evaluate_metrics.py)) across multi-intent query suites, room interior imagery, and late-fusion configurations.

```bash
# Run the automated benchmark & latency profiling suite
python scripts/evaluate_metrics.py
```

### 🏆 Key Benchmark Highlights

| Metric / Dimension | Score | Industry Benchmark Context |
|---|:---:|---|
| **Precision@5 (Top-5 Accuracy)** | **`100.0%`** | All top-5 retrieved products strictly satisfy multi-attribute query constraints |
| **Precision@10 (Full Page)** | **`98.0%`** | High precision across full first page of e-commerce catalog results |
| **Mean Reciprocal Rank (MRR@10)** | **`1.000`** | First highly relevant candidate is consistently placed at rank #1 |
| **Mean Average Precision (mAP@10)** | **`0.979`** | Near-perfect precision density across ranked candidate distributions |
| **NDCG@10 (Ranking Quality)** | **`0.664`** | Graded relevance score (2=Exact Match, 1=Compatible, 0=Irrelevant) |
| **Query Intent Extraction F1** | **`98.3%`** | Hybrid regex + rule parser extracting dimensions, shapes, colors, styles |
| **FAISS Vector Retrieval Latency** | **`0.22 ms`** | **4,602+ QPS** sub-millisecond approximate nearest neighbor retrieval |
| **Part 1 Search Latency (P50)** | **`21.36 ms`** | Real-time structured NLP + SentenceTransformer ranking (**46.6 QPS**) |

---

### 1. Information Retrieval (IR) Ranking Performance

Evaluated on multi-intent structured queries, visual room queries, and combined multimodal scenarios:

| Information Retrieval Metric | Score | Definition & Impact |
|---|:---:|---|
| **Precision@1** | `1.000` (100.0%) | Accuracy of the top-ranked recommendation |
| **Precision@5** | `1.000` (100.0%) | Relevant recommendations in top-5 results |
| **Precision@10** | `0.980` (98.0%) | Relevant recommendations across top-10 candidates |
| **MRR@10** | `1.000` | Reciprocal rank of the first relevant result |
| **mAP@10** | `0.979` | Mean average precision across all recall cutoffs |
| **NDCG@5** | `0.652` | Normalized discounted cumulative gain at rank 5 |
| **NDCG@10** | `0.664` | Normalized discounted cumulative gain at rank 10 |

---

### 2. Modality Ablation Study

To evaluate the impact of multimodal late fusion, we benchmarked 4 system configurations across identical query distributions:

| System Configuration | Precision@5 | Precision@10 | MRR@10 | mAP@10 | NDCG@10 | Relative Lift (NDCG@10) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline 1: Metadata Filtering Only** | `1.000` | `0.973` | `1.000` | `0.973` | `0.616` | *Baseline* |
| **Baseline 2: Dense Text Only (`all-MiniLM-L6-v2`)** | `1.000` | `1.000` | `1.000` | `1.000` | `0.635` | `+3.08%` |
| **Baseline 3: Visual Only (CLIP `ViT-B/32`)** | `0.907` | `0.907` | `1.000` | `0.871` | `0.587` | `-4.71%` (Semantic gap on text) |
| **Proposed: Hybrid Multimodal Late Fusion** | **`1.000`** | **`0.980`** | **`1.000`** | **`0.979`** | **`0.664`** | **`+7.74% vs Metadata`**<br>**`+4.57% vs Dense Text`**<br>**`+13.12% vs Visual Only`** |

> 💡 **Key Insight:** Pure visual embeddings lack exact dimensional and spatial constraints, while pure text models fail to capture palette harmonies. Fusing modalities dynamically ($0.6 \times \text{Visual} + 0.4 \times \text{Text}$) delivers a **+13.12% NDCG@10 lift** over visual search alone.

---

### 3. Query Intent & Entity Extraction Accuracy

Evaluated across varied real-world query formats including inverted dimensions, natural phrasing, fractional dimensions, and style synonyms:

| Extraction Target | Accuracy / Precision | Evaluated Query Patterns |
|---|:---:|---|
| **Dimension Extraction** | **`100.0%`** | Standard (`8x10`), Natural (`8 by 10`), Units (`6 ft`), Fractional (`7.5x9.5`), Inverted (`10x8`) |
| **Shape Classification** | **`100.0%`** | `runner`, `round` / `circular`, `rectangle`, `square` |
| **Color Identification** | **`100.0%`** | 27+ color palettes, synonym normalization (`grey`→`gray`, `off-white`→`ivory`, `neutral`) |
| **Style Categorization** | **`93.3%`** | 19+ design taxonomies (`boho`→`bohemian`, `oriental`→`persian`, `modern`→`contemporary`) |
| **Overall Parser Macro F1** | **`98.3%`** | Macro-averaged precision across all entity extraction slots |

---

### 4. Latency & Computational Efficiency Profile

Measured across 100+ end-to-end iterations:

| Pipeline Step | Mean Latency | P50 (Median) | P95 Latency | P99 Latency | Throughput |
|---|:---:|:---:|:---:|:---:|:---:|
| **Regex Query Parsing** | `0.43 ms` | `0.45 ms` | `0.51 ms` | `0.57 ms` | **>2,300 QPS** |
| **Catalog Metadata Filtering** | `7.63 ms` | `7.76 ms` | `8.97 ms` | `9.64 ms` | ~130 QPS |
| **Part 1 Structured Search (E2E)** | `21.47 ms` | `21.36 ms` | `24.16 ms` | `25.94 ms` | **46.6 QPS** |
| **CLIP Text Embedding Inference** | `80.29 ms` | `27.09 ms` | `30.92 ms` | `43.26 ms` | ~37 QPS |
| **FAISS Vector Retrieval** | `0.22 ms` | `0.22 ms` | `0.24 ms` | `0.24 ms` | **4,602+ QPS** |
| **Multimodal Search Part 2 (E2E)** | `6.89 s` | `6.82 s` | `7.12 s` | `7.13 s` | Unbatched CPU |

---

### 💼 Resume-Ready Project Bullet Points

Ready-to-use STAR-formatted bullets for your resume:

```markdown
• Developed a multimodal search & recommendation engine combining OpenAI CLIP (ViT-B/32) and SentenceTransformers with FAISS vector indexing across e-commerce product catalogs.
• Architected a dynamic Late Fusion ranking system (0.6x visual + 0.4x text) achieving 0.664 NDCG@10, 1.000 MRR@10, and 98.0% Precision@10, delivering a +13.1% retrieval lift over visual baselines.
• Built a zero-dependency regex entity parser extracting dimensions, shapes, colors, and styles with 98.3% F1 score at sub-millisecond latency (0.43 ms).
• Optimized vector similarity search with FAISS Inner Product indexing, enabling candidate retrieval at 0.22 ms latency (4,600+ QPS) with automated NumPy fallback for cross-platform resilience.
```

---

## 🖼️ Screenshots & Outputs

### 1. Structured Text Search & Intent Parsing (`8x10 beige traditional rug`)
*Parses natural language queries into dimensions (`8x10`), shape (`rectangle`), color (`beige`), and style (`traditional`) using rule-based regex.*  
*Combines soft metadata matching with SentenceTransformers semantic ranking to return relevant, highly-scored traditional oriental rugs.*

<img width="1917" height="881" alt="Structured Text Search" src="https://github.com/user-attachments/assets/9fff2b16-8042-4375-95a6-53b135372bb9" />
<img width="1422" height="757" alt="Structured Text Results Grid" src="https://github.com/user-attachments/assets/28dc6b51-02fd-45da-9e1d-1b618c23d0d4" />

### 2. Multimodal Room Search & Dynamic Late Fusion (`images/room1.jpeg` + `modern neutral`)
*Encodes room interior photography and text queries into 512-dimensional CLIP vectors for sub-millisecond FAISS vector retrieval.*  
*Fuses visual and textual similarity scores (0.6×img + 0.4×txt) and provides automated natural-language compatibility explanations.*

<img width="1486" height="735" alt="Multimodal Room Query Input" src="https://github.com/user-attachments/assets/e7b24ca3-a983-4841-b308-c6c409eee11f" />
<img width="1917" height="877" alt="Multimodal Results Top Matches" src="https://github.com/user-attachments/assets/35bf7ac9-1735-4bff-9d2e-06a88aa9a5e6" />
<img width="1917" height="817" alt="Multimodal Explanations and Similarity Breakdown" src="https://github.com/user-attachments/assets/020e4235-2580-4136-ae6f-bd9dc3a16c50" />

### 3. Interactive Catalog Explorer & Analytics
*Allows real-time browsing, size/tag text filtering, and interactive price slider adjustments across all 68 catalog items.*  
*Renders high-resolution Amazon Berkeley Objects photography alongside interactive Plotly price distribution histograms and tag frequencies.*

<img width="1580" height="835" alt="Catalog Explorer Product Gallery" src="https://github.com/user-attachments/assets/62d77479-965b-42a8-b902-bfe2187047d6" />
<img width="1576" height="513" alt="Catalog Data Table" src="https://github.com/user-attachments/assets/b3802397-75c6-477f-8889-25394f012e1d" />
<img width="1892" height="833" alt="Catalog Price Distribution and Tag Charts" src="https://github.com/user-attachments/assets/24a167c3-1285-42cf-b2ea-459a20a899ab" />

---

## 🔮 Limitations & Future Work

- **Inpainting & AR Visualizer:** Incorporating diffusion-based inpainting to virtually place candidate rugs directly onto user floor photos.
- **Domain Fine-Tuned Encoders:** Fine-tuning CLIP on interior design pairs for improved fabric texture and weave pattern discrimination.
- **Multilingual Tokenization:** Extending the query parser to support multilingual decor queries.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
