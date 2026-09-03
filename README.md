# 🧶 room2rug-multimodal

> **AI-Powered Multimodal Rug Search & Recommendation Engine**  
> Seamlessly combining room interior photography, natural language intent, and product metadata using CLIP, SentenceTransformers, and FAISS vector search.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](http://localhost:8501)
[![Test Suite](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/test_pipeline.py)

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

## 🧪 Testing

Run the comprehensive test suite covering query parsing, metadata filtering, CLIP embeddings, indexing, and end-to-end search:

```bash
python tests/test_pipeline.py
```

All 46 unit and integration tests validate the entire pipeline with complete reporting.

---

## 🖼️ Screenshots & Outputs

### 1. Structured Text Search & Intent Parsing (`8x10 beige traditional rug`)
*Parses natural language queries into dimensions (`8x10`), shape (`rectangle`), color (`beige`), and style (`traditional`) using rule-based regex.*  
*Combines soft metadata matching with SentenceTransformers semantic ranking to return relevant, highly-scored traditional oriental rugs.*

![Structured Text Search](https://github.com/user-attachments/assets/9174995e-5138-4531-80a0-2f490c326353)
![Structured Text Grid](https://github.com/user-attachments/assets/570fd7c1-602c-4849-a73b-b1852e1c399a)

### 2. Multimodal Room Search & Dynamic Late Fusion (`images/room1.jpeg` + `modern neutral`)
*Encodes room interior photography and text queries into 512-dimensional CLIP vectors for sub-millisecond FAISS vector retrieval.*  
*Fuses visual and textual similarity scores (0.6×img + 0.4×txt) and provides automated natural-language compatibility explanations.*

![Multimodal Search](https://github.com/user-attachments/assets/6ea990b1-e8e9-4090-a57d-24c9d879a198)

### 3. Interactive Catalog Explorer & Analytics
*Allows real-time browsing, size/tag text filtering, and interactive price slider adjustments across all 68 catalog items.*  
*Renders high-resolution Amazon Berkeley Objects photography alongside interactive Plotly price distribution histograms.*

![Catalog Explorer](https://github.com/user-attachments/assets/7702461c-53af-4450-885d-be6131701282)

---

## 🔮 Limitations & Future Work

- **Inpainting & AR Visualizer:** Incorporating diffusion-based inpainting to virtually place candidate rugs directly onto user floor photos.
- **Domain Fine-Tuned Encoders:** Fine-tuning CLIP on interior design pairs for improved fabric texture and weave pattern discrimination.
- **Multilingual Tokenization:** Extending the query parser to support multilingual decor queries.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
