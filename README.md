# 🧶 Multimodal Rug Search & Recommendation System

A two-part AI-powered search engine for rug e-commerce, combining structured text queries, semantic ranking (SentenceTransformers), and visual similarity (CLIP + FAISS).

---

## 🏗️ Architecture

```
[Text Query] ──► Query Parser ──► Metadata Filter ──►
                                                       ├──► Fusion Score ──► Ranked Results
[Room Image] ──► CLIP Encoder ──► FAISS Search ─────►
[Optional Text]► CLIP Text Enc ──► Cosine Sim ──────►
```

See [`docs/architecture.png`](docs/architecture.png) and the [full technical spec](docs/TECHNICAL_SPEC.md) for details.

---

## 📁 Project Structure

```
MultiModalSearchAssignment/
├── data/
│   └── products.csv          # Shopify export (68 unique products)
├── docs/
│   ├── TECHNICAL_SPEC.md     # Technical write-up
│   ├── architecture.mmd      # Source for architecture diagram
│   └── architecture.png      # System diagram
├── images/
│   └── room1.jpg, room2.JPG, ...   # Room photos for multimodal demo
├── index/
│   └── image.index.npy, text.index.npy, ...  # Pre-built FAISS indexes
├── notebooks/
│   └── demo.ipynb            # Google Colab-friendly demo
├── src/
│   ├── preprocess.py         # CSV → clean product dict (one per handle)
│   ├── query_parser.py       # Regex + rule-based query parser
│   ├── filter_catalog.py     # Soft metadata filters (size, color, style)
│   ├── ranker.py             # SentenceTransformer semantic ranker
│   ├── embedder.py           # CLIP image + text embeddings
│   ├── indexer.py            # FAISS index build, save, load, search
│   ├── fusion.py             # Weighted score combiner (Modes A/B/C)
│   ├── explainer.py          # One-sentence match explanation generator
│   ├── search_part1.py       # Part 1: structured text search pipeline
│   ├── search_part2.py       # Part 2: multimodal image+text pipeline
│   └── search.py             # Unified entry point
├── tests/
│   ├── test_pipeline.py      # Full end-to-end test suite (46 tests)
│   └── test_results.json     # Latest test run results (all PASS)
├── app.py                    # Streamlit frontend app
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows Note:** If `faiss-cpu` fails to install, use a Conda environment:
> ```bash
> conda install -c conda-forge faiss-cpu
> ```
> The system automatically falls back to a NumPy-based index if FAISS is unavailable.

### 3. Data

The Shopify CSV is already at `data/products.csv`. Place additional room photos in `images/`.

> **⚠️ NOTE ON PLACEHOLDER IMAGES:** The default `data/products.csv` contains Shopify image links that have been deleted from the origin server (404 Not Found). Because of this, the frontend UI will default to a placeholder (🧶). **For proper image loading, you must modify the `data/products.csv` file with your own active image links!**

**Using Your Own Product Images:**
If you want to use your own rug images in the UI, you can open `data/products.csv` and update the `Image Src` column with active, publicly accessible URLs to your custom images (e.g., from AWS S3, Imgur, or your own CDN).
Then, re-run the preprocessing script to rebuild the catalog `json`:

```bash
python src/preprocess.py data/products.csv --output data/products.json
```

---

## 🚀 Usage

### Part 1 — Structured Text Search

```python
from src.search_part1 import search_part1

results = search_part1("8x10 beige traditional rug", top_k=5)
for r in results:
    print(r["rank"], r["title"], r["final_score"])
```

Supported query patterns:
- Dimensions: `8x10`, `5 by 8`, `7.5x9.5`
- Colors: beige, blue, grey, ivory, red, navy, brown, cream, black, white, ...
- Styles: traditional, modern, persian, bohemian, geometric, transitional, ...
- Size classes: small, medium, large, xlarge, runner
- Shapes: round, square, runner

### Part 2 — Multimodal Image + Text Search

```python
from src.search_part2 import search_multimodal

# Image only (Mode A)
results = search_multimodal("images/room1.jpeg", top_k=5)

# Image + text (Mode B) — recommended
results = search_multimodal("images/room2.JPG", text_query="warm traditional 8x10", top_k=5)

# Text only (Mode C)
results = search_multimodal(None, text_query="navy geometric modern rug", top_k=5)

for r in results:
    print(r["rank"], r["title"], f"score={r['fusion_score']}")
```

### Command-line interface

```bash
# Part 1
python -m src.search_part1 "8x10 beige traditional rug" --top-k 5

# Part 2
python -m src.search_part2 images/room1.jpeg --text "modern neutral" --top-k 5 --explain
```

---

## 🧪 Running Tests

```bash
python tests/test_pipeline.py
```

All 46 tests pass covering: query parsing, catalog filtering, CLIP embeddings, FAISS indexing, fusion scoring, and end-to-end search (both parts).

---

## 🔬 How It Works

### Part 1 — Scoring Formula

```
final_score = 0.5 × metadata_score + 0.5 × semantic_cosine_similarity
```

- **metadata_score**: weighted sum of size match (40%) + color match (30%) + style match (30%)
- **semantic_cosine_similarity**: cosine similarity between `all-MiniLM-L6-v2` embeddings of query and product text

### Part 2 — Fusion Modes

| Mode | Triggered when | Formula |
|------|---------------|---------|
| A | Image only | `1.0 × image_sim` |
| B | Image + text | `0.6 × image_sim + 0.4 × text_sim` |
| C | Text only | `1.0 × text_sim` |

Image gets higher weight (0.6) because it encodes color palette, room style, and scale simultaneously, while text refines intent.

---

## 📚 Key Dependencies

| Library | Purpose |
|---------|---------|
| `sentence-transformers` | Part 1 semantic ranking (`all-MiniLM-L6-v2`) |
| `transformers` + `torch` | CLIP embeddings (`openai/clip-vit-base-patch32`) |
| `faiss-cpu` | Approximate nearest-neighbor index |
| `pandas` + `beautifulsoup4` | CSV processing & HTML stripping |
| `scikit-learn` | cosine similarity utilities |
| `pillow` + `requests` | Image loading (local + URL) |

---

## 📝 See Also

- [`docs/TECHNICAL_SPEC.md`](docs/TECHNICAL_SPEC.md) — Full technical write-up
- [`notebooks/demo.ipynb`](notebooks/demo.ipynb) — Interactive Colab demo
- [`tests/test_results.json`](tests/test_results.json) — Latest test run (all 46 PASS)
