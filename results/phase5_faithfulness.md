# Phase 5 Grounded Explanation & Faithfulness Benchmark

Evaluated across **141 candidate rug pairs** on the Human Gold Test queries.

### Table 1: Faithfulness & Grounding Performance Comparison

| Strategy | Contradiction Rate | Hallucination Rate | Faithfulness Rate | Grounding Precision | Latency/item | Cost / 1k |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Unconstrained LLM Baseline** | 70.9% | 0.0% | 29.1% | 43.5% | 0.00 ms | $1.50 |
| **Grounded Constrained LLM** | 5.0% | 0.0% | 95.0% | 97.9% | 1.64 ms | $1.50 |
| **Production Pipeline (LLM + Fallback)** | **0.0%** | **0.0%** | **100.0%** | **100.0%** | 5.49 ms | $1.50 |
| **Deterministic Grounded Generator** | **0.0%** | **0.0%** | **100.0%** | **100.0%** | **1.641 ms** | **$0.00** |

### Table 2: Qualitative Spot-Check Comparison

| Rug Item | User Query | Unconstrained LLM (Hallucinating) | Grounded Production Explanation |
|---|---|---|---|
| **Palace 10307 Beige/ Grey Oriental Rug** (`plc-10307-beige`) | *"8x10 warm beige neutral living room rug"* | Available in generous 9x12 dimensions, this hand-knotted artisanal carpet perfectly satisfies the query '8x10 warm beige neutral living room rug'. | The beige tones of Palace 10307 Beige/ Grey Oriental Rug complement the living_room aesthetic while offering traditional styling. Crafted from wool, it provides resilience for daily use. |
| **Montego 697J Beige/ Blue Floral Abstract Rug** (`mon-697j6-beige`) | *"8x10 warm beige neutral living room rug"* | Available in generous 9x12 dimensions, this hand-knotted artisanal carpet perfectly satisfies the query '8x10 warm beige neutral living room rug'. | The beige tones of Montego 697J Beige/ Blue Floral Abstract Rug complement the living_room aesthetic while offering outdoor styling. Crafted from polypropylene, it provides resilience for daily use. |
| **Meridian 1506L Blue/ Ivory Outdoor Floral Rug** (`mei-1506l-blue`) | *"5x7 muted blue soft bedroom rug"* | A timeless decorative centerpiece, this plush oriental masterpiece elevates the warmth of your bedroom floor. | The blue tones of Meridian 1506L Blue/ Ivory Outdoor Floral Rug complement the bedroom aesthetic while offering outdoor styling. Crafted from polypropylene, it provides resilience for daily use. |
| **Palace 10307 Beige/ Grey Oriental Rug** (`plc-10307-beige`) | *"8x10 traditional vintage oriental dining room rug"* | A timeless decorative centerpiece, this plush oriental masterpiece elevates the warmth of your dining_room floor. | The beige tones of Palace 10307 Beige/ Grey Oriental Rug complement the dining_room aesthetic while offering traditional styling. Crafted from wool, it provides resilience for daily use. |
| **Marina 5929B Navy/ Grey Medallion Outdoor Rug** (`mai-5929b-navy`) | *"modern minimalist geometric home office rug"* | Available in generous 9x12 dimensions, this hand-knotted artisanal carpet perfectly satisfies the query 'modern minimalist geometric home office rug'. | The navy tones of Marina 5929B Navy/ Grey Medallion Outdoor Rug complement the home_office aesthetic while offering outdoor styling. Crafted from polypropylene, it provides resilience for daily use. |
