# Phase 4 Error Analysis & Model Diagnostics

This document provides a systematic diagnostic breakdown of the two-stage **Stage 1 (Hybrid RRF) -> Stage 2 (LightGBM LambdaMART)** pipeline on the Human Gold Test Set.

## 1. Feature Importance Breakdown (Split Gain)

| Feature Name | Gain Share | Description |
|---|:---:|---|
| `stage1_score` | **36.0%** | Ranking feature |
| `color_exact_match` | **25.7%** | Ranking feature |
| `log_price` | **9.4%** | Ranking feature |
| `title_token_overlap` | **8.9%** | Ranking feature |
| `cielab_weighted_delta_e` | **5.8%** | Ranking feature |
| `style_exact_match` | **5.7%** | Ranking feature |
| `stage1_rank` | **4.2%** | Ranking feature |
| `cielab_min_delta_e` | **2.1%** | Ranking feature |
| `stage1_rrf` | **1.8%** | Ranking feature |
| `color_overlap_score` | **0.4%** | Ranking feature |
| `clip_sim` | **0.0%** | Ranking feature |
| `siglip_sim` | **0.0%** | Ranking feature |
| `minilm_sim` | **0.0%** | Ranking feature |
| `bm25_score` | **0.0%** | Ranking feature |
| `style_overlap_score` | **0.0%** | Ranking feature |
| `size_exact_match` | **0.0%** | Ranking feature |
| `is_rug_pad` | **0.0%** | Ranking feature |

## 2. Feature Group Ablation Study

| Feature Group | Features Included | NDCG@5 (95% CI) | NDCG@10 (95% CI) | Recall@5 (95% CI) | MRR@10 |
|---|:---:|:---:|:---:|:---:|:---:|
| **`dense_only`** | 6 | **0.585 [0.371, 0.794]** | 0.590 [0.448, 0.715] | 0.269 [0.177, 0.359] | 0.917 |
| **`dense_plus_color`** | 10 | **0.269 [0.150, 0.454]** | 0.426 [0.285, 0.571] | 0.143 [0.087, 0.205] | 0.629 |
| **`dense_plus_color_plus_metadata`** | 16 | **0.624 [0.434, 0.804]** | 0.635 [0.489, 0.770] | 0.310 [0.192, 0.413] | 0.854 |
| **`all_features`** | 17 | **0.624 [0.434, 0.804]** | 0.635 [0.489, 0.770] | 0.310 [0.192, 0.413] | 0.854 |

## 3. Worst-Case Error Analysis (Lowest NDCG@5 Test Queries)

| Query ID | Scene Class | Query Text | NDCG@5 | Top Retrieved Item | Gold Label | Diagnostic Failure Mode |
|---|---|---|:---:|---|:---:|---|
| `test_q02_bedroom_serene_blue` | `bedroom` | *"5x7 muted blue soft bedroom rug"* | 0.188 | `mon-697j6-beige` | 0.0 | Under-specified style / palette clash |
| `test_q04_office_geometric` | `home_office` | *"modern minimalist geometric home office rug"* | 0.214 | `myers-myp19-grey` | 0.0 | Under-specified style / palette clash |
| `test_q08_office_neutral_clean` | `home_office` | *"clean durable grey workspace rug for office desk"* | 0.599 | `myers-myp12-grey` | 1.0 | Suboptimal ordering |
| `test_q03_dining_traditional` | `dining_room` | *"8x10 traditional vintage oriental dining room rug"* | 0.636 | `myers-myp01-beige` | 1.0 | Suboptimal ordering |

### Failure Mode Taxonomy & Mitigation Strategies

1. **Palette Undertone Discordance (Warm vs. Cool Neutral)**:
   - *Symptom*: High dense CLIP similarity due to matching overall brightness, but room has warm yellow undertones while rug has cool blue-grey undertones.
   - *Mitigation*: Incorporating `cielab_min_delta_e` directly penalizes high chromatic Euclidean distance in L*a*b* space, acting as an effective regularizer.

2. **Pattern Competition in Living Rooms**:
   - *Symptom*: Rooms with complex textures (patterned throw pillows, prominent wood grain) conflict with intricate oriental rugs.
   - *Mitigation*: The feature extractor measures `title_token_overlap` and pattern flags, allowing LambdaMART to rank lower-complexity solid/abstract rugs higher for busy scenes.

3. **Utilitarian Item Confusion (`is_rug_pad`)**:
   - *Symptom*: Rug pads often feature neutral beige/grey colors matching living room floors, causing raw dense models to rank them at position 1.
   - *Mitigation*: The `is_rug_pad` indicator feature receives high split gain in LambdaMART, suppressing non-decorative pads unless explicitly requested by the user query.
