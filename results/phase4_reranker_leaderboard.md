# Phase 4 End-to-End Reranker Leaderboard & Benchmark

All metrics evaluated on the **Human Gold Test Set** with **1,000-sample Bootstrap 95% Confidence Intervals**.

### Table 1: End-to-End Pipeline Comparison
| System | Pipeline Stage | NDCG@5 (95% CI) | NDCG@10 (95% CI) | Recall@5 (95% CI) | MRR@10 | MAP@10 | p-val vs v1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **attribute_filter** | Baseline | **0.651** [0.447, 0.856] | 0.659 [0.492, 0.819] | 0.299 [0.266, 0.330] | 0.760 | 0.649 | 0.4625 |
| **bm25** | Baseline | **0.471** [0.286, 0.657] | 0.543 [0.381, 0.723] | 0.225 [0.144, 0.290] | 0.653 | 0.549 | 0.0285 |
| **clip_zero_shot** | Baseline | **0.654** [0.493, 0.824] | 0.650 [0.546, 0.744] | 0.333 [0.276, 0.391] | 0.938 | 0.664 | 0.2545 |
| **baseline_v1** | Baseline | **0.753** [0.594, 0.899] | 0.750 [0.655, 0.841] | 0.353 [0.295, 0.415] | 1.000 | 0.786 | - |
| **stage1_hybrid_rrf** | Stage 1 Hybrid | **0.612** [0.459, 0.778] | 0.590 [0.465, 0.702] | 0.283 [0.225, 0.349] | 0.917 | 0.516 | 0.2545 |
| **full_pipeline_lambdamart** | Stage 2 Reranker | **0.624** [0.434, 0.804] | 0.635 [0.489, 0.770] | 0.310 [0.192, 0.413] | 0.854 | 0.577 | 0.2160 |
