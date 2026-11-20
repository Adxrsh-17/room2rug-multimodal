# Baseline Leaderboard & Statistical Evaluation
**Evaluation Queries**: 8 | **Candidate Catalog Size**: 68
All metrics evaluated strictly on human gold labels with **95% Bootstrap Confidence Intervals** (1,000 resamples).

### Table 1: Baseline Retrieval Leaderboard
| System | NDCG@5 (95% CI) | NDCG@10 (95% CI) | Recall@5 (95% CI) | Recall@10 (95% CI) | MRR@10 | MAP@10 | p-val vs v1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **attribute_filter** | 0.651 [0.447, 0.856] | 0.659 [0.492, 0.819] | 0.299 [0.266, 0.330] | 0.533 [0.409, 0.632] | 0.760 | 0.649 | 0.4625 |
| **bm25** | 0.471 [0.286, 0.657] | 0.543 [0.381, 0.723] | 0.225 [0.144, 0.290] | 0.505 [0.361, 0.633] | 0.653 | 0.549 | 0.0285 |
| **minilm_dense** | 0.568 [0.353, 0.771] | 0.582 [0.427, 0.723] | 0.274 [0.176, 0.356] | 0.469 [0.371, 0.556] | 0.833 | 0.583 | 0.1980 |
| **clip_zero_shot** | 0.654 [0.493, 0.824] | 0.650 [0.546, 0.744] | 0.333 [0.276, 0.391] | 0.516 [0.414, 0.609] | 0.938 | 0.664 | 0.2545 |
| **baseline_v1** | 0.753 [0.594, 0.899] | 0.750 [0.655, 0.841] | 0.353 [0.295, 0.415] | 0.601 [0.472, 0.714] | 1.000 | 0.786 | - |
| **siglip** | 0.654 [0.493, 0.824] | 0.650 [0.546, 0.744] | 0.333 [0.276, 0.391] | 0.516 [0.414, 0.609] | 0.938 | 0.664 | 0.2545 |

### Table 2: Slice Analysis (NDCG@5 by Scene Category)
| Scene Class | baseline_v1 (95% CI) | clip_zero_shot (95% CI) | siglip (95% CI) |
|---|:---:|:---:|:---:|
| `bedroom` | 0.899 [0.798, 1.000] | 0.798 [0.596, 1.000] | 0.798 [0.596, 1.000] |
| `dining_room` | 0.737 [0.475, 1.000] | 0.737 [0.475, 1.000] | 0.737 [0.475, 1.000] |
| `home_office` | 0.679 [0.421, 0.938] | 0.635 [0.431, 0.839] | 0.635 [0.431, 0.839] |
| `living_room` | 0.699 [0.528, 0.869] | 0.444 [0.360, 0.528] | 0.444 [0.360, 0.528] |

### Table 3: Human Inter-Rater Reliability
- **Evaluated Pairs**: 141
- **Exact Agreement**: 83.0%
- **Adjacent Agreement (|r1 - r2| <= 1)**: 100.0%
- **Cohen's Kappa (Quadratic)**: **`0.7040`**
- **Cohen's Kappa (Unweighted)**: `0.6246`

### Table 4: VLM-as-Judge Validation Against Human Gold
- **Evaluated Pairs**: 141
- **Exact Agreement**: 22.0%
- **Cohen's Kappa (Quadratic)**: `0.1928`
- **Spearman Rank Correlation (rho)**: **`0.4168`** (p = 2.72e-07)
- **Severe False Positives (Gold=0, VLM=2)**: 4
- **Severe False Negatives (Gold=2, VLM=0)**: 0
> [!NOTE]
> VLM Judge labels are used strictly as a weak supervision signal for model training in Phase 3. The test set remains 100% human-verified gold.
