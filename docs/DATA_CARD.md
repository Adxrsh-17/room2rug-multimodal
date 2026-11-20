# Data Card: Room2Rug Multimodal Search & Recommendation System

**Version:** 2.0 (Phase 1 Scale)  
**Date:** November 2026  
**Auditor / Maintainer:** Room2Rug Engineering Team  

---

## 1. Overview & Dataset Purpose

The Room2Rug system evaluates multimodal retrieval and learned reranking for interior décor, specifically recommending area rugs and runners matching real-world room scenes. The data pipeline integrates two foundational corpora:
1. **Room Corpus**: Residential interior photographs derived from the MIT CSAIL ADE20K dataset with semantic segmentation masks.
2. **Product Catalog**: Rug product records, attributes, and top-down product photographs sourced from the Amazon Berkeley Objects (ABO) dataset.

---

## 2. Dataset Sources & Official License Terms

### 2.1 ADE20K Room Corpus
- **Official Host**: MIT Computer Science and Artificial Intelligence Laboratory (CSAIL) Vision Group  
  - Website: `http://groups.csail.mit.edu/vision/datasets/ADE20K/`  
  - Benchmark Archive: `http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip`  
  - Codebase: `https://github.com/CSAILVision/ADE20K`
- **License Terms & Restrictions (Official Verification)**:
  - **Non-Commercial Research & Education Only**: The ADE20K image database is licensed strictly for non-commercial research and educational purposes. Commercial exploitation, for-profit software embedding, or commercial service development is prohibited under the standard license without explicit commercial licensing from the MIT Technology Licensing Office (TLO).
  - **Correction Regarding "BSD-3"**: The BSD-3-Clause license found in CSAIL repositories applies **solely to the evaluation code and development tools**, NOT to the underlying image database.
  - **Copyright Ownership**: MIT does not own the copyright to the underlying photographs (which were aggregated from LabelMe, SUN, and Places databases). Access is granted solely for research analysis.
  - **Redistribution**: Raw images may not be publicly mirrored or redistributed outside official hosting channels.
  - **Indemnification**: Users agree to defend and indemnify MIT against any claims arising from data usage.

### 2.2 Amazon Berkeley Objects (ABO) Catalog
- **Official Host**: Amazon & University of California, Berkeley  
  - S3 Host: `https://amazon-berkeley-objects.s3.amazonaws.com/`  
  - Project Site: `https://amazon-berkeley-objects.s3.amazonaws.com/index.html`
- **License**: **Creative Commons Attribution 4.0 International (CC BY 4.0)**.  
  - Permits sharing, adaptation, and commercial use with appropriate attribution to Amazon and UC Berkeley.

### 2.3 Unsplash (Optional Out-of-Distribution Test Slice Only)
- **License**: Unsplash Free License (allows commercial and non-commercial reproduction).  
- **Usage Policy**: In strict accordance with Unsplash API Terms of Service, **no scraping is performed**. If used, a small curated test slice ($\le 50$ images) must be downloaded manually or via the official API with explicit photographer attribution.

---

## 3. Real Dataset Statistics (Empirical Counts)

### 3.1 ADE20K Residential Interior Scene Counts
Derived directly from the canonical `sceneCategories.txt` index (22,210 total annotated scenes in ADE20K / ADEChallengeData2016):

| Scene Class | Training Set | Validation Set | **Total Real Count** |
|---|:---:|:---:|:---:|
| `bedroom` | 1,389 | 139 | **1,528** |
| `living_room` | 697 | 70 | **767** |
| `dining_room` | 412 | 42 | **454** |
| `home_office` | 95 | 10 | **105** |
| **Core 4 Residential Spaces** | **2,593** | **261** | **2,854** |

#### Other Relevant Residential Scenes Available:
- `kitchen`: 728 (661 train, 67 val)
- `bathroom`: 738 (671 train, 67 val)
- `corridor`: 137 (123 train, 14 val)
- `game_room`: 109 (98 train, 11 val)
- `poolroom_home`: 105 (95 train, 10 val)
- `attic`: 97 (87 train, 10 val)
- `childs_room`: 77 (70 train, 7 val)
- `nursery`: 66 (60 train, 6 val)
- `playroom`: 41 (37 train, 4 val)
- `basement`: 32 (29 train, 3 val)
- `hallway`: 30 (28 train, 2 val)
- `sunroom`: 10 (9 train, 1 val)
- `television_room`: 2 (2 train, 0 val)

---

## 4. Segmentation Mask Filtering & Floor Visibility Analysis

To evaluate rug recommendations credibly, rooms must exhibit a clearly visible floor plane. We leverage ADE20K semantic segmentation masks (`objectInfo150.txt`):
- **Class 4**: `floor, flooring`
- **Class 29**: `rug, carpet, carpeting`
- **Class 13**: `person, individual`

The visible floor ratio is defined as:
$$\text{Floor Ratio} = \frac{\sum_{\text{pixels}} (\text{mask} == 4 \lor \text{mask} == 29)}{\text{Total Pixels}}$$

### 4.1 Floor Coverage Threshold Analysis (Core 4 Classes: 2,854 Candidate Images)

| Threshold | Retained Count | Retained \% | Visual Quality & Room Characteristics |
|---|:---:|:---:|---|
| **$< 5\%$** | 246 | 8.6\% | **Rejected**: Floor is virtually absent or blocked by tight close-ups of headboards, wall decor, or ceiling fixtures. Rug placement is scale-incoherent. |
| **$\ge 5\%$** | 2,608 | 91.4\% | Includes tight furniture crops where only a small corner or sliver of floor is visible under desks. Ambiguous for area rugs. |
| **$\ge 8\%$** | 2,389 | 83.7\% | Removes wall crops. Adequate for runner rugs and small accent rugs. |
| **$\ge 10\%$ (Selected)** | **2,175** | **76.2\%** | **Optimal Threshold**: Every retained room presents an expansive, unobstructed floor plane. Enables realistic visual assessment of 8x10, 5x8, and runner placements. |
| **$\ge 15\%$** | 1,755 | 61.5\% | Excludes common furnished rooms where sectional sofas or king beds occupy floor area, biasing the corpus toward unusually sparse or cavernous rooms. |

### 4.2 Person & Clutter Filtering
- **Person Exclusion**: Images with person pixel coverage $> 5\%$ (Class 13) are removed to avoid visual occlusion of the floor.
- **Perceptual Deduplication**: Near-duplicate images (burst shots, identical angle crops) are deduplicated using 64-bit difference hashing (`dHash`) with a Hamming distance threshold $\le 4$.

---

## 5. Group-Aware Splitting & Zero-Leakage Guarantee

To prevent information leakage across recommendation pairs, splits are generated using **group-aware partitioning** (`src/data/splits.py`):
1. Partition unique `rug_id` (product handles) into disjoint sets: $\text{Rugs}_{\text{train}} (70\%)$, $\text{Rugs}_{\text{val}} (15\%)$, $\text{Rugs}_{\text{test}} (15\%)$.
2. Partition unique `room_id` (ADE20K image IDs) into disjoint sets: $\text{Rooms}_{\text{train}} (70\%)$, $\text{Rooms}_{\text{val}} (15\%)$, $\text{Rooms}_{\text{test}} (15\%)$.
3. Form room-rug pairs strictly within partitions:
   $$\text{Pair}(r, m) \in \text{Split}_S \iff r \in \text{Rugs}_S \land m \in \text{Rooms}_S$$

### 5.1 Formal Leakage Verification
An automated unit test ([`tests/test_leakage.py`](file:///c:/Users/Adarsh_Pradeep/OneDrive/Desktop/Career%20Journey/Personal%20Projects/RugMultimodal/room2rug-multimodal/tests/test_leakage.py)) enforces the following invariants:
$$\text{Rugs}_{\text{train}} \cap \text{Rugs}_{\text{test}} = \emptyset, \quad \text{Rugs}_{\text{train}} \cap \text{Rugs}_{\text{val}} = \emptyset, \quad \text{Rugs}_{\text{val}} \cap \text{Rugs}_{\text{test}} = \emptyset$$
$$\text{Rooms}_{\text{train}} \cap \text{Rooms}_{\text{test}} = \emptyset, \quad \text{Rooms}_{\text{train}} \cap \text{Rooms}_{\text{val}} = \emptyset, \quad \text{Rooms}_{\text{val}} \cap \text{Rooms}_{\text{test}} = \emptyset$$

---

## 6. Known Biases & Limitations

1. **Demographic & Architectural Skew**:
   - ADE20K images are disproportionately drawn from North American and European middle-to-high income residential interiors.
   - Non-Western residential flooring traditions (e.g., tatami mats, Middle Eastern majlis seating, tiled verandas) are underrepresented.
2. **Catalog Distribution Skew**:
   - ABO products originate from Amazon private brands (AmazonBasics, Stone & Beam, Rivet), which skew heavily toward modern minimalist, geometric, and transitional styles in neutral, grey, and beige palettes.
   - Antique, hand-knotted, and Persian high-end silk rugs are underrepresented compared to synthetic power-loomed polypropylene rugs.
3. **Domain Gap**:
   - The catalog contains clean, flat-lay or isolated overhead product photos on neutral/white backgrounds.
   - Query rooms contain complex 3D perspective geometry, variable ambient lighting, shadows, and perspective foreshortening.
