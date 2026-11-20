"""Streamlit Gold Labeling Tool for Multimodal Room2Rug Search.

Run with:
    streamlit run src/eval/labeling_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

DATA_DIR = Path("data")
GOLD_LABELS_PATH = DATA_DIR / "gold_labels.json"
TEST_QUERIES_PATH = DATA_DIR / "test_queries.json"
PRODUCTS_PATH = DATA_DIR / "products.json"


def load_json(path: Path, default=None):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default if default is not None else {}


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    st.set_page_config(
        page_title="Room2Rug Gold Labeling Studio",
        page_icon="🛋️",
        layout="wide",
    )

    st.title("🛋️ Room2Rug Gold Labeling Studio")
    st.caption("Human-in-the-loop Ground Truth Annotation for Multimodal Retrieval")

    # Sidebar: Annotator settings & Rubric
    st.sidebar.header("Annotator Configuration")
    annotator_id = st.sidebar.text_input("Annotator ID", value="primary_evaluator").strip()
    if not annotator_id:
        st.sidebar.warning("Please provide an Annotator ID.")
        return

    with st.sidebar.expander("📖 Grading Rubric & Guidelines", expanded=True):
        st.markdown(
            """
            **Rating Scale:**
            - **`0 - Irrelevant / Clashing`**: Color discordance, clashing decor style, or severe scale mismatch.
            - **`1 - Moderately Acceptable`**: Inoffensive neutral, tolerable match, partial color link, or plain baseline.
            - **`2 - Highly Harmonious`**: Intentional palette echo, cohesive aesthetic, correct room scale, high visual synergy.

            **Core Dimensions:**
            1. **Palette Harmony**: Wall color, floor tone, and furniture upholstery integration.
            2. **Style Cohesion**: Contemporary, Traditional, Bohemian, Minimalist synergy.
            3. **Scale & Pattern Balance**: Proportion and pattern density.
            """
        )

    # Load data
    test_queries = load_json(TEST_QUERIES_PATH, default=[])
    products_list = load_json(PRODUCTS_PATH, default=[])
    products_by_id = {p.get("handle"): p for p in products_list if "handle" in p}

    gold_data = load_json(GOLD_LABELS_PATH, default={"annotations": [], "metadata": {}})
    annotations = gold_data.get("annotations", [])

    if not test_queries:
        st.warning("No test queries found in data/test_queries.json. Generate candidate pools first.")
        return

    # Build list of annotation tasks: (query, rug_id)
    tasks = []
    for q in test_queries:
        candidate_ids = q.get("candidate_rug_ids", [])
        for rug_id in candidate_ids:
            tasks.append({"query": q, "rug_id": rug_id})

    if not tasks:
        st.info("No query-rug pairs to annotate.")
        return

    # Create mapping of existing ratings for current annotator
    user_ratings = {(a["query_id"], a["rug_id"]): a for a in annotations if a.get("annotator_id") == annotator_id}

    # Navigation & progress
    total_tasks = len(tasks)
    rated_count = len(user_ratings)
    progress_pct = (rated_count / total_tasks) if total_tasks > 0 else 0.0

    st.sidebar.markdown(f"**Progress**: {rated_count} / {total_tasks} pairs ({progress_pct:.1%})")
    st.sidebar.progress(progress_pct)

    # Session state for task index
    if "task_idx" not in st.session_state:
        # Jump to first unrated task
        first_unrated = next(
            (i for i, t in enumerate(tasks) if (t["query"]["query_id"], t["rug_id"]) not in user_ratings),
            0,
        )
        st.session_state.task_idx = first_unrated

    col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([1, 1, 2, 2])
    with col_nav1:
        if st.button("⬅️ Previous") and st.session_state.task_idx > 0:
            st.session_state.task_idx -= 1
            st.rerun()
    with col_nav2:
        if st.button("Next ➡️") and st.session_state.task_idx < total_tasks - 1:
            st.session_state.task_idx += 1
            st.rerun()
    with col_nav3:
        jump_idx = st.number_input(
            "Jump to pair #", min_value=1, max_value=total_tasks, value=st.session_state.task_idx + 1
        )
        if jump_idx - 1 != st.session_state.task_idx:
            st.session_state.task_idx = jump_idx - 1
            st.rerun()

    current_task = tasks[st.session_state.task_idx]
    q = current_task["query"]
    rug_id = current_task["rug_id"]
    rug = products_by_id.get(rug_id, {})

    pair_key = (q["query_id"], rug_id)
    current_ann = user_ratings.get(pair_key)

    st.divider()

    # Visual Display: Side-by-side room & rug
    col_room, col_rug = st.columns(2)

    with col_room:
        st.subheader("🏠 Query Room Context")
        room_img_path = q.get("room_image_path")
        if room_img_path and Path(room_img_path).exists():
            st.image(room_img_path, caption=f"Room Image: {Path(room_img_path).name}", use_column_width=True)
        else:
            st.info(f"Room image path: {room_img_path} (image placeholder)")

        st.markdown(f"**Query ID**: `{q['query_id']}`")
        st.markdown(f"**Scene Type**: `{q.get('scene_type', 'General')}`")
        st.markdown(f'**Text Query**: *"{q.get("text_query", "N/A")}"*')
        if q.get("target_color"):
            st.markdown(f"**Target Color**: {q['target_color']}")
        if q.get("target_size"):
            st.markdown(f"**Target Size**: {q['target_size']}")

    with col_rug:
        st.subheader("🧶 Candidate Rug (Blinded)")
        rug_img_url = rug.get("image_url")
        if rug_img_url:
            st.image(rug_img_url, caption=rug.get("title", rug_id), use_column_width=True)
        else:
            st.info("No rug image preview available.")

        st.markdown(f"**Title**: **{rug.get('title', 'N/A')}**")
        st.markdown(f"**Rug Handle / ID**: `{rug_id}`")
        st.markdown(f"**Color**: {rug.get('color', 'N/A')} | **Style**: {rug.get('style', 'N/A')}")
        st.markdown(f"**Available Sizes**: {', '.join(rug.get('sizes', [])) or rug.get('size', 'N/A')}")
        st.markdown(f"**Tags**: *{rug.get('tags', 'N/A')}*")

    st.divider()

    # Rating Section
    st.subheader("📝 Record Rating")
    existing_score = current_ann.get("relevance") if current_ann else None
    existing_notes = current_ann.get("notes", "") if current_ann else ""

    col_btn0, col_btn1, col_btn2 = st.columns(3)
    new_rating = None

    with col_btn0:
        if st.button(
            "❌ 0: Irrelevant / Clashing",
            use_container_width=True,
            type="primary" if existing_score == 0 else "secondary",
        ):
            new_rating = 0
    with col_btn1:
        if st.button(
            "⚠️ 1: Moderately Acceptable",
            use_container_width=True,
            type="primary" if existing_score == 1 else "secondary",
        ):
            new_rating = 1
    with col_btn2:
        if st.button(
            "🌟 2: Highly Harmonious", use_container_width=True, type="primary" if existing_score == 2 else "secondary"
        ):
            new_rating = 2

    notes = st.text_input("Styling Rationale / Notes (optional)", value=existing_notes)

    if new_rating is not None:
        # Update or append annotation
        found = False
        for ann in annotations:
            if ann.get("annotator_id") == annotator_id and ann["query_id"] == q["query_id"] and ann["rug_id"] == rug_id:
                ann["relevance"] = new_rating
                ann["notes"] = notes
                found = True
                break

        if not found:
            annotations.append(
                {
                    "query_id": q["query_id"],
                    "rug_id": rug_id,
                    "scene_type": q.get("scene_type", "general"),
                    "relevance": new_rating,
                    "notes": notes,
                    "annotator_id": annotator_id,
                }
            )

        gold_data["annotations"] = annotations
        save_json(GOLD_LABELS_PATH, gold_data)
        st.success(f"Saved rating {new_rating} for pair {st.session_state.task_idx + 1}!")

        # Advance to next unrated task
        if st.session_state.task_idx < total_tasks - 1:
            st.session_state.task_idx += 1
            st.rerun()

    # Inter-rater agreement tab
    st.divider()
    with st.expander("📊 Multi-Rater Agreement Analytics"):
        annotators_found = sorted(list({a.get("annotator_id") for a in annotations if "annotator_id" in a}))
        st.write(f"Active annotators: {', '.join(annotators_found)}")

        if len(annotators_found) >= 2:
            from src.eval.agreement import agreement_report

            r1_id = annotators_found[0]
            r2_id = annotators_found[1]

            map1 = {(a["query_id"], a["rug_id"]): a["relevance"] for a in annotations if a.get("annotator_id") == r1_id}
            map2 = {(a["query_id"], a["rug_id"]): a["relevance"] for a in annotations if a.get("annotator_id") == r2_id}

            common = [p for p in map1 if p in map2]
            if common:
                r1_vals = [map1[p] for p in common]
                r2_vals = [map2[p] for p in common]
                rep = agreement_report(r1_vals, r2_vals, rater1_name=r1_id, rater2_name=r2_id)
                st.write(f"**Overlapping Rated Pairs**: {rep['n_pairs']}")
                st.write(f"**Exact Agreement**: {rep['exact_agreement']:.2%}")
                st.write(f"**Cohen's Kappa (Quadratic)**: `{rep['cohens_kappa_quadratic']:.4f}`")
                st.write(f"**Cohen's Kappa (Unweighted)**: `{rep['cohens_kappa_unweighted']:.4f}`")
                st.table(rep["confusion_matrix_3x3"])
            else:
                st.info("No overlapping pairs annotated between the selected raters yet.")
        else:
            st.info("Need at least 2 distinct annotators to compute Cohen's kappa.")


if __name__ == "__main__":
    main()
