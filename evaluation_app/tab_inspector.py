"""Pipeline inspection and dataset evaluation."""

import asyncio
import json
import os
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional

import cv2
import numpy as np
import streamlit as st

from evaluation_app.config import IMAGES_DIR, LAST_EVAL_FILE
from evaluation_app.db import get_tcgdex_card_details, load_ground_truth, normalize_card_id


@st.cache_resource
def get_pipeline():
    from delta_cards.recognition.pipeline import CardRecognitionPipeline

    return CardRecognitionPipeline()


def _image_from_upload(upload: Any) -> Optional[np.ndarray]:
    if upload is None:
        return None
    return cv2.imdecode(np.frombuffer(upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)


def _is_correct(prediction: str, expected: str) -> bool:
    if expected.lower() == "fake":
        return not prediction
    return normalize_card_id(prediction) == normalize_card_id(expected)


def _inspect(image: np.ndarray, title: str, description: str, check_llm: bool) -> Dict[str, Any]:
    from delta_cards.recognition.text_matcher import rank_candidates

    pipeline = get_pipeline()
    timings = {}

    started = time.perf_counter()
    try:
        crop, annotated = pipeline.isolator.isolate_card_with_debug(image)
        isolation = "isolated"
    except Exception:
        crop, annotated, isolation = image, None, "fallback_full_frame"
    timings["isolate"] = time.perf_counter() - started

    started = time.perf_counter()
    embedding = pipeline.embedder.embed_image(crop)
    timings["embed"] = time.perf_counter() - started

    started = time.perf_counter()
    candidates = pipeline.vector_index.search(embedding, 15)
    timings["search"] = time.perf_counter() - started

    started = time.perf_counter()
    ocr = pipeline.ocr.extract(crop)
    timings["ocr"] = time.perf_counter() - started

    started = time.perf_counter()
    candidate_texts = pipeline.ocr_cache.get_candidate_texts(candidates)
    timings["candidate_ocr"] = time.perf_counter() - started

    started = time.perf_counter()
    match = rank_candidates(ocr.full_text, candidates, candidate_texts, pipeline.acceptance_threshold)
    timings["full_text_match"] = time.perf_counter() - started

    llm_result = None
    if check_llm and match["verified_candidate"]:
        from delta_cards.recognition.llm_verifier import get_llm_verifier

        candidate = match["verified_candidate"]
        offer = SimpleNamespace(title=title, description=description, _query_ocr={"full_text": ocr.full_text}, _candidate_ocr={"full_text": candidate["candidate_text"]})
        started = time.perf_counter()
        llm_result = get_llm_verifier().verify_match(offer)
        timings["llm"] = time.perf_counter() - started

    return {"crop": crop, "annotated": annotated, "isolation": isolation, "embedding": embedding, "candidates": candidates, "ocr": ocr, "candidate_texts": candidate_texts, "match": match, "llm": llm_result, "timings": timings}


def _render_result(result: Dict[str, Any], expected_id: str = "") -> None:
    winner = result["match"]["verified_candidate"]
    winner_id = winner.get("card", {}).get("tcgdex_id", "") if winner else ""
    correct = _is_correct(winner_id, expected_id) if expected_id else None
    columns = st.columns(4)
    columns[0].metric("Prediction", winner_id or "Rejected")
    columns[1].metric("Text score", f"{result['match']['confidence']:.0%}")
    columns[2].metric("OCR language", result["ocr"].detected_language)
    columns[3].metric("Expected", "Correct" if correct else "Incorrect" if correct is False else "Not set")

    left, right = st.columns(2)
    with left:
        st.image(cv2.cvtColor(result["crop"], cv2.COLOR_BGR2RGB), caption="Card crop", width="stretch")
    with right:
        if result["annotated"] is not None:
            st.image(cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB), caption="Card isolation", width="stretch")
        elif winner_id:
            reference = get_tcgdex_card_details(winner_id)
            if reference and reference.get("high_image_url"):
                st.image(reference["high_image_url"], caption="Predicted reference", width="stretch")

    with st.expander("1. Card isolation", expanded=True):
        st.write(result["isolation"])
    with st.expander("2. DINOv3 embedding"):
        st.code(f"shape={result['embedding'].shape}, norm={np.linalg.norm(result['embedding']):.4f}")
    with st.expander("3. Visual candidates"):
        st.dataframe([{"rank": rank + 1, "id": item["card"].get("tcgdex_id"), "name": item["card"].get("name"), "visual_similarity": item["similarity"]} for rank, item in enumerate(result["candidates"])], width="stretch", hide_index=True)
    with st.expander("4. Full OCR text", expanded=True):
        st.text_area("OCR", result["ocr"].full_text, height=160, disabled=True, label_visibility="collapsed")
    with st.expander("5. Candidate full OCR text"):
        st.dataframe([{"id": item["card"].get("tcgdex_id"), "full_text": result["candidate_texts"].get(f"{item['card'].get('tcgdex_id', '')}_{(item['card'].get('lang') or 'en').lower()}", "")} for item in result["candidates"]], width="stretch", hide_index=True)
    with st.expander("6. Full-text decision", expanded=True):
        st.write("Accepted" if winner else "Rejected")
        st.json({"threshold": get_pipeline().acceptance_threshold, "score": result["match"]["confidence"], "candidate": winner_id})
    if result["llm"] is not None:
        with st.expander("7. LLM confirmation", expanded=True):
            st.json(result["llm"].model_dump())
    st.dataframe([{"stage": stage, "seconds": round(seconds, 3)} for stage, seconds in result["timings"].items()], width="stretch", hide_index=True)


def _run_batch(samples: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    pipeline = get_pipeline()
    progress = st.progress(0.0)
    rows = []
    for index, sample in enumerate(samples, 1):
        path = os.path.join(IMAGES_DIR, sample.get("filename", ""))
        image = cv2.imread(path)
        if image is None:
            rows.append({"sample": sample.get("sample_id"), "expected": sample.get("expected_tcgdex_id"), "prediction": "missing image", "correct": False})
            progress.progress(index / len(samples))
            continue
        result = asyncio.run(pipeline.process_card_listing(image, sample.get("title", ""), sample.get("description", "")))
        prediction = (result.get("verified_card") or {}).get("id", "")
        expected = sample.get("expected_tcgdex_id", "")
        correct = _is_correct(prediction, expected)
        rows.append({"sample": sample.get("sample_id"), "expected": expected, "prediction": prediction or "rejected", "correct": correct, "score": result.get("confidence", 0.0)})
        progress.progress(index / len(samples))
    return rows


def _render_evaluations(rows: list[Dict[str, Any]]) -> None:
    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    metrics = st.columns(3)
    metrics[0].metric("Evaluated", total)
    metrics[1].metric("Correct", correct)
    metrics[2].metric("Accuracy", f"{correct / total:.0%}" if total else "N/A")
    result_filter = st.segmented_control("Results", ["All", "Incorrect", "Correct"], default="All")
    visible = [row for row in rows if result_filter == "All" or row["correct"] == (result_filter == "Correct")]
    st.dataframe(visible, width="stretch", hide_index=True)


def _load_evaluation_rows() -> list[Dict[str, Any]]:
    with open(LAST_EVAL_FILE, "r", encoding="utf-8") as evaluation_file:
        saved = json.load(evaluation_file)
    return saved.get("results", []) if isinstance(saved, dict) else saved


def render_inspector_tab() -> None:
    """Render one-image inspection and small batch evaluation controls."""
    dataset = load_ground_truth()
    source, controls = st.columns([3, 2])
    with source:
        mode = st.segmented_control("Image", ["Dataset", "Upload"], default="Dataset")
    selected = None
    with controls:
        use_llm = st.checkbox("LLM confirmation")

    if mode == "Dataset":
        options = {f"{item.get('sample_id')} | {item.get('expected_tcgdex_id', 'unlabeled')}": item for item in dataset}
        selected_label = st.selectbox("Sample", list(options)) if options else None
        selected = options.get(selected_label) if selected_label else None
        image = cv2.imread(os.path.join(IMAGES_DIR, selected.get("filename", ""))) if selected else None
        title = st.text_input("Title", selected.get("title", "") if selected else "")
        description = st.text_area("Description", selected.get("description", "") if selected else "")
        expected_id = selected.get("expected_tcgdex_id", "") if selected else ""
    else:
        image = _image_from_upload(st.file_uploader("Image file", type=["jpg", "jpeg", "png", "webp"]))
        title = st.text_input("Title")
        description = st.text_area("Description")
        expected_id = st.text_input("Expected TCGdex ID")

    if st.button("Run inspection", type="primary", disabled=image is None):
        with st.spinner("Running pipeline..."):
            st.session_state["inspection"] = _inspect(image, title, description, use_llm)
            st.session_state["inspection_expected"] = expected_id
    if "inspection" in st.session_state:
        _render_result(st.session_state["inspection"], st.session_state.get("inspection_expected", ""))

    with st.expander("Evaluate dataset", expanded=True):
        categories = ["All"] + sorted({item.get("category", "") for item in dataset if item.get("category")})
        category = st.selectbox("Category", categories)
        samples = [item for item in dataset if category == "All" or item.get("category") == category]
        mode = st.segmented_control("Scope", ["All samples", "Sample limit"], default="All samples")
        limit = st.number_input("Sample limit", 1, max(1, len(samples)), min(10, len(samples))) if mode == "Sample limit" and samples else len(samples)
        if st.button("Evaluate", type="primary", disabled=not samples):
            with st.spinner(f"Evaluating {limit} samples..."):
                rows = _run_batch(samples[:limit])
                st.session_state["evaluation_rows"] = rows
                with open(LAST_EVAL_FILE, "w", encoding="utf-8") as evaluation_file:
                    json.dump(rows, evaluation_file, indent=2)
        if "evaluation_rows" not in st.session_state and os.path.exists(LAST_EVAL_FILE):
            st.session_state["evaluation_rows"] = _load_evaluation_rows()
        if "evaluation_rows" in st.session_state:
            _render_evaluations(st.session_state["evaluation_rows"])