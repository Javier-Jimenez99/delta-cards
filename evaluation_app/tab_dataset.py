"""Small ground-truth dataset editor."""

import os
from typing import Any, Dict

import streamlit as st

from evaluation_app.config import IMAGES_DIR
from evaluation_app.db import get_tcgdex_card_details, load_ground_truth, save_ground_truth


def _rows(dataset: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return [{"sample": item.get("sample_id"), "expected": item.get("expected_tcgdex_id"), "category": item.get("category"), "title": item.get("title"), "image": item.get("filename")} for item in dataset]


def render_dataset_tab() -> None:
    """Render the dataset table and selected-sample editor."""
    dataset = load_ground_truth()
    st.metric("Samples", len(dataset))
    st.dataframe(_rows(dataset), width="stretch", hide_index=True)
    if not dataset:
        return

    sample_ids = {str(item.get("sample_id")): index for index, item in enumerate(dataset)}
    sample_id = st.selectbox("Sample", sample_ids)
    index = sample_ids[sample_id]
    sample = dataset[index]
    image_path = os.path.join(IMAGES_DIR, sample.get("filename", ""))
    left, right = st.columns(2)
    with left:
        if os.path.exists(image_path):
            st.image(image_path, caption="Listing image", width="stretch")
    with right:
        card = get_tcgdex_card_details(sample.get("expected_tcgdex_id", ""))
        if card and card.get("high_image_url"):
            st.image(card["high_image_url"], caption="Expected card", width="stretch")

    title = st.text_input("Title", sample.get("title", ""), key=f"title_{sample_id}")
    expected = st.text_input("Expected TCGdex ID", sample.get("expected_tcgdex_id", ""), key=f"expected_{sample_id}")
    category = st.text_input("Category", sample.get("category", ""), key=f"category_{sample_id}")
    evidence = st.text_area("Note", sample.get("audit_evidence", ""), key=f"note_{sample_id}")
    save, fake, remove = st.columns(3)
    if save.button("Save", type="primary"):
        sample.update({"title": title, "expected_tcgdex_id": expected, "category": category, "audit_evidence": evidence})
        save_ground_truth(dataset)
        st.rerun()
    if fake.button("Mark fake"):
        sample["expected_tcgdex_id"] = "fake"
        save_ground_truth(dataset)
        st.rerun()
    if remove.button("Delete"):
        dataset.pop(index)
        save_ground_truth(dataset)
        st.rerun()

    with st.expander("Add sample"):
        upload = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"])
        new_id = st.text_input("Sample ID")
        new_expected = st.text_input("Expected ID")
        if st.button("Add", disabled=upload is None or not new_id):
            extension = os.path.splitext(upload.name)[1].lower() or ".jpg"
            filename = f"{new_id}{extension}"
            os.makedirs(IMAGES_DIR, exist_ok=True)
            with open(os.path.join(IMAGES_DIR, filename), "wb") as image_file:
                image_file.write(upload.getvalue())
            dataset.append({"sample_id": new_id, "filename": filename, "expected_tcgdex_id": new_expected, "title": "", "description": "", "category": "", "audit_evidence": ""})
            save_ground_truth(dataset)
            st.rerun()