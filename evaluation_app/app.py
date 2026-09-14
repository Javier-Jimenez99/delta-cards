"""Delta Cards - Evaluation Suite & Dataset Curator.

Streamlit application to visually manage, edit, audit, and expand the Pokémon card
evaluation dataset (ground_truth.json and images), search canonical cards in SQLite,
and run one-click recognition pipeline benchmarks with live diagnostics and side-by-side visual comparison.
"""

import asyncio
import html
import io
import json
import os
import shutil
import sqlite3
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional
import urllib.request

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Ensure src/ is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

GROUND_TRUTH_PATH = os.path.join(PROJECT_ROOT, "data", "evaluation_set", "ground_truth.json")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation_set", "images")
SQLITE_PATH = os.path.join(PROJECT_ROOT, "data", "pokemon_cards_meta.sqlite")
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "images_cache")
LAST_EVAL_FILE = os.path.join(PROJECT_ROOT, "data", "evaluation_last_run.json")
DEBUG_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation_debug")
os.makedirs(DEBUG_DIR, exist_ok=True)

st.set_page_config(
    page_title="Delta Cards — Evaluation Suite",
    page_icon="🎴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clean, unified light theme styles matching Streamlit defaults
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #ffffff;
        padding: 16px 18px;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        border-left: 5px solid #4f46e5;
        margin-bottom: 12px;
    }
    .badge-success {
        background-color: #d1fae5;
        color: #065f46;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: bold;
    }
    .badge-danger {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: bold;
    }
    .badge-warning {
        background-color: #fef3c7;
        color: #92400e;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: bold;
    }
    .info-card-box {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px 14px;
        color: #1f2937;
        box-sizing: border-box;
        font-size: 13px;
        line-height: 1.45;
        margin-bottom: 8px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    .info-card-title {
        font-size: 14px;
        font-weight: 700;
        color: #2563eb;
        margin-bottom: 8px;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 5px;
    }
    .info-card-field {
        margin-bottom: 4px;
        font-size: 12.5px;
    }
    .info-card-label {
        color: #6b7280;
        font-weight: 600;
        font-size: 12px;
    }
    .info-card-desc {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 8px 10px;
        color: #374151;
        max-height: 160px;
        overflow-y: auto;
        font-size: 12px;
        margin-top: 5px;
        white-space: pre-wrap;
        line-height: 1.4;
    }
    .fake-placeholder {
        width: 100%;
        min-height: 380px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: #fff1f2;
        border: 2px dashed #f43f5e;
        border-radius: 8px;
        color: #be123c;
        text-align: center;
        padding: 16px;
        box-sizing: border-box;
    }
    .fake-text {
        font-size: 28px;
        font-weight: 900;
        letter-spacing: 2px;
        color: #e11d48;
        margin-top: 6px;
    }
    .fake-subtitle {
        font-size: 12px;
        color: #be123c;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Helper Functions & Database Cache
# =============================================================================

@st.cache_resource
def get_db_connection():
    """Provides a cached SQLite connection."""
    if not os.path.exists(SQLITE_PATH):
        return None
    con = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def load_ground_truth() -> List[Dict[str, Any]]:
    """Loads ground_truth.json."""
    if not os.path.exists(GROUND_TRUTH_PATH):
        return []
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_ground_truth(dataset: List[Dict[str, Any]]) -> bool:
    """Saves ground_truth.json with automatic backup."""
    try:
        if os.path.exists(GROUND_TRUTH_PATH):
            shutil.copy2(GROUND_TRUTH_PATH, GROUND_TRUTH_PATH + ".bak")
        with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        return True
    except Exception as exc:
        st.error(f"Error saving dataset: {exc}")
        return False


def normalize_card_id(cid: str) -> str:
    """Normalizes card ID e.g. 'swsh4.5-44' vs 'swsh4.5-044'."""
    if not cid:
        return ""
    parts = cid.split("-")
    if len(parts) == 2:
        return f"{parts[0].lower()}-{parts[1].lstrip('0').lower()}"
    return cid.lower()


def to_tcgdex_high_url(url: Optional[str]) -> Optional[str]:
    """Ensures the TCGdex CDN URL points directly to the official high-resolution image."""
    if not url:
        return None
    url = url.strip()
    if "/low." in url:
        return url.replace("/low.", "/high.")
    if url.endswith("/high.webp") or url.endswith("/high.png") or url.endswith("/high.jpg"):
        return url
    return f"{url.rstrip('/')}/high.webp"


def get_tcgdex_card_details(tcgdex_id: str) -> Optional[Dict[str, Any]]:
    """Fetches canonical card metadata and high-resolution TCGdex CDN image URL from SQLite."""
    if not tcgdex_id or tcgdex_id.lower() == "fake":
        return None
    con = get_db_connection()
    if not con:
        return None

    # Exact match
    r = con.execute("SELECT * FROM card_metadata WHERE tcgdex_id = ?", (tcgdex_id,)).fetchone()
    # Fallback to normalized set / local_id
    if not r and "-" in tcgdex_id:
        set_p, num_p = tcgdex_id.split("-", 1)
        r = con.execute(
            "SELECT * FROM card_metadata WHERE set_id = ? AND (local_id = ? OR local_id = ?)",
            (set_p.lower(), num_p, num_p.lstrip("0")),
        ).fetchone()

    if r:
        data = dict(r)
        data["high_image_url"] = to_tcgdex_high_url(data.get("image_url"))
        return data
    return None


def search_cards_in_db(query: str, limit: int = 15) -> List[Dict[str, Any]]:
    """Searches canonical cards in pokemon_cards_meta.sqlite."""
    con = get_db_connection()
    if not con or not query.strip():
        return []

    q = f"%{query.strip()}%"
    sql = """
        SELECT tcgdex_id, lang, name, set_id, set_name, local_id, total_cards, card_fraction, rarity, hp, image_url
        FROM card_metadata
        WHERE tcgdex_id LIKE ? OR name LIKE ? OR local_id = ? OR set_id = ?
        ORDER BY 
            CASE WHEN local_id = ? THEN 0
                 WHEN tcgdex_id LIKE ? THEN 1
                 ELSE 2 END,
            name ASC
        LIMIT ?
    """
    clean_q = query.strip()
    rows = con.execute(sql, (q, q, clean_q, clean_q, clean_q, q, limit)).fetchall()
    return [dict(r) for r in rows]


def render_fake_placeholder():
    """Renders a stylish card-proportioned placeholder banner for counterfeit/fake cards."""
    st.markdown(
        """
        <div class="fake-placeholder">
            <span style="font-size: 42px;">🚫</span>
            <div class="fake-text">FAKE</div>
            <div class="fake-subtitle">Counterfeit / Fake Card</div>
            <div style="font-size: 11px; color: #fb7185; margin-top: 12px; border-top: 1px dashed #f43f5e; padding-top: 8px;">
                No canonical TCGdex counterpart.<br><b>Evaluation Target: Rejection</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_listing_info_box(sample: Dict[str, Any]):
    """Renders marketplace listing metadata box in matching unified dark card style."""
    title = html.escape(str(sample.get("title") or "Untitled"))
    cat = html.escape(str(sample.get("category") or "N/A"))
    fname = html.escape(str(sample.get("filename") or "N/A"))
    desc = html.escape(str(sample.get("description") or "").strip())
    desc_html = f'<div class="info-card-desc">{desc}</div>' if desc else '<div class="info-card-desc" style="color: #6b7280; font-style: italic;">No additional seller description provided.</div>'
    st.markdown(
        f"""
        <div class="info-card-box">
            <div class="info-card-title">📸 Marketplace Listing</div>
            <div class="info-card-field" title="{title}"><span class="info-card-label">Title:</span> <b>{title}</b></div>
            <div class="info-card-field"><span class="info-card-label">Category:</span> <code>{cat}</code></div>
            <div class="info-card-field" title="{fname}"><span class="info-card-label">File:</span> <code>{fname}</code></div>
            <div style="margin-top: 6px; font-weight: 600; color: #4b5563; font-size: 11px;">SELLER DESCRIPTION:</div>
            {desc_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tcgdex_info_box(card: Optional[Dict[str, Any]], sample: Optional[Dict[str, Any]] = None, is_fake: bool = False):
    """Renders canonical TCGdex details box in matching clean light card style."""
    if is_fake:
        st.markdown(
            """
            <div class="info-card-box" style="border-color: #fecdd3; background: #fff1f2;">
                <div class="info-card-title" style="color: #e11d48; border-color: #fecdd3;">🚫 Counterfeit Card (Fake)</div>
                <div class="info-card-field"><span class="info-card-label">Status:</span> <span class="badge-danger">IDENTIFIED FAKE</span></div>
                <div class="info-card-field"><span class="info-card-label">Canonical Counterpart:</span> None</div>
                <div class="info-card-field"><span class="info-card-label">Expected Action:</span> <code>Reject / Fake Detected</code></div>
                <div style="margin-top: 6px; font-weight: 600; color: #be123c; font-size: 11px;">AUDIT MOTIVATION:</div>
                <div class="info-card-desc" style="border-color: #fecdd3; background: #ffe4e6; color: #9f1239;">
                    Unauthorized proxy or counterfeit printing.<br><br>
                    <b>Target:</b> Pipeline must reject or flag as fake with 0 false positives on genuine cards.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if not card:
        exp_id = html.escape(str(sample.get("expected_tcgdex_id", ""))) if sample else ""
        st.markdown(
            f"""
            <div class="info-card-box" style="border-color: #fde68a; background: #fffbeb;">
                <div class="info-card-title" style="color: #d97706; border-color: #fde68a;">⚠️ Unindexed Card</div>
                <div class="info-card-field"><span class="info-card-label">Requested ID:</span> <code>{exp_id}</code></div>
                <div class="info-card-field"><span class="info-card-label">Status:</span> Pending indexation</div>
                <div style="margin-top: 6px; font-weight: 600; color: #b45309; font-size: 11px;">ACTIONS:</div>
                <div class="info-card-desc" style="border-color: #fde68a; background: #fef3c7; color: #92400e;">
                    No metadata found for identifier <b>{exp_id}</b> in SQLite.<br><br>
                    Use the database search tool below to link the canonical card or mark this sample as FAKE.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    c_name = html.escape(str(card.get('name') or ''))
    c_id = html.escape(str(card.get('tcgdex_id') or ''))
    s_name = html.escape(str(card.get('set_name') or ''))
    s_id = html.escape(str(card.get('set_id') or ''))
    local_id = html.escape(str(card.get('local_id') or ''))
    total = html.escape(str(card.get('total_cards') or '?'))
    frac = html.escape(str(card.get('card_fraction') or 'N/A'))
    hp = html.escape(str(card.get('hp') or 'N/A'))
    rarity = html.escape(str(card.get('rarity') or 'N/A'))
    evidence = html.escape(str(sample.get("audit_evidence", "")).strip()) if sample else ""
    evid_html = f'<div class="info-card-desc"><b>Audit Note:</b> {evidence}</div>' if evidence else '<div class="info-card-desc" style="color: #6b7280; font-style: italic;">No additional audit notes.</div>'

    st.markdown(
        f"""
        <div class="info-card-box">
            <div class="info-card-title">🎴 Canonical TCGdex Card</div>
            <div class="info-card-field" title="{c_name}"><span class="info-card-label">Name:</span> <b style="color: #1d4ed8;">{c_name}</b> (<code>{c_id}</code>)</div>
            <div class="info-card-field" title="{s_name}"><span class="info-card-label">Expansion:</span> {s_name} (<code>{s_id}</code>)</div>
            <div class="info-card-field"><span class="info-card-label">Collection:</span> <code>{local_id}</code> / {total} (Fraction: <code>{frac}</code>)</div>
            <div class="info-card-field"><span class="info-card-label">Stats:</span> HP: <b>{hp}</b> | Rarity: <b>{rarity}</b></div>
            <div style="margin-top: 6px; font-weight: 600; color: #4b5563; font-size: 11px;">GROUND TRUTH NOTES:</div>
            {evid_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Streamlit App Layout
# =============================================================================

st.title("🎴 Delta Cards — Dataset Manager & Evaluator")
st.caption("Manage Ground Truth via canonical TCGdex IDs or FAKE labels, side-by-side visual comparison, and 1-Click benchmark evaluation.")

tab_dataset, tab_eval, tab_detect, tab_search = st.tabs([
    "🗂️ Dataset Manager",
    "⚡ Benchmark Evaluator (1-Click)",
    "🔲 Card Detection & Crop (Adrihp06)",
    "🔍 TCGdex Database Explorer",
])


# =============================================================================
# TAB 1: DATASET MANAGER
# =============================================================================
with tab_dataset:
    dataset = load_ground_truth()

    # Metrics banner
    col1, col2, col3, col4, col5 = st.columns(5)
    cat_counts = defaultdict(int)
    for d in dataset:
        cat_counts[d.get("category", "other")] += 1

    col1.metric("Total Samples", len(dataset))
    col2.metric("Real Marketplace", cat_counts.get("real_wallapop", 0))
    col3.metric("Counterfeits (Fakes)", cat_counts.get("marketplace_fakes", 0))
    col4.metric("Multilingual (ES)", cat_counts.get("multilingual_es", 0))
    col5.metric("Reprints / Holos", cat_counts.get("multi_version_holo", 0))

    st.divider()

    # Filter & Search bar
    fcol1, fcol2, fcol3 = st.columns([2, 3, 2])
    available_cats = ["All"] + sorted(list(set(d.get("category", "") for d in dataset if d.get("category"))))
    selected_cat = fcol1.selectbox("Filter by Category", available_cats, index=0)
    search_text = fcol2.text_input("Search by ID, title, or filename", "").strip().lower()

    # Filter dataset
    filtered_indices = []
    for idx, d in enumerate(dataset):
        if selected_cat != "All" and d.get("category") != selected_cat:
            continue
        if search_text:
            text_pool = f"{d.get('sample_id', '')} {d.get('title', '')} {d.get('expected_tcgdex_id', '')} {d.get('filename', '')}".lower()
            if search_text not in text_pool:
                continue
        filtered_indices.append(idx)

    fcol3.info(f"Showing **{len(filtered_indices)}** of {len(dataset)} cards")

    # =========================================================================
    # ADD NEW SAMPLE EXPANDER
    # =========================================================================
    with st.expander("➕ Add New Sample to Dataset", expanded=False):
        st.subheader("Register New Evaluation Sample")
        ancol1, ancol2 = st.columns(2)

        with ancol1:
            new_title = st.text_input("Listing Title", key="new_title")
            new_desc = st.text_area("Description", "", key="new_desc")
            new_cat = st.selectbox("Category", [
                "real_wallapop",
                "marketplace_fakes",
                "multilingual_es",
                "multi_version_holo",
                "severe_glare_degradations",
                "perspective_angles",
            ], index=0, key="new_cat")
            upload_img = st.file_uploader("Upload Image (.jpg, .png, .webp)", type=["jpg", "jpeg", "png", "webp"])
            img_url = st.text_input("Or paste Marketplace Image CDN URL")

        with ancol2:
            st.markdown("**Ground Truth Target:**")
            st.caption("Provide canonical TCGdex ID (e.g. `base1-2`, `paf-127`) or click FAKE button.")
            
            gt_c1, gt_c2 = st.columns([3, 1])
            new_tcgdex = gt_c1.text_input("Canonical TCGdex ID", value="", key="new_tcgdex_id")
            is_new_fake = gt_c2.button("🚫 Set FAKE", key="new_fake_btn")
            if is_new_fake:
                new_tcgdex = "fake"

            new_evidence = st.text_input("Audit Evidence / Reason", placeholder="e.g. Card verified visually from base1 set #2/102", key="new_evidence")

            st.markdown("**TCGdex Card Search Assistant:**")
            search_assistant_query = st.text_input("Search canonical cards by name or set ID", key="new_search_db", placeholder="e.g. Blastoise or base1")
            if search_assistant_query:
                assistant_results = search_cards_in_db(search_assistant_query, limit=5)
                if assistant_results:
                    for ar in assistant_results:
                        ar_c1, ar_c2 = st.columns([3, 1])
                        ar_c1.write(f"**{ar['name']}** (`{ar['tcgdex_id']}`) — *{ar['set_name']}* #{ar['local_id']}")
                        if ar_c2.button("Select ID", key=f"sel_new_{ar['tcgdex_id']}"):
                            st.session_state["new_tcgdex_id"] = ar["tcgdex_id"]
                            st.rerun()

            if st.button("Save Sample to Dataset", type="primary"):
                final_id = new_tcgdex.strip()
                if not new_title:
                    st.error("Title is required.")
                elif not final_id:
                    st.error("Canonical TCGdex ID or FAKE label is required.")
                elif not upload_img and not img_url:
                    st.error("An image is required (upload file or provide URL).")
                else:
                    sid = f"sample_{len(dataset) + 1:03d}"
                    fname = f"{sid}_custom_{int(time.time())}.jpg"
                    save_path = os.path.join(IMAGES_DIR, fname)
                    saved_ok = False

                    if upload_img is not None:
                        with open(save_path, "wb") as f:
                            f.write(upload_img.read())
                        saved_ok = True
                    elif img_url:
                        try:
                            req = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
                            with urllib.request.urlopen(req, timeout=10) as resp:
                                with open(save_path, "wb") as f:
                                    f.write(resp.read())
                            saved_ok = True
                        except Exception as e:
                            st.error(f"Error downloading image: {e}")

                    if saved_ok:
                        derived_card = get_tcgdex_card_details(final_id) if final_id != "fake" else None
                        set_id = derived_card["set_id"] if derived_card else ("fake" if final_id == "fake" else final_id.split("-")[0])
                        local_id = str(derived_card["local_id"]) if derived_card else ("fake" if final_id == "fake" else (final_id.split("-")[1] if "-" in final_id else final_id))

                        new_item = {
                            "sample_id": sid,
                            "category": new_cat,
                            "filename": fname,
                            "expected_tcgdex_id": final_id,
                            "expected_set_id": set_id,
                            "expected_local_id": local_id,
                            "title": new_title,
                            "description": new_desc,
                            "audit_status": "manual_verified",
                            "audit_evidence": new_evidence or f"Ground truth set to {final_id}",
                        }
                        dataset.append(new_item)
                        if save_ground_truth(dataset):
                            st.success(f"Sample {sid} saved successfully with ground truth: {final_id}!")
                            st.rerun()

    st.write("### Visual Comparison & Dataset Audit")

    # Display cards with pagination
    samples_per_page = 20
    total_pages = max(1, (len(filtered_indices) + samples_per_page - 1) // samples_per_page)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start_idx = (page - 1) * samples_per_page
    end_idx = min(start_idx + samples_per_page, len(filtered_indices))

    for fi in filtered_indices[start_idx:end_idx]:
        sample = dataset[fi]
        sid = sample.get("sample_id", f"idx_{fi}")
        filename = sample.get("filename", "")
        exp_id = sample.get("expected_tcgdex_id", "")
        is_fake = (exp_id.lower() == "fake" or sample.get("category") == "marketplace_fakes")
        img_path = os.path.join(IMAGES_DIR, filename)

        # Retrieve canonical TCGdex card details
        tcgdex_card = get_tcgdex_card_details(exp_id) if not is_fake else None

        badge_txt = "🚫 FAKE" if is_fake else f"🎯 {exp_id}"
        expander_title = f"{sid} — {sample.get('title', 'Untitled')} | [{sample.get('category')}] | {badge_txt}"

        with st.expander(expander_title, expanded=False):
            # 3 Columns Layout:
            # Col 1: Marketplace Image
            # Col 2: Canonical Reference Image (TCGdex)
            # Col 3: Data of both & Edition (all collapsed)
            col_market, col_orig, col_data = st.columns([1, 1, 1.2])

            # 1. Marketplace Image
            with col_market:
                st.markdown("##### 📸 Marketplace Image")
                if os.path.exists(img_path):
                    st.image(img_path, width="stretch")
                else:
                    st.warning(f"Image file not found: {filename}")

            # 2. Canonical TCGdex Reference Image
            with col_orig:
                st.markdown("##### 🎴 Canonical Reference (TCGdex)")
                if is_fake:
                    render_fake_placeholder()
                elif tcgdex_card:
                    img_src = tcgdex_card.get("high_image_url")
                    if img_src:
                        st.image(img_src, width="stretch")
                    else:
                        st.info("No official high-res CDN image available.")
                else:
                    st.warning(f"No canonical card found for ID `{exp_id}` in database.")

            # 3. Data of both & Edition
            with col_data:
                st.markdown("##### 📋 Listing & Reference Details")
                if is_fake:
                    st.markdown("<span class='badge-danger'>🚫 GROUND TRUTH: FAKE</span>", unsafe_allow_html=True)
                elif tcgdex_card:
                    st.markdown(f"<span class='badge-success'>✅ GROUND TRUTH: {exp_id}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class='badge-warning'>⚠️ UNVERIFIED: {exp_id}</span>", unsafe_allow_html=True)

                st.caption(f"**Set / Local ID:** `{sample.get('expected_set_id')}` / `{sample.get('expected_local_id')}`")

                # Item 1: Marketplace Listing Details (collapsed)
                with st.expander("📸 Marketplace Listing Details", expanded=False):
                    render_listing_info_box(sample)

                # Item 2: Canonical TCGdex Reference Details (collapsed)
                with st.expander("🎴 Canonical TCGdex Details", expanded=False):
                    render_tcgdex_info_box(tcgdex_card, sample=sample, is_fake=is_fake)

                # Item 3: Ground Truth & Edition (collapsed)
                with st.expander("✏️ Ground Truth & Edition", expanded=False):
                    edit_title = st.text_input("Title", value=sample.get("title", ""), key=f"title_{fi}")
                    edit_tcgdex = st.text_input("Canonical TCGdex ID", value=sample.get("expected_tcgdex_id", ""), key=f"tcg_{fi}")
                    edit_evidence = st.text_input("Audit Evidence / Notes", value=sample.get("audit_evidence", ""), key=f"evid_{fi}")

                    # Database search assistant
                    st.markdown("**TCGdex Database Search Assistant:**")
                    db_query = st.text_input("Search canonical cards by name/set", key=f"dbsearch_{fi}", placeholder="e.g. Meowth 127 or PAF")
                    if db_query:
                        db_results = search_cards_in_db(db_query, limit=5)
                        if db_results:
                            for cr in db_results:
                                cr_c1, cr_c2 = st.columns([3, 1])
                                cr_c1.write(f"**{cr['name']}** (`{cr['tcgdex_id']}`) — *{cr['set_name']}* #{cr['local_id']}")
                                if cr_c2.button("Assign", key=f"apply_{fi}_{cr['tcgdex_id']}"):
                                    sample["expected_tcgdex_id"] = cr["tcgdex_id"]
                                    sample["expected_set_id"] = cr["set_id"]
                                    sample["expected_local_id"] = cr["local_id"]
                                    sample["audit_evidence"] = f"Visual audit: {cr['name']} ({cr['set_name']}) #{cr['local_id']}"
                                    save_ground_truth(dataset)
                                    st.success(f"Assigned to {cr['tcgdex_id']}!")
                                    st.rerun()

                    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                    # 3 Action Buttons aligned side-by-side at the same height
                    c_save, c_fake, c_del = st.columns(3)

                    if c_save.button("Save Changes", key=f"save_{fi}", type="primary", use_container_width=True):
                        cleaned_id = edit_tcgdex.strip()
                        sample["title"] = edit_title
                        sample["expected_tcgdex_id"] = cleaned_id
                        sample["audit_evidence"] = edit_evidence

                        if cleaned_id.lower() == "fake":
                            sample["expected_set_id"] = "fake"
                            sample["expected_local_id"] = "fake"
                            sample["category"] = "marketplace_fakes"
                        else:
                            meta = get_tcgdex_card_details(cleaned_id)
                            if meta:
                                sample["expected_set_id"] = meta["set_id"]
                                sample["expected_local_id"] = str(meta["local_id"])
                            elif "-" in cleaned_id:
                                sample["expected_set_id"] = cleaned_id.split("-")[0]
                                sample["expected_local_id"] = cleaned_id.split("-")[1]

                        if save_ground_truth(dataset):
                            st.success("Ground Truth updated successfully!")
                            st.rerun()

                    if c_fake.button("🚫 Mark FAKE", key=f"fakebtn_{fi}", use_container_width=True):
                        sample["expected_tcgdex_id"] = "fake"
                        sample["expected_set_id"] = "fake"
                        sample["expected_local_id"] = "fake"
                        sample["category"] = "marketplace_fakes"
                        sample["audit_evidence"] = "Manually marked as counterfeit (FAKE)"
                        save_ground_truth(dataset)
                        st.success("Sample marked as FAKE!")
                        st.rerun()

                    if c_del.button("🗑️ Delete", key=f"del_{fi}", use_container_width=True):
                        dataset.pop(fi)
                        if save_ground_truth(dataset):
                            st.warning("Sample deleted.")
                            st.rerun()


# =============================================================================
# TAB 2: SYSTEM BENCHMARK EVALUATOR (1-CLICK)
# =============================================================================
with tab_eval:
    st.subheader("⚡ Comprehensive Pipeline Benchmark (Adrihp06 + DINOv3 FGIR)")
    st.write("Execute photographic recognition on evaluation dataset samples with real-time visual comparison and full diagnostic trace (Top-1 visual retrieval with >= 70% threshold).")

    dataset = load_ground_truth()

    # Auto-load previous evaluation run if present
    if "last_eval_results" not in st.session_state and os.path.exists(LAST_EVAL_FILE):
        try:
            with open(LAST_EVAL_FILE, "r", encoding="utf-8") as f:
                saved_eval = json.load(f)
                st.session_state["last_eval_results"] = saved_eval.get("results", [])
                st.session_state["last_cat_stats"] = saved_eval.get("stats", {})
                st.session_state["last_eval_timestamp"] = saved_eval.get("timestamp", "")
        except Exception:
            pass

    eval_col1, eval_col2, eval_col3 = st.columns([2, 2, 2])
    eval_cat = eval_col1.selectbox(
        "Category to Evaluate",
        ["All Categories"] + sorted(list(set(d.get("category", "") for d in dataset if d.get("category")))),
        index=0,
        key="eval_cat_select",
    )
    eval_limit = eval_col2.number_input("Sample Limit (0 = all)", min_value=0, max_value=len(dataset), value=0, step=5)

    samples_to_eval = dataset
    if eval_cat != "All Categories":
        samples_to_eval = [d for d in dataset if d.get("category") == eval_cat]
    if eval_limit > 0:
        samples_to_eval = samples_to_eval[:eval_limit]

    eval_col3.write(f"**Selected Samples:** {len(samples_to_eval)}")
    if "last_eval_timestamp" in st.session_state and st.session_state["last_eval_timestamp"]:
        eval_col3.caption(f"Last benchmark saved: *{st.session_state['last_eval_timestamp']}*")

    use_llm_verifier = eval_col3.checkbox(
        "🛡️ Enable LLM Verification (Qwen2.5-1.5B)",
        value=False,
        help="Runs the 20-question checklist LLM verifier on all candidate card matches",
    )

    if st.button("▶️ Run 1-Click Benchmark Evaluation", type="primary", use_container_width=True):
        from delta_cards.recognition.pipeline import CardRecognitionPipeline

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        cat_stats = defaultdict(lambda: {"total": 0, "correct": 0, "latencies": []})
        eval_results = []

        status_text.info("Initializing Pipeline: Cardcaptor-v3 + DINOv3 + USearch + Multilingual RapidOCR...")
        pipe = CardRecognitionPipeline(use_fgir=True)

        llm_verifier = None
        if use_llm_verifier:
            from delta_cards.recognition.llm_verifier import get_llm_verifier
            status_text.info("Loading LLM False-Positive Verifier (Qwen2.5-1.5B-Instruct)...")
            llm_verifier = get_llm_verifier(min_confidence=60.0)

        t_start = time.perf_counter()

        for idx, item in enumerate(samples_to_eval, 1):
            sid = item["sample_id"]
            cat = item.get("category", "unknown")
            fname = item.get("filename", "")
            exp_id = item.get("expected_tcgdex_id", "")
            exp_norm = normalize_card_id(exp_id)
            img_path = os.path.join(IMAGES_DIR, fname)

            status_text.markdown(f"Evaluating sample **{idx}/{len(samples_to_eval)}** (`{sid}`): *{item.get('title', '')}*...")

            if not os.path.exists(img_path):
                eval_results.append({
                    "sample_id": sid,
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "audit_evidence": item.get("audit_evidence", ""),
                    "category": cat,
                    "filename": fname,
                    "expected_id": exp_id,
                    "predicted_id": "FILE_MISSING",
                    "correct": False,
                    "latency_ms": 0.0,
                    "method": "file_error",
                    "certainty": 0.0,
                    "ocr_metadata": {},
                })
                continue

            img = cv2.imread(img_path)
            t0 = time.perf_counter()

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                res = loop.run_until_complete(
                    pipe.process_card_listing(
                        img,
                        title=item.get("title", ""),
                        description=item.get("description", ""),
                    )
                )
            finally:
                loop.close()

            lat_ms = (time.perf_counter() - t0) * 1000.0

            pred_card = res.get("verified_card") or res.get("card")
            debug_trace = res.get("debug_trace", {})

            llm_info = {}
            if pred_card and llm_verifier is not None:
                status_text.markdown(f"Evaluating sample **{idx}/{len(samples_to_eval)}** (`{sid}`): *Checking with LLM Verifier...*")
                offer_info = {
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "price": item.get("price", 0.0),
                }
                tcgdex_info = {
                    "id": pred_card.get("id"),
                    "name": pred_card.get("name"),
                    "set_name": pred_card.get("set_name"),
                    "local_id": pred_card.get("local_id"),
                    "hp": pred_card.get("hp"),
                    "rarity": pred_card.get("rarity"),
                    "types": pred_card.get("types", []),
                }
                query_ocr = debug_trace.get("query_full_text", "")
                target_ocr = debug_trace.get("winner_ocr_text", "")
                llm_res = llm_verifier.verify_match(
                    offer_info=offer_info,
                    tcgdex_info=tcgdex_info,
                    query_ocr=query_ocr,
                    target_ocr=target_ocr,
                )
                llm_info = {
                    "llm_verified": llm_res.confirmed,
                    "llm_confidence": round(llm_res.confidence_score, 1),
                    "llm_verdict": llm_res.verdict,
                    "llm_reasoning": llm_res.reasoning,
                    "llm_answers": llm_res.answers,
                }
                if not llm_res.confirmed:
                    pred_card = None

            pred_id = pred_card.get("id") if pred_card else None
            pred_norm = normalize_card_id(pred_id) if pred_id else None

            # Determine success
            if exp_norm == "fake":
                is_correct = (pred_card is None)
            else:
                is_correct = (pred_norm == exp_norm)

            cat_stats[cat]["total"] += 1
            if is_correct:
                cat_stats[cat]["correct"] += 1
            cat_stats[cat]["latencies"].append(lat_ms)

            # Process and persist debug crops (YOLO crop, annotated original)
            debug_trace = res.get("debug_trace", {})
            card_img = debug_trace.get("card_image")
            annotated_img = debug_trace.get("annotated_image")

            yolo_crop_rel = None
            annotated_rel = None

            if card_img is not None:
                yolo_crop_rel = f"data/evaluation_debug/{sid}_yolo.jpg"
                cv2.imwrite(os.path.join(PROJECT_ROOT, yolo_crop_rel), card_img)
            if annotated_img is not None:
                annotated_rel = f"data/evaluation_debug/{sid}_annotated.jpg"
                cv2.imwrite(os.path.join(PROJECT_ROOT, annotated_rel), annotated_img)

            eval_results.append({
                "sample_id": sid,
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "audit_evidence": item.get("audit_evidence", ""),
                "category": cat,
                "filename": fname,
                "expected_id": exp_id,
                "predicted_id": pred_id or ("(Fake Rejected)" if is_correct and exp_norm == "fake" else "(Undetected / Fake)"),
                "correct": is_correct,
                "latency_ms": round(lat_ms, 1),
                "method": res.get("verification_method") or debug_trace.get("verification_method", "visual_retrieval"),
                "certainty": res.get("confidence") or debug_trace.get("certainty", 0.0),
                "profiling": res.get("profiling", {}),
                "isolation_status": debug_trace.get("isolation_status"),
                "yolo_crop_path": yolo_crop_rel,
                "annotated_path": annotated_rel,
                "query_full_text": debug_trace.get("query_full_text", ""),
                "query_top_text": debug_trace.get("query_top_text", ""),
                "ocr_lines": debug_trace.get("ocr_lines", []),
                "query_hp": debug_trace.get("query_hp"),
                "query_fractions": debug_trace.get("query_fractions", []),
                "query_set_codes": debug_trace.get("query_set_codes", []),
                "winner_language": debug_trace.get("winner_language", "en"),
                "candidates": [
                    {
                        **c,
                        "high_image_url": to_tcgdex_high_url(c.get("image_url")),
                    }
                    for c in debug_trace.get("candidates", [])
                ],
                **llm_info,
            })

            progress_bar.progress(float(idx) / len(samples_to_eval))

        total_time = time.perf_counter() - t_start
        status_text.success(f"Benchmark completed in {total_time:.2f}s!")

        # Save results in session_state and persist to disk
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["last_eval_results"] = eval_results
        st.session_state["last_cat_stats"] = dict(cat_stats)
        st.session_state["last_eval_timestamp"] = timestamp_str

        try:
            with open(LAST_EVAL_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": timestamp_str,
                    "results": eval_results,
                    "stats": dict(cat_stats),
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Error saving evaluation_last_run.json: {e}")

    # Render results if available in session_state
    if "last_eval_results" in st.session_state:
        results = st.session_state["last_eval_results"]
        stats = st.session_state["last_cat_stats"]

        total_tested = len(results)
        total_correct = sum(1 for r in results if r["correct"])
        acc_pct = (total_correct / total_tested * 100.0) if total_tested > 0 else 0.0
        avg_lat = sum(r["latency_ms"] for r in results) / total_tested if total_tested > 0 else 0.0

        st.divider()
        st.subheader("📊 Benchmark Results Summary")

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Overall Accuracy", f"{acc_pct:.1f}%", delta=f"{total_correct}/{total_tested} Matches")
        kpi2.metric("Mean Latency", f"{avg_lat:.0f} ms")
        kpi3.metric("Total Samples Tested", f"{total_tested}")

        # Table by category
        cat_rows = []
        for cat, s in stats.items():
            t = s["total"]
            c = s["correct"]
            c_acc = (c / t * 100.0) if t > 0 else 0.0
            c_lat = sum(s["latencies"]) / t if t > 0 else 0.0
            cat_rows.append({
                "Category": cat,
                "Samples": t,
                "Correct": c,
                "Accuracy (%)": f"{c_acc:.1f}%",
                "Mean Latency (ms)": f"{c_lat:.0f}",
            })
        st.table(pd.DataFrame(cat_rows))

        st.divider()
        st.subheader("🔍 Visual Diagnostic & Sample Comparison")

        view_filter = st.radio("Display:", ["Errors Only (Fails)", "All Samples"], horizontal=True)
        displayed_res = [r for r in results if not r["correct"]] if view_filter.startswith("Errors") else results

        st.write(f"Displaying **{len(displayed_res)}** samples")

        for r in displayed_res:
            img_p = os.path.join(IMAGES_DIR, r["filename"])
            exp_id = r["expected_id"]
            pred_id = r["predicted_id"]
            is_exp_fake = (exp_id.lower() == "fake" or r["category"] == "marketplace_fakes")
            is_pred_fake = (pred_id in ["(Fake Rejected)", "(Undetected / Fake)", "(No Detectada / Falsa)", None])

            exp_card = get_tcgdex_card_details(exp_id) if not is_exp_fake else None
            pred_card = get_tcgdex_card_details(pred_id) if (not is_pred_fake and pred_id) else None

            icon = "✅" if r["correct"] else "❌"
            exp_header = f"{icon} {r['sample_id']} — {r['title']} | Expected: {r['expected_id']} | Predicted: {r['predicted_id']} ({r['latency_ms']} ms)"

            with st.expander(exp_header, expanded=False):
                v_col1, v_col2, v_col3 = st.columns(3)

                # Column 1: Input photo (Marketplace)
                with v_col1:
                    st.markdown("##### 📸 Input Image (Marketplace)")
                    if os.path.exists(img_p):
                        st.image(img_p, width="stretch")
                    else:
                        st.warning("Image not found")
                    render_listing_info_box(r)

                # Column 2: Expected TCGdex card
                with v_col2:
                    st.markdown(f"##### 🎯 Expected: `{exp_id}`")
                    if is_exp_fake:
                        render_fake_placeholder()
                    elif exp_card:
                        img_s = exp_card.get("high_image_url")
                        if img_s:
                            st.image(img_s, width="stretch")
                        else:
                            st.info("No CDN photo")
                    else:
                        st.info(f"ID: `{exp_id}` (Not in DB)")
                    render_tcgdex_info_box(exp_card, sample={"expected_tcgdex_id": exp_id, "audit_evidence": r.get("audit_evidence", "")}, is_fake=is_exp_fake)

                # Column 3: Predicted card
                with v_col3:
                    st.markdown(f"##### 🤖 Predicted: `{pred_id}`")
                    if is_pred_fake:
                        render_fake_placeholder()
                    elif pred_card:
                        img_s = pred_card.get("high_image_url")
                        if img_s:
                            st.image(img_s, width="stretch")
                        else:
                            st.info(f"ID: `{pred_id}`")
                    else:
                        st.info(f"ID: `{pred_id}`")
                    render_tcgdex_info_box(pred_card, sample={"expected_tcgdex_id": pred_id}, is_fake=is_pred_fake)

                # Below: Quick actions & correction toolbar
                if not r["correct"]:
                    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                    st.warning("⚠️ Incorrect ground truth annotation? Correct it here:")
                    col_q_id, col_q_fake, col_q_save = st.columns([3, 1.5, 2])
                    suggested_fix = pred_id if (pred_id and not is_pred_fake) else ""
                    quick_gt = col_q_id.text_input("Correct TCGdex ID", value=suggested_fix, key=f"fix_{r['sample_id']}")

                    if col_q_fake.button("🚫 Mark FAKE", key=f"qfake_{r['sample_id']}", use_container_width=True):
                        for d in dataset:
                            if d["sample_id"] == r["sample_id"]:
                                d["expected_tcgdex_id"] = "fake"
                                d["expected_set_id"] = "fake"
                                d["expected_local_id"] = "fake"
                                d["category"] = "marketplace_fakes"
                                d["audit_evidence"] = "Marked as FAKE from evaluation suite"
                                break
                        save_ground_truth(dataset)
                        st.success("Marked as FAKE!")
                        st.rerun()

                    if col_q_save.button("Update Ground Truth", key=f"btnfix_{r['sample_id']}", type="primary", use_container_width=True):
                        clean_gt = quick_gt.strip()
                        meta = get_tcgdex_card_details(clean_gt) if clean_gt != "fake" else None
                        for d in dataset:
                            if d["sample_id"] == r["sample_id"]:
                                d["expected_tcgdex_id"] = clean_gt
                                if clean_gt == "fake":
                                    d["expected_set_id"] = "fake"
                                    d["expected_local_id"] = "fake"
                                    d["category"] = "marketplace_fakes"
                                elif meta:
                                    d["expected_set_id"] = meta["set_id"]
                                    d["expected_local_id"] = str(meta["local_id"])
                                elif "-" in clean_gt:
                                    d["expected_set_id"] = clean_gt.split("-")[0]
                                    d["expected_local_id"] = clean_gt.split("-")[1]
                                d["audit_evidence"] = f"Corrected via benchmark UI: expected {clean_gt}"
                                break
                        save_ground_truth(dataset)
                        st.success(f"Updated ground truth to {clean_gt}!")
                        st.rerun()

                # Detailed Step-by-Step Diagnostic Trace Expander
                with st.expander("🔬 Inspect Pipeline Trace (Visual Isolation, Micro-OCR & Fuzzy Rerank)", expanded=False):
                    # Step 1: YOLO Detection & Direct Crop
                    st.markdown("##### 1️⃣ Visual Detection & Direct Crop (Cardcaptor-v3)")
                    annotated_p = r.get("annotated_path")
                    yolo_p = r.get("yolo_crop_path")
                    ann_col, crop_col = st.columns(2)
                    with ann_col:
                        if annotated_p and os.path.exists(os.path.join(PROJECT_ROOT, annotated_p)):
                            st.image(os.path.join(PROJECT_ROOT, annotated_p), caption="🟢 Detected Card Polygon", width="stretch")
                        else:
                            st.info("No bbox annotation available.")
                    with crop_col:
                        if yolo_p and os.path.exists(os.path.join(PROJECT_ROOT, yolo_p)):
                            st.image(os.path.join(PROJECT_ROOT, yolo_p), caption="✂️ Rectified card crop", width="stretch")
                        else:
                            st.info("No crop available.")

                    # Step 2: Full Card OCR (RapidOCR Multilingual)
                    st.markdown("##### 2️⃣ Full Card OCR (RapidOCR Multilingüe: ES / EN / JA)")
                    full_txt = r.get("query_full_text", "")
                    ocr_lines = r.get("ocr_lines", [])
                    q_top = r.get("query_top_text") or (ocr_lines[0] if ocr_lines else "")
                    q_hp = r.get("query_hp") or r.get("query_top_hp")
                    q_frac = r.get("query_fractions", [])
                    q_sets = r.get("query_set_codes", [])
                    w_lang = (r.get("winner_language") or r.get("detected_language") or "").upper()

                    ocr_col1, ocr_col2 = st.columns([1, 2])
                    with ocr_col1:
                        st.markdown("**Metadatos Clave Detectados:**")
                        st.markdown(f"• **HP:** `{q_hp if q_hp else 'N/A'}`")
                        st.markdown(f"• **Colección / Fracción:** `{', '.join(q_frac) if q_frac else 'N/A'}`")
                        st.markdown(f"• **Códigos / Set / Promo:** `{', '.join(q_sets) if q_sets else 'N/A'}`")
                        if w_lang:
                            st.markdown(f"• **Idioma Carta:** `[{w_lang}]`")
                    with ocr_col2:
                        st.markdown("**Texto Extraído de la Carta Rectificada:**")
                        if full_txt:
                            st.text_area(
                                "OCR Completo",
                                full_txt,
                                height=110,
                                disabled=True,
                                key=f"full_ocr_txt_{r.get('sample_id', '')}",
                                label_visibility="collapsed",
                            )
                        else:
                            st.caption("(No se detectó texto con OCR)")

                    # Step 3: Top Visual Candidates & Multimodal Fuzzy Reranking
                    st.markdown("##### 3️⃣ Top Visual Candidates (Índice Unificado: 39.705 Cartas ES/EN/JA) & Fuzzy Reranking")
                    cands = r.get("candidates", [])
                    if cands:
                        num_cols = min(len(cands), 6)
                        cand_cols = st.columns(num_cols)
                        for ci, cand in enumerate(cands[:num_cols]):
                            with cand_cols[ci]:
                                sim_val = cand.get('similarity', 0.0)
                                comb_val = cand.get('combined_score', sim_val)
                                f_metrics = cand.get('fuzzy_metrics', {})
                                is_winner = (ci == 0 and comb_val >= 0.70)
                                badge = " 🏆 (Winner)" if is_winner else ""
                                cand_lang = (cand.get('lang') or 'en').upper()
                                st.markdown(f"**#{cand.get('rank')}** ({comb_val*100:.1f}%){badge}")
                                st.caption(f"Idioma: `[{cand_lang}]`")
                                if cand.get("high_image_url"):
                                    st.image(cand.get("high_image_url"), width="stretch")
                                text_sc = f_metrics.get('total_text_score', f_metrics.get('text_composite', 0.0))
                                name_sc = f_metrics.get('name_score', 0.0)
                                frac_sc = f_metrics.get('fraction_score', 0.0)
                                st.caption(
                                    f"**{cand.get('name')}**\n\n"
                                    f"`{cand.get('tcgdex_id')}`\n\n"
                                    f"Visual: **{sim_val*100:.1f}%**\n\n"
                                    f"Fuzzy Text: **{text_sc:.0f}%**\n\n"
                                    f"Name: {name_sc:.0f}% | Frac: {frac_sc:.0f}%"
                                )
                    else:
                        st.info("No candidates available.")

                    # Interactive inspector: What is being compared with what (field by field)
                    if cands:
                        with st.expander("🔎 Ver detalle: ¿Qué se está comparando con qué en cada candidato?", expanded=False):
                            st.markdown("###### 📝 Texto de Consulta (Extraído de la Foto vía OCR):")
                            q_c1, q_c2, q_c3 = st.columns(3)
                            q_c1.write(f"**Nombre OCR:** `{q_top or '(en texto completo)'}`")
                            q_c2.write(f"**HP detectado:** `{q_hp or 'N/A'}`")
                            q_c3.write(f"**Colección/Fracción:** `{', '.join(q_frac) if q_frac else 'N/A'}`")
                            if full_txt:
                                st.caption(f"**Texto OCR:** *{full_txt[:200]}...*")

                            st.markdown("###### 📊 Comparación Campo por Campo con los Candidatos de USearch:")
                            table_rows = []
                            for cand in cands:
                                fm = cand.get("fuzzy_metrics", {})
                                t_sc = fm.get("total_text_score", fm.get("text_composite", 0.0))
                                table_rows.append({
                                    "Rank": f"#{cand.get('rank')}",
                                    "Candidato TCGdex": f"{cand.get('name')} ({cand.get('tcgdex_id')} [{cand.get('lang', '').upper()}])",
                                    "Visual (DINOv3)": f"{cand.get('similarity', 0.0)*100:.1f}%",
                                    "Match Nombre": f"{fm.get('name_score', 0):.0f}%",
                                    "Match Fracción": f"{fm.get('fraction_score', 0):.0f}%",
                                    "Match HP": f"{fm.get('hp_score', 50):.0f}%",
                                    "Match Cuerpo/Ataques": f"{fm.get('full_text_sim', 0):.0f}%",
                                    "Fuzzy Total (Ponderado)": f"{t_sc:.1f}%",
                                    "Score Combinado": f"{cand.get('combined_score', 0)*100:.1f}%",
                                    "Método Decisión": cand.get("verification_method", ""),
                                })
                            st.dataframe(table_rows, hide_index=True, use_container_width=True)

                            # Candidate official scan OCR text snippets
                            st.markdown("###### 🖨️ OCR Extraído del Scan Oficial de cada Candidato:")
                            for cand in cands[:4]:
                                fm = cand.get("fuzzy_metrics", {})
                                rec_t = fm.get("cand_ocr_text") or fm.get("reconstructed_text") or cand.get("reconstructed_text")
                                c_l = (cand.get("lang") or "en").upper()
                                with st.expander(f"Scan OCR #{cand.get('rank')} — {cand.get('name')} ({cand.get('tcgdex_id')} [{c_l}])", expanded=False):
                                    if rec_t:
                                        st.code(rec_t, language=None)
                                    else:
                                        st.caption("(Metadatos básicos: nombre, set y número)")

                    # Step 4: LLM False-Positive Verification (if performed)
                    if r.get("llm_verdict"):
                        st.markdown("##### 4️⃣ LLM False-Positive Verification (Qwen2.5-1.5B 20-Question Checklist)")
                        llm_v = r.get("llm_verdict", "N/A")
                        llm_sc = r.get("llm_confidence", 0.0)
                        llm_c1, llm_c2 = st.columns([1, 2])
                        with llm_c1:
                            if r.get("llm_verified"):
                                st.success(f"**Verdict:** CONFIRMED ({llm_sc:.1f}%)")
                            else:
                                st.error(f"**Verdict:** REJECTED ({llm_sc:.1f}%)")
                        with llm_c2:
                            st.caption(f"**Reasoning:** {r.get('llm_reasoning', '')}")
                        if r.get("llm_answers"):
                            with st.expander("📋 View 20 Verification Questions & Answers", expanded=False):
                                st.table(pd.DataFrame(list(r["llm_answers"].items()), columns=["Question", "Answer"]))

                    # Step 5: Decision & Timing Breakdown
                    st.markdown("##### 5️⃣ Verification Decision & Latency Breakdown")
                    d_c1, d_c2 = st.columns(2)
                    with d_c1:
                        st.write(f"**Verification Method:** `{r.get('method')}`")
                        st.write(f"**Certainty Score:** `{r.get('certainty', 0.0):.4f}`")
                        st.write(f"**Benchmark Evaluation:** {'✅ CORRECT' if r.get('correct') else '❌ FAILED'}")
                    with d_c2:
                        prof = r.get("profiling", {})
                        if prof:
                            st.write(f"⏱️ **Isolation (YOLO):** {prof.get('isolation_s', 0)*1000:.1f} ms")
                            st.write(f"⏱️ **DINOv3 Embedding:** {prof.get('embedding_s', 0)*1000:.1f} ms")
                            st.write(f"⏱️ **USearch Search:** {prof.get('search_s', 0)*1000:.2f} ms")
                            if "ocr_s" in prof:
                                lang_label = r.get('detected_language', 'en').upper()
                                st.write(f"⏱️ **Full Card OCR ({lang_label}):** {prof.get('ocr_s', 0)*1000:.1f} ms")
                                st.write(f"⏱️ **TCGdex Text Reconstruct:** {prof.get('reconstruct_s', 0)*1000:.1f} ms")
                                st.write(f"⏱️ **Fuzzy Rerank:** {prof.get('matching_s', 0)*1000:.2f} ms")
                            else:
                                st.write(f"⏱️ **Micro-OCR & Fuzzy:** {prof.get('micro_ocr_s', 0)*1000:.1f} ms")
                            st.write(f"⏱️ **Total Pipeline:** {prof.get('total_s', 0)*1000:.1f} ms")


# =============================================================================
# TAB 3: ADRIHP06 CARD DETECTOR & CROP (DIRECT TO USEARCH)
# =============================================================================
@st.cache_resource(show_spinner="Loading Adrihp06 detector...")
def get_adrihp_detector():
    from delta_cards.recognition.visual_detector import VisualCardDetector
    return VisualCardDetector()


@st.cache_resource(show_spinner="Loading DINOv3 embedder...")
def get_dinov3_embedder():
    from delta_cards.recognition.dinov3_embedder import DINOv3Embedder
    return DINOv3Embedder()


@st.cache_resource(show_spinner="Loading USearch vector index...")
def get_vector_index():
    from delta_cards.recognition.vector_index import USearchCardIndex
    return USearchCardIndex(view_only=True)


with tab_detect:
    st.subheader("🔲 Adrihp06 Card Detection & Direct USearch Crop")
    st.caption("Official detector: Adrihp06/TCGscanner-detector (ONNX). Crops the card bounding box with zero rotation/dewarping and sends it directly to DINOv3 + USearch.")

    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        img_source = st.radio("Image Source", ["Evaluation Dataset", "Upload Custom Image"], horizontal=True, key="detect_img_src")
    with col_ctrl2:
        conf_threshold = st.slider("Confidence Threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05, key="detect_conf_slider")

    selected_image_pil = None
    sample_info_label = ""

    if img_source == "Evaluation Dataset":
        gt_data = load_ground_truth()
        if not gt_data:
            st.error("No samples found in ground_truth.json.")
        else:
            categories = ["All"] + sorted(list(set(d.get("category", "other") for d in gt_data)))
            filter_col1, filter_col2 = st.columns([1, 2])
            with filter_col1:
                selected_cat = st.selectbox("Filter by Category", categories, key="detect_cat_filter")

            filtered_samples = [d for d in gt_data if selected_cat == "All" or d.get("category") == selected_cat]
            sample_options = {
                f"{s.get('sample_id', i)} — {s.get('filename', '')} ({s.get('category', '')})": s
                for i, s in enumerate(filtered_samples)
            }

            with filter_col2:
                selected_sample_label = st.selectbox("Select Sample", list(sample_options.keys()), key="detect_sample_select")

            if selected_sample_label and selected_sample_label in sample_options:
                target_sample = sample_options[selected_sample_label]
                sample_filename = target_sample.get("filename")
                sample_img_path = os.path.join(IMAGES_DIR, sample_filename)
                if os.path.exists(sample_img_path):
                    selected_image_pil = Image.open(sample_img_path).convert("RGB")
                    sample_info_label = f"Sample: `{sample_filename}` | Category: `{target_sample.get('category')}`"
                else:
                    st.error(f"Image not found at {sample_img_path}")
    else:
        uploaded_file = st.file_uploader("Upload Card Image", type=["jpg", "jpeg", "png", "webp"], key="detect_upload")
        if uploaded_file is not None:
            try:
                selected_image_pil = Image.open(uploaded_file).convert("RGB")
                sample_info_label = f"Uploaded image: `{uploaded_file.name}`"
            except Exception as e:
                st.error(f"Error loading uploaded image: {e}")

    if selected_image_pil is not None:
        img_w, img_h = selected_image_pil.size
        st.write(f"🖼️ {sample_info_label} (Dimensions: **{img_w} × {img_h}** px)")

        detector = get_adrihp_detector()
        detector.conf_threshold = conf_threshold

        img_bgr = cv2.cvtColor(np.array(selected_image_pil), cv2.COLOR_RGB2BGR)
        t0 = time.perf_counter()
        bbox = detector.detect_card_bbox(img_bgr)
        det_time_ms = (time.perf_counter() - t0) * 1000

        if bbox is None:
            st.warning(f"No trading card detected with confidence >= {conf_threshold:.2f}")
        else:
            x1, y1, x2, y2, conf = bbox
            card_crop_bgr = img_bgr[y1:y2, x1:x2]
            card_crop_rgb = cv2.cvtColor(card_crop_bgr, cv2.COLOR_BGR2RGB)

            # Draw green box on original
            annotated_rgb = np.array(selected_image_pil).copy()
            cv2.rectangle(annotated_rgb, (x1, y1), (x2, y2), (16, 185, 129), 4)

            # Columns: 1) Detection Over Original, 2) Clean Rectangular Crop, 3) Direct USearch Match
            col_orig, col_crop, col_usearch = st.columns([1.2, 1.0, 1.0])

            with col_orig:
                st.markdown("### 1️⃣ Bounding Box Detección")
                st.caption(f"**Adrihp06** (`riftbound_regions.onnx`) | Conf: **{conf:.2f}** | BBox: `[{x1}, {y1}, {x2}, {y2}]` ({det_time_ms:.1f} ms)")
                st.image(annotated_rgb, width="stretch")

            with col_crop:
                st.markdown("### 2️⃣ Recorte Directo (Sin rotación)")
                st.caption(f"Dimensiones recorte: **{card_crop_rgb.shape[1]} × {card_crop_rgb.shape[0]}** px")
                st.image(card_crop_rgb, width="stretch")

            with col_usearch:
                st.markdown("### 3️⃣ Búsqueda USearch (Directa)")
                v_index = get_vector_index()
                if len(v_index) > 0:
                    embedder = get_dinov3_embedder()
                    t_emb0 = time.perf_counter()
                    q_vec = embedder.embed_image(card_crop_bgr)
                    cands = v_index.search(q_vec, top_k=1)
                    if isinstance(cands, tuple):
                        cands = cands[0]
                    search_ms = (time.perf_counter() - t_emb0) * 1000
                    if cands:
                        top = cands[0]
                        card_meta = top.get("card", {})
                        card_id = card_meta.get("tcgdex_id") or card_meta.get("id") or ""
                        lang_badge = f" [{card_meta.get('lang', '').upper()}]" if card_meta.get("lang") else ""
                        st.caption(f"Match Top-1: **{top.get('similarity', 0):.3f}** sim ({search_ms:.1f} ms)")
                        st.markdown(f"**{card_meta.get('name')}** (`{card_id}`){lang_badge}")
                        st.write(f"Set: {card_meta.get('set_name')} (#{card_meta.get('local_id')})")
                        cdn_url = to_tcgdex_high_url(card_meta.get("image_url"))
                        if cdn_url:
                            st.image(cdn_url, width="stretch")
                    else:
                        st.info("No candidates returned from index.")
                else:
                    st.info("USearch vector index is empty or not built.")
    else:
        st.info("Select a sample from the dataset or upload an image to view card detections.")


# =============================================================================
# TAB 4: TCGDEX DATABASE EXPLORER (30,000+ CARDS)
# =============================================================================
with tab_search:
    st.subheader("🔍 TCGdex Database Explorer (30,000+ Cards)")
    st.write("Query and inspect canonical cards indexed in SQLite (`pokemon_cards_meta.sqlite`).")

    search_query = st.text_input("Search by card name, set ID or collection number (e.g. 'Meowth', 'paf', '127', 'Charizard')", key="tcgdex_global_search")

    if search_query:
        results = search_cards_in_db(search_query, limit=20)
        st.write(f"Found **{len(results)}** cards:")

        for r in results:
            lang_badge = f"[{r.get('lang', 'en').upper()}]" if r.get("lang") else ""
            with st.expander(f"{r['name']} — `{r['tcgdex_id']}` {lang_badge} ({r['set_name']} #{r['local_id']})", expanded=False):
                s_c1, s_c2 = st.columns([1, 4])
                with s_c1:
                    high_cdn_url = to_tcgdex_high_url(r.get("image_url"))
                    if high_cdn_url:
                        st.image(high_cdn_url, width=140)
                    else:
                        st.info("No CDN photo")
                with s_c2:
                    st.markdown(f"### {r['name']} — `{r['tcgdex_id']}`")
                    st.write(f"**Expansion:** {r['set_name']} (`{r['set_id']}`)")
                    st.write(f"**Collection Number:** `{r['local_id']}` / Total: `{r['total_cards']}` (Fraction: `{r['card_fraction']}`)")
                    if r.get("hp"):
                        st.write(f"**HP:** {r['hp']} | **Rarity:** {r.get('rarity')}")
