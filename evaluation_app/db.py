"""Database and data access helpers for Delta Cards Evaluation App."""

import json
import os
import shutil
import sqlite3
import streamlit as st
from typing import Any, Dict, List, Optional

from evaluation_app.config import GROUND_TRUTH_PATH, SQLITE_PATH


@st.cache_resource
def get_db_connection() -> Optional[sqlite3.Connection]:
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
