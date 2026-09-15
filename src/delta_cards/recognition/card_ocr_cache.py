"""Persistent full-text OCR cache for candidate card scans."""

import os
import sqlite3
import threading
import urllib.request
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .card_ocr import CardOCR


class CandidateOCRCache:
    """Caches only the complete OCR text of official scans."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        cache_dir: str = "data/images_cache",
        ocr: Optional[CardOCR] = None,
    ):
        self.db_path = os.path.abspath(db_path or os.path.join("data", "pokemon_cards_meta.sqlite"))
        self.cache_dir = os.path.abspath(cache_dir)
        self._ocr = ocr
        self._lock = threading.Lock()
        self._init_db()

    @property
    def ocr(self) -> CardOCR:
        if self._ocr is None:
            self._ocr = CardOCR()
        return self._ocr

    def _get_conn(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS candidate_ocr_cache (
                    tcgdex_id TEXT NOT NULL,
                    lang TEXT NOT NULL,
                    full_text TEXT NOT NULL,
                    PRIMARY KEY (tcgdex_id, lang)
                );
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(candidate_ocr_cache)")}
            if columns != {"tcgdex_id", "lang", "full_text"}:
                rows = conn.execute("SELECT tcgdex_id, lang, full_text FROM candidate_ocr_cache").fetchall()
                conn.execute("DROP TABLE candidate_ocr_cache")
                conn.execute("CREATE TABLE candidate_ocr_cache (tcgdex_id TEXT NOT NULL, lang TEXT NOT NULL, full_text TEXT NOT NULL, PRIMARY KEY (tcgdex_id, lang))")
                conn.executemany("INSERT INTO candidate_ocr_cache VALUES (?, ?, ?)", rows)

    def get_cached_text(self, tcgdex_id: str, lang: str = "en") -> Optional[str]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT full_text FROM candidate_ocr_cache WHERE tcgdex_id = ? AND lang = ?", (tcgdex_id, lang.lower())).fetchone()
        return row["full_text"] if row else None

    def save_text(self, tcgdex_id: str, lang: str, full_text: str) -> None:
        with self._get_conn() as conn:
            conn.execute("INSERT OR REPLACE INTO candidate_ocr_cache VALUES (?, ?, ?)", (tcgdex_id, lang.lower(), full_text))

    def resolve_candidate_image_path(self, card_info: Dict[str, Any], lang: str) -> Optional[str]:
        """Resolves local filepath for candidate card image, downloading if absent."""
        clean_lang = (lang or card_info.get("lang") or "en").lower()
        set_id = str(card_info.get("set_id") or "").lower()
        local_id = str(card_info.get("local_id") or "").strip()
        tcgdex_id = str(card_info.get("tcgdex_id") or card_info.get("id") or "")

        if not set_id and "-" in tcgdex_id:
            set_id = tcgdex_id.split("-", 1)[0].lower()
            if not local_id:
                local_id = tcgdex_id.split("-", 1)[1]

        candidates_to_try = [
            f"{set_id}_{local_id}",
            f"{set_id}_{local_id.lstrip('0')}",
            f"{set_id}_{local_id.zfill(3)}",
            f"{set_id}_{local_id.zfill(4)}",
        ]

        for stem in candidates_to_try:
            for ext in (".webp", ".png", ".jpg", ".jpeg"):
                p = os.path.join(self.cache_dir, clean_lang, f"{stem}{ext}")
                if os.path.isfile(p) and os.path.getsize(p) > 1000:
                    return p

        image_url = card_info.get("image_url") or card_info.get("image")
        if image_url:
            dest_dir = os.path.join(self.cache_dir, clean_lang)
            os.makedirs(dest_dir, exist_ok=True)
            dest_file = os.path.join(dest_dir, f"{set_id}_{local_id}.webp")

            try:
                with urllib.request.urlopen(f"{image_url.rstrip('/')}/high.webp", timeout=8) as response:
                    content = response.read()
                with open(dest_file, "wb") as image_file:
                    image_file.write(content)
                return dest_file
            except Exception:
                pass

        return None

    def get_candidate_text(self, card_info: Dict[str, str]) -> str:
        tcgdex_id = card_info.get("tcgdex_id") or card_info.get("id") or ""
        clean_lang = (card_info.get("lang") or "en").lower()
        if not tcgdex_id:
            return ""
        cached = self.get_cached_text(tcgdex_id, clean_lang)
        if cached is not None:
            return cached
        img_path = self.resolve_candidate_image_path(card_info, clean_lang)
        if img_path and os.path.exists(img_path):
            with self._lock:
                cached = self.get_cached_text(tcgdex_id, clean_lang)
                if cached is not None:
                    return cached
                text = self.ocr.extract(cv2.imread(img_path)).full_text
                self.save_text(tcgdex_id, clean_lang, text)
                return text
        return ""

    def get_candidate_texts(self, candidates: List[Dict[str, str]]) -> Dict[str, str]:
        texts = {}
        for candidate in candidates:
            card = candidate.get("card", candidate)
            key = f"{card.get('tcgdex_id', '')}_{(card.get('lang') or 'en').lower()}"
            texts[key] = self.get_candidate_text(card)
        return texts
