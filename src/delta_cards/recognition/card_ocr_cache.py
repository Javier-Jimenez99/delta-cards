"""On-demand OCR extraction and SQLite caching for candidate cards.

When USearch retrieves the nearest visual neighbor candidates, this service extracts
the full OCR text directly from the candidate's clean official scan image and caches
the result in SQLite so it is only computed once per card across the system.
"""

import json
import logging
import os
import sqlite3
import threading
import urllib.request
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .card_ocr import CardOCR, CardOCRResult

logger = logging.getLogger(__name__)


class CandidateOCRCache:
    """Manages on-demand OCR extraction and persistent SQLite caching for candidate cards."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        cache_dir: str = "data/images_cache",
        ocr: Optional[CardOCR] = None,
    ):
        """Initializes the candidate OCR cache service.

        Args:
            db_path: Path to the SQLite database (defaults to data/pokemon_cards_meta.sqlite).
            cache_dir: Root directory for cached card images (defaults to data/images_cache).
            ocr: CardOCR instance (initialized lazily if None).
        """
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
        """Ensures the candidate_ocr_cache table exists in SQLite."""
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS candidate_ocr_cache (
                    tcgdex_id TEXT NOT NULL,
                    lang TEXT NOT NULL,
                    full_text TEXT NOT NULL,
                    top_text TEXT NOT NULL,
                    bottom_text TEXT NOT NULL,
                    hp INTEGER,
                    fractions_json TEXT NOT NULL,
                    promo_codes_json TEXT NOT NULL,
                    lines_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (tcgdex_id, lang)
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cand_ocr_lookup ON candidate_ocr_cache (tcgdex_id, lang);"
            )

    def get_cached_ocr(self, tcgdex_id: str, lang: str = "en") -> Optional[Dict[str, Any]]:
        """Retrieves cached OCR result for (tcgdex_id, lang) from SQLite."""
        clean_lang = (lang or "en").lower()
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT * FROM candidate_ocr_cache WHERE tcgdex_id = ? AND lang = ?",
                    (tcgdex_id, clean_lang),
                ).fetchone()
                if not row:
                    return None
                return {
                    "tcgdex_id": row["tcgdex_id"],
                    "lang": row["lang"],
                    "full_text": row["full_text"],
                    "top_text": row["top_text"],
                    "bottom_text": row["bottom_text"],
                    "hp": row["hp"],
                    "fractions": json.loads(row["fractions_json"]) if row["fractions_json"] else [],
                    "promo_codes": json.loads(row["promo_codes_json"]) if row["promo_codes_json"] else [],
                    "lines": json.loads(row["lines_json"]) if row["lines_json"] else [],
                    "reconstructed_text": row["full_text"],
                }
        except Exception as e:
            logger.debug(f"Error reading candidate OCR cache for {tcgdex_id} ({clean_lang}): {e}")
            return None

    def get_cached_ocr_ids(self, lang: Optional[str] = None) -> set:
        """Returns the set of tcgdex_ids already cached in candidate_ocr_cache."""
        try:
            with self._get_conn() as conn:
                if lang:
                    rows = conn.execute(
                        "SELECT tcgdex_id FROM candidate_ocr_cache WHERE lang = ?",
                        (lang.lower(),),
                    ).fetchall()
                else:
                    rows = conn.execute("SELECT tcgdex_id FROM candidate_ocr_cache").fetchall()
                return {r["tcgdex_id"] for r in rows}
        except Exception as e:
            logger.debug(f"Error querying cached OCR IDs: {e}")
            return set()

    def save_cached_ocr_batch(
        self,
        items: List[Tuple[str, str, CardOCRResult]],
    ) -> None:
        """Batch saves multiple OCR extraction results in a single transaction."""
        if not items:
            return
        params = [
            (
                tcgdex_id,
                (lang or "en").lower(),
                ocr_res.full_text,
                ocr_res.top_text,
                ocr_res.bottom_text,
                ocr_res.hp,
                json.dumps(ocr_res.fractions, ensure_ascii=False),
                json.dumps(ocr_res.promo_codes, ensure_ascii=False),
                json.dumps(ocr_res.lines, ensure_ascii=False),
            )
            for tcgdex_id, lang, ocr_res in items
        ]
        try:
            with self._get_conn() as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO candidate_ocr_cache (
                        tcgdex_id, lang, full_text, top_text, bottom_text,
                        hp, fractions_json, promo_codes_json, lines_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    params,
                )
        except Exception as e:
            logger.warning(f"Failed to batch persist candidate OCR: {e}")

    def save_cached_ocr(
        self,
        tcgdex_id: str,
        lang: str,
        ocr_result: CardOCRResult,
    ) -> None:
        """Saves an OCR extraction result into the SQLite cache."""
        clean_lang = (lang or "en").lower()
        try:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO candidate_ocr_cache (
                        tcgdex_id, lang, full_text, top_text, bottom_text,
                        hp, fractions_json, promo_codes_json, lines_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tcgdex_id,
                        clean_lang,
                        ocr_result.full_text,
                        ocr_result.top_text,
                        ocr_result.bottom_text,
                        ocr_result.hp,
                        json.dumps(ocr_result.fractions, ensure_ascii=False),
                        json.dumps(ocr_result.promo_codes, ensure_ascii=False),
                        json.dumps(ocr_result.lines, ensure_ascii=False),
                    ),
                )
        except Exception as e:
            logger.warning(f"Failed to persist candidate OCR for {tcgdex_id} ({clean_lang}): {e}")

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

        # Candidate filenames to check on disk
        candidates_to_try = [
            f"{set_id}_{local_id}",
            f"{set_id}_{local_id.lstrip('0')}",
            f"{set_id}_{local_id.zfill(3)}",
            f"{set_id}_{local_id.zfill(4)}",
        ]

        # 1. Check local images_cache/{lang}
        for stem in candidates_to_try:
            for ext in (".webp", ".png", ".jpg", ".jpeg"):
                p = os.path.join(self.cache_dir, clean_lang, f"{stem}{ext}")
                if os.path.isfile(p) and os.path.getsize(p) > 1000:
                    return p

        # 2. Check across other languages in cache_dir as visual fallback
        for alt_lang in ("en", "es", "ja"):
            if alt_lang == clean_lang:
                continue
            for stem in candidates_to_try:
                for ext in (".webp", ".png", ".jpg", ".jpeg"):
                    p = os.path.join(self.cache_dir, alt_lang, f"{stem}{ext}")
                    if os.path.isfile(p) and os.path.getsize(p) > 1000:
                        return p

        # 3. If missing from disk, download high-res scan from image_url
        image_url = card_info.get("image_url") or card_info.get("image")
        if image_url:
            dest_dir = os.path.join(self.cache_dir, clean_lang)
            os.makedirs(dest_dir, exist_ok=True)
            dest_file = os.path.join(dest_dir, f"{set_id}_{local_id}.webp")

            # Try direct high resolution webp
            urls_to_try = [
                image_url,
                image_url.replace("/low.", "/high."),
                f"{image_url.rstrip('/')}/high.webp" if not image_url.endswith(".webp") else image_url,
            ]

            opener = urllib.request.build_opener()
            opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DeltaCards/2.0")]
            for url in urls_to_try:
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 DeltaCards/2.0"})
                    with opener.open(req, timeout=8) as resp:
                        if resp.status == 200:
                            content = resp.read()
                            if len(content) > 1000:
                                with open(dest_file, "wb") as f:
                                    f.write(content)
                                logger.info(f"Downloaded missing candidate image to {dest_file} ({len(content)} bytes)")
                                return dest_file
                except Exception:
                    continue

        return None

    def get_candidate_ocr(self, card_info: Dict[str, Any], lang: Optional[str] = None) -> Dict[str, Any]:
        """Gets OCR text for a candidate card, computing and caching it on-demand if needed."""
        tcgdex_id = card_info.get("tcgdex_id") or card_info.get("id") or ""
        clean_lang = (lang or card_info.get("lang") or "en").lower()

        if not tcgdex_id:
            return {
                "tcgdex_id": "",
                "lang": clean_lang,
                "full_text": "",
                "top_text": "",
                "bottom_text": "",
                "hp": None,
                "fractions": [],
                "promo_codes": [],
                "lines": [],
                "name": card_info.get("name", ""),
                "local_id": card_info.get("local_id", ""),
                "total_cards": card_info.get("total_cards", ""),
                "card_fraction": card_info.get("card_fraction", ""),
                "reconstructed_text": card_info.get("name", ""),
            }

        # 1. Fast path: check SQLite cache
        cached = self.get_cached_ocr(tcgdex_id, clean_lang)
        if cached:
            # Augment with basic metadata
            cached["name"] = card_info.get("name") or cached.get("name", "")
            cached["local_id"] = card_info.get("local_id") or cached.get("local_id", "")
            cached["total_cards"] = card_info.get("total_cards") or cached.get("total_cards", "")
            fractions = cached.get("fractions") or []
            fallback_frac = fractions[0][2] if fractions and len(fractions[0]) > 2 else ""
            cached["card_fraction"] = card_info.get("card_fraction") or cached.get("card_fraction") or fallback_frac
            return cached

        # 2. Slow path (only first time): run OCR on the clean scan
        img_path = self.resolve_candidate_image_path(card_info, clean_lang)
        if img_path and os.path.exists(img_path):
            try:
                card_img = cv2.imread(img_path)
                if card_img is not None and card_img.size > 0:
                    with self._lock:
                        # Double check cache inside lock
                        cached_again = self.get_cached_ocr(tcgdex_id, clean_lang)
                        if cached_again:
                            cached_again["name"] = card_info.get("name") or cached_again.get("name", "")
                            cached_again["local_id"] = card_info.get("local_id") or cached_again.get("local_id", "")
                            cached_again["total_cards"] = card_info.get("total_cards") or cached_again.get("total_cards", "")
                            cached_again["card_fraction"] = card_info.get("card_fraction") or cached_again.get("card_fraction", "")
                            return cached_again

                        ocr_res = self.ocr.extract(card_img)
                        self.save_cached_ocr(tcgdex_id, clean_lang, ocr_res)

                    return {
                        "tcgdex_id": tcgdex_id,
                        "lang": clean_lang,
                        "full_text": ocr_res.full_text,
                        "top_text": ocr_res.top_text,
                        "bottom_text": ocr_res.bottom_text,
                        "hp": ocr_res.hp,
                        "fractions": ocr_res.fractions,
                        "promo_codes": ocr_res.promo_codes,
                        "lines": ocr_res.lines,
                        "name": card_info.get("name", ""),
                        "local_id": card_info.get("local_id", ""),
                        "total_cards": card_info.get("total_cards", ""),
                        "card_fraction": card_info.get("card_fraction", ""),
                        "reconstructed_text": ocr_res.full_text,
                    }
            except Exception as exc:
                logger.warning(f"Error running OCR on candidate {tcgdex_id} ({img_path}): {exc}")

        # 3. Fallback: return metadata representation if image cannot be read
        fb_text = f"{card_info.get('name', '')} {card_info.get('local_id', '')}/{card_info.get('total_cards', '')}".strip()
        return {
            "tcgdex_id": tcgdex_id,
            "lang": clean_lang,
            "full_text": fb_text,
            "top_text": card_info.get("name", ""),
            "bottom_text": "",
            "hp": card_info.get("hp"),
            "fractions": [],
            "promo_codes": [],
            "lines": [card_info.get("name", "")],
            "name": card_info.get("name", ""),
            "local_id": card_info.get("local_id", ""),
            "total_cards": card_info.get("total_cards", ""),
            "card_fraction": card_info.get("card_fraction", ""),
            "reconstructed_text": fb_text,
        }

    def get_batch_candidate_ocr(
        self,
        candidates: List[Union[str, Dict[str, Any]]],
        lang: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Batch-retrieves OCR for candidate cards, caching newly seen cards."""
        results_map = {}
        for cand in candidates:
            if isinstance(cand, dict):
                card_info = cand.get("card", cand)
                cid = card_info.get("tcgdex_id") or card_info.get("id")
                c_lang = (card_info.get("lang") or lang or "en").lower()
            else:
                cid = str(cand)
                c_lang = (lang or "en").lower()
                card_info = self._get_metadata_by_id(cid, c_lang) or {"tcgdex_id": cid}

            if not cid:
                continue

            ocr_data = self.get_candidate_ocr(card_info, lang=c_lang)
            results_map[f"{cid}_{c_lang}"] = ocr_data
            if cid not in results_map:
                results_map[cid] = ocr_data

        return results_map

    def _get_metadata_by_id(self, tcgdex_id: str, lang: str = "en") -> Optional[Dict[str, Any]]:
        """Helper to fetch card metadata from SQLite card_metadata table if available."""
        clean_lang = (lang or "en").lower()
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT * FROM card_metadata WHERE tcgdex_id = ? AND lang = ?",
                    (tcgdex_id, clean_lang),
                ).fetchone()
                if not row:
                    row = conn.execute(
                        "SELECT * FROM card_metadata WHERE tcgdex_id = ?",
                        (tcgdex_id,),
                    ).fetchone()
                if row:
                    return {
                        "tcgdex_id": row["tcgdex_id"],
                        "lang": row["lang"],
                        "name": row["name"],
                        "set_id": row["set_id"],
                        "set_name": row["set_name"],
                        "local_id": row["local_id"],
                        "total_cards": row["total_cards"],
                        "card_fraction": row["card_fraction"],
                        "hp": row["hp"],
                        "image_url": row["image_url"],
                    }
        except Exception as e:
            logger.debug(f"Could not load metadata for {tcgdex_id} from SQLite: {e}")

        # Fallback: query TCGdex SDK online
        try:
            from tcgdexsdk import TCGdex
            card = TCGdex(clean_lang).card.getSync(tcgdex_id)
            if card:
                set_obj = getattr(card, "set", None)
                set_id = getattr(set_obj, "id", "") if set_obj else ""
                set_name = getattr(set_obj, "name", "") if set_obj else ""
                card_count = getattr(set_obj, "cardCount", None) if set_obj else None
                total_cards = str(getattr(card_count, "official", "") or "") if card_count else ""
                local_id = str(getattr(card, "localId", "") or "")
                card_fraction = f"{local_id}/{total_cards}" if local_id and total_cards else ""
                return {
                    "tcgdex_id": tcgdex_id,
                    "lang": clean_lang,
                    "name": getattr(card, "name", ""),
                    "set_id": set_id,
                    "set_name": set_name,
                    "local_id": local_id,
                    "total_cards": total_cards,
                    "card_fraction": card_fraction,
                    "hp": getattr(card, "hp", None),
                    "image_url": getattr(card, "image", "") or "",
                }
        except Exception as e:
            logger.debug(f"Could not fetch metadata for {tcgdex_id} from tcgdexsdk: {e}")

        return None

    async def get_reconstructed_card(
        self,
        tcgdex_id: str,
        lang: str = "es",
        card_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Asynchronously retrieves candidate card OCR text, resolving metadata if needed."""
        clean_lang = (lang or "en").lower()
        if card_info is None:
            card_info = self._get_metadata_by_id(tcgdex_id, clean_lang) or {
                "tcgdex_id": tcgdex_id,
                "lang": clean_lang,
            }
        import asyncio
        return await asyncio.to_thread(self.get_candidate_ocr, card_info, clean_lang)

    async def get_reconstructed_cards_batch(
        self,
        candidate_ids: Union[List[str], List[Dict[str, Any]]],
        lang: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Asynchronously batch-fetches candidate card OCRs."""
        import asyncio
        return await asyncio.to_thread(self.get_batch_candidate_ocr, candidate_ids, lang)
