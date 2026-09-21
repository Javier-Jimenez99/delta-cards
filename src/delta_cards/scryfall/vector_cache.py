"""Persistent SQLite cache for Scryfall card DINOv3 embeddings."""

import logging
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SCRYFALL_CACHE_DB = os.path.join("data", "scryfall_vectors_cache.sqlite")
EMBEDDING_DIM = 384


def get_scryfall_image_url(card_data: Dict[str, Any]) -> Optional[str]:
    """Extracts the best available card scan image URL from a Scryfall card dictionary.

    Supports both single-faced and double-faced / transform cards.
    """
    if not card_data or not isinstance(card_data, dict):
        return None

    image_uris = card_data.get("image_uris") or {}
    url = (
        image_uris.get("normal")
        or image_uris.get("large")
        or image_uris.get("png")
        or image_uris.get("small")
    )
    if not url and isinstance(card_data.get("card_faces"), list) and card_data["card_faces"]:
        first_face = card_data["card_faces"][0]
        if isinstance(first_face, dict):
            face_uris = first_face.get("image_uris") or {}
            url = (
                face_uris.get("normal")
                or face_uris.get("large")
                or face_uris.get("png")
                or face_uris.get("small")
            )
    return url


class ScryfallVectorCache:
    """Manages persistent SQLite storage and fast memory caching for Scryfall DINOv3 vectors."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = os.path.abspath(db_path or DEFAULT_SCRYFALL_CACHE_DB)
        self._lock = threading.Lock()
        self._mem_cache: Dict[str, np.ndarray] = {}
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        """Initializes the database schema if not already present."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS scryfall_card_vectors (
                        scryfall_id TEXT PRIMARY KEY,
                        name TEXT,
                        set_code TEXT,
                        collector_number TEXT,
                        image_url TEXT,
                        embedding BLOB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_scryfall_vectors_set_num
                    ON scryfall_card_vectors (set_code, collector_number);
                    """
                )

    def get_vector(self, scryfall_id: str) -> Optional[np.ndarray]:
        """Retrieves a cached DINOv3 vector by Scryfall card ID."""
        if not scryfall_id:
            return None

        clean_id = str(scryfall_id).strip()
        if clean_id in self._mem_cache:
            return self._mem_cache[clean_id]

        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT embedding FROM scryfall_card_vectors WHERE scryfall_id = ?",
                    (clean_id,),
                ).fetchone()
                if row and row["embedding"]:
                    vec = np.frombuffer(row["embedding"], dtype=np.float32).copy()
                    if vec.shape == (EMBEDDING_DIM,):
                        self._mem_cache[clean_id] = vec
                        return vec
        except Exception as exc:
            logger.debug(f"Error reading vector for Scryfall ID '{clean_id}': {exc}")

        return None

    def save_vector(
        self,
        scryfall_id: str,
        embedding: np.ndarray,
        name: Optional[str] = None,
        set_code: Optional[str] = None,
        collector_number: Optional[str] = None,
        image_url: Optional[str] = None,
    ) -> None:
        """Persists a DINOv3 vector into SQLite and in-memory cache."""
        if not scryfall_id or embedding is None:
            return

        clean_id = str(scryfall_id).strip()
        vec = np.asarray(embedding, dtype=np.float32)
        if vec.shape != (EMBEDDING_DIM,):
            raise ValueError(f"Expected embedding of shape ({EMBEDDING_DIM},), got {vec.shape}")

        blob = vec.tobytes()
        clean_set = set_code.lower().strip() if set_code else None
        clean_num = str(collector_number).lower().strip() if collector_number else None

        with self._lock:
            self._mem_cache[clean_id] = vec
            try:
                with self._get_conn() as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO scryfall_card_vectors
                        (scryfall_id, name, set_code, collector_number, image_url, embedding)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (clean_id, name, clean_set, clean_num, image_url, blob),
                    )
            except Exception as exc:
                logger.warning(f"Error saving vector for Scryfall ID '{clean_id}': {exc}")

    def get_or_compute_vector(
        self,
        card_data: Dict[str, Any],
        embedder: Any,
    ) -> Optional[np.ndarray]:
        """Retrieves vector from cache or computes it with DINOv3 from card image and caches it."""
        if not card_data or not isinstance(card_data, dict):
            return None

        card_id = card_data.get("id")
        if card_id:
            cached = self.get_vector(card_id)
            if cached is not None:
                return cached

        image_url = get_scryfall_image_url(card_data)
        if not image_url:
            return None

        try:
            vec = embedder.embed_image(image_url)
            if card_id and vec is not None:
                self.save_vector(
                    scryfall_id=card_id,
                    embedding=vec,
                    name=card_data.get("name"),
                    set_code=card_data.get("set"),
                    collector_number=card_data.get("collector_number"),
                    image_url=image_url,
                )
            return vec
        except Exception as exc:
            logger.debug(f"Failed to compute embedding for Scryfall card '{card_id}': {exc}")
            return None

    def get_or_compute_vectors_batch(
        self,
        cards: List[Dict[str, Any]],
        embedder: Any,
    ) -> Dict[str, np.ndarray]:
        """Retrieves or computes embeddings for multiple Scryfall cards."""
        results: Dict[str, np.ndarray] = {}
        missing_cards: List[Tuple[str, str, Dict[str, Any]]] = []

        for card in cards:
            if not isinstance(card, dict):
                continue
            card_id = card.get("id")
            if not card_id:
                continue

            cached = self.get_vector(card_id)
            if cached is not None:
                results[card_id] = cached
            else:
                img_url = get_scryfall_image_url(card)
                if img_url:
                    missing_cards.append((card_id, img_url, card))

        for card_id, img_url, card in missing_cards:
            try:
                vec = embedder.embed_image(img_url)
                if vec is not None:
                    self.save_vector(
                        scryfall_id=card_id,
                        embedding=vec,
                        name=card.get("name"),
                        set_code=card.get("set"),
                        collector_number=card.get("collector_number"),
                        image_url=img_url,
                    )
                    results[card_id] = vec
            except Exception as exc:
                logger.debug(f"Failed to compute embedding for Scryfall card '{card_id}': {exc}")

        return results


_default_vector_cache: Optional[ScryfallVectorCache] = None
_cache_lock = threading.Lock()


def get_default_scryfall_vector_cache(db_path: Optional[str] = None) -> ScryfallVectorCache:
    """Returns a shared singleton instance of ScryfallVectorCache."""
    global _default_vector_cache
    if _default_vector_cache is None:
        with _cache_lock:
            if _default_vector_cache is None:
                _default_vector_cache = ScryfallVectorCache(db_path=db_path)
    return _default_vector_cache
