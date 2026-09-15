"""Card vector storage and index building."""

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from usearch.index import Index

DEFAULT_INDEX_PATH = os.path.join(os.getcwd(), "data", "pokemon_cards.usearch")
DEFAULT_META_PATH = os.path.join(os.getcwd(), "data", "pokemon_cards_meta.sqlite")
EMBEDDING_DIM = 384


class USearchCardIndex:
    """USearch vectors with card metadata in SQLite."""

    def __init__(self, index_path: Optional[str] = None, meta_path: Optional[str] = None, ndim: int = EMBEDDING_DIM, metric: str = "cos", view_only: bool = True, lang: str = "en"):
        self.index_path = index_path or DEFAULT_INDEX_PATH
        self.meta_path = meta_path or DEFAULT_META_PATH
        self.view_only = view_only
        self.index = Index(ndim=ndim, metric=metric)
        self._db_conn: Optional[sqlite3.Connection] = None
        if os.path.isfile(self.index_path) and os.path.isfile(self.meta_path):
            self.load(self.index_path, self.meta_path, view_only)
        elif not view_only:
            self._connect()

    def _connect(self, read_only: bool = False) -> None:
        if self._db_conn:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.meta_path)), exist_ok=True)
        self._db_conn = sqlite3.connect(f"file:{os.path.abspath(self.meta_path)}?mode=ro", uri=True, check_same_thread=False) if read_only else sqlite3.connect(self.meta_path, check_same_thread=False)
        self._db_conn.row_factory = sqlite3.Row
        if not read_only:
            self._db_conn.executescript("""CREATE TABLE IF NOT EXISTS card_metadata (key INTEGER PRIMARY KEY, tcgdex_id TEXT NOT NULL, lang TEXT NOT NULL DEFAULT 'en', name TEXT NOT NULL, set_id TEXT NOT NULL, set_name TEXT NOT NULL, local_id TEXT NOT NULL, total_cards TEXT, card_fraction TEXT, rarity TEXT, hp INTEGER, cardmarket_id INTEGER, image_url TEXT, raw_json TEXT, UNIQUE(tcgdex_id, lang)); CREATE TABLE IF NOT EXISTS unindexable_cards (tcgdex_id TEXT PRIMARY KEY, reason TEXT);""")

    def load(self, index_path: str, meta_path: Optional[str] = None, view: bool = True) -> None:
        self.index_path, self.meta_path = index_path, meta_path or index_path.rsplit(".", 1)[0] + "_meta.sqlite"
        self.index.view(index_path) if view else self.index.load(index_path)
        self._connect(read_only=view)

    def add_card(self, key: int, embedding: np.ndarray, metadata: Dict[str, Any]) -> None:
        self._connect()
        lang = (metadata.get("lang") or "en").lower()
        if self._db_conn.execute("SELECT 1 FROM card_metadata WHERE tcgdex_id = ? AND lang = ?", (metadata["tcgdex_id"], lang)).fetchone():
            return
        while key in self.index:
            key += 1
        self.index.add(key, np.ascontiguousarray(embedding, dtype=np.float32))
        with self._db_conn:
            self._db_conn.execute("INSERT INTO card_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (key, metadata["tcgdex_id"], lang, metadata.get("name", ""), metadata.get("set_id", ""), metadata.get("set_name", ""), str(metadata.get("local_id", "")), metadata.get("total_cards"), metadata.get("card_fraction"), metadata.get("rarity"), metadata.get("hp"), metadata.get("cardmarket_id"), metadata.get("image_url"), json.dumps(metadata)))

    def search(self, query_vector: np.ndarray, top_k: int = 5, exact: bool = True, **_: Any) -> List[Dict[str, Any]]:
        if not len(self.index):
            return []
        matches = self.index.search(np.ascontiguousarray(query_vector, dtype=np.float32), min(top_k, len(self.index)), exact=exact)
        results = []
        for key, distance in zip(matches.keys, matches.distances):
            card = self.get_metadata(int(key))
            if key != -1 and card:
                results.append({"key": int(key), "distance": float(distance), "similarity": round(max(0.0, 1.0 - float(distance)), 4), "card": card})
        return results

    def get_metadata(self, key: int) -> Optional[Dict[str, Any]]:
        row = self._db_conn.execute("SELECT * FROM card_metadata WHERE key = ?", (key,)).fetchone() if self._db_conn else None
        return dict(row) if row else None

    def get_card_by_id_and_lang(self, tcgdex_id: str, lang: str = "en") -> Optional[Dict[str, Any]]:
        row = self._db_conn.execute("SELECT * FROM card_metadata WHERE tcgdex_id = ? AND lang = ?", (tcgdex_id, lang.lower())).fetchone() if self._db_conn else None
        return dict(row) if row else None

    def get_indexed_tcgdex_ids(self, lang: Optional[str] = None) -> set:
        if not self._db_conn:
            return set()
        query, params = ("SELECT tcgdex_id FROM card_metadata WHERE lang = ?", (lang.lower(),)) if lang else ("SELECT tcgdex_id FROM card_metadata", ())
        return {row[0] for row in self._db_conn.execute(query, params)}

    def get_unindexable_tcgdex_ids(self) -> set:
        return {row[0] for row in self._db_conn.execute("SELECT tcgdex_id FROM unindexable_cards")} if self._db_conn else set()

    def record_unindexable_cards(self, cards: List[tuple]) -> None:
        self._connect()
        with self._db_conn:
            self._db_conn.executemany("INSERT OR IGNORE INTO unindexable_cards VALUES (?, ?)", cards)

    def get_next_key(self) -> int:
        return len(self.index)

    def save(self, index_path: Optional[str] = None, meta_path: Optional[str] = None) -> None:
        target = index_path or self.index_path
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        self.index.save(target)

    def __len__(self) -> int:
        return len(self.index)


async def build_index(sets_filter: Optional[List[str]] = None, limit_cards: Optional[int] = None, index_path: Optional[str] = None, cache_dir: str = "data/images_cache", concurrency: int = 16, lang: str = "en", language: Optional[str] = None, **_: Any) -> None:
    """Download card scans and add their DINOv3 vectors to one index."""
    import aiohttp
    from tcgdexsdk import TCGdex
    from .dinov3_embedder import DINOv3Embedder

    language, index_path = (language or lang).lower(), index_path or DEFAULT_INDEX_PATH
    index = USearchCardIndex(index_path, index_path.rsplit(".", 1)[0] + "_meta.sqlite", view_only=False)
    cards = await TCGdex(language).card.list()
    if sets_filter:
        cards = [card for card in cards if card.id.rsplit("-", 1)[0] in set(sets_filter)]
    if limit_cards:
        cards = cards[:limit_cards]
    embedder = DINOv3Embedder()
    os.makedirs(os.path.join(cache_dir, language), exist_ok=True)
    async with aiohttp.ClientSession() as session:
        for card in cards:
            if card.id in index.get_indexed_tcgdex_ids(language) or not getattr(card, "image", None):
                continue
            image_url = f"{card.image.rstrip('/')}/high.webp"
            async with session.get(image_url) as response:
                image = cv2.imdecode(np.frombuffer(await response.read(), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            set_id = card.id.rsplit("-", 1)[0]
            cv2.imwrite(os.path.join(cache_dir, language, f"{set_id}_{card.localId}.webp"), image)
            index.add_card(index.get_next_key(), embedder.embed_image(image), {"tcgdex_id": card.id, "lang": language, "name": getattr(card, "name", ""), "set_id": set_id, "set_name": set_id, "local_id": card.localId, "image_url": image_url})
    index.save()