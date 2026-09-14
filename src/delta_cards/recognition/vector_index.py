"""USearch HNSW vector index with Mmap zero-RAM runtime support and metadata persistence."""

import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from usearch.index import Index

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = os.path.join(os.getcwd(), "data", "pokemon_cards.usearch")
DEFAULT_META_PATH = os.path.join(os.getcwd(), "data", "pokemon_cards_meta.sqlite")
EMBEDDING_DIM = 384


class USearchCardIndex:
    """Zero-RAM Mmap HNSW vector index for fine-grained image retrieval of Pokémon cards."""

    def __init__(
        self,
        index_path: Optional[str] = None,
        meta_path: Optional[str] = None,
        ndim: int = EMBEDDING_DIM,
        metric: str = "cos",
        view_only: bool = True,
        lang: str = "en",
    ):
        """Initializes USearch index.

        Args:
            index_path: Path to .usearch binary file.
            meta_path: Path to companion SQLite metadata database.
            ndim: Dimensionality of embeddings (384 for DINOv3 ViT-Small).
            metric: Distance metric ('cos' for cosine distance).
            view_only: If True, uses mmap view (Zero-RAM). If False, loads into RAM for editing/saving.
            lang: Default card language for this index (e.g. 'en', 'es').
        """
        self.ndim = ndim
        self.metric = metric
        self.index_path = index_path or DEFAULT_INDEX_PATH
        self.meta_path = meta_path or DEFAULT_META_PATH
        self.view_only = view_only
        self.lang = (lang or "en").lower()

        self.index = Index(ndim=self.ndim, metric=self.metric)
        self._db_conn: Optional[sqlite3.Connection] = None
        self._is_loaded = False

        if os.path.isfile(self.index_path) and os.path.isfile(self.meta_path):
            self.load(self.index_path, self.meta_path, view=self.view_only)

    def load(self, index_path: str, meta_path: Optional[str] = None, view: bool = True) -> None:
        """Loads index file. When view=True, memory-maps the file directly from disk."""
        self.index_path = index_path
        self.meta_path = meta_path or index_path.rsplit(".", 1)[0] + "_meta.sqlite"

        if not os.path.isfile(self.index_path):
            raise FileNotFoundError(f"USearch index file not found: {self.index_path}")

        if view:
            logger.info(f"Memory-mapping USearch index from '{self.index_path}' (Zero-RAM mode)...")
            self.index.view(self.index_path)
        else:
            logger.info(f"Loading USearch index into RAM from '{self.index_path}'...")
            self.index.load(self.index_path)

        self._init_sqlite(self.meta_path, read_only=view)
        self._is_loaded = True
        logger.info(f"USearch index active: {len(self.index)} card embeddings loaded.")

    def _init_sqlite(self, db_path: str, read_only: bool = False) -> None:
        """Initializes the companion SQLite database for fast O(1) metadata retrieval."""
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        is_existing = os.path.isfile(db_path)
        if read_only and is_existing:
            uri = f"file:{os.path.abspath(db_path)}?mode=ro"
            self._db_conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        else:
            self._db_conn = sqlite3.connect(db_path, check_same_thread=False)

        self._db_conn.row_factory = sqlite3.Row
        if not read_only or not is_existing:
            with self._db_conn:
                self._db_conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS card_metadata (
                        key INTEGER PRIMARY KEY,
                        tcgdex_id TEXT NOT NULL,
                        lang TEXT NOT NULL DEFAULT 'en',
                        name TEXT NOT NULL,
                        set_id TEXT NOT NULL,
                        set_name TEXT NOT NULL,
                        local_id TEXT NOT NULL,
                        total_cards TEXT,
                        card_fraction TEXT,
                        rarity TEXT,
                        hp INTEGER,
                        cardmarket_id INTEGER,
                        image_url TEXT,
                        raw_json TEXT,
                        UNIQUE(tcgdex_id, lang)
                    )
                    """
                )
                self._db_conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_card_tcgdex_lang ON card_metadata (tcgdex_id, lang)"
                )
                self._db_conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_card_lang ON card_metadata (lang)"
                )
                self._db_conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_card_set_local ON card_metadata (set_id, local_id)"
                )
                self._db_conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_card_fraction ON card_metadata (card_fraction)"
                )
                self._db_conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS unindexable_cards (
                        tcgdex_id TEXT PRIMARY KEY,
                        reason TEXT,
                        checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

    def add_card(
        self,
        key: int,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
    ) -> None:
        """Adds a single card embedding and metadata into index."""
        if self._db_conn is None:
            self._init_sqlite(self.meta_path, read_only=False)

        lang = str(metadata.get("lang") or "en").lower()

        # Check if card (tcgdex_id, lang) is already indexed to prevent duplicate entries
        cursor = self._db_conn.execute(
            "SELECT key FROM card_metadata WHERE tcgdex_id = ? AND lang = ?",
            (metadata["tcgdex_id"], lang),
        )
        row = cursor.fetchone()
        if row is not None:
            # Card already exists in this index/metadata: skip duplicate insertion
            return

        actual_key = key
        while actual_key in self.index:
            actual_key += 1

        vec = np.ascontiguousarray(embedding, dtype=np.float32)
        self.index.add(actual_key, vec)

        card_fraction = metadata.get("card_fraction")
        if not card_fraction and metadata.get("local_id") and metadata.get("total_cards"):
            card_fraction = f"{metadata['local_id']}/{metadata['total_cards']}"

        with self._db_conn:
            self._db_conn.execute(
                """
                INSERT OR REPLACE INTO card_metadata (
                    key, tcgdex_id, lang, name, set_id, set_name,
                    local_id, total_cards, card_fraction,
                    rarity, hp, cardmarket_id, image_url, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    actual_key,
                    metadata["tcgdex_id"],
                    lang,
                    metadata.get("name", ""),
                    metadata.get("set_id", ""),
                    metadata.get("set_name", ""),
                    str(metadata.get("local_id", "")),
                    str(metadata.get("total_cards", "")) if metadata.get("total_cards") else None,
                    card_fraction,
                    metadata.get("rarity"),
                    metadata.get("hp"),
                    metadata.get("cardmarket_id"),
                    metadata.get("image_url"),
                    json.dumps(metadata),
                ),
            )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        exact: bool = True,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Performs SIMD nearest neighbor search on mmap index and fetches Top-K candidates.

        Args:
            query_vector: 384-dimensional DINOv3 query vector.
            top_k: Number of visually similar candidates to return.
            exact: If True, uses exact SIMD search (1.2ms, 100% recall). If False, uses HNSW graph.
            lang: Optional language filter (reserved for compatibility).

        Returns:
            List of candidate dictionaries ordered by similarity (descending).
        """
        if len(self.index) == 0:
            return []

        vec = np.ascontiguousarray(query_vector, dtype=np.float32)
        search_count = min(len(self.index), max(top_k * 4, 16) if lang else top_k)
        matches = self.index.search(vec, search_count, exact=exact)

        matching_lang_results = []
        other_results = []

        for key, distance in zip(matches.keys, matches.distances):
            if key == -1:
                continue
            meta = self.get_metadata(int(key))
            if not meta:
                continue
            similarity = max(0.0, 1.0 - float(distance))
            item = {
                "key": int(key),
                "distance": float(distance),
                "similarity": round(similarity, 4),
                "card": meta,
            }
            if lang and meta.get("lang") and meta.get("lang").lower() == lang.lower():
                matching_lang_results.append(item)
            else:
                other_results.append(item)

        # Prioritize matching language candidates, fill remainder with nearest visual neighbors (fallback)
        combined = matching_lang_results + other_results
        return combined[:top_k]

    def get_cards_by_local_id(self, local_id: str, lang: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves all indexed cards matching a collector number / local ID, optionally filtered by language."""
        if self._db_conn is None:
            return []
        clean_id = local_id.lstrip("0")
        if lang:
            cursor = self._db_conn.execute(
                "SELECT * FROM card_metadata WHERE (local_id = ? OR local_id = ?) AND lang = ?",
                (local_id, clean_id, lang.lower()),
            )
        else:
            cursor = self._db_conn.execute(
                "SELECT * FROM card_metadata WHERE local_id = ? OR local_id = ?",
                (local_id, clean_id),
            )
        rows = cursor.fetchall()
        results = []
        for row in rows:
            results.append({
                "key": row["key"],
                "tcgdex_id": row["tcgdex_id"],
                "lang": row["lang"] if "lang" in row.keys() else "en",
                "name": row["name"],
                "set_id": row["set_id"],
                "set_name": row["set_name"],
                "local_id": row["local_id"],
                "total_cards": row["total_cards"],
                "card_fraction": row["card_fraction"],
                "rarity": row["rarity"],
                "hp": row["hp"],
                "cardmarket_id": row["cardmarket_id"],
                "image_url": row["image_url"],
                "raw_json": json.loads(row["raw_json"]) if row["raw_json"] else {},
            })
        return results

    def search_cards_by_name(self, name: str, lang: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Searches card metadata by Pokémon name substring."""
        if self._db_conn is None:
            return []
        clean_name = name.strip()
        if not clean_name:
            return []
        if lang:
            cursor = self._db_conn.execute(
                "SELECT * FROM card_metadata WHERE name LIKE ? AND lang = ? LIMIT ?",
                (f"%{clean_name}%", lang.lower(), limit),
            )
        else:
            cursor = self._db_conn.execute(
                "SELECT * FROM card_metadata WHERE name LIKE ? LIMIT ?",
                (f"%{clean_name}%", limit),
            )
        rows = cursor.fetchall()
        results = []
        for row in rows:
            results.append({
                "key": row["key"],
                "tcgdex_id": row["tcgdex_id"],
                "lang": row["lang"] if "lang" in row.keys() else "en",
                "name": row["name"],
                "set_id": row["set_id"],
                "set_name": row["set_name"],
                "local_id": row["local_id"],
                "total_cards": row["total_cards"],
                "card_fraction": row["card_fraction"],
                "rarity": row["rarity"],
                "hp": row["hp"],
                "cardmarket_id": row["cardmarket_id"],
                "image_url": row["image_url"],
                "raw_json": json.loads(row["raw_json"]) if row["raw_json"] else {},
            })
        return results

    def get_metadata(self, key: int) -> Optional[Dict[str, Any]]:
        """Retrieves card metadata by its integer index key in O(1)."""
        if self._db_conn is None:
            return None

        cursor = self._db_conn.execute(
            "SELECT * FROM card_metadata WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        if row is None:
            return None

        return {
            "key": row["key"],
            "tcgdex_id": row["tcgdex_id"],
            "lang": row["lang"] if "lang" in row.keys() else "en",
            "name": row["name"],
            "set_id": row["set_id"],
            "set_name": row["set_name"],
            "local_id": row["local_id"],
            "total_cards": row["total_cards"],
            "card_fraction": row["card_fraction"],
            "rarity": row["rarity"],
            "hp": row["hp"],
            "cardmarket_id": row["cardmarket_id"],
            "image_url": row["image_url"],
            "raw_json": json.loads(row["raw_json"]) if row["raw_json"] else {},
        }

    def get_card_by_id_and_lang(self, tcgdex_id: str, lang: str = "en") -> Optional[Dict[str, Any]]:
        """Retrieves card metadata by its tcgdex_id and language."""
        if self._db_conn is None:
            self._init_sqlite(self.meta_path, read_only=True)

        cursor = self._db_conn.execute(
            "SELECT * FROM card_metadata WHERE tcgdex_id = ? AND lang = ?",
            (tcgdex_id, lang.lower()),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        return {
            "key": row["key"],
            "tcgdex_id": row["tcgdex_id"],
            "lang": row["lang"],
            "name": row["name"],
            "set_id": row["set_id"],
            "set_name": row["set_name"],
            "local_id": row["local_id"],
            "total_cards": row["total_cards"],
            "card_fraction": row["card_fraction"],
            "rarity": row["rarity"],
            "hp": row["hp"],
            "cardmarket_id": row["cardmarket_id"],
            "image_url": row["image_url"],
            "raw_json": json.loads(row["raw_json"]) if row["raw_json"] else {},
        }

    def get_indexed_tcgdex_ids(self, lang: Optional[str] = None) -> set:
        """Returns a set of card tcgdex_ids recorded in the metadata database, optionally filtered by language."""
        if self._db_conn is None:
            self._init_sqlite(self.meta_path, read_only=self.view_only)
        if lang:
            cursor = self._db_conn.execute("SELECT tcgdex_id FROM card_metadata WHERE lang = ?", (lang.lower(),))
        else:
            cursor = self._db_conn.execute("SELECT tcgdex_id FROM card_metadata")
        return {row[0] for row in cursor.fetchall()}

    def get_indexed_cards_by_lang(self, lang: str) -> set:
        """Returns set of all tcgdex_ids indexed for a specific language."""
        return self.get_indexed_tcgdex_ids(lang=lang)

    def get_unindexable_tcgdex_ids(self) -> set:
        """Returns a set of all card tcgdex_ids recorded as lacking images or unindexable."""
        if self._db_conn is None:
            self._init_sqlite(self.meta_path, read_only=self.view_only)
        cursor = self._db_conn.execute("SELECT tcgdex_id FROM unindexable_cards")
        return {row[0] for row in cursor.fetchall()}

    def record_unindexable_cards(self, cards: List[tuple]) -> None:
        """Records a list of (tcgdex_id, reason) tuples into unindexable_cards table."""
        if not cards:
            return
        if self._db_conn is None:
            self._init_sqlite(self.meta_path, read_only=False)
        with self._db_conn:
            self._db_conn.executemany(
                "INSERT OR IGNORE INTO unindexable_cards (tcgdex_id, reason) VALUES (?, ?)",
                cards,
            )

    def get_next_key(self) -> int:
        """Returns the next available integer key for vector insertion."""
        if self._db_conn is None:
            self._init_sqlite(self.meta_path, read_only=False)
        cursor = self._db_conn.execute("SELECT COALESCE(MAX(key), -1) + 1 FROM card_metadata")
        candidate_key = max(len(self.index), int(cursor.fetchone()[0]))
        while candidate_key in self.index:
            candidate_key += 1
        return candidate_key

    def save(self, index_path: Optional[str] = None, meta_path: Optional[str] = None) -> None:
        """Saves compiled USearch binary index to disk."""
        target_index = index_path or self.index_path
        os.makedirs(os.path.dirname(os.path.abspath(target_index)), exist_ok=True)
        self.index.save(target_index)
        logger.debug(f"USearch index saved to '{target_index}' ({os.path.getsize(target_index)} bytes).")

    def __len__(self) -> int:
        return len(self.index)


class MultiLanguageCardIndex:
    """Manages language-partitioned USearch indices (e.g. 'es', 'en', 'fr', etc.) with on-demand loading and fallback."""

    def __init__(
        self,
        base_dir: str = "data",
        default_lang: str = "en",
        fallback_index_path: Optional[str] = None,
        view_only: bool = True,
    ):
        """Initializes the multi-language index manager.

        Args:
            base_dir: Directory containing per-language indices (e.g. data/pokemon_cards_es.usearch).
            default_lang: Language to fallback to if a specific language index is not built.
            fallback_index_path: Legacy single-index fallback path (default: data/pokemon_cards.usearch).
            view_only: If True, uses Zero-RAM memory-mapping mode.
        """
        self.base_dir = os.path.abspath(base_dir)
        self.default_lang = default_lang.lower()
        self.fallback_index_path = fallback_index_path or os.path.join(self.base_dir, "pokemon_cards.usearch")
        self.view_only = view_only
        self._indices: Dict[str, USearchCardIndex] = {}

    def get_index_for_language(self, lang: Optional[Union[str, int]] = None) -> Tuple[USearchCardIndex, str]:
        """Resolves and loads the USearchCardIndex for the target language with fallback.

        Returns:
            Tuple of (USearchCardIndex instance, resolved_language_code).
        """
        if isinstance(lang, int):
            clean_lang = self.default_lang
        else:
            clean_lang = (lang or self.default_lang).lower()

        # 1. Already cached in memory?
        if clean_lang in self._indices:
            return self._indices[clean_lang], clean_lang

        # 2. Check if specific language index exists on disk (e.g. data/pokemon_cards_es.usearch)
        lang_idx_path = os.path.join(self.base_dir, f"pokemon_cards_{clean_lang}.usearch")
        lang_meta_path = os.path.join(self.base_dir, f"pokemon_cards_{clean_lang}_meta.sqlite")
        if os.path.isfile(lang_idx_path) and os.path.isfile(lang_meta_path):
            idx = USearchCardIndex(
                index_path=lang_idx_path,
                meta_path=lang_meta_path,
                lang=clean_lang,
                view_only=self.view_only,
            )
            self._indices[clean_lang] = idx
            return idx, clean_lang

        # 3. Check default language index (e.g. data/pokemon_cards_en.usearch)
        def_idx_path = os.path.join(self.base_dir, f"pokemon_cards_{self.default_lang}.usearch")
        def_meta_path = os.path.join(self.base_dir, f"pokemon_cards_{self.default_lang}_meta.sqlite")
        if os.path.isfile(def_idx_path) and os.path.isfile(def_meta_path):
            if self.default_lang not in self._indices:
                self._indices[self.default_lang] = USearchCardIndex(
                    index_path=def_idx_path,
                    meta_path=def_meta_path,
                    lang=self.default_lang,
                    view_only=self.view_only,
                )
            return self._indices[self.default_lang], self.default_lang

        # 4. Fallback to unified single index (data/pokemon_cards.usearch)
        fallback_meta = self.fallback_index_path.rsplit(".", 1)[0] + "_meta.sqlite"
        if os.path.isfile(self.fallback_index_path) and os.path.isfile(fallback_meta):
            if "unified" not in self._indices:
                self._indices["unified"] = USearchCardIndex(
                    index_path=self.fallback_index_path,
                    meta_path=fallback_meta,
                    lang="en",
                    view_only=self.view_only,
                )
            return self._indices["unified"], clean_lang

        # 5. Empty index initialized for this language
        idx = USearchCardIndex(
            index_path=lang_idx_path,
            meta_path=lang_meta_path,
            lang=clean_lang,
            view_only=self.view_only,
        )
        self._indices[clean_lang] = idx
        return idx, clean_lang

    def search(
        self,
        query_vector: np.ndarray,
        first_arg: Optional[Union[str, int]] = None,
        top_k: int = 5,
        exact: bool = True,
        lang: Optional[str] = None,
        **kwargs,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Routes search to the index for `lang` (or fallback).

        Supports:
          - search(query_vector, lang="es", top_k=5)
          - search(query_vector, "es", 5)
          - search(query_vector, 5)
          - search(query_vector, top_k=5)

        Returns:
            Tuple of (list of candidates, language_actually_queried).
        """
        if isinstance(first_arg, int):
            top_k = first_arg
        elif isinstance(first_arg, str):
            lang = first_arg

        idx, actual_lang = self.get_index_for_language(lang)
        candidates = idx.search(query_vector, top_k=top_k, exact=exact, lang=lang)
        return candidates, actual_lang

    def get_card_by_id_and_lang(self, tcgdex_id: str, lang: str = "en") -> Optional[Dict[str, Any]]:
        """Retrieves card metadata by tcgdex_id and language."""
        idx, _ = self.get_index_for_language(lang)
        return idx.get_card_by_id_and_lang(tcgdex_id, lang)

    def get_cards_by_local_id(self, local_id: str, lang: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves all indexed cards matching a collector number / local ID."""
        idx, _ = self.get_index_for_language(lang or self.default_lang)
        return idx.get_cards_by_local_id(local_id, lang=lang)

    def search_cards_by_name(self, name: str, lang: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Searches card metadata by Pokémon name substring."""
        idx, _ = self.get_index_for_language(lang or self.default_lang)
        return idx.search_cards_by_name(name, lang=lang, limit=limit)

    def get_indexed_tcgdex_ids(self, lang: Optional[str] = None) -> set:
        """Returns indexed tcgdex IDs for the given language."""
        idx, _ = self.get_index_for_language(lang)
        return idx.get_indexed_tcgdex_ids(lang=lang)

    def __len__(self) -> int:
        """Returns the number of cards in the default or active index."""
        idx, _ = self.get_index_for_language(self.default_lang)
        return len(idx)


