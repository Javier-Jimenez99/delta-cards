"""Deterministic card matching by edition fraction (num/denom), promo codes, and Pokémon name."""

import os
import re
import sqlite3
import unicodedata
from typing import Any, Dict, List, Optional

# Sets where official printed card count differs from total count (with secret rares)
SET_OFFICIAL_COUNTS: Dict[str, int] = {
    "base5": 82, "neo3": 64, "neo4": 105, "ecard2": 147, "ecard3": 144, "ex3": 97, "ex4": 95, "ex5": 101, "ex6": 112,
    "ex7": 109, "ex8": 107, "ex9": 106, "ex10": 115, "ex11": 113, "ex12": 92, "ex13": 110, "dp2": 123, "dp7": 100,
    "pl1": 127, "pl2": 111, "pl3": 147, "pl4": 99, "hgss1": 123, "hgss2": 95, "hgss3": 90, "hgss4": 102, "col1": 95,
    "bw1": 114, "bw3": 101, "bw4": 99, "bw5": 108, "bw6": 124, "dv1": 20, "bw7": 149, "bw8": 135, "bw9": 116,
    "bw10": 101, "xyp": 211, "bw11": 113, "xy2": 106, "xy3": 111, "xy4": 119, "xy5": 160, "xy6": 108, "xy7": 98,
    "xy8": 162, "xy9": 122, "g1": 83, "xy10": 124, "xy11": 114, "xy12": 108, "sm1": 149, "sm2": 145, "sm3": 147,
    "sm3.5": 73, "sm4": 111, "sm5": 156, "sm6": 131, "sm7": 168, "sm7.5": 70, "sm8": 214, "sm9": 181, "sm10": 214,
    "sm11": 236, "sm115": 68, "sm12": 236, "swsh1": 202, "swsh2": 192, "swsh3": 189, "swsh3.5": 73, "swsh4": 185,
    "swsh4.5": 72, "swsh5": 163, "swsh6": 198, "swsh7": 203, "swsh8": 264, "swsh9": 172, "swsh10": 189, "swsh10.5": 78,
    "swsh11": 196, "swsh12": 195, "swsh12.5": 159, "sv01": 198, "svp": 225, "sv02": 193, "sv03": 197, "sv03.5": 165,
    "sv04": 182, "sv04.5": 91, "sv05": 162, "sv06": 167, "sv06.5": 64, "sv07": 142, "A1": 226, "sv08": 191,
    "A1a": 68, "sv08.5": 131, "A2": 140, "A2a": 75, "A2b": 72, "sv09": 159, "A3": 155, "A3a": 69, "sv10": 182,
    "A3b": 69, "sv10.5b": 86, "sv10.5w": 86, "A4": 161, "A4a": 71, "me01": 132, "B1": 226, "me02": 94, "B1a": 69,
    "B2": 155, "me02.5": 217, "B2a": 93, "me03": 88, "me04": 86, "me05": 84
}

OFFICIAL_TOTAL_TO_SETS: Dict[int, List[str]] = {}
for _sid, _count in SET_OFFICIAL_COUNTS.items():
    OFFICIAL_TOTAL_TO_SETS.setdefault(_count, []).append(_sid)

_FRACTION_REGEX = re.compile(r"(?<!/)\b(\d{1,4})\s*/\s*(\d{1,4})\b(?!/)")
_PROMO_REGEX = re.compile(r"\b(SVP|SWSH|SM|XY|BW|DP|HGSS)[-\s]?(\d{1,4})\b", re.IGNORECASE)


def _clean_text(s: str) -> str:
    """Normalizes text: lowercase, strip accents, replace punctuation with spaces."""
    s = unicodedata.normalize("NFKD", (s or "").lower())
    clean = "".join(c for c in s if not unicodedata.combining(c))
    return " " + " ".join(re.sub(r"[^\w\s]", " ", clean).split()) + " "


def match_card_by_index_and_name(
    title: str,
    description: str = "",
    db_conn: Optional[sqlite3.Connection] = None,
    meta_path: Optional[str] = None,
    preferred_lang: str = "es",
) -> Optional[Dict[str, Any]]:
    """Matches a card deterministically using edition fraction (num/denom) or promo code and Pokémon name.

    If candidates share the exact same name and index, returns None to defer to visual matching.
    """
    full_text = f"{title or ''} {description or ''}"
    norm_text = _clean_text(full_text)

    db_file = os.path.abspath(meta_path or os.path.join("data", "pokemon_cards_meta.sqlite"))
    conn = db_conn or sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    try:
        # 1. Check Fraction (num/denom)
        fractions = set(_FRACTION_REGEX.findall(full_text))
        if fractions:
            matched_cards = []
            for num_str, denom_str in fractions:
                num, denom = int(num_str), int(denom_str)
                if denom == 0:
                    continue

                variants = [
                    f"{num}/{denom}",
                    f"{num:02d}/{denom}",
                    f"{num:03d}/{denom}",
                    f"{num}/{denom:03d}",
                    f"{num:03d}/{denom:03d}",
                ]
                rows = conn.execute(
                    f"SELECT * FROM card_metadata WHERE card_fraction IN ({','.join('?' for _ in variants)})",
                    variants,
                ).fetchall()

                # Also check sets whose printed official card count equals denom
                target_sets = OFFICIAL_TOTAL_TO_SETS.get(denom, [])
                if target_sets:
                    num_vars = [str(num), f"{num:02d}", f"{num:03d}"]
                    set_ph = ",".join("?" for _ in target_sets)
                    num_ph = ",".join("?" for _ in num_vars)
                    set_rows = conn.execute(
                        f"SELECT * FROM card_metadata WHERE set_id IN ({set_ph}) AND local_id IN ({num_ph})",
                        target_sets + num_vars,
                    ).fetchall()
                    if set_rows:
                        rows = list(rows) + list(set_rows)

                cards_by_id: Dict[str, Dict[str, Any]] = {}
                for r in rows:
                    cid = r["tcgdex_id"]
                    cards_by_id.setdefault(cid, {"cards": {}, "names": set()})
                    cards_by_id[cid]["cards"][r["lang"]] = dict(r)
                    if r["name"]:
                        cards_by_id[cid]["names"].add(r["name"])

                matching_for_frac = []
                for cid, data in cards_by_id.items():
                    if any(_clean_text(n) in norm_text for n in data["names"] if n):
                        chosen = data["cards"].get(preferred_lang) or next(iter(data["cards"].values()))
                        matching_for_frac.append(chosen)

                if len(matching_for_frac) == 1:
                    matched_cards.append(matching_for_frac[0])
                elif len(matching_for_frac) > 1:
                    return None  # Collision: multiple cards share index and matched name

            unique_cids = {c["tcgdex_id"] for c in matched_cards}
            if len(unique_cids) == 1:
                return matched_cards[0]

        # 2. Check Promo Codes (SWSH, SVP, SM, XY, BW, DP, HGSS)
        promos = _PROMO_REGEX.findall(full_text)
        if promos:
            matched_promos = []
            for prefix, num_str in promos:
                prefix = prefix.upper()
                num_int = int(num_str)
                p_vars = [
                    f"{prefix}{num_str}",
                    f"{prefix}{num_int}",
                    f"{prefix}{num_int:02d}",
                    f"{prefix}{num_int:03d}",
                    f"{prefix}{num_int:04d}",
                ]
                p_ph = ",".join("?" for _ in p_vars)
                rows = conn.execute(
                    f"SELECT * FROM card_metadata WHERE UPPER(local_id) IN ({p_ph})", p_vars
                ).fetchall()
                if not rows:
                    continue

                cards_by_id = {}
                for r in rows:
                    cid = r["tcgdex_id"]
                    cards_by_id.setdefault(cid, {"cards": {}, "names": set()})
                    cards_by_id[cid]["cards"][r["lang"]] = dict(r)
                    if r["name"]:
                        cards_by_id[cid]["names"].add(r["name"])

                matching_for_promo = []
                for cid, data in cards_by_id.items():
                    if any(_clean_text(n) in norm_text for n in data["names"] if n):
                        chosen = data["cards"].get(preferred_lang) or next(iter(data["cards"].values()))
                        matching_for_promo.append(chosen)

                if len(matching_for_promo) == 1:
                    matched_promos.append(matching_for_promo[0])
                elif len(matching_for_promo) > 1:
                    return None  # Collision

            unique_cids = {c["tcgdex_id"] for c in matched_promos}
            if len(unique_cids) == 1:
                return matched_promos[0]

    finally:
        if db_conn is None:
            conn.close()

    return None
