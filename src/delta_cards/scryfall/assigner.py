"""Offer assignment service to enrich Magic: The Gathering card offers using Scryfall API."""

from dataclasses import dataclass
import logging
import re
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from ..database.connection import get_engine
from ..database.models import PlatformOffers
from ..database.repository import upsert_cardmarket_product, upsert_cardmarket_price
from ..models import OfferListing, is_card_offer
from .client import (
    ScryfallClient,
    build_magic_cardmarket_url,
    extract_scryfall_cardmarket_prices,
    fetch_scryfall_card_pricing,
    _default_client,
)
from .vector_cache import (
    ScryfallVectorCache,
    get_default_scryfall_vector_cache,
    get_scryfall_image_url,
)

logger = logging.getLogger(__name__)

_default_embedder = None
_embedder_lock = threading.Lock()


def get_default_dinov3_embedder():
    """Lazily initializes and returns the shared DINOv3 embedder instance."""
    global _default_embedder
    if _default_embedder is None:
        with _embedder_lock:
            if _default_embedder is None:
                from ..recognition.dinov3_embedder import DINOv3Embedder
                _default_embedder = DINOv3Embedder()
    return _default_embedder


# Regex pattern to match set code (3-4 alphanumeric) and collector number (e.g. MH3 138, LTR #001)
SET_NUMBER_PATTERN = re.compile(
    r"\b([a-zA-Z0-9]{3,5})\s*(?:#|nº|no\.?|num\.?)?\s*(\d{1,4}[a-zA-Z]?)\b",
    re.IGNORECASE,
)

# Strip out boilerplate marketplace terms from Magic titles to extract the card name
CLEANUP_TITLE_TERMS = [
    r"\bcarta\s+magic\s+the\s+gathering\b",
    r"\bcartas\s+magic\s+the\s+gathering\b",
    r"\bmagic\s+the\s+gathering\b",
    r"\bcarta\s+magic\b",
    r"\bcartas\s+magic\b",
    r"\bmagic\s+carta\b",
    r"\bmtg\s+carta\b",
    r"\bcarta\s+mtg\b",
    r"\bmtg\s+single\b",
    r"\bmagic\b",
    r"\bmtg\b",
    r"\bthe\s+gathering\b",
    r"\bfoil\b",
    r"\bholo\b",
    r"\betched\b",
    r"\btextured\b",
    r"\bshowcase\b",
    r"\bborderless\b",
    r"\bextended\s+art\b",
    r"\bretro\s+frame\b",
    r"\bretro\b",
    r"\bsecret\s+lair\b",
    r"\bplaneswalker\b",
    r"\bcommander\b",
    r"\bedh\b",
    r"\bmodern\b",
    r"\blegacy\b",
    r"\bstandard\b",
    r"\bvintage\b",
    r"\bnear\s+mint\b",
    r"\bnm\b",
    r"\bmint\b",
    r"\bex\b",
    r"\bexcellent\b",
    r"\bgd\b",
    r"\blp\b",
    r"\bspanish\b",
    r"\bespañol\b",
    r"\bingles\b",
    r"\benglish\b",
    r"\ben\s+castellano\b",
    r"\ben\s+ingles\b",
    r"\bprecintada?\b",
    r"\bnueva?\b",
    r"\boriginal\b",
    r"\bvendo\b",
    r"\bcompro\b",
    r"\bperfecto\s+estado\b",
]


# Patterns for card treatments, variants, finishes, and art cards
ART_CARD_PATTERN = re.compile(
    r"\b(?:cartas?\s+de\s+arte|tarjetas?\s+de\s+arte|art\s+cards?|art\s+series)\b",
    re.IGNORECASE,
)
FOIL_PATTERN = re.compile(
    r"\b(?:foil|foilada|brillante|surge\s+foil|halo\s+foil|galaxy\s+foil|confetti\s+foil|etched\s+foil|foil\s+grabado)\b",
    re.IGNORECASE,
)
NON_FOIL_PATTERN = re.compile(
    r"\b(?:no\s+foil|non\s*[- ]?foil|sin\s+foil)\b",
    re.IGNORECASE,
)
BORDERLESS_PATTERN = re.compile(r"\b(?:borderless|sin\s+bordes?)\b", re.IGNORECASE)
EXTENDED_ART_PATTERN = re.compile(r"\b(?:extended\s+art|arte\s+extendido|extended)\b", re.IGNORECASE)
SHOWCASE_PATTERN = re.compile(
    r"\b(?:showcase|vitrina|resaltad[oa]|scroll|pergamino|posters?|cartel(?:es)?|stained\s+glass|vidriera)\b",
    re.IGNORECASE,
)
RETRO_PATTERN = re.compile(
    r"\b(?:retro\s+frame|old\s+frame|old\s+border|marco\s+retro|borde\s+antiguo|retro)\b",
    re.IGNORECASE,
)
FULL_ART_PATTERN = re.compile(r"\b(?:full\s+art|arte\s+completo)\b", re.IGNORECASE)
ANIME_PATTERN = re.compile(r"\b(?:anime|manga|japanese\s+art|arte\s+japon[eé]s|waifu)\b", re.IGNORECASE)
ETCHED_PATTERN = re.compile(r"\b(?:etched|foil\s+grabado|grabado)\b", re.IGNORECASE)
TEXTURED_PATTERN = re.compile(r"\b(?:textured|texturizad[oa]|foil\s+texturizad[oa])\b", re.IGNORECASE)
SERIALIZED_PATTERN = re.compile(
    r"\b(?:serialized|serializad[oa]|numerad[oa]|\d{1,3}\s*/\s*500|001\s*/\s*001)\b",
    re.IGNORECASE,
)
PRERELEASE_PATTERN = re.compile(
    r"\b(?:prerelease|pre\s*[- ]?release|presentaci[oó]n|pack\s+de\s+presentaci[oó]n)\b",
    re.IGNORECASE,
)
BUNDLE_PATTERN = re.compile(r"\b(?:bundle|gift\s+bundle|fat\s+pack)\b", re.IGNORECASE)
PROMO_PATTERN = re.compile(
    r"\b(?:promo|promocional|buy\s+a\s+box|fnm|store\s+championship|judge\s+gift)\b",
    re.IGNORECASE,
)
SECRET_LAIR_PATTERN = re.compile(r"\b(?:secret\s+lair|sld)\b", re.IGNORECASE)
SPECIAL_GUESTS_PATTERN = re.compile(r"\b(?:special\s+guests|invitados\s+especiales|spg)\b", re.IGNORECASE)
MASTERPIECE_PATTERN = re.compile(
    r"\b(?:masterpiece|inventions|invenciones|invocations|invocaciones|expeditions|expediciones)\b",
    re.IGNORECASE,
)

STANDALONE_NUMBER_PATTERN = re.compile(
    r"(?:#|nº|no\.?|num\.?)\s*(\d{1,4}[a-zA-Z]?)\b|\b(\d{1,4}[a-zA-Z]?)\s*/\s*\d{2,4}\b",
    re.IGNORECASE,
)

# Known set aliases mapped to (compiled_regex, set_code, set_name)
SET_ALIASES = [
    (re.compile(r"\b(?:modern\s+horizons?\s*3|horizontes\s+de\s+modern\s*3|mh3)\b", re.IGNORECASE), "mh3", "Modern Horizons 3"),
    (re.compile(r"\b(?:modern\s+horizons?\s*2|horizontes\s+de\s+modern\s*2|mh2)\b", re.IGNORECASE), "mh2", "Modern Horizons 2"),
    (re.compile(r"\b(?:modern\s+horizons?\s*1|horizontes\s+de\s+modern\s*1|mh1)\b", re.IGNORECASE), "mh1", "Modern Horizons"),
    (re.compile(r"\b(?:lord\s+of\s+the\s+rings|el\s+se[ñn]or\s+de\s+los\s+anillos|tales\s+of\s+middle[- ]earth|ltr)\b", re.IGNORECASE), "ltr", "The Lord of the Rings: Tales of Middle-earth"),
    (re.compile(r"\b(?:dominaria\s+remaster(?:ed|izada)|dmr)\b", re.IGNORECASE), "dmr", "Dominaria Remastered"),
    (re.compile(r"\b(?:double\s+masters\s*2022|2x2)\b", re.IGNORECASE), "2x2", "Double Masters 2022"),
    (re.compile(r"\b(?:double\s+masters|2xm)\b", re.IGNORECASE), "2xm", "Double Masters"),
    (re.compile(r"\b(?:commander\s+masters|cmm)\b", re.IGNORECASE), "cmm", "Commander Masters"),
    (re.compile(r"\b(?:ravnica\s+remaster(?:ed|izada)|rvr)\b", re.IGNORECASE), "rvr", "Ravnica Remastered"),
    (re.compile(r"\b(?:time\s+spiral\s+remaster(?:ed|izada)|tsr)\b", re.IGNORECASE), "tsr", "Time Spiral Remastered"),
    (re.compile(r"\b(?:foundations|fundaciones|fdn)\b", re.IGNORECASE), "fdn", "Foundations"),
    (re.compile(r"\b(?:bloomburrow|blb)\b", re.IGNORECASE), "blb", "Bloomburrow"),
    (re.compile(r"\b(?:duskmourn|dsk)\b", re.IGNORECASE), "dsk", "Duskmourn: House of Horror"),
    (re.compile(r"\b(?:outlaws\s+of\s+thunder\s+junction|forajidos\s+de\s+cruce\s+de\s+truenos|otj)\b", re.IGNORECASE), "otj", "Outlaws of Thunder Junction"),
    (re.compile(r"\b(?:murders\s+at\s+karlov\s+manor|asesinatos\s+en\s+la\s+mansi[oó]n\s+karlov|mkm)\b", re.IGNORECASE), "mkm", "Murders at Karlov Manor"),
    (re.compile(r"\b(?:lost\s+caverns\s+of\s+ixalan|cavernas\s+perdidas\s+de\s+ixalan|lci)\b", re.IGNORECASE), "lci", "The Lost Caverns of Ixalan"),
    (re.compile(r"\b(?:wilds\s+of\s+eldraine|tierras\s+salvajes\s+de\s+eldraine|woe)\b", re.IGNORECASE), "woe", "Wilds of Eldraine"),
    (re.compile(r"\b(?:march\s+of\s+the\s+machine|marcha\s+de\s+las\s+m[aá]quinas)\b", re.IGNORECASE), "mom", "March of the Machine"),
    (re.compile(r"\b(?:multiverse\s+legends|mul)\b", re.IGNORECASE), "mul", "Multiverse Legends"),
    (re.compile(r"\b(?:phyrexia\s+all\s+will\s+be\s+one|phyrexia\s+todos\s+ser[aá]n\s+uno)\b", re.IGNORECASE), "one", "Phyrexia: All Will Be One"),
    (re.compile(r"\b(?:brothers\'?\s+war|guerra\s+de\s+los\s+hermanos)\b", re.IGNORECASE), "bro", "The Brothers' War"),
    (re.compile(r"\b(?:retro\s+artifacts|brr)\b", re.IGNORECASE), "brr", "The Brothers' War Retro Artifacts"),
    (re.compile(r"\b(?:dominaria\s+united|dominaria\s+unida|dmu)\b", re.IGNORECASE), "dmu", "Dominaria United"),
    (re.compile(r"\b(?:kamigawa\s+neon\s+dynasty|kamigawa\s+dinast[ií]a\s+de\s+ne[oó]n|neo)\b", re.IGNORECASE), "neo", "Kamigawa: Neon Dynasty"),
    (re.compile(r"\b(?:final\s+fantasy|fca)\b", re.IGNORECASE), "fca", "Final Fantasy"),
    (re.compile(r"\b(?:fallout|pip)\b", re.IGNORECASE), "pip", "Fallout"),
    (re.compile(r"\b(?:doctor\s+who|dr\s+who)\b", re.IGNORECASE), "who", "Doctor Who"),
    (re.compile(r"\b(?:warhammer\s*(?:40000|40k)?|40k)\b", re.IGNORECASE), "40k", "Warhammer 40,000"),
    (re.compile(r"\b(?:alpha)\b", re.IGNORECASE), "lea", "Limited Edition Alpha"),
    (re.compile(r"\b(?:beta)\b", re.IGNORECASE), "leb", "Limited Edition Beta"),
    (re.compile(r"\b(?:unlimited)\b", re.IGNORECASE), "2ed", "Unlimited Edition"),
    (re.compile(r"\b(?:revised|revisada)\b", re.IGNORECASE), "3ed", "Revised Edition"),
    (re.compile(r"\b(?:kaladesh\s+inventions|inventions|invenciones)\b", re.IGNORECASE), "mps", "Kaladesh Inventions"),
    (re.compile(r"\b(?:special\s+guests|invitados\s+especiales|spg)\b", re.IGNORECASE), "spg", "Special Guests"),
    (re.compile(r"\b(?:secret\s+lair\s+drop|secret\s+lair|sld)\b", re.IGNORECASE), "sld", "Secret Lair Drop"),
]


@dataclass
class MagicListingFeatures:
    """Structured features extracted from Magic: The Gathering marketplace listing text."""
    candidate_name: Optional[str] = None
    set_code: Optional[str] = None
    set_name: Optional[str] = None
    collector_number: Optional[str] = None
    is_borderless: bool = False
    is_extended_art: bool = False
    is_showcase: bool = False
    is_retro: bool = False
    is_full_art: bool = False
    is_anime: bool = False
    is_etched: bool = False
    is_textured: bool = False
    is_serialized: bool = False
    is_promo: bool = False
    is_prerelease: bool = False
    is_bundle: bool = False
    is_secret_lair: bool = False
    is_special_guests: bool = False
    is_masterpiece: bool = False
    is_art_card: bool = False
    is_foil: Optional[bool] = None


def extract_magic_listing_features(
    title: str, description: Optional[str] = None
) -> MagicListingFeatures:
    """Extracts rich version features, treatments, finishes, sets, and clean candidate name."""
    combined = f"{title} {description or ''}".strip()

    is_art_card = bool(ART_CARD_PATTERN.search(combined))
    is_non_foil = bool(NON_FOIL_PATTERN.search(combined))
    is_foil = True if bool(FOIL_PATTERN.search(combined)) and not is_non_foil else (False if is_non_foil else None)

    set_code: Optional[str] = None
    collector_num: Optional[str] = None
    set_name: Optional[str] = None

    # 1. Set code + collector number pattern
    m_set_num = SET_NUMBER_PATTERN.search(title)
    if not m_set_num and description:
        m_set_num = SET_NUMBER_PATTERN.search(description)
    if m_set_num:
        set_code = m_set_num.group(1).lower()
        collector_num = m_set_num.group(2).lower()

    # 2. Standalone collector number pattern if not found yet
    if not collector_num:
        m_num = STANDALONE_NUMBER_PATTERN.search(title)
        if not m_num and description:
            m_num = STANDALONE_NUMBER_PATTERN.search(description)
        if m_num:
            collector_num = (m_num.group(1) or m_num.group(2)).lower()

    # 3. Known set names and aliases
    for pat, code, sname in SET_ALIASES:
        if pat.search(combined):
            if not set_code:
                set_code = code
            if not set_name:
                set_name = sname
            break

    # 4. Version and treatment indicators
    is_borderless = bool(BORDERLESS_PATTERN.search(combined))
    is_extended_art = bool(EXTENDED_ART_PATTERN.search(combined))
    is_showcase = bool(SHOWCASE_PATTERN.search(combined))
    is_retro = bool(RETRO_PATTERN.search(combined))
    is_full_art = bool(FULL_ART_PATTERN.search(combined))
    is_anime = bool(ANIME_PATTERN.search(combined))
    is_etched = bool(ETCHED_PATTERN.search(combined))
    is_textured = bool(TEXTURED_PATTERN.search(combined))
    is_serialized = bool(SERIALIZED_PATTERN.search(combined))
    is_prerelease = bool(PRERELEASE_PATTERN.search(combined))
    is_bundle = bool(BUNDLE_PATTERN.search(combined))
    is_promo = bool(PROMO_PATTERN.search(combined))
    is_secret_lair = bool(SECRET_LAIR_PATTERN.search(combined))
    is_special_guests = bool(SPECIAL_GUESTS_PATTERN.search(combined))
    is_masterpiece = bool(MASTERPIECE_PATTERN.search(combined))

    # 5. Extract clean card name
    cleaned = title
    for pat in CLEANUP_TITLE_TERMS:
        cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)

    cleaned = SET_NUMBER_PATTERN.sub(" ", cleaned)
    for pat, _, _ in SET_ALIASES:
        cleaned = pat.sub(" ", cleaned)

    for pat in [
        BORDERLESS_PATTERN, EXTENDED_ART_PATTERN, SHOWCASE_PATTERN, RETRO_PATTERN,
        FULL_ART_PATTERN, ANIME_PATTERN, ETCHED_PATTERN, TEXTURED_PATTERN,
        SERIALIZED_PATTERN, PRERELEASE_PATTERN, BUNDLE_PATTERN, PROMO_PATTERN,
        SECRET_LAIR_PATTERN, SPECIAL_GUESTS_PATTERN, MASTERPIECE_PATTERN,
        ART_CARD_PATTERN, FOIL_PATTERN, NON_FOIL_PATTERN, STANDALONE_NUMBER_PATTERN,
    ]:
        cleaned = pat.sub(" ", cleaned)

    if collector_num:
        cleaned = re.sub(r"\b" + re.escape(collector_num) + r"\b", " ", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"[#\(\)\[\]\{\}\/\\\"\'\|•·\-_]", " ", cleaned)
    cleaned = " ".join(cleaned.split()).strip()

    candidate_name = cleaned if len(cleaned) >= 2 else None

    return MagicListingFeatures(
        candidate_name=candidate_name,
        set_code=set_code,
        set_name=set_name,
        collector_number=collector_num,
        is_borderless=is_borderless,
        is_extended_art=is_extended_art,
        is_showcase=is_showcase,
        is_retro=is_retro,
        is_full_art=is_full_art,
        is_anime=is_anime,
        is_etched=is_etched,
        is_textured=is_textured,
        is_serialized=is_serialized,
        is_promo=is_promo,
        is_prerelease=is_prerelease,
        is_bundle=is_bundle,
        is_secret_lair=is_secret_lair,
        is_special_guests=is_special_guests,
        is_masterpiece=is_masterpiece,
        is_art_card=is_art_card,
        is_foil=is_foil,
    )


def extract_magic_identifiers(
    title: str, description: Optional[str] = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extracts candidate set code, collector number, and clean card name from listing text.

    Maintains backward compatibility with callers expecting (set_code, collector_number, candidate_name).
    """
    features = extract_magic_listing_features(title, description)
    return features.set_code, features.collector_number, features.candidate_name


def score_magic_printing(p: Dict[str, Any], f: MagicListingFeatures) -> float:
    """Scores how closely a candidate Scryfall printing matches the listing's extracted version features."""
    p_set = (p.get("set") or "").lower().strip()
    p_set_name = (p.get("set_name") or "").lower().strip()
    p_num = str(p.get("collector_number") or "").lower().strip()
    p_border = (p.get("border_color") or "").lower().strip()
    p_frame = str(p.get("frame") or "").lower().strip()
    p_frame_effects = [str(e).lower() for e in (p.get("frame_effects") or [])]
    p_promo_types = [str(t).lower() for t in (p.get("promo_types") or [])]
    p_finishes = [str(fin).lower() for fin in (p.get("finishes") or [])]
    p_promo = bool(p.get("promo"))
    p_full_art = bool(p.get("full_art"))

    score = 0.0

    # 1. Collector Number Match
    if f.collector_number:
        f_clean = f.collector_number.lower().lstrip("0")
        p_clean = p_num.lstrip("0")
        if f_clean == p_clean or f.collector_number.lower() == p_num:
            score += 250.0
        else:
            score -= 150.0

    # 2. Set Code & Set Name Match
    if f.set_code:
        if p_set == f.set_code.lower():
            score += 120.0
        else:
            score -= 80.0
    elif f.set_name and f.set_name.lower() in p_set_name:
        score += 100.0

    # 3. Treatments
    is_p_borderless = (p_border == "borderless")
    if f.is_borderless:
        score += 60.0 if is_p_borderless else -50.0
    elif is_p_borderless:
        score -= 20.0

    is_p_ea = ("extendedart" in p_frame_effects)
    if f.is_extended_art:
        score += 60.0 if is_p_ea else -50.0
    elif is_p_ea:
        score -= 20.0

    is_p_showcase = (
        "showcase" in p_frame_effects
        or "showcase" in p_promo_types
        or "scroll" in p_promo_types
        or "poster" in p_promo_types
    )
    if f.is_showcase:
        score += 60.0 if is_p_showcase else -50.0
    elif is_p_showcase:
        score -= 20.0

    is_p_retro = (
        p_frame in ("1993", "1997")
        or "retro" in p_frame_effects
        or p_set in ("h2r", "brr", "dmr")
    )
    if f.is_retro:
        score += 70.0 if is_p_retro else -60.0
    elif is_p_retro and (p_set in ("h2r", "brr", "dmr", "mh2", "tsr") or p_frame in ("1993", "1997")):
        score -= 25.0

    is_p_full_art = p_full_art
    if f.is_full_art:
        score += 50.0 if is_p_full_art else -30.0

    is_p_anime = ("anime" in p_promo_types or "manga" in p_promo_types or "japaneseart" in p_promo_types)
    if f.is_anime:
        score += 80.0 if is_p_anime else -60.0
    elif is_p_anime:
        score -= 30.0

    is_p_etched = ("etched" in p_finishes or "etched" in p_frame_effects)
    if f.is_etched:
        score += 60.0 if is_p_etched else -40.0
    elif is_p_etched:
        score -= 15.0

    is_p_textured = ("textured" in p_promo_types or "textured" in p_frame_effects)
    if f.is_textured:
        score += 80.0 if is_p_textured else -60.0
    elif is_p_textured:
        score -= 30.0

    # Serialized safeguard: Heavily penalize serialized versions unless explicitly stated!
    is_p_serialized = (
        "serialized" in p_promo_types
        or "serialized" in p_frame_effects
        or "doublerainbow" in p_promo_types
    )
    if f.is_serialized:
        score += 150.0 if is_p_serialized else -100.0
    elif is_p_serialized:
        score -= 1000.0

    is_p_bundle = ("bundle" in p_promo_types)
    if f.is_bundle:
        score += 70.0 if is_p_bundle else -30.0

    is_p_prerelease = ("prerelease" in p_promo_types or "datestamped" in p_promo_types)
    if f.is_prerelease:
        score += 70.0 if is_p_prerelease else -40.0
    elif is_p_prerelease:
        score -= 20.0

    is_p_sld = (p_set in ("sld", "slz", "slp") or "secret lair" in p_set_name)
    if f.is_secret_lair:
        score += 90.0 if is_p_sld else -80.0
    elif is_p_sld:
        score -= 40.0

    is_p_spg = (p_set == "spg" or "special guests" in p_set_name)
    if f.is_special_guests:
        score += 90.0 if is_p_spg else -80.0
    elif is_p_spg:
        score -= 40.0

    is_p_mps = (p_set in ("mps", "mp2", "exp") or "masterpiece" in p_set_name)
    if f.is_masterpiece:
        score += 100.0 if is_p_mps else -80.0
    elif is_p_mps:
        score -= 80.0

    # 4. Finishes compatibility
    if f.is_foil is True:
        if "foil" in p_finishes or "etched" in p_finishes:
            score += 10.0
        elif "nonfoil" in p_finishes and "foil" not in p_finishes:
            score -= 30.0

    # 5. Baseline / Normal preferences
    if not p_promo:
        score += 5.0
    if p_border == "black":
        score += 5.0
    if p.get("cardmarket_id"):
        score += 10.0
    if p.get("prices", {}).get("eur") is not None:
        score += 5.0

    return score


def select_best_magic_printing(
    printings: List[Dict[str, Any]], features: MagicListingFeatures
) -> Optional[Dict[str, Any]]:
    """Selects the best matching Scryfall printing for a listing based on scored version features."""
    if not printings:
        return None
    if len(printings) == 1:
        return printings[0]

    scored = []
    for p in printings:
        s = score_magic_printing(p, features)
        rel_date = str(p.get("released_at") or "")
        scored.append((s, rel_date, p))

    # Sort descending by score, then release date (newest first for ties)
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2]


def select_best_magic_printing_visual(
    printings: List[Dict[str, Any]],
    listing_image: Union[str, bytes, np.ndarray],
    features: Optional[MagicListingFeatures] = None,
    embedder: Optional[Any] = None,
    vector_cache: Optional[ScryfallVectorCache] = None,
    detector: Optional[Any] = None,
) -> Tuple[Optional[Dict[str, Any]], float]:
    """Selects the best matching Scryfall printing for a listing using DINOv3 visual embeddings.

    Extracts or retrieves DINOv3 dense vector representations for each candidate printing,
    computes cosine similarity with the listing's image, and selects the version with the highest score.
    In case of near-identical visual similarity (e.g. shared artwork between regular and foil versions),
    features-based scoring is used as a tiebreaker.

    Args:
        printings: List of candidate Scryfall card dictionaries.
        listing_image: Image URL, bytes, or NumPy BGR array of the listing photograph.
        features: Optional extracted text features for tie-breaking.
        embedder: Optional DINOv3Embedder instance.
        vector_cache: Optional ScryfallVectorCache instance.
        detector: Optional VisualCardDetector to crop card boundaries.

    Returns:
        Tuple of (selected_card_dict, similarity_score).
    """
    if not printings:
        return None, 0.0

    if len(printings) == 1 and not listing_image:
        return printings[0], 1.0

    emb = embedder or get_default_dinov3_embedder()
    vcache = vector_cache or get_default_scryfall_vector_cache()

    listing_crop = listing_image
    if detector is not None:
        try:
            listing_crop = detector.isolate_card(listing_image)
        except Exception as e:
            logger.debug(f"Detector could not isolate card, using original image: {e}")
            listing_crop = listing_image

    try:
        listing_vec = emb.embed_image(listing_crop)
    except Exception as exc:
        logger.warning(f"Failed to extract DINOv3 embedding from listing image: {exc}")
        fallback = select_best_magic_printing(printings, features or MagicListingFeatures()) if features else printings[0]
        return fallback, 0.0

    scored_prints: List[Tuple[float, float, str, Dict[str, Any]]] = []
    for p in printings:
        try:
            p_vec = vcache.get_or_compute_vector(p, emb)
            if p_vec is not None:
                sim = float(np.dot(listing_vec, p_vec))
            else:
                sim = -1.0
        except Exception as exc:
            logger.debug(f"Error getting vector for card printing {p.get('id')}: {exc}")
            sim = -1.0

        t_score = score_magic_printing(p, features) if features else 0.0
        rel_date = str(p.get("released_at") or "")
        scored_prints.append((sim, t_score, rel_date, p))

    if not scored_prints:
        fallback = select_best_magic_printing(printings, features or MagicListingFeatures()) if features else printings[0]
        return fallback, 0.0

    # Sort descending by visual similarity
    scored_prints.sort(key=lambda item: item[0], reverse=True)
    best_sim = scored_prints[0][0]

    # In case of near-identical visual score (e.g. difference < 0.01 for same artwork), use text features as tie-breaker
    top_candidates = [item for item in scored_prints if item[0] >= best_sim - 0.01]
    top_candidates.sort(key=lambda item: (item[1], item[2]), reverse=True)
    winner = top_candidates[0][3]

    return winner, best_sim


def resolve_scryfall_card(
    title: str,
    description: Optional[str] = None,
    client: Optional[ScryfallClient] = None,
    language: str = "es",
    image_url: Optional[Union[str, bytes, np.ndarray]] = None,
    vector_cache: Optional[ScryfallVectorCache] = None,
    embedder: Optional[Any] = None,
    detector: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Resolves a marketplace listing to a Scryfall card and Cardmarket pricing,
    taking into account printings, variants, arts, finishes, sets, and DINOv3 visual matching."""
    cli = client or _default_client
    features = extract_magic_listing_features(title, description)

    # 1. Reject non-card / art-card items to avoid false positives
    if features.is_art_card:
        logger.debug(f"Rejecting art card listing: '{title}'")
        return None

    card_data: Optional[Dict[str, Any]] = None
    candidate_prints: List[Dict[str, Any]] = []

    # 2. Strategy 1: Look up by set code and collector number
    if features.set_code and features.collector_number:
        try:
            c = cli.get_card_by_set_number(features.set_code, features.collector_number, lang=language)
            if not c:
                c = cli.get_card_by_set_number(features.set_code, features.collector_number)
            if c:
                card_data = c
                candidate_prints = [c]
        except Exception as err:
            logger.debug(f"Error checking Scryfall set/number ({features.set_code}/{features.collector_number}): {err}")

    # 3. Strategy 2: Version-aware printings resolution using canonical name
    if not card_data and features.candidate_name:
        try:
            base_card = cli.get_card_by_name(features.candidate_name, set_code=features.set_code)
            canonical_name = base_card.get("name") if isinstance(base_card, dict) else features.candidate_name

            prints = cli.get_card_prints(canonical_name, set_code=features.set_code)
            if isinstance(prints, list) and prints and isinstance(prints[0], dict):
                candidate_prints = prints
            elif isinstance(base_card, dict):
                candidate_prints = [base_card]
        except Exception as err:
            logger.debug(f"Error checking Scryfall name '{features.candidate_name}': {err}")

    # 4. Strategy 3: Fallback query search if candidate has multiple tokens
    if not candidate_prints and not card_data and features.candidate_name and len(features.candidate_name.split()) >= 2:
        try:
            results = cli.search_cards(features.candidate_name)
            if isinstance(results, list) and results and isinstance(results[0], dict):
                candidate_prints = results
        except Exception as err:
            logger.debug(f"Error in Scryfall search fallback for '{features.candidate_name}': {err}")

    if not candidate_prints and not card_data:
        return None

    visual_similarity: Optional[float] = None

    # Visual Matching with DINOv3:
    # If listing image is provided:
    # Check if we have multiple versions; if only 1 version was found, expand it to all versions of that card.
    if image_url:
        if len(candidate_prints) <= 1:
            base_c = candidate_prints[0] if candidate_prints else card_data
            if base_c:
                try:
                    expanded = cli.get_card_prints_for_card(base_c)
                    if expanded and len(expanded) > 1:
                        candidate_prints = expanded
                except Exception as exc:
                    logger.debug(f"Could not expand printings for card: {exc}")

        if candidate_prints:
            best_visual_card, sim = select_best_magic_printing_visual(
                printings=candidate_prints,
                listing_image=image_url,
                features=features,
                embedder=embedder,
                vector_cache=vector_cache,
                detector=detector,
            )
            if best_visual_card:
                card_data = best_visual_card
                visual_similarity = sim

    # If card_data was not chosen via visual matching, use text feature scoring
    if not card_data:
        if candidate_prints:
            card_data = select_best_magic_printing(candidate_prints, features) or candidate_prints[0]

    if not card_data:
        return None

    pricing_info = extract_scryfall_cardmarket_prices(card_data, language=language, is_foil=features.is_foil)
    if visual_similarity is not None:
        pricing_info["visual_similarity"] = visual_similarity

    return pricing_info


def _apply_scryfall_result_to_offer(offer: Any, pricing_info: Dict[str, Any]) -> None:
    """Applies Scryfall pricing and metadata to an OfferListing or PlatformOffers model."""
    if not pricing_info:
        return

    cm_id = pricing_info.get("idProduct")
    cm_url = pricing_info.get("url")
    prices = pricing_info.get("prices") or {}

    raw = pricing_info.get("raw") or {}
    scryfall_id = raw.get("id")

    if scryfall_id:
        if hasattr(offer, "tcgdex_id"):
            offer.tcgdex_id = scryfall_id
        if hasattr(offer, "scryfall_id"):
            offer.scryfall_id = scryfall_id

    if cm_id:
        if hasattr(offer, "cardmarket_product_id"):
            offer.cardmarket_product_id = cm_id
    if cm_url:
        if hasattr(offer, "cardmarket_url"):
            offer.cardmarket_url = cm_url
        object.__setattr__(offer, "_temp_cm_url", cm_url)

    low = prices.get("low")
    avg = prices.get("avg")
    trend = prices.get("trend")

    if low is not None:
        if hasattr(offer, "cardmarket_low_price"):
            offer.cardmarket_low_price = low
        object.__setattr__(offer, "_temp_cm_low_price", low)
    if avg is not None:
        if hasattr(offer, "cardmarket_avg_price"):
            offer.cardmarket_avg_price = avg
        object.__setattr__(offer, "_temp_cm_avg_price", avg)
    if trend is not None:
        if hasattr(offer, "cardmarket_trend_price"):
            offer.cardmarket_trend_price = trend
        object.__setattr__(offer, "_temp_cm_trend_price", trend)

    if pricing_info.get("image_url") and not getattr(offer, "image_url", None):
        offer.image_url = pricing_info["image_url"]

    vis_sim = pricing_info.get("visual_similarity")
    if vis_sim is not None:
        object.__setattr__(offer, "_temp_visual_similarity", vis_sim)


def assign_scryfall_to_offers(
    offers: List[OfferListing],
    client: Optional[ScryfallClient] = None,
    progress_callback: Optional[Any] = None,
    vector_cache: Optional[ScryfallVectorCache] = None,
    embedder: Optional[Any] = None,
    detector: Optional[Any] = None,
) -> List[OfferListing]:
    """Enriches a list of OfferListing objects with Scryfall ID and Cardmarket Product ID."""
    if not offers:
        return []

    cli = client or _default_client
    card_offers = [
        o for o in offers
        if is_card_offer(o.product_type, title=o.title, description=o.description)
        and not getattr(o, "cardmarket_product_id", None)
    ]

    total = len(card_offers)
    for idx, offer in enumerate(card_offers, 1):
        try:
            res = resolve_scryfall_card(
                title=offer.title,
                description=offer.description,
                client=cli,
                language=offer.language or "es",
                image_url=getattr(offer, "image_url", None),
                vector_cache=vector_cache,
                embedder=embedder,
                detector=detector,
            )
            if res:
                _apply_scryfall_result_to_offer(offer, res)
        except Exception as exc:
            logger.debug(f"Failed to assign Scryfall for offer '{offer.title}': {exc}")

        if progress_callback:
            progress_callback(idx, total, offer)

    return offers


def enrich_magic_offers_in_db(
    engine: Optional[Engine] = None,
    limit: Optional[int] = None,
    client: Optional[ScryfallClient] = None,
    vector_cache: Optional[ScryfallVectorCache] = None,
    embedder: Optional[Any] = None,
    detector: Optional[Any] = None,
) -> int:
    """Enriches existing Magic card offers stored in the PlatformOffers database table

    Finds listings with assigned_state == 'pending', matches them with Scryfall,
    saves CardmarketProduct and CardmarketPrice, and marks deal candidates.
    """
    eng = engine or get_engine()
    cli = client or _default_client

    from sqlmodel import or_

    with Session(eng) as session:
        statement = select(PlatformOffers).where(
            PlatformOffers.assigned_state == "pending",
            PlatformOffers.product_type == "single_card",
            or_(
                PlatformOffers.game == "magic",
                PlatformOffers.title.ilike("%magic%"),
                PlatformOffers.title.ilike("%mtg%"),
                PlatformOffers.search_profile_id.ilike("%magic%"),
            ),
        )
        if limit is not None:
            statement = statement.limit(limit)

        offers_to_enrich = session.exec(statement).all()
        if not offers_to_enrich:
            logger.info("No unassigned Magic card offers found in database.")
            return 0

        valid_offers: List[PlatformOffers] = []
        for db_o in offers_to_enrich:
            if is_card_offer(db_o.product_type, title=db_o.title, description=db_o.description):
                valid_offers.append(db_o)
            else:
                db_o.assigned_state = "not_assigned"
                session.add(db_o)

        if not valid_offers:
            session.commit()
            return 0

        updated_count = 0
        for db_o in valid_offers:
            try:
                res = resolve_scryfall_card(
                    title=db_o.title,
                    description=db_o.description,
                    client=cli,
                    language=db_o.language or "es",
                    image_url=getattr(db_o, "image_url", None),
                    vector_cache=vector_cache,
                    embedder=embedder,
                    detector=detector,
                )
                if res and res.get("idProduct"):
                    _apply_scryfall_result_to_offer(db_o, res)
                    cm_id = res["idProduct"]
                    cm_url = res.get("url") or build_magic_cardmarket_url(cm_id, language=db_o.language or "es")

                    upsert_cardmarket_product(
                        product_id=cm_id,
                        cardmarket_url=cm_url,
                        session=session,
                    )

                    prices = res.get("prices") or {}
                    low_p = prices.get("low")
                    avg_p = prices.get("avg")
                    trend_p = prices.get("trend")

                    if low_p is not None or avg_p is not None or trend_p is not None:
                        upsert_cardmarket_price(
                            product_id=cm_id,
                            language=db_o.language or "es",
                            min_price_all=low_p,
                            avg_price=avg_p,
                            trend_price=trend_p,
                            session=session,
                        )

                    ref_price = avg_p if avg_p is not None else low_p
                    if ref_price is not None and ref_price > 0 and db_o.price < ref_price:
                        if getattr(db_o, "alert_status", None) != "sent":
                            db_o.alert_status = "pending"
                    else:
                        if getattr(db_o, "alert_status", None) != "sent":
                            db_o.alert_status = "none"

                    db_o.assigned_state = "assigned"
                    updated_count += 1
                else:
                    db_o.assigned_state = "not_assigned"
                    if getattr(db_o, "alert_status", None) != "sent":
                        db_o.alert_status = "none"

                session.add(db_o)
            except Exception as exc:
                logger.warning(f"Error enriching offer {db_o.id} ({db_o.title[:30]}): {exc}")

        session.commit()
        return updated_count

