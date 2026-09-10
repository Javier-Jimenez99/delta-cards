"""Title and description parsing for extracting Pokemon card information."""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ParsedCardInfo:
    """Extracted search hints from a card listing title/description."""

    raw_title: str
    cleaned_title: str
    card_name: str
    local_id: Optional[str] = None
    total_cards_in_set: Optional[str] = None
    set_hints: List[str] = None
    modifier_hints: List[str] = None

    def __post_init__(self):
        if self.set_hints is None:
            self.set_hints = []
        if self.modifier_hints is None:
            self.modifier_hints = []


# Common noise words in Spanish/English marketplace titles
STOP_WORDS = {
    "carta",
    "cartas",
    "card",
    "cards",
    "pokemon",
    "pokémon",
    "tcg",
    "ccg",
    "original",
    "originales",
    "oficial",
    "autentica",
    "auténtica",
    "es",
    "en",
    "esp",
    "espana",
    "españa",
    "espanol",
    "español",
    "english",
    "ingles",
    "inglés",
    "japones",
    "japonés",
    "japanese",
    "nueva",
    "nuevo",
    "mint",
    "near mint",
    "nm",
    "psa",
    "bgs",
    "cgc",
    "gem",
    "lote",
    "caja",
    "sobre",
    "sobres",
    "vendo",
    "pack",
    "precio",
    "barato",
    "oferta",
}

# Special card suffixes to preserve with the card name (e.g. "Charizard VMAX", "Pikachu EX")
CARD_MODIFIERS = [
    "vmax",
    "vstar",
    "v-union",
    "break",
    "prism",
    "gx",
    "ex",
    "v",
    "lv.x",
    "holo",
    "reverse",
    "secret",
    "gold",
    "rainbow",
    "shiny",
    "radiant",
    "radiante",
    "sar",
    "ar",
    "illustration rare",
    "special illustration rare",
    "full art",
    "alt art",
]

# Regex pattern to match card numbers such as 136/189, 025/165, 4/102, #136, etc.
CARD_NUMBER_PATTERN = re.compile(
    r"(?:(?:no\.?|#)\s*)?(\b\d{1,3}[a-z]?)\s*/\s*(\d{2,3})\b",
    re.IGNORECASE,
)
STANDALONE_NUMBER_PATTERN = re.compile(
    r"(?:(?:no\.?|#)\s*)(\d{1,3})\b",
    re.IGNORECASE,
)


def extract_card_number(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extracts card number and set total from text (e.g. '136/189' -> ('136', '189')).

    Returns:
        (local_id, total_cards_in_set)
    """
    match = CARD_NUMBER_PATTERN.search(text)
    if match:
        local_id = match.group(1).lstrip("0") or "0"
        total = match.group(2)
        return local_id, total

    standalone = STANDALONE_NUMBER_PATTERN.search(text)
    if standalone:
        local_id = standalone.group(1).lstrip("0") or "0"
        return local_id, None

    return None, None


def parse_card_title(title: str, description: Optional[str] = None) -> ParsedCardInfo:
    """Parses a listing title and optional description to extract Pokemon name,

    card number, and set clues for matching against TCGDex.
    """
    raw_title = title.strip()
    full_text = f"{title} {description or ''}".strip()

    # Extract card number from title or full text
    local_id, total_in_set = extract_card_number(raw_title)
    if not local_id and description:
        local_id, total_in_set = extract_card_number(description)

    # Detect modifier hints (e.g., VMAX, EX, Holo)
    lower_title = raw_title.lower()
    detected_modifiers = []
    for mod in CARD_MODIFIERS:
        if re.search(rf"\b{re.escape(mod)}\b", lower_title):
            detected_modifiers.append(mod.upper())

    # Clean title for Pokemon name extraction
    cleaned = raw_title
    # Remove number patterns like 136/189
    cleaned = CARD_NUMBER_PATTERN.sub(" ", cleaned)
    cleaned = STANDALONE_NUMBER_PATTERN.sub(" ", cleaned)

    # Remove non-alphanumeric punctuation except hyphens
    cleaned = re.sub(r"[^\w\s-]", " ", cleaned)

    tokens = cleaned.split()
    name_tokens = []
    for token in tokens:
        t_lower = token.lower()
        if t_lower in STOP_WORDS:
            continue
        name_tokens.append(token)

    # Reconstruct candidate card name
    candidate_name = " ".join(name_tokens).strip()

    # Extract set hints if mentioned
    set_hints = []
    # If total in set is found (e.g. 102, 165, 189), that's a strong hint
    if total_in_set:
        set_hints.append(f"total:{total_in_set}")

    return ParsedCardInfo(
        raw_title=raw_title,
        cleaned_title=cleaned,
        card_name=candidate_name,
        local_id=local_id,
        total_cards_in_set=total_in_set,
        set_hints=set_hints,
        modifier_hints=detected_modifiers,
    )
