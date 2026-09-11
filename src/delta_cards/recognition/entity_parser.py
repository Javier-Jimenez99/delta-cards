"""Entity extraction and normalization using Qwen2.5-0.5B-Instruct via llama-cpp-python and Instructor/Pydantic."""

import json
import logging
import os
import re
from typing import Any, Optional

from huggingface_hub import hf_hub_download
from langdetect import DetectorFactory, LangDetectException, detect
from llama_cpp import Llama
from pydantic import BaseModel, Field, field_validator

DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
DEFAULT_FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
DEFAULT_MODEL_DIR = os.path.join(os.getcwd(), "models")


class ExtractedCardMetadata(BaseModel):
    """Normalized structured metadata extracted from card visual OCR and marketplace text."""

    card_name: str = Field(
        description="The primary name of the Pokémon or Trainer card (in English or romanized if recognized)."
    )
    set_code: Optional[str] = Field(
        default=None,
        description="The official set code if identified (e.g., 'sv2a', 'sv01', 'swsh3', 'BS').",
    )
    collector_number: Optional[str] = Field(
        default=None,
        description="The card number within the set, without leading zeros if possible (e.g., '25', '025/165', '4/102').",
    )
    attacks: list[str] = Field(
        default_factory=list,
        description="Names of the attacks or abilities found on the card.",
    )
    language: str = Field(
        default="es",
        description="Primary inferred language code of the physical card: 'es', 'en', 'ja', etc.",
    )
    is_holo: bool = Field(
        default=False,
        description="Whether the seller description or card indicates a foil/holo/reverse variant.",
    )

    @field_validator("language", mode="before")
    @classmethod
    def normalize_language(cls, v: Any) -> str:
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
        return "es"


STRICT_DENOMINATOR_SET_MAP = {
    "102": "base1",      # Base Set (102)
    "64": "base2",       # Jungle (64)
    "62": "base3",       # Fossil (62)
    "130": "base4",      # Base Set 2 (130)
    "82": "base5",       # Team Rocket (82)
    "132": "gym1",       # Gym Heroes (132)
    "111": "neo1",       # Neo Genesis (111)
    "75": "neo2",        # Neo Discovery (75)
    "105": "neo4",       # Neo Destiny (105)
    "165": "sv03.5",     # 151 (165)
    "207": "sv03.5",     # 151 secret rares (207)
    "191": "sv08",       # Surging Sparks (191)
    "252": "sv08",       # Surging Sparks secret rares (252)
    "73": "sm3.5",       # Shining Legends (73)
    "78": "sm3.5",       # Shining Legends secret rares (78)
    "189": "swsh3",      # Darkness Ablaze (189)
    "201": "swsh3",      # Darkness Ablaze secret rares (201)
    "195": "swsh12",     # Silver Tempest (195)
    "215": "swsh12",     # Silver Tempest secret rares (215)
    "185": "swsh4",      # Vivid Voltage (185)
    "203": "swsh4",      # Vivid Voltage secret rares (203)
    "193": "sv06",       # Twilight Masquerade (193)
    "197": "sv03",       # Obsidian Flames (197)
    "162": "sv05",       # Temporal Forces (162)
    "264": "swsh08",     # Fusion Strike (264)
    "123": "hgss1",      # HeartGold & SoulSilver (123)
    "114": "bw1",        # Black & White (114)
}

SYSTEM_PROMPT = """Extract Pokemon card metadata from OCR and marketplace listing into structured JSON:
- card_name: Pokemon/Trainer name only (e.g. 'Pikachu ex', 'Charizard'). No HP, set names, or seller adjectives.
- set_code: Official set code if explicit (e.g. 'sv08', 'base1'). Otherwise null.
- collector_number: Card number (e.g. '057/191', '11/102'). Only extract numbers explicitly in text.
- attacks: List of attack/ability names.
- language: 2-letter lowercase code ('es', 'en', 'ja', 'fr', 'de', 'it') of the PHYSICAL CARD itself (based on card OCR markers like 'HP' vs 'PS', attack text, or explicit statements like 'en inglés'). Do NOT use the seller description language to determine card language.
- is_holo: Boolean indicating foil/holo/reverse."""


def ensure_model_file(
    model_path: Optional[str] = None,
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
    model_dir: str = DEFAULT_MODEL_DIR,
) -> str:
    """Ensures the quantized GGUF model exists locally, downloading if necessary."""
    if model_path and os.path.isfile(model_path):
        return model_path

    # Check common locations
    candidate_paths = [
        os.path.join(model_dir, filename),
        os.path.join(os.getcwd(), "models", filename),
        os.path.join(os.getcwd(), "data", "models", filename),
        os.path.expanduser(f"~/.cache/delta_cards/models/{filename}"),
    ]
    for path in candidate_paths:
        if os.path.isfile(path) and os.path.getsize(path) > 10_000_000:
            return path

    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Downloading {filename} from HuggingFace ({repo_id})...")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=model_dir,
    )
    return downloaded_path


class EntityParser:
    """Extracts structured Pokémon card metadata using quantized Qwen2.5-0.5B-Instruct on CPU."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
    ):
        """Initializes the entity parser.

        Args:
            model_path: Path to the .gguf model file. If None, resolves or downloads default.
            n_ctx: Context window length.
            n_threads: Number of CPU execution threads (defaults to physical cores, max 4).
        """
        # Tuning: cap threads to 4 physical cores to avoid AVX2 CPU cache thrashing on Linux
        cpu_cnt = os.cpu_count() or 4
        self.threads = n_threads or min(cpu_cnt // 2 if cpu_cnt > 4 else cpu_cnt, 4)
        self.n_ctx = n_ctx
        self.model_path = ensure_model_file(model_path)

        logger.info(
            f"Initializing Llama model from {self.model_path} on CPU with {self.threads} threads..."
        )
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.threads,
            n_batch=512,
            n_ubatch=512,
            use_mmap=True,
            verbose=False,
        )

    @staticmethod
    def detect_card_language(
        ocr_text: str = "",
        seller_title: str = "",
        seller_description: str = "",
    ) -> str:
        """Accurately detects the language of the PHYSICAL CARD itself using langdetect.

        CRITICAL: Does NOT use general seller marketplace prose (e.g. Spanish descriptions)
        to classify the card language.
        1. Physical card OCR text analysed via langdetect (with noise filtering) -> ('es', 'en', 'ja', 'de', 'fr', 'it', 'pt').
        2. Japanese script check (Kanji/Kana >= 6 chars) for cards with Japanese text.
        3. Physical card HP / PS / KP markers (for short or noisy OCR text).
        4. Explicit seller declaration of card language in title/desc (e.g. 'en inglés', 'japonés').
        5. Copyright line markers (e.g. '1999, España').
        6. Default fallback: 'es'.
        """
        # 1. Primary: Physical card OCR text language detection via langdetect
        # Filter to alphabetic words of length >= 3 to eliminate 1-2 char OCR noise fragments
        words = [w for w in (ocr_text or "").split() if len(w) >= 3 and w.isalpha()]
        filtered_ocr = " ".join(words)
        if len(filtered_ocr) >= 15:
            try:
                detected = detect(filtered_ocr)
                if detected in ("es", "en", "ja", "de", "fr", "it", "pt"):
                    return detected
            except LangDetectException:
                pass

        # 2. Japanese script check (Kanji/Kana >= 6 chars)
        if len(re.findall(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf]", ocr_text or "")) >= 6:
            return "ja"

        # 3. Physical card HP / PS / KP markers (especially useful for short OCR snippets)
        ocr_lower = (ocr_text or "").lower()
        if re.search(r"\b\d{2,3}\s*(ps|pv)\b|\b(ps|pv)\s*\d{2,3}\b", ocr_lower):
            return "es"
        if re.search(r"\b\d{2,3}\s*hp\b|\bhp\s*\d{2,3}\b", ocr_lower):
            return "en"
        if re.search(r"\b\d{2,3}\s*kp\b", ocr_lower):
            return "de"

        # 4. Explicit seller declarations about the physical card's language in title/description
        title_desc = f"{seller_title or ''} {seller_description or ''}".lower()
        if re.search(r"\b(en\s+japon[eé]s|idioma\s*:?\s*japon[eé]s|versi[oó]n\s+japonesa|\bjapanese\b|\bjapon[eé]s\b|\bjaponesa\b)\b", title_desc):
            return "ja"
        if re.search(r"\b(en\s+ingl[eé]s|idioma\s*:?\s*ingl[eé]s|versi[oó]n\s+ingl[eé]sa|en\s+english|\benglish\b|\bingl[eé]s\b|\bingles\b)\b", title_desc):
            return "en"
        if re.search(r"\b(en\s+italiano|idioma\s*:?\s*italiano|lingua\s+italiana|\bitaliana\b|\bitaliano\b)\b", title_desc):
            return "it"
        if re.search(r"\b(en\s+alem[aá]n|idioma\s*:?\s*alem[aá]n|\bgerman\b|\balem[aá]n\b)\b", title_desc):
            return "de"
        if re.search(r"\b(en\s+franc[eé]s|idioma\s*:?\s*franc[eé]s|\bfrench\b|\bfranc[eé]s\b)\b", title_desc):
            return "fr"
        if re.search(r"\b(en\s+español|en\s+castellano|idioma\s*:?\s*español|idioma\s*:?\s*castellano|versi[oó]n\s+española|versi[oó]n\s+en\s+español|\bespañol\b|\bcastellano\b)\b", title_desc):
            return "es"

        # 5. Copyright line markers (e.g. "1999, España" on Base Set cards)
        if re.search(r"\b(1999|2000)\s*,\s*españa\b", title_desc + " " + ocr_lower):
            return "es"

        # 6. Default fallback
        return "es"

    def try_strict_deterministic_extraction(
        self,
        ocr_text: str,
        seller_title: str,
        seller_description: str,
    ) -> Optional[ExtractedCardMetadata]:
        """Attempts strict, zero-false-positive deterministic entity extraction.

        Requirements for a valid deterministic extraction:
        1. An explicit, unambiguous collector fraction (num/den) in title, OCR, or desc.
        2. Denominator MUST match a known official set card count.
        3. Numerator must be within realistic set bounds (1 <= num <= den + 50).
        4. Clean card name extracted with length >= 3 and not generic noise words.

        Returns:
            ExtractedCardMetadata if strictly validated, otherwise None (triggers LLM).
        """
        combined = f"{seller_title} {seller_description} {ocr_text}"
        combined_lower = combined.lower()

        # 1. Look for explicit collector number fraction (e.g. 057/191, 11/102, 136/201)
        frac_match = None
        for source in [seller_title, ocr_text, seller_description]:
            if source:
                m = re.search(r"\b(\d{1,3})/(\d{2,3})\b", source)
                if m:
                    frac_match = m
                    break

        if not frac_match:
            return None

        num_str, den_str = frac_match.group(1).strip(), frac_match.group(2).strip()
        if den_str not in STRICT_DENOMINATOR_SET_MAP:
            return None

        # Validate numerator bounds
        try:
            num_int = int(num_str)
            den_int = int(den_str)
            if num_int <= 0 or num_int > (den_int + 50):
                return None
        except ValueError:
            return None

        set_code = STRICT_DENOMINATOR_SET_MAP[den_str]
        collector_number = f"{num_str}/{den_str}"

        # 2. Extract and clean card name from title (or OCR header if title empty)
        name_source = seller_title.strip() if seller_title else ocr_text.strip()
        name = re.split(r"\b\d{1,3}/\d{2,3}\b", name_source)[0].strip()
        name = re.split(r"\b\d{2,4}\s*(hp|ps)\b", name, flags=re.IGNORECASE)[0].strip()
        name = re.sub(r"\s+\d{2,4}$", "", name).strip()
        name = re.sub(r"\(.*?\)", "", name).strip()

        # Remove leading marketplace noise words iteratively
        while True:
            cleaned = re.sub(
                r"^(carta\s+(de\s+)?|carte\s+(de\s+)?|tarjeta\s+(de\s+)?|pok[eéè]mon\s+tcg\s+|pok[eéè]mon\s+|tcg\s+)",
                "",
                name,
                flags=re.IGNORECASE,
            ).strip()
            if cleaned == name:
                break
            name = cleaned

        # Strip trailing noise suffixes
        trailing_noise = [
            r"\b(holo\s+rare|holo|reverse\s+holo|reverse|ultra\s+rara|ultra\s+rare|rare|rara)\b.*$",
            r"\b(teracristal|terastal)\b.*$",
            r"\b(base\s+set(\s*2)?)\b.*$",
            r"\b(full\s+art|secret\s+rare|special\s+illustration\s+rare)\b.*$",
            r"\b(sar|sir|hr|sr|ur|ar|ir)\b.*$",
            r"\b(psa\s*\d+|cgc\s*\d+|bgs\s*\d+|nm\s*[-–]?\s*mint|\bnm\b|\bmint\b).*$",
            r"\b\d{4}\s*,\s*españa\b.*$",
            r"\b\d{4}\s*,\s*spain\b.*$",
        ]
        for pattern in trailing_noise:
            name = re.sub(pattern, "", name, flags=re.IGNORECASE).strip()

        name = re.sub(r"[\.\,\-\:]+$", "", name).strip()
        name = re.sub(r"\s+", " ", name)

        # Strict name validation: name must be >= 3 characters and not generic marketplace noise
        generic_words = {"carta", "pokemon", "tcg", "lote", "pack", "sobre", "caja", "box", "tin", "album", "carpeta"}
        if len(name) < 3 or name.lower() in generic_words:
            return None

        # Language inference for physical card
        lang = self.detect_card_language(
            ocr_text=ocr_text,
            seller_title=seller_title,
            seller_description=seller_description,
        )

        # Holo inference
        holo_keywords = ["holo", "foil", "reverse", "brillante", "rare", "rara", "sar", "sir", "vmax", "vstar", "ex", "gx"]
        is_holo = any(w in combined_lower for w in holo_keywords)

        return ExtractedCardMetadata(
            card_name=name,
            set_code=set_code,
            collector_number=collector_number,
            attacks=[],
            language=lang,
            is_holo=is_holo,
        )

    def parse_listing(
        self,
        ocr_text: str,
        seller_title: str,
        seller_description: str,
        use_fast_path: bool = True,
    ) -> ExtractedCardMetadata:
        """Synthesizes OCR text and seller listing into normalized ExtractedCardMetadata.

        Args:
            ocr_text: Text extracted from the card via OCR.
            seller_title: Seller's item title (e.g. from Wallapop/eBay).
            seller_description: Seller's description text.
            use_fast_path: If True, uses strict deterministic extraction if 100% matched.

        Returns:
            ExtractedCardMetadata instance.
        """
        if use_fast_path:
            strict = self.try_strict_deterministic_extraction(ocr_text, seller_title, seller_description)
            if strict is not None:
                return strict

        # Truncate noisy OCR text to first 250 chars (header/name) + last 100 chars (numbers)
        # to dramatically reduce prompt-evaluation time on CPU
        cleaned_ocr = ocr_text.strip() if ocr_text else ""
        if len(cleaned_ocr) > 350:
            cleaned_ocr = cleaned_ocr[:250] + "\n...\n" + cleaned_ocr[-100:]

        user_prompt = f"""Synthesize the following information to extract Pokemon card metadata:
[SELLER TITLE]: {seller_title.strip() if seller_title else "N/A"}
[SELLER DESCRIPTION]: {seller_description.strip() if seller_description else "N/A"}
[OCR TEXT FROM CARD]:
{cleaned_ocr if cleaned_ocr else "N/A"}
"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Use llama-cpp grammar-constrained JSON schema generation
        schema = ExtractedCardMetadata.model_json_schema()

        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                response_format={"type": "json_object", "schema": schema},
                temperature=0.0,
                max_tokens=128,
            )
            raw_json = response["choices"][0]["message"]["content"]
            metadata = ExtractedCardMetadata.model_validate_json(raw_json)

            # Post-normalization heuristics to refine card_name and collector_number
            metadata = self._refine_metadata(metadata, ocr_text, seller_title, seller_description)
            return metadata

        except Exception as exc:
            logger.warning(
                f"LLM entity extraction error: {exc}. Using heuristic fallback extractor."
            )
            return self._heuristic_fallback(ocr_text, seller_title, seller_description)

    def _refine_metadata(
        self,
        metadata: ExtractedCardMetadata,
        ocr_text: str,
        seller_title: str,
        seller_description: str = "",
    ) -> ExtractedCardMetadata:
        """Performs post-processing cleanup on extracted metadata."""
        combined = f"{seller_title} {ocr_text}"

        # If set_code has a fraction like "025/165", it is actually the collector_number!
        if metadata.set_code and "/" in metadata.set_code:
            if not metadata.collector_number or "/" not in metadata.collector_number:
                temp = metadata.collector_number
                metadata.collector_number = metadata.set_code
                metadata.set_code = temp if temp and temp.lower() not in ["null", "none"] else None

        # If OCR text or title has an explicit fraction (e.g. 136/201), verify against LLM output
        fraction_matches = re.findall(r"\b(\d{1,3}/\d{2,3})\b", combined)
        if fraction_matches:
            # If LLM didn't extract a fraction or extracted one not in text, use the explicit fraction
            if not metadata.collector_number or metadata.collector_number not in combined:
                metadata.collector_number = fraction_matches[0]

        # Clean collector number if it has leading zeroes in a single integer
        if metadata.collector_number:
            col = metadata.collector_number.strip()
            # If format is "025/165", leave as is or normalize
            if "/" in col:
                num, den = col.split("/", 1)
                num = num.strip()
                den = den.strip()
                col = f"{num.lstrip('0') or '0'}/{den}"
            elif col.isdigit():
                col = col.lstrip("0") or "0"
            metadata.collector_number = col

        # Clean card_name: strip HP, attacks, fractions, or trailing numbers accidentally captured by LLM
        name = metadata.card_name.strip()
        # Cut before card number fraction e.g. "Lugia V 138/195..." -> "Lugia V"
        name = re.split(r"\b\d{1,3}/\d{2,3}\b", name)[0].strip()
        # Cut at HP indicator (e.g. "330 HP", "220 HP", "60 HP", "110HP", "130 PS")
        name = re.split(r"\b\d{2,4}\s*(hp|ps)\b", name, flags=re.IGNORECASE)[0].strip()
        # Remove trailing lone numbers e.g. "Lugia V 220" -> "Lugia V"
        name = re.sub(r"\s+\d{2,4}$", "", name).strip()
        # Strip parenthesized text like "(m6 107/076 SAR)"
        name = re.sub(r"\(.*?\)", "", name).strip()

        # Remove leading marketplace noise words iteratively (handling accents & casing)
        while True:
            cleaned = re.sub(
                r"^(carta\s+(de\s+)?|carte\s+(de\s+)?|tarjeta\s+(de\s+)?|pok[eéè]mon\s+tcg\s+|pok[eéè]mon\s+|tcg\s+)",
                "",
                name,
                flags=re.IGNORECASE,
            ).strip()
            if cleaned == name:
                break
            name = cleaned

        # Strip trailing noise suffixes: rarity, sets, mechanics, condition, country
        trailing_noise = [
            r"\b(holo\s+rare|holo|reverse\s+holo|reverse|ultra\s+rara|ultra\s+rare|rare|rara)\b.*$",
            r"\b(teracristal|terastal)\b.*$",
            r"\b(base\s+set(\s*2)?)\b.*$",
            r"\b(full\s+art|secret\s+rare|special\s+illustration\s+rare)\b.*$",
            r"\b(sar|sir|hr|sr|ur|ar|ir)\b.*$",
            r"\b(psa\s*\d+|cgc\s*\d+|bgs\s*\d+|nm\s*[-–]?\s*mint|\bnm\b|\bmint\b).*$",
            r"\b\d{4}\s*,\s*españa\b.*$",
            r"\b\d{4}\s*,\s*spain\b.*$",
        ]
        for pattern in trailing_noise:
            name = re.sub(pattern, "", name, flags=re.IGNORECASE).strip()

        # Strip trailing punctuation
        name = re.sub(r"[\.\,\-\:]+$", "", name).strip()
        name = re.sub(r"\s+", " ", name)
        if name:
            metadata.card_name = name

        # Detect if set_code is invalid, generic, or a grading/condition artifact (e.g. "NM - Mint 8.0", "7.5")
        set_is_invalid = False
        if not metadata.set_code:
            set_is_invalid = True
        else:
            set_lower = metadata.set_code.lower().strip()
            if set_lower in ["pokemon", "tcg", "null", "none", "unknown"]:
                set_is_invalid = True
            elif any(g in set_lower for g in ["mint", "nm", "psa", "cgc", "bgs", "grade", "españa"]):
                set_is_invalid = True
            elif re.match(r"^\d+(\.\d+)?$", set_lower):  # e.g. "7.5" or "8.0"
                set_is_invalid = True

        combined_lower = combined.lower()
        den_map = {
            "102": "base1",      # Base Set
            "64": "base2",       # Jungle
            "62": "base3",       # Fossil
            "130": "base4",      # Base Set 2
            "82": "base5",       # Team Rocket
            "132": "gym1",       # Gym Heroes
            "111": "neo1",       # Neo Genesis
            "75": "neo2",        # Neo Discovery
            "64": "neo3",        # Neo Revelation
            "105": "neo4",       # Neo Destiny
            "165": "sv03.5",     # 151
            "207": "sv03.5",     # 151 (secret rares)
            "191": "sv08",       # Surging Sparks / Chispas Fulgurantes
            "252": "sv08",       # Surging Sparks (secret rares)
            "73": "sm3.5",       # Shining Legends / Leyendas Luminosas
            "78": "sm3.5",       # Shining Legends (secret rares)
            "189": "swsh3",      # Darkness Ablaze / Oscuridad Incandescente
            "201": "swsh3",      # Darkness Ablaze (secret rares)
            "195": "swsh12",     # Silver Tempest / Tempestad Plateada
            "215": "swsh12",     # Silver Tempest (secret rares)
            "185": "swsh4",      # Vivid Voltage
            "203": "swsh4",      # Vivid Voltage (secret rares)
            "193": "sv06",       # Twilight Masquerade
            "197": "sv03",       # Obsidian Flames
            "162": "sv05",       # Temporal Forces
            "264": "swsh08",     # Fusion Strike
            "123": "hgss1",      # HeartGold & SoulSilver
            "114": "bw1",        # Black & White
        }

        # 1. First priority for set_code: check collector_number denominator
        if metadata.collector_number and "/" in metadata.collector_number:
            _, den = metadata.collector_number.split("/", 1)
            den = den.strip()
            if den in den_map:
                metadata.set_code = den_map[den]
                set_is_invalid = False

        # 2. Second priority if still invalid: text matching in seller title / description
        if set_is_invalid:
            known_sets = {
                "base set 2": "base4",
                "base set": "base1",
                "shining legends": "sm3.5",
                "leyendas luminosas": "sm3.5",
                "darkness ablaze": "swsh3",
                "oscuridad incandescente": "swsh3",
                "silver tempest": "swsh12",
                "tempestad plateada": "swsh12",
                "vivid voltage": "swsh4",
                "voltaje vivido": "swsh4",
                "voltaje vívido": "swsh4",
                "evolving skies": "swsh7",
                "cielos evolutivos": "swsh7",
                "lost origin": "swsh11",
                "origen perdido": "swsh11",
                "brilliant stars": "swsh9",
                "astros brillantes": "swsh9",
                "crown zenith": "swsh12.5",
                "zenit real": "swsh12.5",
                "151": "sv03.5",
                "paldean fates": "sv04.5",
                "destinos de paldea": "sv04.5",
                "temporal forces": "sv05",
                "fuerzas temporales": "sv05",
                "twilight masquerade": "sv06",
                "mascarada crepuscular": "sv06",
                "shrouded fable": "sv06.5",
                "fabula sombria": "sv06.5",
                "stellar crown": "sv07",
                "corona astral": "sv07",
                "surging sparks": "sv08",
                "chispas fulgurantes": "sv08",
                "obsidian flames": "sv03",
                "llamas obsidianas": "sv03",
                "fusion strike": "swsh08",
                "golpe fusion": "swsh08",
                "golpe fusión": "swsh08",
                "heartgold": "hgss1",
                "soulsilver": "hgss1",
                "black & white": "bw1",
                "black and white": "bw1",
                "nero e bianco": "bw1",
            }
            for set_keyword, code in known_sets.items():
                if set_keyword in combined_lower:
                    metadata.set_code = code
                    set_is_invalid = False
                    break

        if set_is_invalid:
            metadata.set_code = None

        # Ensure physical card language is accurately detected using physical card cues & langdetect
        metadata.language = self.detect_card_language(
            ocr_text=ocr_text,
            seller_title=seller_title,
            seller_description=seller_description,
        )

        return metadata



    @staticmethod
    def _heuristic_fallback(
        ocr_text: str,
        seller_title: str,
        seller_description: str,
    ) -> ExtractedCardMetadata:
        """Deterministic regex fallback when local LLM is unavailable or fails."""
        combined = f"{seller_title} {seller_description} {ocr_text}"

        # 1. Card number pattern: e.g. "025/165", "4/102", "136/201", "25"
        num_match = re.search(r"\b(\d{1,3}/\d{2,3})\b", combined)
        collector_number = num_match.group(1) if num_match else None
        if not collector_number:
            single_match = re.search(r"№?\s*(\d{1,3})\b", ocr_text)
            if single_match:
                collector_number = single_match.group(1).lstrip("0") or "0"

        # 2. Language detection for physical card
        lang = EntityParser.detect_card_language(
            ocr_text=ocr_text,
            seller_title=seller_title,
            seller_description=seller_description,
        )

        # 3. Holo detection
        holo_keywords = ["holo", "holografica", "holográfica", "foil", "reverse", "brillante"]
        is_holo = any(kw in combined.lower() for kw in holo_keywords)

        # 4. Name extraction
        # Try finding primary Pokemon name from title
        title_tokens = [
            w for w in seller_title.split()
            if w.lower() not in ["carta", "pokemon", "en", "de", "la", "el", "pack", "lote", "con"]
        ]
        card_name = " ".join(title_tokens[:3]) if title_tokens else "Pokemon Card"

        return ExtractedCardMetadata(
            card_name=card_name,
            set_code=None,
            collector_number=collector_number,
            attacks=[],
            language=lang,
            is_holo=is_holo,
        )


def parse_listing(
    ocr_text: str,
    seller_title: str,
    seller_description: str,
    model_path: Optional[str] = None,
) -> ExtractedCardMetadata:
    """Convenience function to parse a listing using EntityParser."""
    parser = EntityParser(model_path=model_path)
    return parser.parse_listing(ocr_text, seller_title, seller_description)


detect_card_language = EntityParser.detect_card_language
