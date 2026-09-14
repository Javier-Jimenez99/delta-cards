"""Full-card OCR extraction and language detection using RapidOCR ONNX."""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Patterns for key Pokémon card elements
RE_FRACTION = re.compile(r"\b([A-Za-z0-9]{1,8})\s*[/|\\lI]\s*(\d{1,4})[A-Za-z]*\b", re.IGNORECASE)
RE_SPACE_FRACTION = re.compile(r"\b(\d{1,4})\s+(\d{2,4})\b")
RE_HP = re.compile(
    r"(?:(\d{2,3})\s*(?:hp|ps|pv|kp)\b|\b(?:hp|ps|pv|kp)\s*(\d{2,3}))",
    re.IGNORECASE,
)
RE_PROMO = re.compile(r"\b(SVP|SWSH|SM|XY|BW|DP|HGSS|P-A|WP)\s*#?([0-9]{1,4})\b", re.IGNORECASE)
RE_JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")

# Language marker words
SPANISH_KEYWORDS = {
    "básico", "basico", "roba", "daño", "dano", "desecha", "descarta",
    "entrenador", "energía", "energia", "partidario", "evolución", "evolucion",
    "durante", "turno", "habilidad", "objeto", "combate", "fase", "resistencia",
    "debilidad", "retirada", "puntos", "moneda"
}
ENGLISH_KEYWORDS = {
    "basic", "stage", "draw", "damage", "discard", "trainer", "energy",
    "supporter", "during", "turn", "ability", "item", "weakness", "resistance",
    "retreat", "counters", "coin", "flip", "attached", "active", "benched"
}
FRENCH_KEYWORDS = {
    "base", "niveau", "piochez", "dégâts", "degats", "faiblesse", "retraite",
    "talent", "défaussez", "defaussez", "tour", "banc", "énergie"
}
GERMAN_KEYWORDS = {
    "basis", "phase", "ziehe", "schaden", "schwäche", "schwaeche", "rückzug",
    "rueckzug", "fähigkeit", "faehigkeit", "ablagestapel", "energie"
}


@dataclass
class CardOCRResult:
    """Consolidated result from full-card OCR and language detection."""

    full_text: str
    lines: List[str] = field(default_factory=list)
    detected_language: str = "es"
    hp: Optional[int] = None
    fractions: List[Tuple[str, str, str]] = field(default_factory=list)  # (num, total, full)
    promo_codes: List[str] = field(default_factory=list)
    top_text: str = ""
    bottom_text: str = ""
    raw_results: List[Any] = field(default_factory=list)


class CardOCR:
    """Extracts complete text from rectified card images across English, Spanish, and Japanese."""

    def __init__(self, limit_side_len: int = 960):
        """Initializes the RapidOCR engine with multilingual (ES/EN/JA) support."""
        from rapidocr_onnxruntime import RapidOCR

        models_dir = os.path.join(os.getcwd(), "models", "ocr")
        rec_model = os.path.join(models_dir, "japan_PP-OCRv4_rec.onnx")
        rec_keys = os.path.join(models_dir, "japan_dict.txt")

        # Auto-download multilingual model if not present
        if not (os.path.isfile(rec_model) and os.path.isfile(rec_keys)):
            os.makedirs(models_dir, exist_ok=True)
            import urllib.request
            try:
                if not os.path.isfile(rec_keys):
                    urllib.request.urlretrieve(
                        "https://huggingface.co/cycloneboy/japan_PP-OCRv4_rec_infer/resolve/main/japan_dict.txt",
                        rec_keys,
                    )
                if not os.path.isfile(rec_model):
                    urllib.request.urlretrieve(
                        "https://huggingface.co/cycloneboy/japan_PP-OCRv4_rec_infer/resolve/main/model.onnx",
                        rec_model,
                    )
            except Exception as exc:
                logger.warning(f"Could not auto-download multilingual OCR model: {exc}. Using default.")

        rapid_kwargs = {
            "use_cls": False,
            "det_limit_side_len": limit_side_len,
        }
        if os.path.isfile(rec_model) and os.path.isfile(rec_keys):
            rapid_kwargs["rec_model_path"] = rec_model
            rapid_kwargs["rec_keys_path"] = rec_keys

        # Detect whether CUDAExecutionProvider is actually functional.
        # rapidocr-onnxruntime picks up CUDAExecutionProvider automatically when
        # onnxruntime-gpu is installed — no extra kwargs needed.
        # We only store _gpu to optimise the OCR worker count in index_builder.
        self._gpu = False
        try:
            import onnxruntime as ort
            import onnx
            from onnx import helper, TensorProto

            if "CUDAExecutionProvider" in ort.get_available_providers():
                _X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
                _Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
                _graph = helper.make_graph(
                    [helper.make_node("Identity", ["X"], ["Y"])], "probe", [_X], [_Y]
                )
                _model_bytes = helper.make_model(
                    _graph, opset_imports=[helper.make_opsetid("", 13)]
                ).SerializeToString()
                _sess = ort.InferenceSession(
                    _model_bytes,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                if "CUDAExecutionProvider" in _sess.get_providers():
                    self._gpu = True
                    logger.info("CardOCR: GPU (CUDAExecutionProvider) available. ✓")
        except Exception:
            pass

        self.ocr = RapidOCR(**rapid_kwargs)


    def extract(self, card_image: np.ndarray) -> CardOCRResult:
        """Executes full-card OCR on the rectified upright card image.

        Args:
            card_image: BGR NumPy array of the rectified card.

        Returns:
            CardOCRResult with full text, detected fields, and language.
        """
        if card_image is None or card_image.size == 0:
            return CardOCRResult(full_text="", detected_language="es")

        h, w = card_image.shape[:2]

        results, _ = self.ocr(card_image)
        if not results:
            return CardOCRResult(full_text="", detected_language="es")

        lines = []
        top_lines = []
        bottom_lines = []

        top_threshold = int(h * 0.32)
        bottom_threshold = int(h * 0.70)

        for item in results:
            if len(item) < 2 or not item[1]:
                continue
            text = str(item[1]).strip()
            if not text:
                continue
            lines.append(text)

            # Spatial partition based on bounding box
            box = item[0]
            try:
                # box is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                y_center = sum(pt[1] for pt in box) / 4.0
                if y_center <= top_threshold:
                    top_lines.append(text)
                elif y_center >= bottom_threshold:
                    bottom_lines.append(text)
            except Exception:
                pass

        full_text = " ".join(lines)
        top_text = " ".join(top_lines)
        bottom_text = " ".join(bottom_lines)

        # 1. Detect HP
        hp = self._extract_hp(top_text, is_top=True) or self._extract_hp(full_text, is_top=False)

        # 2. Detect collector numbers and fractions
        fractions = self._extract_fractions(bottom_text) or self._extract_fractions(full_text)

        # 3. Detect promo codes
        promos = self._extract_promos(bottom_text) or self._extract_promos(full_text)

        # 4. Detect language
        detected_lang = self._detect_language(full_text, top_text, hp)

        return CardOCRResult(
            full_text=full_text,
            lines=lines,
            detected_language=detected_lang,
            hp=hp,
            fractions=fractions,
            promo_codes=promos,
            top_text=top_text,
            bottom_text=bottom_text,
            raw_results=results,
        )

    def _extract_hp(self, text: str, is_top: bool = False) -> Optional[int]:
        """Extracts HP integer value from text."""
        m = RE_HP.search(text)
        if m:
            val_str = m.group(1) or m.group(2)
            try:
                val = int(val_str)
                if 30 <= val <= 350:
                    return val
            except (ValueError, TypeError):
                pass

        # Handle letter O instead of digit 0 (e.g. "7OHP" -> 70)
        m_o = re.search(r"\b([3-9])O\s*(?:hp|ps|pv|kp)\b", text, re.IGNORECASE)
        if m_o:
            return int(m_o.group(1)) * 10

        # Loose numeric fallback: ONLY allowed in top_text (header) to avoid matching attack damage in full_text
        if is_top:
            candidates = re.findall(r"(?:[円¥hHpPsSvVkKoO0]|\b)(3[0-4]0|[12][0-9]0|[3-9]0)\b", text)
            if candidates:
                valid_hps = [int(c) for c in candidates if 30 <= int(c) <= 340 and int(c) % 10 == 0]
                if valid_hps:
                    return valid_hps[0]

        return None

    def _extract_fractions(self, text: str) -> List[Tuple[str, str, str]]:
        """Extracts (numerator, denominator, full_str) fractions."""
        fractions = []
        seen = set()

        for m in RE_FRACTION.finditer(text):
            num = m.group(1).strip()
            total = m.group(2).strip()
            # Clean non-digit prefixes from num if any
            clean_num = re.sub(r"^[A-Za-z]+", "", num)
            if clean_num and (clean_num, total) not in seen:
                seen.add((clean_num, total))
                fractions.append((clean_num, total, f"{clean_num}/{total}"))

        for m in RE_SPACE_FRACTION.finditer(text):
            n, tot = m.group(1), m.group(2)
            if int(n) <= int(tot) and int(tot) >= 30 and (n, tot) not in seen:
                seen.add((n, tot))
                fractions.append((n, tot, f"{n}/{tot}"))

        return fractions

    def _extract_promos(self, text: str) -> List[str]:
        """Extracts promo codes like SVP 025 or SWSH 123."""
        promos = []
        for m in RE_PROMO.finditer(text):
            prefix = m.group(1).upper()
            num = m.group(2)
            promos.append(f"{prefix}{num}")
            promos.append(f"{prefix} {num}")
            promos.append(f"{prefix}-{num}")
        return promos

    def _detect_language(self, full_text: str, top_text: str, hp: Optional[int]) -> str:
        """Determines the physical card language from text markers and keywords."""
        text_lower = full_text.lower()

        # 1. Japanese (Kanji, Hiragana, Katakana)
        if RE_JAPANESE.search(full_text):
            return "ja"

        # 2. HP Marker specifics
        # German uses KP (e.g. "120 KP")
        if re.search(r"\b\d{2,3}\s*kp\b|\bkp\s*\d{2,3}\b", text_lower):
            return "de"

        # French often uses PV
        if re.search(r"\b\d{2,3}\s*pv\b|\bpv\s*\d{2,3}\b", text_lower):
            return "fr"

        # Spanish uses PS (e.g. "110 PS")
        has_ps = bool(re.search(r"\b\d{2,3}\s*ps\b|\bps\s*\d{2,3}\b", text_lower))

        # 3. Vocabulary word counts
        tokens = set(re.findall(r"[a-záéíóúñü]+", text_lower))

        es_count = len(tokens.intersection(SPANISH_KEYWORDS))
        en_count = len(tokens.intersection(ENGLISH_KEYWORDS))
        fr_count = len(tokens.intersection(FRENCH_KEYWORDS))
        de_count = len(tokens.intersection(GERMAN_KEYWORDS))

        if has_ps:
            es_count += 3

        scores = [
            ("es", es_count),
            ("en", en_count),
            ("fr", fr_count),
            ("de", de_count),
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_lang, top_score = scores[0]

        if top_score >= 2:
            return top_lang

        # 4. Fallback: langdetect on alphabetical words
        try:
            import langdetect

            clean_alpha = re.sub(r"[0-9\W]+", " ", full_text).strip()
            if len(clean_alpha) > 15:
                lang = langdetect.detect(clean_alpha)
                if lang in ("es", "en", "ja", "de", "fr", "it", "pt"):
                    return lang
        except Exception:
            pass

        # Default: if PS is present -> es, else en
        if has_ps:
            return "es"
        return "en"
