"""Full-card OCR extraction using OpenCV image pre-processing and Tesseract OCR."""

import logging
import os
import re
import shutil
from typing import List, Optional

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)


def _configure_tesseract_binary() -> str:
    """Configures pytesseract binary location, checking local environment and system PATH."""
    candidates = [
        os.path.expanduser("~/.local/bin/tesseract"),
        shutil.which("tesseract"),
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return candidate

    return pytesseract.pytesseract.tesseract_cmd


class CardOCRReader:
    """Extracts high-accuracy text from rectified trading cards using OpenCV and Tesseract."""

    def __init__(
        self,
        tesseract_cmd: Optional[str] = None,
        default_psm: int = 6,
        languages: Optional[List[str]] = None,
    ):
        """Initializes the OCR reader.

        Args:
            tesseract_cmd: Custom path to tesseract binary, if any.
            default_psm: Page segmentation mode (default 6: uniform block of text, 11: sparse text).
            languages: Priority list of OCR language codes (e.g., ["eng", "spa", "jpn"]).
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        else:
            _configure_tesseract_binary()

        self.default_psm = default_psm
        self.desired_languages = languages or ["eng", "spa", "jpn"]
        self.lang_arg = self._resolve_available_languages()

    def _resolve_available_languages(self) -> str:
        """Determines which desired languages are installed in Tesseract with graceful fallback."""
        try:
            installed = pytesseract.get_languages()
        except Exception as exc:
            logger.debug(f"Failed to query tesseract languages: {exc}. Defaulting to eng.")
            return "eng"

        matched = [lang for lang in self.desired_languages if lang in installed]
        if matched:
            return "+".join(matched)

        if "eng" in installed:
            return "eng"

        return installed[0] if installed else "eng"

    def preprocess_image(self, cropped_card: np.ndarray) -> np.ndarray:
        """Preprocesses rectified card image to optimize OCR accuracy.

        Steps:
        1. Grayscale conversion.
        2. Contrast Limited Adaptive Histogram Equalization (CLAHE).
        3. Bilateral filtering for noise suppression while preserving sharp text edges.
        4. Adaptive binarization to cleanly separate printed text from card holo foil/textures.

        Args:
            cropped_card: BGR image of the rectified card.

        Returns:
            Preprocessed binary image suitable for OCR.
        """
        # 1. Grayscale
        if cropped_card.ndim == 3:
            gray = cv2.cvtColor(cropped_card, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropped_card.copy()

        # 2. Contrast enhancement via CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray)

        # 3. Bilateral filter to smooth texture while keeping character edges crisp
        smoothed = cv2.bilateralFilter(contrast_enhanced, d=7, sigmaColor=50, sigmaSpace=50)

        # 4. Adaptive thresholding (Gaussian binarization)
        binary = cv2.adaptiveThreshold(
            smoothed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=4,
        )

        return binary

    def extract_text(
        self,
        cropped_card: np.ndarray,
        psm: Optional[int] = None,
        use_multi_pass: bool = True,
    ) -> str:
        """Executes full-card Tesseract OCR and normalizes textual output.

        Args:
            cropped_card: NumPy array of the cropped card (BGR).
            psm: Optional override for Page Segmentation Mode.
            use_multi_pass: If True, combines preprocessed binary and grayscale passes
                           for maximal text retention across holo/foil textures.

        Returns:
            Cleaned and normalized raw text output across the entire card face.
        """
        if cropped_card is None or cropped_card.size == 0:
            return ""

        active_psm = psm if psm is not None else self.default_psm
        config = f"--psm {active_psm} -l {self.lang_arg}"

        # First pass: preprocessed adaptive binary
        binary_img = self.preprocess_image(cropped_card)
        try:
            text_pass_1 = pytesseract.image_to_string(binary_img, config=config)
        except Exception as exc:
            logger.warning(f"Tesseract OCR failed on binary pass: {exc}. Trying fallback.")
            text_pass_1 = ""

        if not use_multi_pass:
            return self.normalize_text(text_pass_1)

        # Second pass: grayscale with CLAHE directly (handles dark backgrounds and inverted font)
        if cropped_card.ndim == 3:
            gray = cv2.cvtColor(cropped_card, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropped_card
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        try:
            # Mode 11 (sparse text) or Mode 6 on enhanced grayscale
            fallback_psm = 11 if active_psm == 6 else 6
            text_pass_2 = pytesseract.image_to_string(
                enhanced_gray, config=f"--psm {fallback_psm} -l {self.lang_arg}"
            )
        except Exception:
            text_pass_2 = ""

        # Merge results uniquely preserving line structure
        combined_lines = []
        seen = set()
        for raw in (text_pass_1 + "\n" + text_pass_2).splitlines():
            clean_line = raw.strip()
            if clean_line and clean_line.lower() not in seen:
                seen.add(clean_line.lower())
                combined_lines.append(clean_line)

        return self.normalize_text("\n".join(combined_lines))

    @staticmethod
    def normalize_text(text: str) -> str:
        """Cleans and normalizes noisy OCR output while preserving card numbers and names."""
        if not text:
            return ""

        # Remove non-printable characters
        cleaned = re.sub(r"[^\x20-\x7E\n\u00C0-\u017F\u3040-\u30FF\u4E00-\u9FFF]", " ", text)
        # Collapse excessive spaces within lines
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.splitlines()]
        # Remove empty lines and trailing spaces
        return "\n".join(line for line in lines if line)


def extract_text(cropped_card: np.ndarray) -> str:
    """Convenience function to extract text using CardOCRReader."""
    reader = CardOCRReader()
    return reader.extract_text(cropped_card)
