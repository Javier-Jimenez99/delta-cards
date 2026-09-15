"""Small local LLM confirmation step for an OCR match."""

import os
import re
import threading
import time
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field


class LLMVerificationResult(BaseModel):
    is_verified: bool
    confidence_score: float = Field(ge=0.0, le=100.0)
    confirmed_count: int = 0
    total_questions: int = 1
    checks: Dict[str, bool] = {}
    questions_detail: list = []
    verdict_summary: str = ""
    inference_time_s: float = 0.0
    raw_response: str = ""


def _full_text(value: Union[Dict[str, Any], str, None]) -> str:
    return value.get("full_text", "") if isinstance(value, dict) else str(value or "")


class LLMCardVerifier:
    """Ask one yes/no question about two complete OCR texts."""

    _model = None
    _lock = threading.Lock()

    def __init__(self, model_path: str = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf", min_confidence: float = 70.0, n_threads: int = 8, n_gpu_layers: int = 0):
        self.model_path, self.min_confidence, self.n_threads, self.n_gpu_layers = model_path, min_confidence, n_threads, n_gpu_layers

    def _ensure_model(self):
        if self._model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"LLM model not found: {self.model_path}")
            from llama_cpp import Llama
            self._model = Llama(model_path=self.model_path, n_ctx=1024, n_threads=self.n_threads, n_gpu_layers=self.n_gpu_layers, verbose=False)
        return self._model

    def build_verification_prompt(self, query_ocr: Union[Dict[str, Any], str], candidate_ocr: Union[Dict[str, Any], str], wallapop_info: Any = None, tcgdex_info: Any = None) -> str:
        return f"Do these two OCR texts describe the same Pokemon card? Reply only YES or NO.\nPhoto: {_full_text(query_ocr)[:500]}\nOfficial: {_full_text(candidate_ocr)[:500]}"

    def verify_match(self, offer: Any, query_ocr: Optional[Union[Dict[str, Any], str]] = None, candidate_ocr: Optional[Union[Dict[str, Any], str]] = None, tcgdex_card: Optional[Dict[str, Any]] = None) -> LLMVerificationResult:
        started = time.perf_counter()
        prompt = self.build_verification_prompt(query_ocr or getattr(offer, "_query_ocr", ""), candidate_ocr or getattr(offer, "_candidate_ocr", ""))
        raw_response = self._ensure_model().create_chat_completion(messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=3)["choices"][0]["message"]["content"].strip()
        confirmed = bool(re.match(r"yes\b", raw_response, re.IGNORECASE))
        return LLMVerificationResult(is_verified=confirmed, confidence_score=100.0 if confirmed else 0.0, confirmed_count=int(confirmed), checks={"full_ocr_text_match": confirmed}, verdict_summary="Match confirmed" if confirmed else "Match rejected", inference_time_s=round(time.perf_counter() - started, 3), raw_response=raw_response)


_default_verifier: Optional[LLMCardVerifier] = None


def get_llm_verifier(model_path: str = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf", min_confidence: float = 70.0, n_threads: int = 8, n_gpu_layers: int = 0) -> LLMCardVerifier:
    global _default_verifier
    if _default_verifier is None:
        _default_verifier = LLMCardVerifier(model_path, min_confidence, n_threads, n_gpu_layers)
    return _default_verifier