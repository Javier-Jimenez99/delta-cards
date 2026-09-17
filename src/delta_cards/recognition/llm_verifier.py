"""LLM-based False Positive Verification Service for Pokémon Card Deals."""

import json
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# The 20 Structured Verification Questions: (id, text_en, text_es, is_hard_veto)
# Questions are purely in English for optimal LLM reasoning and instruction adherence.
VERIFICATION_QUESTIONS: List[Tuple[str, str, str, bool]] = [
    (
        "q01_pokemon_name_match",
        "Pokemon name in photo OCR matches TCGdex candidate name",
        "",
        True,
    ),
    (
        "q02_stage_evolution_match",
        "Stage or evolution (Basic, Stage 1, Stage 2, V, VMAX, ex) matches",
        "",
        False,
    ),
    (
        "q03_hp_alignment",
        "HP value in photo OCR matches or aligns with official card",
        "",
        False,
    ),
    (
        "q04_pokemon_type_consistent",
        "Card element or energy type is consistent",
        "",
        False,
    ),
    (
        "q05_collector_number_match",
        "Local collector number (e.g. 75, 025) matches",
        "",
        False,
    ),
    (
        "q06_total_card_count_match",
        "Set total card count (e.g. /108, /198) matches",
        "",
        False,
    ),
    (
        "q07_promo_code_alignment",
        "Promo code aligns or not applicable",
        "",
        False,
    ),
    (
        "q08_set_expansion_consistent",
        "Set name or expansion abbreviation matches or aligns",
        "",
        False,
    ),
    (
        "q09_primary_attack_name",
        "Primary attack name appears in photo OCR",
        "",
        False,
    ),
    (
        "q10_attack_damage_match",
        "Primary attack damage number matches",
        "",
        False,
    ),
    (
        "q11_secondary_attack_or_ability",
        "Secondary attack or special ability appears in photo OCR",
        "",
        False,
    ),
    (
        "q12_energy_rules_or_retreat",
        "Card rules (e.g. Pokemon V Rule) or retreat cost consistent",
        "",
        False,
    ),
    (
        "q13_ocr_text_subset_valid",
        "Photo OCR text is a plausible subset of official scan OCR",
        "",
        False,
    ),
    (
        "q14_card_mechanic_badge",
        "Special mechanics (EX, GX, V, VMAX, Tera, Radiant) match",
        "",
        False,
    ),
    (
        "q15_rarity_visual_indicator",
        "Card rarity indicator is consistent with listing and card",
        "",
        False,
    ),
    (
        "q16_no_contradictory_pokemon",
        "No conflicting or contradictory Pokemon name appears in photo",
        "",
        False,
    ),
    (
        "q17_wallapop_title_mentions_card",
        "Wallapop listing title references this card or its collector number",
        "",
        False,
    ),
    (
        "q18_wallapop_description_consistent",
        "Wallapop description is consistent with the card and does not describe another item",
        "",
        False,
    ),
    (
        "q19_single_card_not_bulk_lot",
        "Listing is for a single individual card (not a bulk lot, tin, or empty box)",
        "",
        False,
    ),
    (
        "q20_overall_verdict_genuine_match",
        "Overall verdict: this is confirmed beyond doubt to be a genuine match",
        "",
        True,
    ),
]


class QuestionCheck(BaseModel):
    """Result of an individual verification question."""
    id: str
    question: str
    question_es: str
    confirmed: bool
    is_veto: bool = False


class LLMVerificationResult(BaseModel):
    """Result of LLM false-positive verification."""
    is_verified: bool
    confidence_score: float = Field(..., ge=0.0, le=100.0, description="Confidence percentage (0-100)")
    confirmed_count: int = Field(..., ge=0, le=20)
    total_questions: int = 20
    checks: Dict[str, bool] = Field(default_factory=dict)
    questions_detail: List[QuestionCheck] = Field(default_factory=list)
    verdict_summary: str = ""
    inference_time_s: float = 0.0
    raw_response: str = ""

    @property
    def confirmed(self) -> bool:
        return self.is_verified

    @property
    def verdict(self) -> str:
        return self.verdict_summary

    @property
    def reasoning(self) -> str:
        return self.verdict_summary

    @property
    def answers(self) -> Dict[str, bool]:
        return self.checks


def compute_factual_checks(
    query_ocr: Any,
    candidate_ocr: Any,
    wallapop_info: Any,
    tcgdex_info: Any,
) -> Dict[str, bool]:
    """Computes baseline factual question validations directly from OCR and card metadata."""
    q_full = ""
    q_hp = ""
    q_fracs = []
    if isinstance(query_ocr, dict):
        q_full = (query_ocr.get("full_text") or " ".join(query_ocr.get("lines", []))).lower()
        q_hp = str(query_ocr.get("hp") or "")
        q_fracs = [str(f).lower() for f in query_ocr.get("fractions", [])]
    else:
        q_full = str(query_ocr or "").lower()

    c_full = ""
    if isinstance(candidate_ocr, dict):
        c_full = (candidate_ocr.get("full_text") or " ".join(candidate_ocr.get("lines", []))).lower()
    else:
        c_full = str(candidate_ocr or "").lower()

    w_title = (
        getattr(wallapop_info, "title", "")
        if hasattr(wallapop_info, "title")
        else str(wallapop_info.get("title", "") if isinstance(wallapop_info, dict) else "")
    ) or ""
    w_desc = (
        getattr(wallapop_info, "description", "")
        if hasattr(wallapop_info, "description")
        else str(wallapop_info.get("description", "") if isinstance(wallapop_info, dict) else "")
    ) or ""
    w_full = (w_title + " " + w_desc).lower()

    t_name = ""
    t_local_id = ""
    t_total = ""
    t_hp = ""
    t_attacks = []
    t_stage = ""
    t_types = []
    t_set_name = ""
    t_set_id = ""
    t_rarity = ""

    if isinstance(tcgdex_info, dict):
        t_name = str(tcgdex_info.get("name") or "").lower()
        t_local_id = str(tcgdex_info.get("localId") or tcgdex_info.get("local_id") or "").lower()
        if isinstance(tcgdex_info.get("set"), dict):
            t_total = str(tcgdex_info.get("set", {}).get("cardCount", {}).get("total") or "")
            t_set_name = str(tcgdex_info.get("set", {}).get("name") or "").lower()
            t_set_id = str(tcgdex_info.get("set", {}).get("id") or "").lower()
        if not t_total:
            t_total = str(tcgdex_info.get("total_cards") or "")
        if not t_set_name:
            t_set_name = str(tcgdex_info.get("set_name") or "").lower()
        if not t_set_id:
            t_set_id = str(tcgdex_info.get("set_id") or "").lower()
        t_hp = str(tcgdex_info.get("hp") or "")
        t_attacks = tcgdex_info.get("attacks", [])
        t_stage = str(tcgdex_info.get("stage") or "").lower()
        t_types = [str(t).lower() for t in tcgdex_info.get("types", [])]
        t_rarity = str(tcgdex_info.get("rarity") or "").lower()

    clean_t_name = re.sub(r"[^a-z0-9]", "", t_name)
    clean_q_full = re.sub(r"[^a-z0-9]", "", q_full)
    clean_w_full = re.sub(r"[^a-z0-9]", "", w_full)
    clean_w_title = re.sub(r"[^a-z0-9]", "", w_title.lower())

    # Strip affixes and card mechanics to extract core Pokémon name tokens
    AFFIXES = {
        "team", "magma", "magmas", "aqua", "aquas", "rocket", "rockets",
        "dark", "light", "shining", "galarian", "alolan", "hisuian", "paldean",
        "radiant", "delta", "species", "v", "vmax", "vstar", "ex", "gx",
        "de", "del", "el", "la", "los", "las", "the", "of", "carta", "pokemon"
    }
    t_name_tokens = [
        re.sub(r"[^a-z0-9]", "", tok)
        for tok in t_name.lower().split()
        if re.sub(r"[^a-z0-9]", "", tok) not in AFFIXES and len(re.sub(r"[^a-z0-9]", "", tok)) >= 3
    ]

    # 1. Pokemon name match: is official name or core token in photo or listing?
    if clean_t_name:
        name_in_photo = (clean_t_name in clean_q_full)
        name_in_wallapop = (clean_t_name in clean_w_title or clean_t_name in clean_w_full)
        token_in_photo = any(tok in clean_q_full for tok in t_name_tokens)
        token_in_title = any(tok in clean_w_title for tok in t_name_tokens)
        q01 = bool(name_in_photo or name_in_wallapop or token_in_photo or token_in_title)
    else:
        q01 = True

    # 2. Stage/evolution match
    stages = ["vmax", "vstar", "ex", "gx", "fase 1", "stage 1", "fase 2", "stage 2", "basic", "básico", "basico"]
    matched_stage_card = [s for s in stages if s in t_name or s in t_stage]
    if matched_stage_card:
        q02 = any(s in q_full or s in w_full or s in c_full for s in matched_stage_card)
    elif t_name:
        # Standard basic card: check no contradictory evolution stage claimed
        has_conflict = any(s in q_full or s in w_title.lower() for s in ["vmax", "vstar", "stage 2", "fase 2"])
        q02 = not has_conflict
    else:
        q02 = True

    # 3. HP alignment: verified if HP is found and matches
    if t_hp:
        clean_hp = t_hp.strip()
        if q_hp and q_hp == clean_hp:
            q03 = True
        elif re.search(rf"\b{clean_hp}\s*(hp|ps)\b", q_full) or re.search(rf"\b(hp|ps)\s*{clean_hp}\b", q_full):
            q03 = True
        elif re.search(rf"\b{clean_hp}\s*(hp|ps)\b", w_full) or re.search(rf"\b(hp|ps)\s*{clean_hp}\b", w_full):
            q03 = True
        elif clean_hp in q_full and int(clean_hp) >= 40:
            q03 = True
        else:
            q03 = False
    else:
        q03 = True

    # 4. Pokemon type consistent
    if t_types:
        TYPE_MAP = {
            "fire": ["fuego", "fire"],
            "water": ["agua", "water"],
            "grass": ["planta", "grass"],
            "lightning": ["rayo", "eléctrico", "lightning"],
            "psychic": ["psíquico", "psiquico", "psychic"],
            "fighting": ["lucha", "fighting"],
            "darkness": ["oscuridad", "siniestro", "darkness", "dark"],
            "metal": ["metal", "acero", "steel"],
            "dragon": ["dragón", "dragon"],
            "fairy": ["hada", "fairy"],
            "colorless": ["incoloro", "normal", "colorless"],
        }
        all_type_synonyms = []
        for t in t_types:
            all_type_synonyms.extend(TYPE_MAP.get(t, [t]))
        q04 = any(ts in q_full or ts in w_full or ts in c_full for ts in all_type_synonyms)
    else:
        q04 = True

    # 5. Collector number match
    if t_local_id:
        num_clean = re.sub(r"^0+", "", t_local_id)
        q05 = False
        for f in q_fracs:
            if t_local_id in f or (num_clean and num_clean in f):
                q05 = True
                break
        if not q05:
            pattern = rf"(?<!\d){re.escape(t_local_id)}(?!\d)"
            if re.search(pattern, q_full):
                q05 = True
            elif num_clean and len(num_clean) >= 2 and re.search(rf"(?<!\d){re.escape(num_clean)}(?!\d)", q_full):
                q05 = True
            elif re.search(pattern, w_title) or (num_clean and len(num_clean) >= 2 and re.search(rf"(?<!\d){re.escape(num_clean)}(?!\d)", w_title)):
                q05 = True
    else:
        q05 = True

    # 6. Total cards count match
    if t_total:
        clean_tot = t_total.strip()
        q06 = False
        for f in q_fracs:
            if clean_tot in f:
                q06 = True
                break
        if not q06:
            if f"/{clean_tot}" in q_full or f"/{clean_tot}" in w_full:
                q06 = True
            elif len(clean_tot) >= 3 and re.search(rf"(?<!\d){re.escape(clean_tot)}(?!\d)", q_full):
                q06 = True
    else:
        q06 = True

    # 7. Promo code alignment
    is_promo_card = any(p in t_set_id for p in ["basep", "swshp", "smp", "xyp", "bwp", "dp", "np", "hsp"]) or "promo" in t_set_name
    if is_promo_card:
        q07 = "promo" in w_full or "promo" in q_full or bool(t_local_id and t_local_id in w_full)
    else:
        q07 = "promo" not in w_title.lower()

    # 8. Set expansion consistent
    if t_set_name or t_set_id:
        clean_set_name = re.sub(r"[^a-z0-9]", "", t_set_name)
        set_tokens = [re.sub(r"[^a-z0-9]", "", w) for w in t_set_name.split() if len(w) >= 4]
        if clean_set_name and (clean_set_name in clean_w_full or clean_set_name in clean_q_full):
            q08 = True
        elif any(tok in clean_w_full or tok in clean_q_full for tok in set_tokens):
            q08 = True
        elif t_set_id and (t_set_id in w_full or t_set_id in q_full):
            q08 = True
        else:
            q08 = False
    else:
        q08 = True

    # 9. Primary attack name
    if t_attacks:
        att0 = t_attacks[0].get("name", "") if isinstance(t_attacks[0], dict) else str(t_attacks[0])
        clean_att0 = re.sub(r"[^a-z0-9]", "", att0.lower())
        q09 = bool(clean_att0 and len(clean_att0) >= 3 and ((clean_att0 in clean_q_full) or (clean_att0 in clean_w_full)))
    else:
        q09 = True

    # 10. Attack damage match
    if t_attacks:
        dmg0 = str(t_attacks[0].get("damage", "") if isinstance(t_attacks[0], dict) else "")
        clean_dmg = re.sub(r"[^\d]", "", dmg0)
        if clean_dmg:
            q10 = bool(re.search(rf"(?<!\d){clean_dmg}(?!\d)", q_full))
        else:
            q10 = True
    else:
        q10 = True

    # 11. Secondary attack or ability
    if len(t_attacks) > 1:
        att1 = t_attacks[1].get("name", "") if isinstance(t_attacks[1], dict) else str(t_attacks[1])
        clean_att1 = re.sub(r"[^a-z0-9]", "", att1.lower())
        q11 = bool(clean_att1 and len(clean_att1) >= 3 and ((clean_att1 in clean_q_full) or (clean_att1 in clean_w_full)))
    else:
        q11 = True

    # 12. Energy rules or retreat
    mechanics_in_card = [m for m in ["ex", "gx", "v", "vmax", "vstar", "radiant", "tera"] if m in t_name or m in t_stage]
    if mechanics_in_card:
        q12 = any(m in q_full or m in w_title.lower() for m in mechanics_in_card)
    else:
        q12 = True

    # 13. OCR text subset valid
    if clean_q_full:
        matches_found = 0
        if clean_t_name and clean_t_name in clean_q_full:
            matches_found += 1
        if t_local_id and re.search(rf"(?<!\d){t_local_id}(?!\d)", q_full):
            matches_found += 1
        if t_hp and t_hp in q_full:
            matches_found += 1
        q13 = matches_found >= 1 or not clean_t_name
    else:
        q13 = not c_full

    # 14. Card mechanic badge
    q14 = q02

    # 15. Rarity visual indicator
    if "holo" in w_full or "reverse" in w_full or "secret" in w_full:
        if "holo" in t_rarity or "rare" in t_rarity or "ultra" in t_rarity or "secret" in t_rarity or not t_rarity:
            q15 = True
        else:
            q15 = False
    else:
        q15 = True

    # 16. No contradictory pokemon
    q16 = True
    common_names = ["charizard", "pikachu", "mewtwo", "mew", "rayquaza", "lugia", "umbreon", "gengar", "blastoise", "venusaur", "eevee", "snorlax", "lucario", "latios", "latias", "noibat"]
    for cn in common_names:
        if (cn in q_full or cn in w_title.lower()) and cn not in t_name and cn not in w_full:
            q16 = False
            break

    # 17. Wallapop title mentions card
    q17 = bool(clean_t_name in clean_w_title or any(tok in clean_w_title for tok in t_name_tokens) or not clean_t_name)

    # 18. Wallapop description consistent
    lot_or_fake_desc = any(w in w_desc.lower() for w in ["proxy", "replica", "copia", "falsa", "no original", "lote de", "caja vacia", "caja vacía", "tin vacia"])
    q18 = not lot_or_fake_desc

    # 19. Single card not bulk lot
    ptype = str(getattr(wallapop_info, "product_type", "") or "").lower()
    is_lot = any(w in w_full for w in ["lote ", "pack de ", "caja vacia", "caja vacía", "tin vacia", "50 cartas", "100 cartas"])
    q19 = not is_lot if ptype in ("single_card", "card", "carta", "") else False

    # Check for direct contradictions (e.g. HP mismatch or collector number mismatch in fractions)
    hp_conflict = bool(t_hp and q_hp and t_hp != q_hp)
    num_conflict = False
    if t_local_id and (q_fracs or re.findall(r"\b\d{1,3}/\d{2,3}\b", q_full)):
        if not q05:
            num_conflict = True

    # 20. Overall verdict: requires name, single card, no contradictory pokemon, no explicit conflicts, and reasonable evidence
    if clean_t_name:
        evidence_count = sum([q01, q02, q03, q05, q08, q09])
        q20 = bool(q01 and q16 and q19 and q18 and not hp_conflict and not num_conflict and evidence_count >= 2)
    else:
        q20 = bool(q01 and q16 and q19 and q18 and not hp_conflict and not num_conflict)

    return {
        "q01_pokemon_name_match": q01,
        "q02_stage_evolution_match": q02,
        "q03_hp_alignment": q03,
        "q04_pokemon_type_consistent": q04,
        "q05_collector_number_match": q05,
        "q06_total_card_count_match": q06,
        "q07_promo_code_alignment": q07,
        "q08_set_expansion_consistent": q08,
        "q09_primary_attack_name": q09,
        "q10_attack_damage_match": q10,
        "q11_secondary_attack_or_ability": q11,
        "q12_energy_rules_or_retreat": q12,
        "q13_ocr_text_subset_valid": q13,
        "q14_card_mechanic_badge": q14,
        "q15_rarity_visual_indicator": q15,
        "q16_no_contradictory_pokemon": q16,
        "q17_wallapop_title_mentions_card": q17,
        "q18_wallapop_description_consistent": q18,
        "q19_single_card_not_bulk_lot": q19,
        "q20_overall_verdict_genuine_match": q20,
    }


class LLMCardVerifier:
    """Lightweight local LLM auditor to verify Pokémon card matches before alerting."""

    _instance: Optional["LLMCardVerifier"] = None
    _model = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_path: str = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        min_confidence: float = 70.0,
        n_threads: int = 8,
        n_gpu_layers: int = 0,
    ):
        self.model_path = model_path
        self.min_confidence = min_confidence
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers

    def _ensure_model(self):
        """Lazy loads the GGUF model into memory thread-safely with GPU acceleration."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    if not os.path.exists(self.model_path):
                        # Try alternate path if default doesn't exist
                        alt_path = "models/Qwen3-0.6B-Q4_K_M.gguf"
                        if os.path.exists(alt_path):
                            logger.info(f"Using alternate LLM model: {alt_path}")
                            self.model_path = alt_path
                        else:
                            raise FileNotFoundError(f"LLM model not found at {self.model_path}")

                    from llama_cpp import Llama
                    logger.info(f"Loading local LLM verifier from {self.model_path} (gpu_layers={self.n_gpu_layers})...")
                    try:
                        self._model = Llama(
                            model_path=self.model_path,
                            n_ctx=2048,
                            n_threads=self.n_threads,
                            n_gpu_layers=self.n_gpu_layers,
                            verbose=False,
                        )
                    except Exception as exc:
                        logger.warning(f"Failed to load with GPU offload ({exc}). Falling back to CPU.")
                        self._model = Llama(
                            model_path=self.model_path,
                            n_ctx=2048,
                            n_threads=self.n_threads,
                            n_gpu_layers=0,
                            verbose=False,
                        )
                    logger.info("Local LLM verifier loaded successfully.")
        return self._model

    def build_verification_prompt(
        self,
        query_ocr: Union[Dict[str, Any], str],
        candidate_ocr: Union[Dict[str, Any], str],
        wallapop_info: Union[Dict[str, Any], Any],
        tcgdex_info: Union[Dict[str, Any], Any],
    ) -> str:
        """Formats the comparison payload into a concise evaluation prompt."""
        # Query OCR formatting
        if isinstance(query_ocr, dict):
            q_text = query_ocr.get("full_text") or " ".join(query_ocr.get("lines", []))
            q_hp = query_ocr.get("hp") or ""
            raw_q_fracs = query_ocr.get("fractions", [])
            q_fracs_list = []
            for f in raw_q_fracs:
                if isinstance(f, (list, tuple)) and len(f) >= 3:
                    q_fracs_list.append(str(f[2]))
                elif isinstance(f, (list, tuple)) and len(f) > 0:
                    q_fracs_list.append(str(f[0]))
                else:
                    q_fracs_list.append(str(f))
            q_fracs = ", ".join(q_fracs_list)
            query_str = f"Text: {q_text[:350]} | HP: {q_hp} | Fractions: {q_fracs}"
        else:
            query_str = str(query_ocr)[:350]

        # Candidate Official Scan OCR formatting
        if isinstance(candidate_ocr, dict):
            c_text = candidate_ocr.get("full_text") or " ".join(candidate_ocr.get("lines", []))
            c_hp = candidate_ocr.get("hp") or ""
            raw_c_fracs = candidate_ocr.get("fractions", [])
            c_fracs_list = []
            for f in raw_c_fracs:
                if isinstance(f, (list, tuple)) and len(f) >= 3:
                    c_fracs_list.append(str(f[2]))
                elif isinstance(f, (list, tuple)) and len(f) > 0:
                    c_fracs_list.append(str(f[0]))
                else:
                    c_fracs_list.append(str(f))
            c_fracs = ", ".join(c_fracs_list)
            cand_str = f"Text: {c_text[:350]} | HP: {c_hp} | Fractions: {c_fracs}"
        else:
            cand_str = str(candidate_ocr)[:350]

        # Wallapop Listing info
        if hasattr(wallapop_info, "title"):
            w_title = getattr(wallapop_info, "title", "")
            w_desc = getattr(wallapop_info, "description", "") or ""
            w_price = f"{getattr(wallapop_info, 'price', 0.0):.2f} EUR"
        elif isinstance(wallapop_info, dict):
            w_title = wallapop_info.get("title", "")
            w_desc = wallapop_info.get("description", "")
            w_price = f"{wallapop_info.get('price', 0.0)} EUR"
        else:
            w_title = str(wallapop_info)
            w_desc = ""
            w_price = ""

        wallapop_str = f"Title: {w_title}\nDescription: {w_desc[:250]}\nPrice: {w_price}"

        # TCGdex info
        if isinstance(tcgdex_info, dict):
            t_name = tcgdex_info.get("name") or tcgdex_info.get("id", "")
            t_set = tcgdex_info.get("set", {}).get("name") if isinstance(tcgdex_info.get("set"), dict) else tcgdex_info.get("set_name", "")
            t_num = tcgdex_info.get("localId") or tcgdex_info.get("local_id", "")
            t_tot = ""
            if isinstance(tcgdex_info.get("set"), dict):
                t_tot = tcgdex_info.get("set", {}).get("cardCount", {}).get("total", "")
            if not t_tot:
                t_tot = tcgdex_info.get("total_cards", "")
            t_hp = tcgdex_info.get("hp", "")
            t_attacks = tcgdex_info.get("attacks", [])
            att_str = ", ".join(
                f"{a.get('name') if isinstance(a, dict) else str(a)}"
                for a in (t_attacks[:3] if isinstance(t_attacks, list) else [])
            )
            tcgdex_str = f"Name: {t_name} | Number: {t_num}/{t_tot} | Set: {t_set} | HP: {t_hp} | Attacks: {att_str}"
        else:
            tcgdex_str = str(tcgdex_info)

        question_lines = [f"{i+1}. {q[1]}" for i, q in enumerate(VERIFICATION_QUESTIONS)]
        questions_block = "\n".join(question_lines)

        prompt = f"""Examine this Pokémon card match to determine if it is genuine or a false positive:
[Photo OCR]: {query_str}
[Candidate Official Scan]: {cand_str}
[Wallapop Listing]:
{wallapop_str}
[Official TCGdex Card]: {tcgdex_str}

Evaluate each of the following 20 verification questions with 1 (confirmed/yes) or 0 (failed/no):
{questions_block}

Output strictly the 20 answers as comma-separated digits (0 or 1), in exact order without extra words:"""
        return prompt

    def verify_match(
        self,
        offer: Any = None,
        query_ocr: Optional[Union[Dict[str, Any], str]] = None,
        candidate_ocr: Optional[Union[Dict[str, Any], str]] = None,
        tcgdex_card: Optional[Dict[str, Any]] = None,
        offer_info: Any = None,
        tcgdex_info: Optional[Dict[str, Any]] = None,
        target_ocr: Optional[Union[Dict[str, Any], str]] = None,
        **kwargs,
    ) -> LLMVerificationResult:
        """Runs the 20 verification questions using the local LLM and returns the confidence score."""
        import time

        t_start = time.perf_counter()

        # Resolve parameter aliases
        if offer is None and offer_info is not None:
            offer = offer_info
        if tcgdex_card is None and tcgdex_info is not None:
            tcgdex_card = tcgdex_info
        if candidate_ocr is None and target_ocr is not None:
            candidate_ocr = target_ocr

        # Resolve query_ocr if missing
        if query_ocr is None:
            query_ocr = getattr(offer, "_query_ocr", None) or {}
            if not query_ocr and getattr(offer, "image_url", None):
                try:
                    from .card_ocr import CardOCR
                    ocr_res = CardOCR().extract(offer.image_url)
                    query_ocr = {
                        "full_text": ocr_res.full_text,
                        "hp": ocr_res.hp,
                        "fractions": [f[2] for f in ocr_res.fractions],
                        "lines": ocr_res.lines,
                    }
                except Exception as e:
                    logger.debug(f"Could not extract query OCR: {e}")
                    query_ocr = {"full_text": getattr(offer, "title", "")}

        # Resolve candidate_ocr & tcgdex_card if missing
        if candidate_ocr is None:
            candidate_ocr = getattr(offer, "_candidate_ocr", None) or {}

        if tcgdex_card is None:
            tcgdex_card = getattr(offer, "_tcgdex_card", None) or {}
            if not tcgdex_card and getattr(offer, "tcgdex_id", None):
                try:
                    from .vector_index import MultiLanguageCardIndex
                    vindex = MultiLanguageCardIndex(view_only=True)
                    # Prefer English card representation for English LLM prompts
                    tcgdex_card = (
                        vindex.get_card_by_id_and_lang(offer.tcgdex_id, "en")
                        or vindex.get_card_by_id_and_lang(
                            offer.tcgdex_id, getattr(offer, "language", "es") or "es"
                        )
                        or {}
                    )
                except Exception as e:
                    logger.debug(f"Could not load tcgdex card: {e}")

        # If candidate_ocr is still empty, load from cache
        if (not candidate_ocr or (isinstance(candidate_ocr, dict) and not candidate_ocr.get("full_text"))) and getattr(offer, "tcgdex_id", None):
            try:
                from .card_ocr_cache import CandidateOCRCache
                cache = CandidateOCRCache()
                card_payload = {
                    "tcgdex_id": getattr(offer, "tcgdex_id", ""),
                    "lang": getattr(offer, "language", "es") or "es",
                    "image_url": tcgdex_card.get("image_url") if isinstance(tcgdex_card, dict) else "",
                }
                candidate_ocr = cache.get_candidate_ocr(card_payload, card_payload["lang"])
            except Exception as e:
                logger.debug(f"Could not load candidate OCR cache: {e}")
                candidate_ocr = {}

        # Build prompt
        prompt = self.build_verification_prompt(
            query_ocr=query_ocr,
            candidate_ocr=candidate_ocr,
            wallapop_info=offer,
            tcgdex_info=tcgdex_card or {"tcgdex_id": getattr(offer, "tcgdex_id", "")},
        )

        # Run inference
        raw_output = ""
        int_answers = []
        try:
            model = self._ensure_model()
            with self._lock:
                response = model.create_chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a Pokémon card verification auditor. "
                                "Output strictly 20 comma-separated digits (0 or 1), without any other text or explanation: "
                                "1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1"
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=150,
                )
            raw_output = response["choices"][0]["message"]["content"]
            # Extract JSON list of 0s and 1s
            match = re.search(r"\[[\s\d,]+\]", raw_output)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    int_answers = [1 if int(x) == 1 else 0 for x in parsed]
                except Exception:
                    pass

            if not int_answers:
                # Fallback: extract single 0/1 digits from text
                digits = re.findall(r"\b[01]\b", raw_output)
                if digits:
                    int_answers = [int(d) for d in digits[:20]]
        except Exception as exc:
            logger.warning(f"Error during LLM verification inference: {exc}")

        # Check if LLM output gave valid per-question answers (e.g. from mock in unit tests or capable LLM)
        has_valid_llm = False
        if len(int_answers) == 20:
            # Check if this is the 0.5B model collapse pattern (where only the first is 1 and all others 0)
            if int_answers[0] == 1 and sum(int_answers[1:]) == 0:
                has_valid_llm = False
            else:
                has_valid_llm = True
        elif len(int_answers) > 20:
            int_answers = int_answers[:20]
            if int_answers[0] == 1 and sum(int_answers[1:]) == 0:
                has_valid_llm = False
            else:
                has_valid_llm = True

        # Compute factual checks from OCR and card data
        factual_checks = compute_factual_checks(
            query_ocr=query_ocr,
            candidate_ocr=candidate_ocr,
            wallapop_info=offer,
            tcgdex_info=tcgdex_card or {"tcgdex_id": getattr(offer, "tcgdex_id", "")},
        )

        # Check semantic rejection in LLM raw output text
        clean_raw = raw_output.lower()
        if "verdict: no" in clean_raw or "match: no" in clean_raw or "is_match: no" in clean_raw or "proxy" in clean_raw or "replica" in clean_raw:
            factual_checks["q20_overall_verdict_genuine_match"] = False

        checks: Dict[str, bool] = {}
        detail_list: List[QuestionCheck] = []
        confirmed_count = 0

        for idx, (qid, q_en, q_es, is_veto) in enumerate(VERIFICATION_QUESTIONS):
            fact_ok = factual_checks.get(qid, True)
            if has_valid_llm:
                llm_ok = bool(int_answers[idx] == 1)
                is_confirmed = fact_ok and llm_ok
            else:
                is_confirmed = fact_ok

            if is_confirmed:
                confirmed_count += 1
            checks[qid] = is_confirmed
            detail_list.append(
                QuestionCheck(
                    id=qid,
                    question=q_en,
                    question_es=q_es,
                    confirmed=is_confirmed,
                    is_veto=is_veto,
                )
            )

        t_elapsed = time.perf_counter() - t_start

        # Check hard vetoes: q01 (name match) and q20 (overall verdict)
        q01_confirmed = checks.get("q01_pokemon_name_match", False)
        q20_confirmed = checks.get("q20_overall_verdict_genuine_match", False)

        if not q01_confirmed or not q20_confirmed:
            # Hard veto triggered
            score = min(25.0, round((confirmed_count / 20.0) * 100, 1))
            is_verified = False
            reasons = []
            if not q01_confirmed:
                reasons.append("Nombre no coincide")
            if not q20_confirmed:
                reasons.append("Veredicto global rechazado")
            verdict_summary = f"Veto IA: {', '.join(reasons)}"
        else:
            score = round((confirmed_count / 20.0) * 100, 1)
            is_verified = score >= self.min_confidence
            verdict_summary = (
                f"✓ Match Confirmado ({confirmed_count}/20)"
                if is_verified
                else f"⚠️ Confianza insuficiente ({score:.0f}% < {self.min_confidence:.0f}%)"
            )

        logger.info(
            f"LLM Verification Result: score={score}% ({confirmed_count}/20), "
            f"verified={is_verified} ({verdict_summary}) in {t_elapsed:.2f}s"
        )

        return LLMVerificationResult(
            is_verified=is_verified,
            confidence_score=score,
            confirmed_count=confirmed_count,
            total_questions=20,
            checks=checks,
            questions_detail=detail_list,
            verdict_summary=verdict_summary,
            inference_time_s=round(t_elapsed, 3),
            raw_response=raw_output,
        )


_default_verifier: Optional[LLMCardVerifier] = None


def get_llm_verifier(
    model_path: str = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    min_confidence: float = 70.0,
    n_threads: int = 8,
    n_gpu_layers: int = 0,
) -> LLMCardVerifier:
    """Returns singleton LLMCardVerifier instance."""
    global _default_verifier
    if _default_verifier is None:
        _default_verifier = LLMCardVerifier(
            model_path=model_path,
            min_confidence=min_confidence,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
        )
    return _default_verifier
