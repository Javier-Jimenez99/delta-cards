"""Fuzzy matching engine comparing Query OCR text against TCGdex reconstructed card candidates."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from fuzzywuzzy import fuzz

logger = logging.getLogger(__name__)


class CardTextMatcher:
    """Evaluates and reranks candidate cards by comparing OCR text with TCGdex reconstructed data."""

    def __init__(self, acceptance_threshold: float = 0.60):
        """Initializes matcher with customizable acceptance threshold."""
        self.acceptance_threshold = acceptance_threshold

    def match_candidates(
        self,
        ocr_result: Any,
        candidates: List[Dict[str, Any]],
        reconstructed_data_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compares query OCR text against all USearch candidates and ranks them.

        Args:
            ocr_result: CardOCRResult instance from CardOCR.
            candidates: List of candidate dicts from USearch (containing "card", "similarity").
            reconstructed_data_map: Dict mapping tcgdex_id to reconstructed card data.

        Returns:
            Dict containing:
                "verified_candidate": Best candidate dict or None.
                "confidence": Float between 0.0 and 1.0.
                "is_verified": Boolean True if above threshold.
                "method": Explanation of matching rationale.
                "ranked_candidates": Full list of scored and sorted candidates.
        """
        if not candidates:
            return {
                "verified_candidate": None,
                "confidence": 0.0,
                "is_verified": False,
                "method": "no_candidates",
                "ranked_candidates": [],
            }

        query_full = getattr(ocr_result, "full_text", "") or ""
        query_top = getattr(ocr_result, "top_text", "") or ""
        query_full_lower = query_full.lower()
        query_top_lower = query_top.lower()
        query_hp = getattr(ocr_result, "hp", None)
        query_fractions = getattr(ocr_result, "fractions", [])
        query_promos = getattr(ocr_result, "promo_codes", [])

        scored_candidates = []

        for cand in candidates:
            card_info = cand.get("card", {})
            tcgdex_id = card_info.get("tcgdex_id")
            vis_sim = float(cand.get("similarity", 0.0))
            cand_lang = (card_info.get("lang") or "").lower()
            rec_data = reconstructed_data_map.get(f"{tcgdex_id}_{cand_lang}") or reconstructed_data_map.get(tcgdex_id, {})
            cand_name = str(rec_data.get("name") or card_info.get("name") or "")
            cand_hp = rec_data.get("hp") or card_info.get("hp")
            cand_local_id = str(rec_data.get("local_id") or card_info.get("local_id") or "").lstrip("0")
            cand_total = str(rec_data.get("total_cards") or card_info.get("total_cards") or "")
            cand_fraction = str(rec_data.get("card_fraction") or card_info.get("card_fraction") or "")
            cand_rec_text = str(rec_data.get("reconstructed_text") or "")
            cand_attacks = rec_data.get("attacks", [])
            cand_abilities = rec_data.get("abilities", [])

            # 1. Name Similarity & Modifier Specificity
            name_score = 0.0
            modifier_bonus = 0.0
            if cand_name:
                cand_name_lower = cand_name.lower()
                tsr = fuzz.token_set_ratio(cand_name_lower, query_top_lower or query_full_lower)
                pr = fuzz.partial_ratio(cand_name_lower, query_top_lower or query_full_lower)
                name_score = float(max(tsr, pr))

                # Boost specific card variants (e.g. "team magma", "dark", "ex", "vmax", "gx")
                cand_words = set(re.findall(r"[a-z]{2,}", cand_name_lower))
                special_modifiers = {"magma", "aqua", "rocket", "dark", "ex", "vmax", "vstar", "gx", "prime", "mega", "shining"}
                cand_mods = cand_words.intersection(special_modifiers)
                if cand_mods:
                    matched_mods = [m for m in cand_mods if m in query_full_lower or m in query_top_lower]
                    if len(matched_mods) == len(cand_mods):
                        modifier_bonus = 15.0  # Boost specific variant
                    elif not matched_mods:
                        modifier_bonus = -10.0  # Candidate has modifier not present on card
                name_score = min(100.0, max(0.0, name_score + modifier_bonus))

            # 2. Fraction & Collector Number Matching
            fraction_score = 0.0
            fraction_conflict = False
            matched_fraction = ""

            for q_num, q_tot, q_frac in query_fractions:
                norm_q_num = q_num.lstrip("0")
                if cand_fraction and q_frac.lower() == cand_fraction.lower():
                    fraction_score = 100.0
                    matched_fraction = q_frac
                    break
                elif norm_q_num == cand_local_id and q_tot == cand_total:
                    fraction_score = 100.0
                    matched_fraction = q_frac
                    break
                elif norm_q_num == cand_local_id:
                    fraction_score = max(fraction_score, 85.0)
                    matched_fraction = norm_q_num

            # Check promo matching
            if fraction_score < 90.0 and query_promos:
                cid_clean = tcgdex_id.lower().replace("-", "").replace(" ", "")
                for promo in query_promos:
                    p_clean = promo.lower().replace("-", "").replace(" ", "")
                    if p_clean in cid_clean or cid_clean in p_clean:
                        fraction_score = 100.0
                        matched_fraction = promo
                        break

            # Fallback substring in full text
            if fraction_score < 70.0:
                if cand_fraction and cand_fraction.lower() in query_full_lower:
                    fraction_score = 95.0
                    matched_fraction = cand_fraction
                elif cand_local_id and re.search(r"\b0*" + re.escape(cand_local_id) + r"\b", query_full_lower):
                    fraction_score = 75.0
                    matched_fraction = cand_local_id

            # Detect fraction conflict: query has clear fraction but candidate has completely different one
            # Only treat as conflict if name is NOT strongly confirmed!
            # If the Pokémon name is an exact match (e.g. Meltan), a conflicting fraction
            # is almost always OCR noise from reflections/holo patterns.
            if query_fractions and fraction_score == 0.0 and cand_fraction:
                if name_score < 75.0:
                    fraction_conflict = True

            # 3. HP Matching
            hp_score = 50.0  # neutral by default
            if query_hp is not None and cand_hp is not None:
                try:
                    if int(query_hp) == int(cand_hp):
                        hp_score = 100.0
                    else:
                        hp_score = 0.0
                except (ValueError, TypeError):
                    hp_score = 50.0

            # 4. Attack / Ability Names Matching
            cand_lines = rec_data.get("lines", [])
            attack_score = 0.0
            attack_matches = []
            for at in cand_attacks:
                at_name = at.get("name", "")
                if at_name and len(at_name) >= 3:
                    sim = fuzz.partial_ratio(at_name.lower(), query_full_lower)
                    attack_matches.append(sim)
            for ab in cand_abilities:
                ab_name = ab.get("name", "")
                if ab_name and len(ab_name) >= 3:
                    sim = fuzz.partial_ratio(ab_name.lower(), query_full_lower)
                    attack_matches.append(sim)

            # If no attacks metadata, match intermediate lines from candidate scan OCR
            if not attack_matches and cand_lines and len(cand_lines) > 2:
                for line in cand_lines[1:-1]:
                    if 4 <= len(line) <= 25 and not line.isdigit():
                        sim = fuzz.partial_ratio(line.lower(), query_full_lower)
                        attack_matches.append(sim)

            if attack_matches:
                attack_score = float(max(attack_matches))

            # 5. Full Reconstructed Scan Text Similarity
            full_text_sim = 0.0
            if cand_rec_text and query_full:
                token_sim = float(fuzz.token_set_ratio(cand_rec_text.lower(), query_full_lower))
                partial_sim = float(fuzz.partial_ratio(cand_rec_text.lower(), query_full_lower))
                full_text_sim = max(token_sim, partial_sim * 0.85)

            # 6. Compute Composite Score
            text_composite = (
                0.35 * name_score
                + 0.35 * fraction_score
                + 0.15 * full_text_sim
                + 0.10 * hp_score
                + 0.05 * attack_score
            )

            # High-confidence exact deterministic booster
            is_deterministic_match = (fraction_score >= 95.0 and name_score >= 45.0)
            is_strong_text_match = (name_score >= 80.0 and (fraction_score >= 75.0 or not query_fractions))

            if is_deterministic_match:
                combined_score = 0.95 + 0.05 * vis_sim
                method = "fuzzy_exact_fraction_and_name"
            elif is_strong_text_match:
                combined_score = (0.40 * vis_sim) + (0.60 * (text_composite / 100.0))
                method = "fuzzy_strong_text_confirmed"
            elif fraction_conflict:
                # Heavy penalty if physical card fraction directly contradicts candidate fraction
                combined_score = (vis_sim * 0.45)
                method = "penalized_fraction_conflict"
            else:
                combined_score = (0.45 * vis_sim) + (0.55 * (text_composite / 100.0))
                method = "fuzzy_composite_rerank"

            # Base Set 2 (base4) tie-breaker:
            # If candidate is from Base Set 2 (reprint), check if "2000" or "Base Set 2" appears in query OCR.
            # If not detected, slightly deprioritize Base Set 2 to break micro-ties in favor of original Base Set (base1), Jungle (base2), or Fossil (base3).
            set_id_lower = str(rec_data.get("set_id") or card_info.get("set_id") or "").lower()
            if set_id_lower == "base4":
                has_2000 = "2000" in query_full_lower or "1999-2000" in query_full_lower
                has_set2_marker = bool(re.search(r"\bbase\s*set\s*2\b|\bset\s*2\b", query_full_lower))
                if not (has_2000 or has_set2_marker):
                    combined_score -= 0.015

            # Unmatched Pokémon name safeguard:
            # A candidate whose name does not match the card (name_score < 60) must never beat a verified Pokémon
            # simply due to an incidental number in OCR noise.
            if name_score < 60.0 and fraction_score < 95.0:
                combined_score = min(combined_score, 0.52)

            metrics = {
                "name_score": round(name_score, 1),
                "fraction_score": round(fraction_score, 1),
                "hp_score": round(hp_score, 1),
                "attack_score": round(attack_score, 1),
                "full_text_sim": round(full_text_sim, 1),
                "text_composite": round(text_composite, 1),
                "total_text_score": round(text_composite, 1),
                "matched_fraction": matched_fraction,
                "reconstructed_text": cand_rec_text,
                "cand_ocr_text": cand_rec_text,
                "cand_lines": cand_lines,
            }

            cand_entry = dict(cand)
            cand_entry["combined_score"] = round(combined_score, 4)
            cand_entry["fuzzy_metrics"] = metrics
            cand_entry["verification_method"] = method
            cand_entry["reconstructed_card"] = rec_data

            scored_candidates.append((combined_score, cand_entry, method))

        # Sort descending by combined score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_cand, best_method = scored_candidates[0]

        is_verified = (best_score >= self.acceptance_threshold)

        return {
            "verified_candidate": best_cand if is_verified else None,
            "confidence": round(best_score, 4),
            "is_verified": is_verified,
            "method": best_method if is_verified else "rejected_below_threshold",
            "ranked_candidates": [c[1] for c in scored_candidates],
        }
