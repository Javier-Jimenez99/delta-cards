"""Full-OCR-text candidate ranking."""

from typing import Any, Dict, List

from fuzzywuzzy import fuzz


def rank_candidates(
    query_text: str,
    candidates: List[Dict[str, Any]],
    candidate_texts: Dict[str, str],
    threshold: float,
) -> Dict[str, Any]:
    scored = []
    for candidate in candidates:
        card = candidate.get("card", {})
        key = f"{card.get('tcgdex_id', '')}_{(card.get('lang') or 'en').lower()}"
        candidate_text = candidate_texts.get(key, "")
        score = fuzz.token_set_ratio(query_text, candidate_text) / 100 if query_text and candidate_text else 0.0
        scored.append((score, {**candidate, "combined_score": round(score, 4), "candidate_text": candidate_text}))
    scored.sort(key=lambda item: item[0], reverse=True)
    score, candidate = scored[0] if scored else (0.0, None)
    return {"verified_candidate": candidate if score >= threshold else None, "confidence": round(score, 4), "is_verified": score >= threshold, "ranked_candidates": [item[1] for item in scored]}