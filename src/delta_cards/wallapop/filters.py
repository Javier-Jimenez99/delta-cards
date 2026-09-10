"""Filtering heuristics based strictly on keywords and criteria defined in the JSON configuration."""

import re
from typing import Dict, Any, Tuple
from ..models import SearchProfile


def is_offer_valid(
    raw_item: Dict[str, Any],
    profile: SearchProfile,
) -> Tuple[bool, str]:
    """Validates an offer item against search profile criteria.

    All keywords and exclusion terms are driven purely by the JSON configuration profile.
    No hardcoded words or sealed flags exist in this code.

    Args:
        raw_item: The raw dictionary returned by WallaPy.
        profile: The active SearchProfile.

    Returns:
        Tuple[bool, str]: (is_valid, reason)
    """
    title = (raw_item.get("title") or "").lower()
    description = (raw_item.get("description") or "").lower()
    full_text = f"{title} {description}"

    price = raw_item.get("price")
    if price is None:
        return False, "Missing price"

    try:
        price_num = float(price)
    except (ValueError, TypeError):
        return False, f"Invalid price format: {price}"

    # 1. Price boundaries
    if profile.min_price is not None and price_num < profile.min_price:
        return False, f"Price {price_num} is below min_price {profile.min_price}"
    if profile.max_price is not None and price_num > profile.max_price:
        return False, f"Price {price_num} exceeds max_price {profile.max_price}"

    # 2. Excluded keywords check (disqualify if any excluded keyword matches)
    for ex in profile.excluded_keywords:
        pattern = rf"\b{re.escape(ex.lower())}\b"
        if re.search(pattern, full_text):
            return False, f"Matched excluded keyword: '{ex}'"

    # 3. Positive keywords check (at least one keyword from JSON must match)
    if profile.keywords:
        matched = False
        for kw in profile.keywords:
            kw_lower = kw.lower().strip()
            if not kw_lower:
                continue
            pattern = rf"\b{re.escape(kw_lower)}\b" if " " not in kw_lower else re.escape(kw_lower)
            if re.search(pattern, full_text):
                matched = True
                break
        if not matched:
            return False, "None of the profile keywords matched title or description"

    return True, "Valid"
