"""Constants and mappings for Cardmarket scraping and URL parameters."""

from typing import Dict

# Cardmarket numeric language identifier map
CARDMARKET_LANGUAGE_MAP: Dict[str, int] = {
    "en": 1,
    "english": 1,
    "fr": 2,
    "french": 2,
    "de": 3,
    "german": 3,
    "es": 4,
    "spanish": 4,
    "it": 5,
    "italian": 5,
    "zh-hans": 6,
    "s-chinese": 6,
    "ja": 7,
    "japanese": 7,
    "pt": 8,
    "portuguese": 8,
    "ru": 9,
    "russian": 9,
    "ko": 10,
    "korean": 10,
    "zh-hant": 11,
    "t-chinese": 11,
    "nl": 12,
    "dutch": 12,
    "pl": 13,
    "polish": 13,
    "cs": 14,
    "czech": 14,
    "hu": 15,
    "hungarian": 15,
}

# Country code for Spain on Cardmarket
SPAIN_SELLER_COUNTRY_ID: int = 10

# Base Cardmarket URL
CARDMARKET_BASE_URL: str = "https://www.cardmarket.com"
