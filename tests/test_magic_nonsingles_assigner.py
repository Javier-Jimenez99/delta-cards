"""Unit tests for Magic: The Gathering Cardmarket Non-Singles (Sealed Products) Matcher and Assigner."""

import pytest
from delta_cards.models import OfferListing
from delta_cards.nonsingles import (
    MAGIC_CATEGORY_TO_PRODUCT_TYPE,
    MAGIC_FORMAT_TO_CATEGORIES,
    NonSinglesMatcher,
    assign_nonsingles_to_offers,
    resolve_product_type_from_category,
)
from delta_cards.settings import Settings


def test_magic_matcher_loads():
    """Verify Magic matcher initializes catalog and keywords without errors."""
    matcher = NonSinglesMatcher.get_instance(game="magic")
    assert matcher is not None
    assert len(matcher.df_catalog) > 5000
    assert matcher.bm25 is not None
    assert matcher.game == "magic"


def test_magic_resolve_product_type_from_category():
    """Verify Magic Cardmarket categories are mapped to normalized product types."""
    assert resolve_product_type_from_category("Magic Display", game="magic") == "booster_box"
    assert resolve_product_type_from_category("Magic Booster", game="magic") == "booster_pack"
    assert resolve_product_type_from_category("Magic Fatpack", game="magic") == "bundle"
    assert resolve_product_type_from_category("Magic Intropack", game="magic") == "deck"
    assert resolve_product_type_from_category("Magic Starter Deck", game="magic") == "deck"
    assert resolve_product_type_from_category("Magic TournamentPack", game="magic") == "box_set"
    assert resolve_product_type_from_category("Magic Theme Deck Display", game="magic") == "box_set"
    assert resolve_product_type_from_category("MtG Set", game="magic") == "box_set"
    assert resolve_product_type_from_category("Magic Lot", game="magic") == "bulk"
    assert resolve_product_type_from_category("Unknown Category", game="magic") == "sealed"


def test_magic_matcher_booster_box():
    """Verify Magic booster box listings match display category."""
    matcher = NonSinglesMatcher.get_instance(game="magic")

    # Bloomburrow Play Booster Box
    ents, matches = matcher.search_product("Caja de sobres Bloomburrow Play booster precintada", top_k=1)
    assert matches, "Expected match for Bloomburrow Play booster box"
    assert "Bloomburrow Play Booster Box" in matches[0]["name"]
    assert matches[0]["categoryName"] == "Magic Display"

    # Modern Horizons 3 Display
    ents, matches = matcher.search_product("Display Modern Horizons 3 precintado", top_k=1)
    assert matches, "Expected match for Modern Horizons 3 display"
    assert "Modern Horizons 3" in matches[0]["name"]
    assert matches[0]["categoryName"] == "Magic Display"

    # Duskmourn Collector Booster Box
    ents, matches = matcher.search_product("Duskmourn Collector Booster Box sealed", top_k=1)
    assert matches, "Expected match for Duskmourn Collector Booster Box"
    assert "Duskmourn: House of Horror Collector Booster Box" in matches[0]["name"]
    assert matches[0]["categoryName"] == "Magic Display"


def test_magic_matcher_bundle():
    """Verify Magic bundle listings match fatpack category."""
    matcher = NonSinglesMatcher.get_instance(game="magic")

    # Foundations Bundle
    ents, matches = matcher.search_product("Bundle Foundations español precintado", top_k=1)
    assert matches, "Expected match for Foundations bundle"
    assert "Foundations Fat Pack Bundle" in matches[0]["name"]
    assert matches[0]["categoryName"] == "Magic Fatpack"

    # Modern Horizons 3 Gift Bundle
    ents, matches = matcher.search_product("Modern Horizons 3 Gift Bundle sealed", top_k=1)
    assert matches, "Expected match for Modern Horizons 3 Gift Edition Bundle"
    assert "Modern Horizons 3 Fat Pack Bundle: Gift Edition" in matches[0]["name"]
    assert matches[0]["categoryName"] == "Magic Fatpack"


def test_magic_matcher_booster_pack():
    """Verify Magic individual booster packs match booster category."""
    matcher = NonSinglesMatcher.get_instance(game="magic")

    # Modern Horizons 2 Collector Booster
    ents, matches = matcher.search_product("Sobre de Modern Horizons 2 Collector", top_k=1)
    assert matches, "Expected match for MH2 Collector Booster"
    assert "Modern Horizons 2 Collector Booster" in matches[0]["name"]
    assert matches[0]["categoryName"] == "Magic Booster"

    # Aetherdrift Play Booster
    ents, matches = matcher.search_product("Sobre Aetherdrift Play Booster precintado", top_k=1)
    assert matches, "Expected match for Aetherdrift Play Booster"
    assert "Aetherdrift Play Booster" in matches[0]["name"]
    assert matches[0]["categoryName"] == "Magic Booster"


def test_magic_matcher_exclusions():
    """Verify non-sealed Magic listings (empty boxes, lots, bulk, sleeves) are safely rejected."""
    matcher = NonSinglesMatcher.get_instance(game="magic")
    excluded_titles = [
        "Caja vacia de sobres Magic Modern Horizons 3",
        "Lote 500 cartas magic the gathering bulk",
        "Morralla de cartas magic sueltas",
        "Carpeta y fundas Magic the Gathering",
        "Caja sola Bloomburrow sin sobres",
    ]
    for title in excluded_titles:
        ents, matches = matcher.search_product(title, top_k=1)
        assert not matches, f"Listing '{title}' should have been rejected as non-sealed, but matched: {matches}"


def test_magic_matcher_format_only_skipped():
    """Verify format-only listings without a set do not produce blind false positives."""
    matcher = NonSinglesMatcher.get_instance(game="magic")
    format_only_titles = [
        "Booster Box Magic precintada",
        "Magic Bundle precintado",
        "Sobre Magic precintado",
    ]
    for title in format_only_titles:
        ents, matches = matcher.search_product(title, top_k=1)
        assert not matches, f"Format-only title '{title}' should not match blindly, but matched: {matches}"


def test_magic_assign_nonsingles_to_offers():
    """Verify assign_nonsingles_to_offers enriches Magic listings with Cardmarket URLs and prices."""
    offers = [
        OfferListing(
            id="magic_bb_1",
            title="Caja de sobres Bloomburrow Play booster precintada",
            price=120.0,
            currency="EUR",
            product_type="non_single",
            game="magic",
            link="https://es.wallapop.com/item/magic_bb_1",
        ),
        OfferListing(
            id="magic_bundle_1",
            title="Bundle Foundations español precintado",
            price=35.0,
            currency="EUR",
            product_type="non_single",
            game="magic",
            link="https://es.wallapop.com/item/magic_bundle_1",
        ),
        OfferListing(
            id="magic_pack_1",
            title="Sobre de Modern Horizons 2 Collector",
            price=30.0,
            currency="EUR",
            product_type="non_single",
            game="magic",
            link="https://es.wallapop.com/item/magic_pack_1",
        ),
    ]

    assigned = assign_nonsingles_to_offers(offers)
    assert len(assigned) == 3

    # Bloomburrow Play Booster Box
    assert assigned[0].assigned_state == "assigned"
    assert assigned[0].cardmarket_product_id is not None
    assert assigned[0].product_type == "booster_box"
    assert "Magic/Products" in assigned[0].cardmarket_url
    assert assigned[0].cardmarket_trend_price is not None

    # Foundations Bundle
    assert assigned[1].assigned_state == "assigned"
    assert assigned[1].cardmarket_product_id is not None
    assert assigned[1].product_type == "bundle"
    assert "Magic/Products" in assigned[1].cardmarket_url

    # MH2 Collector Booster Pack
    assert assigned[2].assigned_state == "assigned"
    assert assigned[2].cardmarket_product_id is not None
    assert assigned[2].product_type == "booster_pack"
    assert "Magic/Products" in assigned[2].cardmarket_url


def test_magic_webhook_routing():
    """Verify Discord webhook URL routing for Magic sealed products."""
    settings = Settings(
        _env_file=None,
        magic_cards_webhook="https://discord.com/api/webhooks/magic_cards",
        magic_sealed_webhook="https://discord.com/api/webhooks/magic_sealed",
    )

    # Magic sealed products route to magic_sealed_webhook
    assert settings.get_webhook_url("magic", "booster_box") == "https://discord.com/api/webhooks/magic_sealed"
    assert settings.get_webhook_url("magic", "bundle") == "https://discord.com/api/webhooks/magic_sealed"
    assert settings.get_webhook_url("magic", "booster_pack") == "https://discord.com/api/webhooks/magic_sealed"

    # Magic singles route to magic_cards_webhook
    assert settings.get_webhook_url("magic", "single_card") == "https://discord.com/api/webhooks/magic_cards"

    # Fallback to magic_cards_webhook when magic_sealed_webhook is None
    settings_no_sealed = Settings(
        _env_file=None,
        magic_cards_webhook="https://discord.com/api/webhooks/magic_cards",
    )
    assert settings_no_sealed.get_webhook_url("magic", "booster_box") == "https://discord.com/api/webhooks/magic_cards"
