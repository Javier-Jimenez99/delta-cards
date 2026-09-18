"""Unit tests for Cardmarket Non-Singles (Sealed Products) Matcher and Assigner."""

import pytest
from delta_cards.models import OfferListing
from delta_cards.nonsingles import (
    FORMAT_TO_CATEGORIES,
    NonSinglesMatcher,
    assign_nonsingles_to_offers,
    resolve_product_type_from_category,
)
from delta_cards.settings import Settings


def test_nonsingles_matcher_loads():
    """Verify matcher initializes catalog and keywords without errors."""
    matcher = NonSinglesMatcher.get_instance()
    assert matcher is not None
    assert len(matcher.df_catalog) > 0
    assert matcher.bm25 is not None


def test_resolve_product_type_from_category():
    """Verify Cardmarket categories are mapped to normalized product types."""
    assert resolve_product_type_from_category("Pokémon Booster") == "booster_pack"
    assert resolve_product_type_from_category("Pokémon Blisters") == "booster_pack"
    assert resolve_product_type_from_category("Pokémon Display") == "booster_box"
    assert resolve_product_type_from_category("Pokémon Elite Trainer Boxes") == "etb"
    assert resolve_product_type_from_category("Pokémon Tins") == "tin"
    assert resolve_product_type_from_category("Pokémon Box Set") == "box_set"
    assert resolve_product_type_from_category("Pokémon Theme Decks") == "deck"
    assert resolve_product_type_from_category("Unknown Category") == "sealed"


def test_matcher_booster_box():
    """Verify booster box listing matches display category."""
    matcher = NonSinglesMatcher.get_instance()
    ents, matches = matcher.search_product("Booster Box Pokémon Mega Evolución", top_k=1)
    assert matches, "Expected a match for Mega Evolución booster box"
    assert "Mega Evolution Booster Box" in matches[0]["name"]
    assert matches[0]["categoryName"] == "Pokémon Display"


def test_matcher_etb():
    """Verify ETB listing matches elite trainer box category."""
    matcher = NonSinglesMatcher.get_instance()
    ents, matches = matcher.search_product("Caja Entrenador Élite Pokémon Fuerzas Fantasmales", top_k=1)
    assert matches, "Expected a match for Fuerzas Fantasmales ETB"
    assert "Phantom Forces Elite Trainer Box" in matches[0]["name"]
    assert matches[0]["categoryName"] == "Pokémon Elite Trainer Boxes"


def test_matcher_booster():
    """Verify booster pack listing matches booster category."""
    matcher = NonSinglesMatcher.get_instance()
    ents, matches = matcher.search_product("Sobre Pokémon Tempestad Plateada Español", top_k=1)
    assert matches, "Expected a match for Tempestad Plateada booster"
    assert "Silver Tempest Booster" in matches[0]["name"]
    assert matches[0]["categoryName"] == "Pokémon Booster"


def test_matcher_ambiguous_no_match():
    """Verify ambiguous listing without a set is safely filtered out (no false positive)."""
    matcher = NonSinglesMatcher.get_instance()
    ents, matches = matcher.search_product("Sobre Pokémon Venusaur Nuevo", top_k=1)
    assert not matches, "Ambiguous Venusaur booster pack should return no match to prevent false positives"


def test_matcher_non_sealed_exclusions():
    """Verify non-sealed listings (single cards, empty boxes, lots) are safely rejected."""
    matcher = NonSinglesMatcher.get_instance()
    excluded_titles = [
        "Carta #Pokemon #Zorua di N SVP 189 - Promo ETB Avventure Insieme ITA Sealed NM Prezzo Cadauno",
        "Promo upc 151 sealed e scatola con accessori ita",
        "Lot de 9 Carte Pokémon Ex",
        "Lote Pokémon Fuegos Fantasmales + Promo Charcadet Sellada + Taco Cartas + ETB Accesorios",
        "Pack Marco Expositor Pokemon TCG Pikachu VMAX Chino Simplificado Precintado",
        "Caja y sobres sueltos Violet Ex TCG Japonés precintada Pokémon auténticas",
    ]
    for title in excluded_titles:
        ents, matches = matcher.search_product(title, top_k=1)
        assert not matches, f"Listing '{title}' should have been rejected as non-sealed, but matched: {matches}"


def test_matcher_format_only_without_expansion_skipped():
    """Verify generic format-only listings without expansion do not trigger blind expensive defaults."""
    matcher = NonSinglesMatcher.get_instance()
    format_only_titles = [
        "ETB Pokemon Sealed",
        "Pokemon Booster Bundle Sealed",
        "Booster Box Pokémon Sellada",
    ]
    for title in format_only_titles:
        ents, matches = matcher.search_product(title, top_k=1)
        assert not matches, f"Format-only title '{title}' should not blindly match a specific product, but matched: {matches}"


def test_matcher_multilingual_modern_sets():
    """Verify modern Italian and Spanish set names are recognized correctly."""
    matcher = NonSinglesMatcher.get_instance()

    # Surging Sparks (Italian: Scintille Folgoranti)
    ents, matches = matcher.search_product("Pokémon Scintille Folgoranti Booster Bundle 6 Buste ITA Box Sealed Neuf", top_k=1)
    assert matches, "Expected match for Scintille Folgoranti Booster Bundle"
    assert "Surging Sparks" in matches[0]["name"]

    # White Flare ETB
    ents, matches = matcher.search_product("Pokémon White Flare ETB", top_k=1)
    assert matches, "Expected match for White Flare ETB"
    assert "White Flare Elite Trainer Box" in matches[0]["name"]

    # 30th Anniversary Italian ETB
    ents, matches = matcher.search_product("ETB 30 Anniversario Pokèmon ITA Sealed", top_k=1)
    assert matches, "Expected match for 30th Celebration ETB"
    assert "30th Celebration Elite Trainer Box" in matches[0]["name"]

    # Collection Box should NOT match Booster Box
    ents, matches = matcher.search_product("Caja pokemon greninja ex 30 aniversario sellada.", top_k=1)
    assert matches, "Expected match for Greninja ex collection box"
    assert "Greninja ex Box" in matches[0]["name"]
    assert matches[0]["categoryName"] == "Pokémon Box Set"


def test_assign_nonsingles_to_offers():
    """Verify assign_nonsingles_to_offers enriches listings in place."""
    offers = [
        OfferListing(
            id="offer_bb_1",
            title="Booster Box Pokémon Mega Evolución Sellada",
            price=150.0,
            currency="EUR",
            product_type="non_single",
            link="https://es.wallapop.com/item/offer_bb_1",
        ),
        OfferListing(
            id="offer_pack_1",
            title="Sobre Pokémon Tempestad Plateada Español",
            price=4.5,
            currency="EUR",
            product_type="non_single",
            link="https://es.wallapop.com/item/offer_pack_1",
        ),
    ]

    assigned = assign_nonsingles_to_offers(offers)
    assert len(assigned) == 2

    # Booster box offer
    assert assigned[0].assigned_state == "assigned"
    assert assigned[0].cardmarket_product_id is not None
    assert assigned[0].product_type == "booster_box"
    assert assigned[0].cardmarket_trend_price is not None

    # Booster pack offer
    assert assigned[1].assigned_state == "assigned"
    assert assigned[1].cardmarket_product_id is not None
    assert assigned[1].product_type == "booster_pack"


def test_webhook_routing_by_category():
    """Verify Discord webhook URLs are routed deterministically based on product_type."""
    settings = Settings(
        _env_file=None,
        pokemon_cards_webhook="https://discord.com/api/webhooks/cards",
        pokemon_booster_webhook="https://discord.com/api/webhooks/boosters",
        pokemon_booster_box_webhook="https://discord.com/api/webhooks/boxes",
        magic_cards_webhook="https://discord.com/api/webhooks/magic",
    )

    # Magic
    assert settings.get_webhook_url("magic") == "https://discord.com/api/webhooks/magic"

    # Single cards
    assert settings.get_webhook_url("pokemon", "single_card") == "https://discord.com/api/webhooks/cards"
    assert settings.get_webhook_url("pokemon", "card") == "https://discord.com/api/webhooks/cards"

    # Boosters / Blisters
    assert settings.get_webhook_url("pokemon", "booster_pack") == "https://discord.com/api/webhooks/boosters"
    assert settings.get_webhook_url("pokemon", "booster") == "https://discord.com/api/webhooks/boosters"
    assert settings.get_webhook_url("pokemon", "blister") == "https://discord.com/api/webhooks/boosters"

    # Booster Boxes / ETBs / Sealed Boxes
    assert settings.get_webhook_url("pokemon", "booster_box") == "https://discord.com/api/webhooks/boxes"
    assert settings.get_webhook_url("pokemon", "etb") == "https://discord.com/api/webhooks/boxes"
    assert settings.get_webhook_url("pokemon", "tin") == "https://discord.com/api/webhooks/boxes"
    assert settings.get_webhook_url("pokemon", "box_set") == "https://discord.com/api/webhooks/boxes"
