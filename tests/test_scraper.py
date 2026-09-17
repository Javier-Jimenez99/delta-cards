"""Unit tests for delta_cards filters, config loader, and models."""

import pytest
from pathlib import Path
from delta_cards.config_loader import load_config
from delta_cards.models import SearchProfile
from delta_cards.wallapop.filters import is_offer_valid
from delta_cards.wallapop.scraper import raw_to_offer_listing


def test_load_default_config():
    config_path = Path(__file__).resolve().parent.parent / "config" / "pokemon_cards_search.json"
    assert config_path.is_file(), f"Config file not found at {config_path}"

    config = load_config(config_path)
    assert config.name == "Pokemon Cards Search Config"
    assert config.game == "pokemon"
    assert len(config.search_profiles) == 1

    profile_ids = [p.id for p in config.search_profiles]
    assert "single_card" in profile_ids

    # Verify extra_config has the non-single-card profiles
    extra_config_path = Path(__file__).resolve().parent.parent / "extra_config" / "pokemon_sealed_search.json"
    assert extra_config_path.is_file(), f"Extra config file not found at {extra_config_path}"
    extra_config = load_config(extra_config_path)
    assert len(extra_config.search_profiles) == 4
    extra_profile_ids = [p.id for p in extra_config.search_profiles]
    assert "booster_box" in extra_profile_ids
    assert "booster_pack" in extra_profile_ids
    assert "full_deck" in extra_profile_ids
    assert "elite_trainer_box" in extra_profile_ids

    bb_profile = next(p for p in extra_config.search_profiles if p.id == "booster_box")
    assert isinstance(bb_profile.queries, list)
    assert len(bb_profile.queries) > 1
    assert "pokemon booster box" in bb_profile.queries


def test_load_all_configs_ignores_extra_config():
    from delta_cards.config_loader import load_all_configs
    configs = load_all_configs("config")
    assert len(configs) >= 2
    games = {c.game for c in configs}
    assert "pokemon" in games
    assert "magic" in games
    for c in configs:
        for p in c.search_profiles:
            assert p.product_type == "single_card"


def test_query_variations_normalization():
    # String query
    p1 = SearchProfile(
        id="test1",
        name="Test 1",
        product_type="cards",
        query="single query",
    )
    assert p1.queries == ["single query"]

    # List query
    p2 = SearchProfile(
        id="test2",
        name="Test 2",
        product_type="cards",
        query=["query 1", "  query 2  ", "", "query 3"],
    )
    assert p2.queries == ["query 1", "query 2", "query 3"]


def test_keywords_filter():
    profile = SearchProfile(
        id="booster_box",
        name="Booster Box",
        product_type="booster_box",
        query=["pokemon booster box", "caja sobres pokemon"],
        keywords=["precintada", "precintado", "sin abrir", "sealed"],
        excluded_keywords=["vacia", "sin sobres"],
        min_price=70.0,
        max_price=500.0,
    )

    # Valid item matching positive keywords from JSON
    valid_item = {
        "id": "item1",
        "title": "Caja Pokemon Booster Box 151 Precintada",
        "description": "Nueva a estrenar, completamente sin abrir y sellada de fabrica.",
        "price": 180.0,
    }
    is_valid, reason = is_offer_valid(valid_item, profile)
    assert is_valid is True, f"Expected valid, got: {reason}"

    # Disqualified: matches excluded keyword from JSON
    empty_item = {
        "id": "item2",
        "title": "Caja Pokemon Booster Box 151 vacia",
        "description": "Solo caja vacia sin sobres para coleccionistas.",
        "price": 80.0,
    }
    is_valid, reason = is_offer_valid(empty_item, profile)
    assert is_valid is False
    assert "vacia" in reason or "sin sobres" in reason

    # Disqualified: price below min_price
    cheap_item = {
        "id": "item3",
        "title": "Caja Pokemon Booster Box Precintada",
        "description": "Sealed booster box.",
        "price": 5.0,  # below 70.0
    }
    is_valid, reason = is_offer_valid(cheap_item, profile)
    assert is_valid is False
    assert "below min_price" in reason

    # Disqualified: none of the keywords match
    unmatched_item = {
        "id": "item4",
        "title": "Caja Pokemon Booster Box 151",
        "description": "Abierta para ver el contenido.",
        "price": 120.0,
    }
    is_valid, reason = is_offer_valid(unmatched_item, profile)
    assert is_valid is False
    assert "None of the profile keywords matched" in reason


def test_single_card_filter_matches_card_keywords():
    profile = SearchProfile(
        id="single_card",
        name="Single Cards",
        product_type="single_card",
        query="pokemon carta",
        keywords=["carta", "pokemon", "charizard"],
        excluded_keywords=["codigo", "proxy", "falsa"],
        min_price=1.0,
    )

    # Valid single card matching keyword 'carta'
    card_item = {
        "id": "card1",
        "title": "Charizard Base Set Holo Español",
        "description": "Carta en buen estado, perfecta para coleccionistas.",
        "price": 120.0,
    }
    is_valid, reason = is_offer_valid(card_item, profile)
    assert is_valid is True, f"Single card should be valid, got: {reason}"

    # Disqualified: proxy card
    fake_card = {
        "id": "card2",
        "title": "Charizard VMAX Shiny Proxy",
        "description": "Carta no original proxy de alta calidad.",
        "price": 10.0,
    }
    is_valid, reason = is_offer_valid(fake_card, profile)
    assert is_valid is False
    assert "proxy" in reason


def test_raw_to_offer_listing_conversion():
    raw = {
        "id": "test_123",
        "title": "Pokemon Booster Pack 151 Precintado",
        "price": 8.5,
        "currency": "EUR",
        "location": "Madrid",
        "state": "ES",
        "description": "Sobre precintado sin pesar.",
        "link": "https://wallapop.com/item/pokemon-booster-pack-123",
        "main_image": "https://cdn.wallapop.com/img.jpg",
        "is_reserved": False,
        "shipping_available": True,
        "seller_link": "https://es.wallapop.com/user/seller_xyz",
    }

    offer = raw_to_offer_listing(
        raw=raw,
        profile_id="booster_pack",
        product_type="booster_pack",
    )

    assert offer.id == "test_123"
    assert offer.title == "Pokemon Booster Pack 151 Precintado"
    assert offer.price == 8.5
    assert offer.currency == "EUR"
    assert offer.product_type == "booster_pack"
    assert offer.seller_user_id == "seller_xyz"
    assert offer.location == "Madrid"
    assert offer.country == "ES"
    assert offer.shipping_available is True


@pytest.mark.anyio
async def test_scraper_run_with_progress_callback(monkeypatch):
    from delta_cards.models import SearchConfig, WallapopSettings, OutputSettings
    from delta_cards.wallapop.scraper import CardOffersScraper

    config = SearchConfig(
        name="Test Scraper",
        wallapop_settings=WallapopSettings(delay_between_requests_ms=0),
        output_settings=OutputSettings(output_dir="tmp/test"),
        search_profiles=[
            SearchProfile(
                id="test_profile",
                name="Test Profile",
                product_type="booster_box",
                query=["query 1", "query 2"],
                keywords=["precintada"],
                min_price=50.0,
            )
        ],
    )

    scraper = CardOffersScraper(config)

    # Mock client.search_offers
    mock_items = [
        {
            "id": "mock_1",
            "title": "Caja Pokemon Precintada",
            "price": 100.0,
            "currency": "EUR",
            "link": "https://wallapop.com/item/1",
        }
    ]

    async def mock_search_offers(*args, **kwargs):
        return mock_items

    monkeypatch.setattr(scraper.client, "search_offers", mock_search_offers)

    events_received = []

    def on_progress(event: str, data: dict):
        events_received.append((event, data))

    offers = await scraper.run(progress_callback=on_progress)

    assert len(offers) == 1
    assert offers[0].id == "mock_1"

    event_names = [e[0] for e in events_received]
    assert "profile_start" in event_names
    assert "query_start" in event_names
    assert "query_done" in event_names
    assert "profile_done" in event_names


@pytest.mark.anyio
async def test_scraper_run_with_incremental_on_batch(monkeypatch):
    from delta_cards.models import SearchConfig, WallapopSettings, OutputSettings
    from delta_cards.wallapop.scraper import CardOffersScraper

    config = SearchConfig(
        name="Test Scraper",
        wallapop_settings=WallapopSettings(delay_between_requests_ms=0),
        output_settings=OutputSettings(output_dir="tmp/test"),
        search_profiles=[
            SearchProfile(
                id="test_profile",
                name="Test Profile",
                product_type="single_card",
                query=["query 1", "query 2"],
                keywords=["carta"],
                min_price=1.0,
            )
        ],
    )

    scraper = CardOffersScraper(config)

    async def mock_search_offers(product_name, *args, **kwargs):
        if product_name == "query 1":
            return [
                {"id": "card_q1_1", "title": "Carta Pikachu", "price": 10.0, "currency": "EUR", "link": "https://wallapop.com/1"},
                {"id": "card_q1_2", "title": "Carta Mew", "price": 20.0, "currency": "EUR", "link": "https://wallapop.com/2"},
            ]
        return [
            {"id": "card_q2_1", "title": "Carta Charizard", "price": 50.0, "currency": "EUR", "link": "https://wallapop.com/3"},
        ]

    monkeypatch.setattr(scraper.client, "search_offers", mock_search_offers)

    batches_received = []

    def on_batch(batch):
        batches_received.append(batch)

    offers = await scraper.run(on_batch=on_batch)

    # 2 queries executed -> 2 batches emitted
    assert len(batches_received) == 2
    assert len(batches_received[0]) == 2
    assert batches_received[0][0].id == "card_q1_1"
    assert batches_received[0][1].id == "card_q1_2"
    assert len(batches_received[1]) == 1
    assert batches_received[1][0].id == "card_q2_1"
    assert len(offers) == 3

