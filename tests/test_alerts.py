"""Tests for Discord alerts and saving condition evaluation."""

import pytest
from unittest.mock import MagicMock, patch
from delta_cards.alerts import (
    AlertConditionResult,
    DiscordAlertService,
    build_discord_embed,
    evaluate_alert_condition,
    format_product_type,
    send_discord_webhook,
)
from delta_cards.models import OfferListing
from delta_cards.settings import Settings


def test_evaluate_condition_percentage_only():
    """Test condition evaluation when only percentage saving threshold is set."""
    settings = Settings(
        _env_file=None,
        alert_min_saving_percentage=25.0,
        alert_min_saving_absolute=None,
    )

    # Market: 100€, Offer: 70€ -> 30% saving (>= 25%) -> Triggered
    res = evaluate_alert_condition(offer_price=70.0, market_price=100.0, settings=settings)
    assert res.is_triggered is True
    assert res.savings_absolute == 30.0
    assert res.savings_percentage == 30.0

    # Market: 100€, Offer: 80€ -> 20% saving (< 25%) -> Not triggered
    res = evaluate_alert_condition(offer_price=80.0, market_price=100.0, settings=settings)
    assert res.is_triggered is False
    assert res.savings_absolute == 20.0
    assert res.savings_percentage == 20.0


def test_evaluate_condition_absolute_only():
    """Test condition evaluation when only absolute saving threshold is set."""
    settings = Settings(
        _env_file=None,
        alert_min_saving_percentage=None,
        alert_min_saving_absolute=15.0,
    )

    # Market: 50€, Offer: 30€ -> 20€ saving (>= 15€) -> Triggered
    res = evaluate_alert_condition(offer_price=30.0, market_price=50.0, settings=settings)
    assert res.is_triggered is True
    assert res.savings_absolute == 20.0

    # Market: 50€, Offer: 40€ -> 10€ saving (< 15€) -> Not triggered
    res = evaluate_alert_condition(offer_price=40.0, market_price=50.0, settings=settings)
    assert res.is_triggered is False
    assert res.savings_absolute == 10.0


def test_evaluate_condition_operator_or():
    """Test condition evaluation with OR operator."""
    settings = Settings(
        _env_file=None,
        alert_min_saving_percentage=50.0,
        alert_min_saving_absolute=20.0,
        alert_condition_operator="or",
    )

    # Meets absolute (20€ >= 20€) but not percentage (20/80 = 25% < 50%) -> Triggered (OR)
    res = evaluate_alert_condition(offer_price=60.0, market_price=80.0, settings=settings)
    assert res.is_triggered is True

    # Meets percentage (50%) on low price (1€ off 2€) but not absolute (1€ < 20€) -> Triggered (OR)
    res = evaluate_alert_condition(offer_price=1.0, market_price=2.0, settings=settings)
    assert res.is_triggered is True

    # Meets neither
    res = evaluate_alert_condition(offer_price=90.0, market_price=100.0, settings=settings)
    assert res.is_triggered is False


def test_evaluate_condition_operator_and():
    """Test condition evaluation with AND operator."""
    settings = Settings(
        _env_file=None,
        alert_min_saving_percentage=50.0,
        alert_min_saving_absolute=20.0,
        alert_condition_operator="and",
    )

    # Meets absolute (20€) but not percentage (25%) -> Not triggered (AND)
    res = evaluate_alert_condition(offer_price=60.0, market_price=80.0, settings=settings)
    assert res.is_triggered is False

    # Meets percentage (50%) but not absolute (1€) -> Not triggered (AND)
    res = evaluate_alert_condition(offer_price=1.0, market_price=2.0, settings=settings)
    assert res.is_triggered is False

    # Meets both: 50€ off 100€ (50% and 50€ >= 20€) -> Triggered (AND)
    res = evaluate_alert_condition(offer_price=50.0, market_price=100.0, settings=settings)
    assert res.is_triggered is True


def test_evaluate_condition_edge_cases():
    """Test edge cases: no discount, invalid market price, negative prices."""
    settings = Settings(_env_file=None)

    # Offer price >= market price
    res = evaluate_alert_condition(offer_price=100.0, market_price=80.0, settings=settings)
    assert res.is_triggered is False
    assert res.savings_absolute <= 0

    # Market price 0 or negative
    res = evaluate_alert_condition(offer_price=50.0, market_price=0.0, settings=settings)
    assert res.is_triggered is False

    res = evaluate_alert_condition(offer_price=50.0, market_price=-10.0, settings=settings)
    assert res.is_triggered is False


def test_format_product_type():
    """Verify translation of raw product type codes into Spanish display labels."""
    assert format_product_type("single_card") == "Carta"
    assert format_product_type("card") == "Carta"
    assert format_product_type("booster_pack") == "Paquete"
    assert format_product_type("sobre") == "Paquete"
    assert format_product_type("booster_box") == "Caja"
    assert format_product_type("box") == "Caja"
    assert format_product_type("deck") == "Mazo"
    assert format_product_type("etb") == "Caja (ETB)"
    assert format_product_type(None) == "Otro"


def test_build_discord_embed():
    """Verify embed formatting in the renovated format."""
    offer = OfferListing(
        id="offer_123",
        title="Charizard Base Set Holo 4/102",
        price=150.0,
        product_type="single_card",
        location="Madrid, Spain",
        link="https://es.wallapop.com/item/charizard-123",
        image_url="https://images.wallapop.com/item.jpg",
        cardmarket_url="https://www.cardmarket.com/en/Pokemon/Products/Singles/Base-Set/Charizard",
        cardmarket_avg_price=200.0,
        cardmarket_low_price=180.0,
    )
    res = AlertConditionResult(
        is_triggered=True,
        savings_absolute=50.0,
        savings_percentage=25.0,
        offer_price=150.0,
        market_price=200.0,
        reason="Triggered",
    )

    embed = build_discord_embed(offer, res)
    assert embed["title"] == "Oferta: Charizard Base Set Holo 4/102"
    assert embed["url"] == offer.link
    assert embed["thumbnail"]["url"] == offer.image_url
    assert embed["footer"]["text"] == "Alerta de ofertas Delta Cards"

    fields_dict = {f["name"]: f["value"] for f in embed["fields"]}

    # Row 1: Wallapop, Precio medio, Precio mínimo
    assert "<:wallapop:1548103689477038080> Wallapop" in fields_dict
    assert "150.00 €" in fields_dict["<:wallapop:1548103689477038080> Wallapop"]

    assert "<:cardmarket:1548103777750487056> Precio medio" in fields_dict
    assert "200.00 €" in fields_dict["<:cardmarket:1548103777750487056> Precio medio"]

    assert "<:cardmarket:1548103777750487056> Precio mínimo" in fields_dict
    assert "180.00 €" in fields_dict["<:cardmarket:1548103777750487056> Precio mínimo"]

    # Flags and related texts must be removed
    assert "🇪🇸 ESP" not in fields_dict
    assert "🇪🇺 UE" not in fields_dict

    # Row 2: Tipo, Ubicación (Ahorro removed)
    assert "💰 Ahorro" not in fields_dict

    assert "📦 Tipo" in fields_dict
    assert fields_dict["📦 Tipo"] == "**Carta**"

    assert "📍 Ubicación" in fields_dict
    assert "Madrid, Spain" in fields_dict["📍 Ubicación"]

    # Row 3: Links without individual emotes
    assert "🔗 Enlaces" in fields_dict
    links_val = fields_dict["🔗 Enlaces"]
    assert offer.link in links_val
    assert offer.cardmarket_url in links_val
    assert "[Oferta en Wallapop]" in links_val
    assert "[Producto en Cardmarket]" in links_val
    assert "🛒" not in links_val
    assert "📈" not in links_val


def test_build_discord_embed_vinted():
    """Verify embed formatting when offer is from Vinted."""
    offer = OfferListing(
        id="vinted_12345",
        title="Umbreon GX 15/149",
        price=15.0,
        product_type="single_card",
        platform="vinted",
        location="Madrid, Spain",
        link="https://www.vinted.es/items/12345-umbreon-gx",
        image_url="https://images1.vinted.net/item.jpg",
        cardmarket_url="https://www.cardmarket.com/es/Pokemon/Products/Singles/Sun-Moon/Umbreon-GX",
        cardmarket_avg_price=25.0,
        cardmarket_low_price=20.0,
    )
    res = AlertConditionResult(
        is_triggered=True,
        savings_absolute=10.0,
        savings_percentage=40.0,
        offer_price=15.0,
        market_price=25.0,
        reason="Triggered",
    )

    embed = build_discord_embed(offer, res)
    assert embed["title"] == "Oferta: Umbreon GX 15/149"
    assert embed["url"] == offer.link

    fields_dict = {f["name"]: f["value"] for f in embed["fields"]}

    # Row 1: Vinted emoji and name
    assert "<:vinted:1549420019518013470> Vinted" in fields_dict
    assert "15.00 €" in fields_dict["<:vinted:1549420019518013470> Vinted"]
    assert "<:wallapop:1548103689477038080> Wallapop" not in fields_dict

    # Row 3: Links with 'Oferta en Vinted'
    assert "🔗 Enlaces" in fields_dict
    links_val = fields_dict["🔗 Enlaces"]
    assert "[Oferta en Vinted]" in links_val
    assert "[Oferta en Wallapop]" not in links_val
    assert offer.link in links_val
    assert offer.cardmarket_url in links_val


def test_build_discord_embed_product_id_fallback():
    """Verify fallback behavior when prices and URLs are missing or product_id fallback is used."""
    offer = OfferListing(
        id="offer_456",
        title="Pikachu Illustrator Promo",
        price=500.0,
        product_type="single_card",
        link="https://es.wallapop.com/item/pikachu-456",
        cardmarket_product_id="869748",
    )
    res = AlertConditionResult(
        is_triggered=True,
        savings_absolute=200.0,
        savings_percentage=28.57,
        offer_price=500.0,
        market_price=700.0,
        reason="Triggered",
    )

    embed = build_discord_embed(offer, res)
    assert embed["title"] == "Oferta: Pikachu Illustrator Promo"
    fields_dict = {f["name"]: f["value"] for f in embed["fields"]}

    # Displays market_price fallback under Precio medio
    assert "<:cardmarket:1548103777750487056> Precio medio" in fields_dict
    assert "700.00 €" in fields_dict["<:cardmarket:1548103777750487056> Precio medio"]

    # Precio mínimo displays N/A when low_price is missing
    assert "<:cardmarket:1548103777750487056> Precio mínimo" in fields_dict
    assert "N/A" in fields_dict["<:cardmarket:1548103777750487056> Precio mínimo"]

    # Ubicación displays N/A when location is missing
    assert "📍 Ubicación" in fields_dict
    assert "N/A" in fields_dict["📍 Ubicación"]

    # Flags and related texts removed
    assert "🇪🇸 ESP" not in fields_dict
    assert "🇪🇺 UE" not in fields_dict

    # Ahorro removed
    assert "💰 Ahorro" not in fields_dict

    # Links row contains auto-generated Cardmarket URL with product ID and no emotes in link text
    links_val = fields_dict["🔗 Enlaces"]
    assert "idProduct=869748" in links_val
    assert "[Oferta en Wallapop]" in links_val
    assert "[Producto en Cardmarket]" in links_val
    assert "🛒" not in links_val
    assert "📈" not in links_val


def test_send_discord_webhook_success():
    """Verify sending webhook successfully invokes httpx."""
    embeds = [{"title": "Test Deal", "fields": []}]

    mock_resp = MagicMock()
    mock_resp.status_code = 204

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        ok = send_discord_webhook("https://discord.com/api/webhooks/test", embeds)
        assert ok is True
        mock_client.post.assert_called_once()
        args, kwargs = mock_client.post.call_args
        assert kwargs["json"]["embeds"] == embeds
        assert kwargs["json"]["content"] == "<:pokeball_light:1548052080357412995> ¡Ha aparecido una oferta salvaje!"


def test_send_discord_webhook_failure_handling():
    """Verify graceful handling when Discord webhook encounters rate limits or errors."""
    embeds = [{"title": "Test Deal"}]

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        ok = send_discord_webhook("https://discord.com/api/webhooks/test", embeds)
        assert ok is False


def test_discord_alert_service_disabled():
    """When discord_alerts_enabled is False, service should not send alerts."""
    settings = Settings(
        _env_file=None,
        discord_webhook_url="https://discord.com/api/webhooks/test",
        discord_alerts_enabled=False,
    )
    service = DiscordAlertService(settings=settings)
    offer = OfferListing(
        id="o1",
        title="Pikachu",
        price=10.0,
        product_type="single_card",
        link="https://wallapop.com/item/1",
        cardmarket_low_price=50.0,
    )

    with patch("delta_cards.alerts.service.send_discord_webhook") as mock_send:
        sent_count = service.process_and_alert_offers([offer])
        assert sent_count == 0
        mock_send.assert_not_called()


def test_discord_alert_service_success():
    """Verify DiscordAlertService processes offers, filters deals, and dispatches webhook."""
    settings = Settings(
        _env_file=None,
        discord_webhook_url="https://discord.com/api/webhooks/test",
        discord_alerts_enabled=True,
        alert_min_saving_percentage=20.0,
        alert_min_saving_absolute=5.0,
    )
    service = DiscordAlertService(settings=settings)

    deal_offer = OfferListing(
        id="deal_1",
        title="Gengar VMAX Alt Art",
        price=100.0,
        product_type="single_card",
        link="https://wallapop.com/item/gengar",
        cardmarket_low_price=200.0,  # 50% discount & 100€ savings -> Deal!
    )
    no_deal_offer = OfferListing(
        id="no_deal_1",
        title="Bulbasaur",
        price=10.0,
        product_type="single_card",
        link="https://wallapop.com/item/bulbasaur",
        cardmarket_low_price=10.0,  # 0€ savings -> No deal
    )

    with patch("delta_cards.alerts.service.send_discord_webhook", return_value=True) as mock_send:
        sent_count = service.process_and_alert_offers([deal_offer, no_deal_offer])
        assert sent_count == 1
        mock_send.assert_called_once()
        args, kwargs = mock_send.call_args
        embeds = kwargs["embeds"]
        assert len(embeds) == 1
        assert "Gengar" in embeds[0]["title"]
        assert kwargs["delay_between_offers"] == 0.0


def test_send_discord_webhook_delay_between_offers():
    """Verify that send_discord_webhook pauses between consecutive offer alerts when delay_between_offers > 0."""
    embeds = [{"title": "Offer 1"}, {"title": "Offer 2"}, {"title": "Offer 3"}]
    mock_resp = MagicMock()
    mock_resp.status_code = 204

    with patch("httpx.Client") as mock_client_cls, patch("time.sleep") as mock_sleep:
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        ok = send_discord_webhook(
            "https://discord.com/api/webhooks/test",
            embeds=embeds,
            delay_between_offers=2.5,
        )
        assert ok is True
        assert mock_client.post.call_count == 3
        # Should sleep twice: once before embed 2, once before embed 3
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(2.5)


def test_discord_alert_service_routes_by_game():
    """Verify that Pokémon and Magic deals are routed to their respective webhooks."""
    settings = Settings(
        _env_file=None,
        pokemon_cards_webhook="https://discord.com/api/webhooks/pokemon",
        magic_cards_webhook="https://discord.com/api/webhooks/magic",
        discord_alerts_enabled=True,
        alert_min_saving_percentage=20.0,
        alert_min_saving_absolute=5.0,
    )
    service = DiscordAlertService(settings=settings)

    pokemon_deal = OfferListing(
        id="poke_1",
        title="Charizard VMAX",
        price=50.0,
        product_type="single_card",
        game="pokemon",
        link="https://wallapop.com/item/charizard",
        cardmarket_low_price=100.0,
    )
    magic_deal = OfferListing(
        id="magic_1",
        title="Black Lotus",
        price=500.0,
        product_type="single_card",
        game="magic",
        link="https://wallapop.com/item/lotus",
        cardmarket_low_price=1000.0,
    )

    with patch("delta_cards.alerts.service.send_discord_webhook", return_value=True) as mock_send:
        sent_count = service.process_and_alert_offers([pokemon_deal, magic_deal])
        assert sent_count == 2
        calls = mock_send.call_args_list
        urls = [call.kwargs.get("webhook_url") or (call.args[0] if call.args else None) for call in calls]
        assert "https://discord.com/api/webhooks/pokemon" in urls
        assert "https://discord.com/api/webhooks/magic" in urls

        for call in calls:
            url = call.kwargs.get("webhook_url") or (call.args[0] if call.args else None)
            bot_user = call.kwargs.get("username")
            if url == "https://discord.com/api/webhooks/pokemon":
                assert bot_user == "El Meowth de la suerte"
            elif url == "https://discord.com/api/webhooks/magic":
                assert bot_user == "Ragavan Buscatesoros"


def test_send_discord_webhook_usernames():
    """Verify bot usernames for Pokemon and Magic webhooks."""
    embeds = [{"title": "Test Deal"}]

    mock_resp = MagicMock()
    mock_resp.status_code = 204

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        # 1. Pokemon bot name
        send_discord_webhook(
            "https://discord.com/api/webhooks/test",
            embeds,
            username="El Meowth de la suerte",
        )
        payload = mock_client.post.call_args[1]["json"]
        assert payload["username"] == "El Meowth de la suerte"

        # 2. Magic bot name
        send_discord_webhook(
            "https://discord.com/api/webhooks/test",
            embeds,
            username="Ragavan Buscatesoros",
        )
        payload = mock_client.post.call_args[1]["json"]
        assert payload["username"] == "Ragavan Buscatesoros"


