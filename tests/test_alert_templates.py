"""Unit tests for alert text templates by channel and placeholder rendering."""

from pathlib import Path
import pytest

from delta_cards.alerts.templates import (
    AlertTemplateManager,
    SafeTemplateFormatter,
    render_alert_text,
    resolve_channel_key,
)
from delta_cards.alerts.service import (
    AlertConditionResult,
    DiscordAlertService,
    send_discord_webhook,
)
from delta_cards.models import OfferListing
from delta_cards.settings import Settings


def test_resolve_channel_key():
    """Test channel key resolution for different games and product types."""
    # Pokémon
    assert resolve_channel_key("pokemon", "single_card") == "pokemon_cards"
    assert resolve_channel_key("pokemon", "card") == "pokemon_cards"
    assert resolve_channel_key("pokemon", None) == "pokemon_cards"
    assert resolve_channel_key("pokemon", "booster_pack") == "pokemon_boosters"
    assert resolve_channel_key("pokemon", "sobre") == "pokemon_boosters"
    assert resolve_channel_key("pokemon", "blister") == "pokemon_boosters"
    assert resolve_channel_key("pokemon", "booster_box") == "pokemon_boxes"
    assert resolve_channel_key("pokemon", "etb") == "pokemon_boxes"
    assert resolve_channel_key("pokemon", "caja") == "pokemon_boxes"
    assert resolve_channel_key("pokemon", "bundle") == "pokemon_boxes"

    # Magic
    assert resolve_channel_key("magic", "single_card") == "magic_cards"
    assert resolve_channel_key("magic", None) == "magic_cards"
    assert resolve_channel_key("mtg", "card") == "magic_cards"
    assert resolve_channel_key("magic", "booster_pack") == "magic_boosters"
    assert resolve_channel_key("magic", "sobre") == "magic_boosters"
    assert resolve_channel_key("magic", "blister") == "magic_boosters"
    assert resolve_channel_key("magic", "booster_box") == "magic_boxes"
    assert resolve_channel_key("mtg", "display") == "magic_boxes"
    assert resolve_channel_key("magic", "bundle") == "magic_boxes"
    assert resolve_channel_key("magic", "deck") == "magic_boxes"

    # Default / Unknown
    assert resolve_channel_key("yugioh", "card") == "default"


def test_safe_template_formatter():
    """Test that SafeTemplateFormatter does not crash on missing placeholders."""
    formatter = SafeTemplateFormatter()
    template = "Offer: {title} for {price} (Code: {code})"
    rendered = formatter.format(template, title="Charizard", price="150.00€")
    assert "Charizard" in rendered
    assert "150.00€" in rendered
    assert "{code}" in rendered  # Unfilled placeholder remains safe


def test_alert_template_manager_render_from_files():
    """Test loading and rendering original template files from config/templates/alerts."""
    manager = AlertTemplateManager(templates_dir=Path("config/templates/alerts"))

    offer = OfferListing(
        id="test-1",
        title="Charizard Base Set",
        price=80.0,
        game="pokemon",
        product_type="single_card",
        link="https://wallapop.com/item/charizard",
        cardmarket_low_price=120.0,
    )
    result = AlertConditionResult(
        is_triggered=True,
        savings_absolute=40.0,
        savings_percentage=33.3,
        offer_price=80.0,
        market_price=120.0,
        reason="Good deal",
    )

    rendered = manager.render(
        channel_key="pokemon_cards",
        offer=offer,
        result=result,
        emoji="<:pokeball_light:1548052080357412995>",
    )

    assert "¡Ha aparecido una oferta salvaje!" in rendered
    assert "Charizard Base Set" in rendered
    assert "80.00 €" in rendered
    assert "Precio medio" in rendered
    assert "Tipo" in rendered
    assert "Ubicación" in rendered
    assert "Enlaces" in rendered


def test_all_config_templates_exist_and_render():
    """Verify each configured template file renders successfully without error."""
    channels = [
        "pokemon_cards",
        "pokemon_boosters",
        "pokemon_boxes",
        "magic_cards",
        "magic_boosters",
        "magic_boxes",
        "magic_sealed",
        "default",
    ]
    manager = AlertTemplateManager(templates_dir=Path("config/templates/alerts"))

    offer = OfferListing(
        id="test-2",
        title="Black Lotus Alpha",
        price=5000.0,
        game="magic",
        product_type="single_card",
        link="https://wallapop.com/item/lotus",
        cardmarket_low_price=10000.0,
    )
    result = AlertConditionResult(
        is_triggered=True,
        savings_absolute=5000.0,
        savings_percentage=50.0,
        offer_price=5000.0,
        market_price=10000.0,
        reason="Mega deal",
    )

    for ch in channels:
        text = manager.render(channel_key=ch, offer=offer, result=result)
        assert "oferta salvaje" in text.lower()


def test_template_manager_fallback():
    """Test fallback to default when channel file is missing."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Create only default.txt
        (tmp_path / "default.txt").write_text("Default Alert: {title} - {price}", encoding="utf-8")

        manager = AlertTemplateManager(templates_dir=tmp_path)
        offer = OfferListing(
            id="t3",
            title="Pikachu Illustrator",
            price=200.0,
            product_type="single_card",
            link="https://test.com/pika",
        )
        result = AlertConditionResult(
            is_triggered=True,
            savings_absolute=50.0,
            savings_percentage=20.0,
            offer_price=200.0,
            market_price=250.0,
            reason="Pikachu",
        )

        rendered = manager.render("nonexistent_channel", offer, result)
        assert rendered == "Default Alert: Pikachu Illustrator - 200.00 €"


def test_render_alert_text_helper():
    """Test the standalone render_alert_text helper."""
    offer = OfferListing(
        id="t4",
        title="Modern Horizons 3 Booster Box",
        price=180.0,
        game="magic",
        product_type="booster_box",
        link="https://wallapop.com/item/mh3",
    )
    result = AlertConditionResult(
        is_triggered=True,
        savings_absolute=40.0,
        savings_percentage=18.2,
        offer_price=180.0,
        market_price=220.0,
        reason="Deal",
    )

    rendered = render_alert_text(
        offer=offer,
        result=result,
        channel_key="magic_boxes",
        templates_dir=Path("config/templates/alerts"),
    )
    assert "¡Ha aparecido una oferta salvaje de Magic!" in rendered
    assert "Modern Horizons 3 Booster Box" in rendered
    assert "180.00 €" in rendered


def test_custom_template_placeholders():
    """Verify that when a user customizes a template with placeholders, they are populated."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "magic_boxes.txt").write_text(
            "🔮 {title} - {price} (-{discount_percentage}%) en {platform}",
            encoding="utf-8",
        )
        offer = OfferListing(
            id="t5",
            title="Modern Horizons 3 Booster Box",
            price=180.0,
            game="magic",
            product_type="booster_box",
            link="https://wallapop.com/item/mh3",
        )
        result = AlertConditionResult(
            is_triggered=True,
            savings_absolute=40.0,
            savings_percentage=18.2,
            offer_price=180.0,
            market_price=220.0,
            reason="Deal",
        )
        rendered = render_alert_text(
            offer=offer,
            result=result,
            channel_key="magic_boxes",
            templates_dir=tmp_path,
        )
        assert rendered == "🔮 Modern Horizons 3 Booster Box - 180.00 € (-18.2%) en Wallapop"


def test_send_discord_webhook_list_content():
    """Test send_discord_webhook when content is passed as a list of strings."""
    from unittest.mock import patch, MagicMock

    with patch("delta_cards.alerts.service.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        embeds = [{"title": "Card 1"}, {"title": "Card 2"}]
        contents = ["Alert 1: Card 1", "Alert 2: Card 2"]

        ok = send_discord_webhook(
            webhook_url="https://discord.com/api/webhooks/test",
            embeds=embeds,
            content=contents,
            delay_between_offers=0,
        )

        assert ok is True
        assert mock_client.post.call_count == 2
        first_call = mock_client.post.call_args_list[0][1]["json"]
        second_call = mock_client.post.call_args_list[1][1]["json"]
        assert first_call["content"] == "Alert 1: Card 1"
        assert second_call["content"] == "Alert 2: Card 2"


def test_build_discord_embed_with_description_in_green_block():
    """Verify that when description is provided, all alert text is placed inside the green block."""
    from delta_cards.alerts.service import build_discord_embed, extract_alert_header

    offer = OfferListing(
        id="t6",
        title="Modern Horizons 3 Play Booster",
        price=5.50,
        game="magic",
        product_type="booster_pack",
        link="https://es.wallapop.com/item/mh3-booster",
        image_url="https://images.wallapop.com/mh3.jpg",
    )
    result = AlertConditionResult(
        is_triggered=True,
        savings_absolute=5.50,
        savings_percentage=50.0,
        offer_price=5.50,
        market_price=11.00,
        reason="Deal",
    )
    rendered_text = (
        "<:mtg_logo:1550271985265807440> ¡Ha aparecido una oferta salvaje de Magic!\n"
        "**[Oferta: Modern Horizons 3 Play Booster](https://es.wallapop.com/item/mh3-booster)**\n\n"
        "<:wallapop:1548103689477038080> **Wallapop:** 5.50 €\n"
        "📦 **Tipo:** Sobre\n"
        "📍 **Ubicación:** Madrid, Spain"
    )

    header = extract_alert_header(rendered_text)
    assert header == "<:mtg_logo:1550271985265807440> ¡Ha aparecido una oferta salvaje de Magic!"

    embed = build_discord_embed(offer, result, description=rendered_text)

    # Header is extracted for outer message content, description in embed is omitted
    assert embed.get("description") is None
    # Title and URL are set on the embed
    assert embed["title"] == "Oferta: Modern Horizons 3 Play Booster"
    assert embed["url"] == offer.link
    # Green block color
    assert embed["color"] == 0x2ECC71
    # Footer and timestamp present
    assert embed["footer"]["text"] == "Alerta de ofertas Delta Cards"
    assert "timestamp" in embed
    # Thumbnail inside the embed
    assert embed["thumbnail"]["url"] == offer.image_url
    # Fields organized in columns with inline: True
    assert len(embed["fields"]) == 3
    assert all(f["inline"] is True for f in embed["fields"])
    fields_dict = {f["name"]: f["value"] for f in embed["fields"]}
    assert "<:wallapop:1548103689477038080> Wallapop" in fields_dict
    assert "**5.50 €**" in fields_dict["<:wallapop:1548103689477038080> Wallapop"]
    assert "📦 Tipo" in fields_dict
    assert "**Sobre**" in fields_dict["📦 Tipo"]
    assert "📍 Ubicación" in fields_dict
    assert "**Madrid, Spain**" in fields_dict["📍 Ubicación"]


def test_send_discord_webhook_header_outside_embed():
    """Verify that send_discord_webhook sends the header outside above the green embed."""
    from unittest.mock import patch, MagicMock

    with patch("delta_cards.alerts.service.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        embed = {
            "title": "Oferta: Producto",
            "color": 0x2ECC71,
            "fields": [{"name": "Precio", "value": "10.00 €", "inline": True}],
        }
        header = "<:mtg_logo:1550271985265807440> ¡Ha aparecido una oferta salvaje de Magic!"

        ok = send_discord_webhook(
            webhook_url="https://discord.com/api/webhooks/test",
            embeds=[embed],
            content=header,
        )

        assert ok is True
        assert mock_client.post.call_count == 1
        payload = mock_client.post.call_args_list[0][1]["json"]
        assert payload["content"] == header
        assert payload["embeds"] == [embed]
        assert payload["username"] == "Delta Cards Alert"

