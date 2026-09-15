"""Tests for the full-text yes/no LLM verifier."""

from unittest.mock import MagicMock, patch

from delta_cards.alerts.service import AlertConditionResult, DiscordAlertService, build_discord_embed
from delta_cards.models import OfferListing
from delta_cards.recognition.llm_verifier import LLMCardVerifier, LLMVerificationResult
from delta_cards.settings import Settings


def test_verifier_prompt_contains_only_complete_ocr_text():
    prompt = LLMCardVerifier().build_verification_prompt({"full_text": "RAYQUAZA EX 75/108"}, {"full_text": "RAYQUAZA EX 75/108"})
    assert "Photo: RAYQUAZA EX 75/108" in prompt
    assert "Official: RAYQUAZA EX 75/108" in prompt
    assert "Wallapop" not in prompt


def test_verifier_accepts_yes_response():
    verifier = LLMCardVerifier()
    verifier._model = MagicMock()
    verifier._model.create_chat_completion.return_value = {"choices": [{"message": {"content": "YES"}}]}
    result = verifier.verify_match(OfferListing(id="1", title="Rayquaza", price=1, product_type="single_card", link="https://example.com"), "RAYQUAZA EX", "RAYQUAZA EX")
    assert result.is_verified is True
    assert result.confidence_score == 100.0
    assert result.checks == {"full_ocr_text_match": True}


def test_verifier_rejects_no_response():
    verifier = LLMCardVerifier()
    verifier._model = MagicMock()
    verifier._model.create_chat_completion.return_value = {"choices": [{"message": {"content": "NO"}}]}
    result = verifier.verify_match(OfferListing(id="1", title="Rayquaza", price=1, product_type="single_card", link="https://example.com"), "RAYQUAZA", "PIKACHU")
    assert result.is_verified is False
    assert result.confidence_score == 0.0


def test_build_discord_embed_with_llm_field():
    offer = OfferListing(id="deal_1", title="Carta Pokemon Rayquaza EX", price=15.0, cardmarket_low_price=30.0, cardmarket_avg_price=35.0, product_type="single_card", link="https://wallapop.com/item/deal_1")
    condition = AlertConditionResult(is_triggered=True, savings_absolute=20.0, savings_percentage=57.1, offer_price=15.0, market_price=35.0, reason="Triggered")
    result = LLMVerificationResult(is_verified=True, confidence_score=100.0, confirmed_count=1, verdict_summary="Match confirmed")
    fields = build_discord_embed(offer, condition, llm_verification=result)["fields"]
    assert any("100%" in field["value"] for field in fields if "Verificación IA" in field["name"])


def test_discord_alert_service_suppresses_rejected_match():
    settings = Settings(_env_file=None, discord_alerts_enabled=True, discord_webhook_url="https://discord.com/api/webhooks/test/123", llm_verifier_enabled=True)
    verifier = MagicMock()
    verifier.verify_match.return_value = LLMVerificationResult(is_verified=False, confidence_score=0.0, verdict_summary="Match rejected")
    service = DiscordAlertService(settings=settings, verifier=verifier)
    offer = OfferListing(id="suspect", title="Falso Positivo", price=10.0, cardmarket_low_price=50.0, cardmarket_avg_price=50.0, product_type="single_card", link="https://wallapop.com/item/suspect")
    with patch("delta_cards.alerts.service.send_discord_webhook") as webhook:
        assert service.process_and_alert_offers([offer]) == 0
        webhook.assert_not_called()