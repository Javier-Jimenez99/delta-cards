"""Unit and integration tests for LLMCardVerifier and false-positive prevention."""

import pytest
from unittest.mock import MagicMock, patch

from delta_cards.alerts.service import (
    AlertConditionResult,
    DiscordAlertService,
    build_discord_embed,
)
from delta_cards.models import OfferListing
from delta_cards.recognition.llm_verifier import (
    VERIFICATION_QUESTIONS,
    LLMCardVerifier,
    LLMVerificationResult,
)
from delta_cards.settings import Settings


def test_verification_questions_count_and_vetoes():
    """Verify that exactly 20 questions are defined and hard vetoes exist."""
    assert len(VERIFICATION_QUESTIONS) == 20
    q_ids = [q[0] for q in VERIFICATION_QUESTIONS]
    assert "q01_pokemon_name_match" in q_ids
    assert "q20_overall_verdict_genuine_match" in q_ids

    # Hard vetoes
    q01 = [q for q in VERIFICATION_QUESTIONS if q[0] == "q01_pokemon_name_match"][0]
    assert q01[3] is True  # is_hard_veto = True

    q20 = [q for q in VERIFICATION_QUESTIONS if q[0] == "q20_overall_verdict_genuine_match"][0]
    assert q20[3] is True  # is_hard_veto = True


def test_build_verification_prompt():
    """Verify that the verification prompt correctly includes all 4 components."""
    verifier = LLMCardVerifier()
    offer = OfferListing(
        id="wallapop_123",
        title="Carta Pokemon Rayquaza EX 75/108 Cielos Rugientes",
        description="En excelente estado",
        price=20.0,
        product_type="single_card",
        link="https://wallapop.com/item/123",
    )
    query_ocr = {
        "full_text": "RAYQUAZA EX 170 HP Dragon Golpe Garra 30 75/108",
        "hp": "170",
        "fractions": ["75/108"],
    }
    candidate_ocr = {
        "full_text": "RAYQUAZA EX 170 HP Basico Dragon 75/108 2015",
        "hp": "170",
        "fractions": ["75/108"],
    }
    tcgdex_info = {
        "name": "Rayquaza EX",
        "local_id": "75",
        "total_cards": "108",
        "set_name": "Cielos Rugientes",
        "hp": "170",
        "attacks": [{"name": "Golpe Garra"}],
    }

    prompt = verifier.build_verification_prompt(
        query_ocr=query_ocr,
        candidate_ocr=candidate_ocr,
        wallapop_info=offer,
        tcgdex_info=tcgdex_info,
    )

    assert "[Photo OCR]:" in prompt
    assert "RAYQUAZA EX" in prompt
    assert "[Candidate Official Scan]:" in prompt
    assert "[Wallapop Listing]:" in prompt
    assert "Carta Pokemon Rayquaza EX 75/108" in prompt
    assert "[Official TCGdex Card]:" in prompt
    assert "75/108" in prompt
    assert "1. Pokemon name in photo OCR matches TCGdex candidate name" in prompt
    assert "20. Overall verdict: this is confirmed beyond doubt to be a genuine match" in prompt


def test_llm_verifier_genuine_match_mocked():
    """Test score computation when 19 of 20 questions are confirmed."""
    verifier = LLMCardVerifier(min_confidence=70.0)

    # Mock Llama model returning 19 ones and one zero
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]"}}]
    }
    verifier._model = mock_model

    offer = OfferListing(
        id="off_1",
        title="Rayquaza EX",
        price=10.0,
        product_type="single_card",
        link="https://wallapop.com/item/off_1",
    )
    res = verifier.verify_match(offer, query_ocr="Rayquaza", candidate_ocr="Rayquaza")

    assert res.is_verified is True
    assert res.confidence_score == 100.0
    assert res.confirmed_count == 20
    assert "✓ Match Confirmado" in res.verdict_summary


def test_llm_verifier_hard_veto_name_mismatch():
    """Test that if q01 (name match) is False, hard veto prevents verification."""
    verifier = LLMCardVerifier(min_confidence=70.0)

    mock_model = MagicMock()
    # First item (q01) is 0 (name mismatch)
    answers = [0] + [1] * 19
    mock_model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": str(answers)}}]
    }
    verifier._model = mock_model

    offer = OfferListing(
        id="off_2",
        title="Pikachu fake",
        price=10.0,
        product_type="single_card",
        link="https://wallapop.com/item/off_2",
    )
    res = verifier.verify_match(offer, query_ocr="Charizard", candidate_ocr="Pikachu")

    assert res.is_verified is False
    assert res.confidence_score <= 25.0
    assert "Veto IA" in res.verdict_summary
    assert "Nombre no coincide" in res.verdict_summary


def test_llm_verifier_hard_veto_verdict_rejected():
    """Test that if q20 (overall verdict) is False, hard veto triggers."""
    verifier = LLMCardVerifier(min_confidence=70.0)

    mock_model = MagicMock()
    # Last item (q20) is 0
    answers = [1] * 19 + [0]
    mock_model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": str(answers)}}]
    }
    verifier._model = mock_model

    offer = OfferListing(
        id="off_3",
        title="Mew",
        price=10.0,
        product_type="single_card",
        link="https://wallapop.com/item/off_3",
    )
    res = verifier.verify_match(offer, query_ocr="Mew", candidate_ocr="Mew")

    assert res.is_verified is False
    assert res.confidence_score <= 25.0
    assert "Veto IA" in res.verdict_summary
    assert "Veredicto global rechazado" in res.verdict_summary


def test_build_discord_embed_with_llm_field():
    """Verify that Discord embed displays the LLM verification field."""
    offer = OfferListing(
        id="deal_1",
        title="Carta Pokemon Rayquaza EX",
        price=15.0,
        cardmarket_low_price=30.0,
        cardmarket_avg_price=35.0,
        product_type="single_card",
        link="https://wallapop.com/item/deal_1",
    )
    cond = AlertConditionResult(
        is_triggered=True,
        savings_absolute=20.0,
        savings_percentage=57.1,
        offer_price=15.0,
        market_price=35.0,
        reason="Triggered",
    )
    llm_res = LLMVerificationResult(
        is_verified=True,
        confidence_score=95.0,
        confirmed_count=19,
        total_questions=20,
        verdict_summary="✓ Match Confirmado (19/20)",
    )

    embed = build_discord_embed(offer, cond, llm_verification=llm_res)

    fields = embed["fields"]
    llm_field = [f for f in fields if "Verificación IA" in f["name"]]
    assert len(llm_field) == 1
    assert "95%" in llm_field[0]["value"]
    assert "19/20 confirmadas" in llm_field[0]["value"]
    assert "Match Confirmado" in llm_field[0]["value"]


def test_discord_alert_service_suppresses_false_positives():
    """Verify that DiscordAlertService does not send alerts when LLM rejects candidate."""
    settings = Settings(
        _env_file=None,
        discord_alerts_enabled=True,
        discord_webhook_url="https://discord.com/api/webhooks/test/123",
        llm_verifier_enabled=True,
        llm_verifier_min_confidence=70.0,
    )

    mock_verifier = MagicMock()
    mock_verifier.verify_match.return_value = LLMVerificationResult(
        is_verified=False,
        confidence_score=20.0,
        confirmed_count=4,
        total_questions=20,
        verdict_summary="Veto IA: Nombre no coincide",
    )

    service = DiscordAlertService(settings=settings, verifier=mock_verifier)

    offer = OfferListing(
        id="suspect_deal",
        title="Falso Positivo Charizard",
        price=10.0,
        cardmarket_low_price=50.0,
        cardmarket_avg_price=50.0,
        product_type="single_card",
        link="https://wallapop.com/item/suspect_deal",
    )

    with patch("delta_cards.alerts.service.send_discord_webhook") as mock_webhook:
        sent = service.process_and_alert_offers([offer])
        assert sent == 0
        mock_webhook.assert_not_called()
        mock_verifier.verify_match.assert_called_once()


def test_discord_alert_service_dispatches_when_verified():
    """Verify that DiscordAlertService dispatches alerts when LLM confirms match."""
    settings = Settings(
        _env_file=None,
        discord_alerts_enabled=True,
        discord_webhook_url="https://discord.com/api/webhooks/test/123",
        llm_verifier_enabled=True,
        llm_verifier_min_confidence=70.0,
    )

    mock_verifier = MagicMock()
    mock_verifier.verify_match.return_value = LLMVerificationResult(
        is_verified=True,
        confidence_score=90.0,
        confirmed_count=18,
        total_questions=20,
        verdict_summary="✓ Match Confirmado (18/20)",
    )

    service = DiscordAlertService(settings=settings, verifier=mock_verifier)

    offer = OfferListing(
        id="genuine_deal",
        title="Rayquaza EX Auténtico",
        price=15.0,
        cardmarket_low_price=40.0,
        cardmarket_avg_price=40.0,
        product_type="single_card",
        link="https://wallapop.com/item/genuine_deal",
    )

    with patch("delta_cards.alerts.service.send_discord_webhook", return_value=True) as mock_webhook:
        sent = service.process_and_alert_offers([offer])
        assert sent == 1
        mock_webhook.assert_called_once()
        # Verify LLM field is in the dispatched embed
        call_args = mock_webhook.call_args
        embeds = call_args.kwargs.get("embeds") or call_args[1].get("embeds")
        assert len(embeds) == 1
        embed_fields = embeds[0]["fields"]
        llm_field = [f for f in embed_fields if "Verificación IA" in f["name"]]
        assert len(llm_field) == 1
        assert "90%" in llm_field[0]["value"]


def test_llm_verifier_real_model_inference():
    """Integration test: runs real local LLM model on CPU to verify genuine match."""
    import os

    model_path = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    if not os.path.exists(model_path):
        pytest.skip("Local model not available for integration test.")

    verifier = LLMCardVerifier(model_path=model_path, min_confidence=70.0)

    offer = OfferListing(
        id="real_test_1",
        title="Carta Pokemon Rayquaza EX 75/108",
        description="Rayquaza EX Roaring Skies original en buen estado",
        price=12.0,
        product_type="single_card",
        link="https://wallapop.com/item/real_test_1",
    )
    query_ocr = {
        "full_text": "RAYQUAZA EX 170 HP Dragon Golpe Garra 30 Ascension Dragon 100 75/108",
        "hp": "170",
        "fractions": ["75/108"],
    }
    candidate_ocr = {
        "full_text": "RAYQUAZA EX 170 HP Basico Dragon Golpe Garra 30 Ascension Dragon 100 75/108",
        "hp": "170",
        "fractions": ["75/108"],
    }
    tcgdex_info = {
        "name": "Rayquaza EX",
        "local_id": "75",
        "total_cards": "108",
        "set_name": "Cielos Rugientes",
        "hp": "170",
        "attacks": [{"name": "Golpe Garra"}, {"name": "Ascensión Dragón"}],
    }

    res = verifier.verify_match(
        offer=offer,
        query_ocr=query_ocr,
        candidate_ocr=candidate_ocr,
        tcgdex_card=tcgdex_info,
    )

    assert res.is_verified is True
    assert res.confidence_score >= 70.0
    assert res.confirmed_count >= 14
    assert res.checks["q01_pokemon_name_match"] is True
    assert res.checks["q20_overall_verdict_genuine_match"] is True
