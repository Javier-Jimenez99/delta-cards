"""Unit tests for decoupled AlertWorkerService and alert_status lifecycle."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from sqlmodel import Session, select, SQLModel, create_engine

from delta_cards.database.models import WallapopOffers, CardmarketProduct, CardmarketPrice
from delta_cards.database.repository import (
    save_offers_to_db,
    get_pending_alert_offers,
    update_offer_alert_status,
)
from delta_cards.models import OfferListing
from delta_cards.settings import Settings
from delta_cards.alerts.service import AlertWorkerService
from delta_cards.recognition.llm_verifier import LLMVerificationResult


@pytest.fixture
def test_engine():
    """In-memory SQLite database engine for testing."""
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def sample_product_and_price(test_engine):
    """Inserts a reference Cardmarket product and price into the test database."""
    with Session(test_engine) as session:
        product = CardmarketProduct(
            id="12345",
            cardmarket_url="https://www.cardmarket.com/es/Pokemon/Products?idProduct=12345",
        )
        price = CardmarketPrice(
            cardmarket_product_id="12345",
            language="es",
            min_price_all=30.0,
            avg_price=50.0,
        )
        session.add(product)
        session.add(price)
        session.commit()
    return "12345"


def test_alert_worker_settings_interval():
    """Test calculation of effective alert interval in seconds."""
    s1 = Settings(_env_file=None, alert_dispatch_interval_seconds=1800)
    assert s1.alert_interval == 1800

    s2 = Settings(_env_file=None, alert_dispatch_interval_minutes=45, alert_dispatch_interval_seconds=1800)
    assert s2.alert_interval == 45 * 60 == 2700


def test_alert_worker_dispatches_one_and_marks_sent(test_engine, sample_product_and_price):
    """Test that AlertWorkerService dispatches 1 verified pending offer and marks it as 'sent'."""
    now = datetime.now(timezone.utc)
    with Session(test_engine) as session:
        offer = WallapopOffers(
            id="offer_1",
            title="Charizard Holo Base Set",
            price=25.0,
            currency="EUR",
            product_type="single_card",
            link="https://es.wallapop.com/item/offer_1",
            cardmarket_product_id="12345",
            language="es",
            assigned_state="assigned",
            alert_status="pending",
            scraped_at=now,
            updated_at=now,
        )
        session.add(offer)
        session.commit()

    settings = Settings(
        _env_file=None,
        discord_webhook_url="https://discord.com/api/webhooks/mock",
        discord_alerts_enabled=True,
        llm_verifier_enabled=True,
    )

    mock_verifier = MagicMock()
    mock_verifier.verify_match.return_value = LLMVerificationResult(
        is_verified=True,
        confidence_score=95.0,
        confirmed_count=19,
        verdict_summary="Genuine Charizard deal",
    )

    worker = AlertWorkerService(engine=test_engine, settings=settings, verifier=mock_verifier)

    with patch("delta_cards.alerts.service.send_discord_webhook", return_value=True) as mock_send:
        result = worker.dispatch_one_pending_alert()

        assert result is not None
        assert result.id == "offer_1"
        assert mock_send.called
        assert mock_verifier.verify_match.called

    # Check database status was updated to 'sent'
    with Session(test_engine) as session:
        updated = session.exec(select(WallapopOffers).where(WallapopOffers.id == "offer_1")).first()
        assert updated.alert_status == "sent"


def test_alert_worker_skips_rejected_and_dispatches_next(test_engine, sample_product_and_price):
    """Test that if the first pending offer is rejected by the LLM, it is marked 'rejected'

    and the worker immediately tests and sends the next pending offer in the same cycle.
    """
    now = datetime.now(timezone.utc)
    with Session(test_engine) as session:
        # First offer: will be rejected by LLM
        offer_rejected = WallapopOffers(
            id="offer_fake",
            title="Fake Charizard Card",
            price=10.0,
            currency="EUR",
            product_type="single_card",
            link="https://es.wallapop.com/item/offer_fake",
            cardmarket_product_id="12345",
            language="es",
            assigned_state="assigned",
            alert_status="pending",
            scraped_at=now,
            updated_at=now,
        )
        # Second offer: genuine deal
        offer_good = WallapopOffers(
            id="offer_real",
            title="Real Charizard Card",
            price=20.0,
            currency="EUR",
            product_type="single_card",
            link="https://es.wallapop.com/item/offer_real",
            cardmarket_product_id="12345",
            language="es",
            assigned_state="assigned",
            alert_status="pending",
            scraped_at=now,
            updated_at=now,
        )
        session.add(offer_rejected)
        session.add(offer_good)
        session.commit()

    settings = Settings(
        _env_file=None,
        discord_webhook_url="https://discord.com/api/webhooks/mock",
        discord_alerts_enabled=True,
        llm_verifier_enabled=True,
    )

    mock_verifier = MagicMock()
    # First call rejects, second call verifies
    res_rejected = LLMVerificationResult(
        is_verified=False,
        confidence_score=35.0,
        confirmed_count=7,
        verdict_summary="Contradictory HP and non-matching card details",
    )
    res_good = LLMVerificationResult(
        is_verified=True,
        confidence_score=90.0,
        confirmed_count=18,
        verdict_summary="Confirmed genuine match",
    )
    mock_verifier.verify_match.side_effect = [res_rejected, res_good]

    worker = AlertWorkerService(engine=test_engine, settings=settings, verifier=mock_verifier)

    with patch("delta_cards.alerts.service.send_discord_webhook", return_value=True) as mock_send:
        result = worker.dispatch_one_pending_alert()

        # Only the good offer was dispatched!
        assert result is not None
        assert result.id == "offer_real"
        assert mock_send.call_count == 1
        assert mock_verifier.verify_match.call_count == 2

    # Verify statuses in database
    with Session(test_engine) as session:
        db_fake = session.exec(select(WallapopOffers).where(WallapopOffers.id == "offer_fake")).first()
        assert db_fake.alert_status == "rejected"

        db_real = session.exec(select(WallapopOffers).where(WallapopOffers.id == "offer_real")).first()
        assert db_real.alert_status == "sent"


def test_alert_worker_returns_none_when_no_pending(test_engine):
    """Test that AlertWorkerService returns None when there are no pending offers."""
    settings = Settings(
        _env_file=None,
        discord_webhook_url="https://discord.com/api/webhooks/mock",
        discord_alerts_enabled=True,
    )
    worker = AlertWorkerService(engine=test_engine, settings=settings)
    assert worker.dispatch_one_pending_alert() is None


def test_repository_pending_alert_queries(test_engine):
    """Test repository functions get_pending_alert_offers and update_offer_alert_status."""
    with Session(test_engine) as session:
        o1 = WallapopOffers(
            id="p1",
            title="Card 1",
            price=15.0,
            product_type="single_card",
            link="http://link1",
            alert_status="pending",
        )
        o2 = WallapopOffers(
            id="s1",
            title="Card 2",
            price=15.0,
            product_type="single_card",
            link="http://link2",
            alert_status="sent",
        )
        session.add(o1)
        session.add(o2)
        session.commit()

    pending = get_pending_alert_offers(test_engine)
    assert len(pending) == 1
    assert pending[0].id == "p1"

    ok = update_offer_alert_status("p1", "sent", test_engine)
    assert ok is True

    pending_after = get_pending_alert_offers(test_engine)
    assert len(pending_after) == 0


def test_save_offers_to_db_preserves_alert_status(test_engine):
    """Test that upserting scraped offers preserves previously set alert_status ('sent' or 'pending')."""
    # 1. Initial offer is pending
    initial_offer = OfferListing(
        id="offer_preserve",
        title="Pikachu Rare",
        price=10.0,
        currency="EUR",
        product_type="single_card",
        link="http://pika",
        alert_status="sent",
    )
    save_offers_to_db([initial_offer], engine=test_engine)

    with Session(test_engine) as session:
        saved = session.exec(select(WallapopOffers).where(WallapopOffers.id == "offer_preserve")).first()
        assert saved.alert_status == "sent"

    # 2. Re-scraping the same offer without alert_status specified (defaults to 'none')
    scraped_again = OfferListing(
        id="offer_preserve",
        title="Pikachu Rare (Updated title)",
        price=9.0,
        currency="EUR",
        product_type="single_card",
        link="http://pika",
        alert_status="none",
    )
    save_offers_to_db([scraped_again], engine=test_engine)

    with Session(test_engine) as session:
        reloaded = session.exec(select(WallapopOffers).where(WallapopOffers.id == "offer_preserve")).first()
        # Must preserve 'sent' and not overwrite back to 'none'!
        assert reloaded.alert_status == "sent"
        assert reloaded.price == 9.0
