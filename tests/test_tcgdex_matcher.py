"""Unit tests for offer card assignment, product filtering, pipeline integration, and persistence."""

import pytest
from unittest.mock import MagicMock
from sqlmodel import create_engine, select, Session

from delta_cards.models import OfferListing
from delta_cards.database.models import WallapopOffers
from delta_cards.database.connection import init_db
from delta_cards.database.repository import (
    save_offers_to_db,
    get_all_offers,
    update_offer_tcgdex,
    get_unassigned_card_offers,
)
from delta_cards.recognition.card_ocr import CardOCR
from delta_cards.recognition.pipeline import CardRecognitionPipeline
from delta_cards.tcgdex.assigner import (
    build_cardmarket_url,
    is_card_offer,
    assign_tcgdex_to_offers,
    enrich_card_offers_in_db,
)


def test_is_card_offer_strict_filtering():
    """Verifies that only single cards are processed, and sealed products (boxes, packs, etc.) are excluded."""
    # Allowed
    assert is_card_offer("single_card") is True
    assert is_card_offer("card") is True
    assert is_card_offer("SINGLE_CARD") is True

    # Strictly Excluded
    assert is_card_offer("booster_box") is False
    assert is_card_offer("booster_pack") is False
    assert is_card_offer("deck") is False
    assert is_card_offer("etb") is False
    assert is_card_offer("elite_trainer_box") is False
    assert is_card_offer("caja_sobres") is False
    assert is_card_offer("pokemon_pack") is False
    assert is_card_offer(None) is False
    assert is_card_offer("") is False

    # Title exclusions (lots, mystery boxes, bulk)
    assert is_card_offer("single_card", title="Lote de cartas Pokemon") is False
    assert is_card_offer("single_card", title="Mystery Box Pokemon") is False
    assert is_card_offer("single_card", title="Bulk 500 cartas") is False
    assert is_card_offer("single_card", title="Pikachu 057/191 Chispas Fulgurantes") is True


def test_card_ocr_fraction_extraction():
    """Verifies CardOCR extraction of card number and fractions."""
    ocr = CardOCR()
    fractions = ocr._extract_fractions("Carta Pokemon Charizard VMAX 020/189 Holo Original")
    assert len(fractions) > 0
    assert fractions[0][2] == "020/189"


def test_build_cardmarket_url():
    """Tests generating direct Cardmarket redirect URLs."""
    # Basic usage
    assert (
        build_cardmarket_url("483559")
        == "https://www.cardmarket.com/es/Pokemon/Products?idProduct=483559"
    )
    # Integer product ID and English language
    assert (
        build_cardmarket_url(483559, language="en")
        == "https://www.cardmarket.com/en/Pokemon/Products?idProduct=483559"
    )
    # Empty or None ID
    assert build_cardmarket_url(None) is None
    assert build_cardmarket_url("") is None


def test_assign_tcgdex_to_offers_skips_boxes_and_packs():
    """Verifies that assign_tcgdex_to_offers skips booster boxes and booster packs,

    and only assigns single cards via the pipeline.
    """
    box_offer = OfferListing(
        id="box_1",
        title="Pokemon Booster Box 151 Precintada",
        price=180.0,
        currency="EUR",
        product_type="booster_box",
        link="https://wallapop.com/item/box_1",
    )
    pack_offer = OfferListing(
        id="pack_1",
        title="Pokemon Booster Pack Oscuridad Incandescente",
        price=5.0,
        currency="EUR",
        product_type="booster_pack",
        link="https://wallapop.com/item/pack_1",
    )
    card_offer = OfferListing(
        id="card_1",
        title="Pokemon Furret 136/189 Oscuridad Incandescente",
        price=2.0,
        currency="EUR",
        product_type="single_card",
        link="https://wallapop.com/item/card_1",
    )

    offers = [box_offer, pack_offer, card_offer]

    # Mock pipeline
    mock_pipeline = MagicMock(spec=CardRecognitionPipeline)
    def mock_process_sync(offers_list, progress_callback=None):
        for o in offers_list:
            if is_card_offer(o.product_type, title=o.title, description=o.description):
                o.tcgdex_id = "swsh3-136"
                o.cardmarket_product_id = "483559"
                o.cardmarket_url = "https://www.cardmarket.com/es/Pokemon/Products?idProduct=483559"
                o.cardmarket_low_price = 0.08
                o.language = "es"
        return offers_list

    mock_pipeline.process_offers_sync.side_effect = mock_process_sync

    enriched = assign_tcgdex_to_offers(offers, pipeline=mock_pipeline)

    # Box and pack must NEVER be touched
    assert box_offer.tcgdex_id is None
    assert box_offer.cardmarket_product_id is None
    assert box_offer.cardmarket_url is None
    assert pack_offer.tcgdex_id is None
    assert pack_offer.cardmarket_product_id is None
    assert pack_offer.cardmarket_url is None

    # Card must be enriched
    assert card_offer.tcgdex_id == "swsh3-136"
    assert card_offer.cardmarket_product_id == "483559"
    assert card_offer.cardmarket_url == "https://www.cardmarket.com/es/Pokemon/Products?idProduct=483559"
    assert card_offer.cardmarket_low_price == 0.08
    assert card_offer.language == "es"


def test_database_persistence_and_repository_helpers():
    """Verifies that enriched card offers correctly persist tcgdex_id and

    cardmarket_product_id into the WallapopOffers database table, and tests
    the repository helpers (update_offer_tcgdex, get_unassigned_card_offers).
    """
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)

    card_listing = OfferListing(
        id="card_100",
        title="Carta Charizard Base Set 4/102",
        price=350.0,
        currency="EUR",
        product_type="single_card",
        link="https://wallapop.com/item/card_100",
        tcgdex_id="base1-4",
        cardmarket_product_id="12345",
        cardmarket_url="https://www.cardmarket.com/es/Pokemon/Products?idProduct=12345",
    )
    box_listing = OfferListing(
        id="box_200",
        title="Caja Booster Box 151",
        price=180.0,
        currency="EUR",
        product_type="booster_box",
        link="https://wallapop.com/item/box_200",
    )

    save_offers_to_db([card_listing, box_listing], engine=engine)

    offers = get_all_offers(engine=engine)
    o_map = {o.id: o for o in offers}

    assert o_map["card_100"].tcgdex_id == "base1-4"
    assert o_map["card_100"].cardmarket_product_id == "12345"
    assert o_map["box_200"].tcgdex_id is None
    assert o_map["box_200"].cardmarket_product_id is None

    # Test update_offer_tcgdex helper
    updated = update_offer_tcgdex(
        offer_id="box_200",
        tcgdex_id="custom-id",
        cardmarket_product_id="cm-999",
        cardmarket_url="https://www.cardmarket.com/es/Pokemon/Products?idProduct=cm-999",
        engine=engine,
    )
    assert updated is True

    with Session(engine) as session:
        box = session.exec(select(WallapopOffers).where(WallapopOffers.id == "box_200")).first()
        assert box.tcgdex_id == "custom-id"
        assert box.cardmarket_product_id == "cm-999"
        assert box.cardmarket_product.cardmarket_url == "https://www.cardmarket.com/es/Pokemon/Products?idProduct=cm-999"

        card = session.exec(select(WallapopOffers).where(WallapopOffers.id == "card_100")).first()
        assert card.cardmarket_product.cardmarket_url == "https://www.cardmarket.com/es/Pokemon/Products?idProduct=12345"


def test_enrich_card_offers_in_db():
    """Verifies that enrich_card_offers_in_db finds unassigned card offers

    and updates them in the database using the pipeline while ignoring sealed products.
    """
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)

    # Insert an unassigned card offer and an unassigned booster box
    unassigned_card = OfferListing(
        id="card_201",
        title="Pokemon Furret 136/189",
        price=2.0,
        currency="EUR",
        product_type="single_card",
        link="https://wallapop.com/item/card_201",
        tcgdex_id=None,
        cardmarket_product_id=None,
    )
    unassigned_box = OfferListing(
        id="box_202",
        title="Pokemon Booster Box",
        price=150.0,
        currency="EUR",
        product_type="booster_box",
        link="https://wallapop.com/item/box_202",
        tcgdex_id=None,
        cardmarket_product_id=None,
    )
    save_offers_to_db([unassigned_card, unassigned_box], engine=engine)

    # Check unassigned cards query
    unassigned_list = get_unassigned_card_offers(engine=engine)
    assert len(unassigned_list) == 1
    assert unassigned_list[0].id == "card_201"

    # Mock pipeline
    mock_pipeline = MagicMock(spec=CardRecognitionPipeline)
    mock_pipeline.process_card_listing.return_value = {
        "status": "success",
        "verified_card": {"id": "swsh3-136", "name": "Furret"},
        "cardmarket": {
            "idProduct": "483559",
            "url": "https://www.cardmarket.com/es/Pokemon/Products?idProduct=483559",
            "prices": {"low": 0.08},
        },
        "extracted_metadata": {"language": "es"},
    }

    updated_count = enrich_card_offers_in_db(engine=engine, pipeline=mock_pipeline)
    assert updated_count == 1

    # Verify database state
    with Session(engine) as session:
        card_db = session.exec(select(WallapopOffers).where(WallapopOffers.id == "card_201")).first()
        assert card_db.tcgdex_id == "swsh3-136"
        assert card_db.cardmarket_product_id == "483559"
        assert card_db.cardmarket_product.cardmarket_url == "https://www.cardmarket.com/es/Pokemon/Products?idProduct=483559"

        box_db = session.exec(select(WallapopOffers).where(WallapopOffers.id == "box_202")).first()
        assert box_db.tcgdex_id is None
        assert box_db.cardmarket_product_id is None
        assert box_db.cardmarket_product is None


def test_cli_tcgdex_arguments():
    """Verifies that the new CLI flags --skip-tcgdex and --assign-existing-cards parse properly."""
    from delta_cards.cli import parse_args

    default_args = parse_args([])
    assert default_args.skip_tcgdex is False
    assert default_args.assign_existing_cards is False
    assert default_args.profile is None
    assert default_args.type is None
    assert default_args.scrape_prices is False

    skip_args = parse_args(["--skip-tcgdex"])
    assert skip_args.skip_tcgdex is True

    assign_args = parse_args(["--assign-existing-cards"])
    assert assign_args.assign_existing_cards is True

    custom_args = parse_args(["--scrape-prices", "-p", "single_card", "-t", "single_card"])
    assert custom_args.scrape_prices is True
    assert custom_args.profile == ["single_card"]
    assert custom_args.type == ["single_card"]


def test_already_scraped_offers_not_reassigned_even_if_unmatched():
    """Verifies that offers already present in the database (whether matched or unmatched)

    do NOT get sent to the CardRecognitionPipeline again, their cardmarket/tcgdex fields
    are preserved, and other fields (such as price and title) are updated upon DB save.
    """
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)

    # 1. Pre-populate database with two offers:
    # - card_matched: previously matched with TCGdex and Cardmarket
    # - card_unmatched: previously scraped but couldn't match (None IDs)
    pre_matched = OfferListing(
        id="card_prev_matched",
        title="Carta Pikachu 057/191",
        price=10.0,
        currency="EUR",
        product_type="single_card",
        link="https://wallapop.com/item/card_prev_matched",
        tcgdex_id="sv08-057",
        cardmarket_product_id="799111",
        cardmarket_url="https://www.cardmarket.com/es/Pokemon/Products?idProduct=799111",
    )
    pre_unmatched = OfferListing(
        id="card_prev_unmatched",
        title="Carta Rara Antigua",
        price=20.0,
        currency="EUR",
        product_type="single_card",
        link="https://wallapop.com/item/card_prev_unmatched",
        tcgdex_id=None,
        cardmarket_product_id=None,
    )
    save_offers_to_db([pre_matched, pre_unmatched], engine=engine)

    # 2. Fresh scrape returns updated prices for the 2 existing offers, PLUS 1 new offer
    scraped_matched = OfferListing(
        id="card_prev_matched",
        title="Carta Pikachu 057/191 (Updated)",
        price=8.5,  # Price dropped
        currency="EUR",
        product_type="single_card",
        link="https://wallapop.com/item/card_prev_matched",
    )
    scraped_unmatched = OfferListing(
        id="card_prev_unmatched",
        title="Carta Rara Antigua (Updated)",
        price=15.0,  # Price dropped
        currency="EUR",
        product_type="single_card",
        link="https://wallapop.com/item/card_prev_unmatched",
    )
    scraped_new = OfferListing(
        id="card_brand_new",
        title="Charmander 004/102 Base Set",
        price=50.0,
        currency="EUR",
        product_type="single_card",
        link="https://wallapop.com/item/card_brand_new",
    )
    offers_batch = [scraped_matched, scraped_unmatched, scraped_new]

    # 3. Caller-side DB check (as done in cli.py):
    # Query DB for already scraped offers and restore their data
    with Session(engine) as s:
        existing_rows = s.exec(
            select(WallapopOffers.id, WallapopOffers.tcgdex_id, WallapopOffers.cardmarket_product_id, WallapopOffers.language)
            .where(WallapopOffers.id.in_([o.id for o in offers_batch]))
        ).all()
        db_map = {r[0]: r for r in existing_rows}
        already_in_db = set(db_map.keys())
        for o in offers_batch:
            if o.id in db_map:
                _, db_tcg, db_cm, db_lang = db_map[o.id]
                if db_tcg:
                    o.tcgdex_id = db_tcg
                if db_cm:
                    o.cardmarket_product_id = db_cm
                if db_lang:
                    o.language = db_lang

    # Offers already in DB or already assigned do NOT go to the assigner
    offers_to_assign = [
        o for o in offers_batch
        if o.id not in already_in_db and not o.tcgdex_id and not o.cardmarket_product_id
    ]

    # Mock pipeline: record calls to verify who gets processed
    processed_calls = []
    mock_pipeline = MagicMock(spec=CardRecognitionPipeline)

    def mock_process_sync(offers_list, progress_callback=None):
        for o in offers_list:
            processed_calls.append(o.id)
            if o.id == "card_brand_new":
                o.tcgdex_id = "base1-4"
                o.cardmarket_product_id = "12345"
                o.language = "es"
        return offers_list

    mock_pipeline.process_offers_sync.side_effect = mock_process_sync

    # Run assign_tcgdex_to_offers ONLY with the new unassigned offers
    assign_tcgdex_to_offers(offers_to_assign, pipeline=mock_pipeline)

    # Only card_brand_new entered the pipeline!
    assert processed_calls == ["card_brand_new"]

    # Pre-matched offer preserved its DB values
    assert scraped_matched.tcgdex_id == "sv08-057"
    assert scraped_matched.cardmarket_product_id == "799111"

    # Pre-unmatched offer remained None (no rerun)
    assert scraped_unmatched.tcgdex_id is None
    assert scraped_unmatched.cardmarket_product_id is None

    # New card was matched
    assert scraped_new.tcgdex_id == "base1-4"
    assert scraped_new.cardmarket_product_id == "12345"

    # 4. Save offers back to DB and verify fields
    save_offers_to_db(offers_batch, engine=engine)

    with Session(engine) as s:
        db_matched = s.exec(select(WallapopOffers).where(WallapopOffers.id == "card_prev_matched")).first()
        assert db_matched.price == 8.5  # Price updated
        assert db_matched.title == "Carta Pikachu 057/191 (Updated)"  # Title updated
        assert db_matched.tcgdex_id == "sv08-057"  # Preserved
        assert db_matched.cardmarket_product_id == "799111"  # Preserved

        db_unmatched = s.exec(select(WallapopOffers).where(WallapopOffers.id == "card_prev_unmatched")).first()
        assert db_unmatched.price == 15.0  # Price updated
        assert db_unmatched.title == "Carta Rara Antigua (Updated)"
        assert db_unmatched.tcgdex_id is None
        assert db_unmatched.cardmarket_product_id is None

        db_new = s.exec(select(WallapopOffers).where(WallapopOffers.id == "card_brand_new")).first()
        assert db_new.price == 50.0
        assert db_new.tcgdex_id == "base1-4"
        assert db_new.cardmarket_product_id == "12345"


def test_assign_tcgdex_to_offers_concurrency():
    """Verifies that assign_tcgdex_to_offers runs with max_concurrency and processes all offers."""
    offers = [
        OfferListing(
            id=f"card_{i}",
            title=f"Pokemon Card {i}",
            price=5.0,
            currency="EUR",
            product_type="single_card",
            link=f"https://wallapop.com/item/card_{i}",
        )
        for i in range(12)
    ]

    mock_pipeline = MagicMock(spec=CardRecognitionPipeline)
    def mock_process_sync(offers_list, progress_callback=None, max_concurrency=4):
        for idx, o in enumerate(offers_list, 1):
            o.tcgdex_id = f"tcg_{o.id}"
            o.cardmarket_product_id = f"cm_{o.id}"
            if progress_callback:
                progress_callback(idx, len(offers_list), o)
        return offers_list

    mock_pipeline.process_offers_sync.side_effect = mock_process_sync

    callback_calls = []
    def on_progress(idx, total, o):
        callback_calls.append((idx, total, o.id))

    enriched = assign_tcgdex_to_offers(
        offers,
        pipeline=mock_pipeline,
        progress_callback=on_progress,
        max_concurrency=4,
    )

    assert len(enriched) == 12
    assert len(callback_calls) == 12
    for o in enriched:
        assert o.tcgdex_id == f"tcg_{o.id}"
        assert o.cardmarket_product_id == f"cm_{o.id}"

