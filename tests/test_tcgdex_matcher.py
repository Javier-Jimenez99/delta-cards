"""Unit tests for the TCGDex matching, parsing, product filtering, and persistence module."""

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
from delta_cards.tcgdex.parser import parse_card_title, extract_card_number
from delta_cards.tcgdex.matcher import TCGDexMatcher, MatchResult
from delta_cards.tcgdex.assigner import (
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


def test_extract_card_number():
    """Verifies extraction of card numbers and set totals."""
    assert extract_card_number("Furret 136/189") == ("136", "189")
    assert extract_card_number("Charizard 004/102 Holo") == ("4", "102")
    assert extract_card_number("Pikachu #025/165") == ("25", "165")
    assert extract_card_number("Mewtwo #150") == ("150", None)
    assert extract_card_number("No. 25 Pikachu") == ("25", None)
    assert extract_card_number("Sin numero aqui") == (None, None)


def test_parse_card_title():
    """Tests title parsing, noise removal, and modifier identification."""
    parsed = parse_card_title("Carta Pokemon Charizard VMAX 020/189 Holo Original")
    assert parsed.local_id == "20"
    assert parsed.total_cards_in_set == "189"
    assert "VMAX" in parsed.modifier_hints
    assert "HOLO" in parsed.modifier_hints
    assert "charizard" in parsed.card_name.lower()
    # Ensure noise words like "Carta", "Pokemon", "Original" were removed from card_name
    assert "carta" not in parsed.card_name.lower()
    assert "original" not in parsed.card_name.lower()


def test_matcher_extract_cardmarket_id():
    """Tests extracting Cardmarket Product ID from raw TCGDex payload."""
    matcher = TCGDexMatcher()

    # Payload with pricing.cardmarket.idProduct
    raw_with_pricing = {
        "id": "swsh3-136",
        "pricing": {
            "cardmarket": {
                "idProduct": 483559,
                "avg": 0.08,
            }
        },
    }
    assert matcher.extract_cardmarket_id(raw_with_pricing) == "483559"

    # Payload with variants_detailed.thirdParty.cardmarket
    raw_with_variant = {
        "id": "swsh3-136",
        "variants_detailed": [
            {
                "thirdParty": {
                    "cardmarket": 999123,
                }
            }
        ],
    }
    assert matcher.extract_cardmarket_id(raw_with_variant) == "999123"

    # Payload with no cardmarket ID
    assert matcher.extract_cardmarket_id({}) is None
    assert matcher.extract_cardmarket_id({"pricing": {}}) is None


def test_assign_tcgdex_to_offers_skips_boxes_and_packs():
    """Verifies that assign_tcgdex_to_offers skips booster boxes and booster packs,

    and only assigns single cards.
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

    # Mock matcher
    mock_matcher = MagicMock(spec=TCGDexMatcher)
    mock_matcher.match.return_value = MatchResult(
        tcgdex_id="swsh3-136",
        cardmarket_product_id="483559",
        confidence_score=95.0,
        matched_name="Furret",
        matched_local_id="136",
        matched_set_name="Oscuridad Incandescente",
    )

    enriched = assign_tcgdex_to_offers(offers, matcher=mock_matcher)

    # Box and pack must NEVER be touched
    assert box_offer.tcgdex_id is None
    assert box_offer.cardmarket_product_id is None
    assert pack_offer.tcgdex_id is None
    assert pack_offer.cardmarket_product_id is None

    # Card must be enriched
    assert card_offer.tcgdex_id == "swsh3-136"
    assert card_offer.cardmarket_product_id == "483559"

    # Matcher should have been called only ONCE (for the card)
    assert mock_matcher.match.call_count == 1


def test_database_persistence_and_repository_helpers():
    """Verifies that enriched card offers correctly persist tcgdex_id and

    cardmarket_product_id into the WallapopOffers database table, and tests
    the new repository helpers (update_offer_tcgdex, get_unassigned_card_offers).
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
        engine=engine,
    )
    assert updated is True

    with Session(engine) as session:
        box = session.exec(select(WallapopOffers).where(WallapopOffers.id == "box_200")).first()
        assert box.tcgdex_id == "custom-id"
        assert box.cardmarket_product_id == "cm-999"


def test_enrich_card_offers_in_db():
    """Verifies that enrich_card_offers_in_db finds unassigned card offers

    and updates them in the database while ignoring sealed products.
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

    # Mock matcher
    mock_matcher = MagicMock(spec=TCGDexMatcher)
    mock_matcher.match.return_value = MatchResult(
        tcgdex_id="swsh3-136",
        cardmarket_product_id="483559",
        confidence_score=90.0,
    )

    updated_count = enrich_card_offers_in_db(engine=engine, matcher=mock_matcher)
    assert updated_count == 1

    # Verify database state
    with Session(engine) as session:
        card_db = session.exec(select(WallapopOffers).where(WallapopOffers.id == "card_201")).first()
        assert card_db.tcgdex_id == "swsh3-136"
        assert card_db.cardmarket_product_id == "483559"

        box_db = session.exec(select(WallapopOffers).where(WallapopOffers.id == "box_202")).first()
        assert box_db.tcgdex_id is None
        assert box_db.cardmarket_product_id is None


def test_cli_tcgdex_arguments():
    """Verifies that the new CLI flags --skip-tcgdex and --assign-existing-cards parse properly."""
    from delta_cards.cli import parse_args

    default_args = parse_args([])
    assert default_args.skip_tcgdex is False
    assert default_args.assign_existing_cards is False

    skip_args = parse_args(["--skip-tcgdex"])
    assert skip_args.skip_tcgdex is True

    assign_args = parse_args(["--assign-existing-cards"])
    assert assign_args.assign_existing_cards is True

