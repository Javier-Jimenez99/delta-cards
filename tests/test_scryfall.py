import os
import tempfile
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from sqlmodel import Session, select

from delta_cards.database.connection import get_engine
from delta_cards.database.models import PlatformOffers, CardmarketProduct, CardmarketPrice
from delta_cards.models import OfferListing
from delta_cards.scryfall.client import (
    ScryfallClient,
    build_magic_cardmarket_url,
    extract_scryfall_cardmarket_prices,
    fetch_scryfall_card_pricing,
)
from delta_cards.scryfall.vector_cache import (
    ScryfallVectorCache,
    get_scryfall_image_url,
)
from delta_cards.scryfall.assigner import (
    extract_magic_identifiers,
    extract_magic_listing_features,
    MagicListingFeatures,
    score_magic_printing,
    select_best_magic_printing,
    select_best_magic_printing_visual,
    resolve_scryfall_card,
    assign_scryfall_to_offers,
    enrich_magic_offers_in_db,
)



def test_build_magic_cardmarket_url():
    """Verify Magic Cardmarket URL construction."""
    assert build_magic_cardmarket_url("771076", "es") == "https://www.cardmarket.com/es/Magic/Products?idProduct=771076"
    assert build_magic_cardmarket_url("771076", "en") == "https://www.cardmarket.com/en/Magic/Products?idProduct=771076"
    assert build_magic_cardmarket_url(None) is None


def test_extract_scryfall_cardmarket_prices():
    """Verify extracting EUR prices and cardmarket product ID from Scryfall card dict."""
    sample_scryfall = {
        "id": "7673784e-db4b-43a1-8d55-1bb9fc1e284f",
        "name": "Lightning Bolt",
        "set": "msc",
        "set_name": "Marvel Super Heroes Commander",
        "collector_number": "806",
        "cardmarket_id": 892161,
        "prices": {
            "usd": "0.99",
            "usd_foil": "4.41",
            "eur": "2.03",
            "eur_foil": "1.86",
        },
        "purchase_uris": {
            "cardmarket": "https://www.cardmarket.com/en/Magic/Products?idProduct=892161&referrer=scryfall",
        },
        "image_uris": {
            "normal": "https://cards.scryfall.io/normal/front/7/6/7673784e-db4b-43a1-8d55-1bb9fc1e284f.jpg",
        },
    }

    result = extract_scryfall_cardmarket_prices(sample_scryfall, language="es")
    assert result["idProduct"] == "892161"
    assert result["url"] == "https://www.cardmarket.com/en/Magic/Products?idProduct=892161&referrer=scryfall"
    assert result["prices"]["eur"] == 2.03
    assert result["prices"]["eur_foil"] == 1.86
    assert result["prices"]["avg"] == 2.03
    assert result["prices"]["low"] == 2.03
    assert result["name"] == "Lightning Bolt"
    assert result["set"] == "msc"
    assert result["collector_number"] == "806"
    assert result["image_url"] == "https://cards.scryfall.io/normal/front/7/6/7673784e-db4b-43a1-8d55-1bb9fc1e284f.jpg"


def test_extract_magic_identifiers():
    """Verify extraction of set codes, numbers, and clean card names from marketplace titles."""
    # Pattern: Set code + collector number
    set_code, num, name = extract_magic_identifiers("Carta Magic The Gathering Ragavan, Nimble Pilferer MH2 138 Foil")
    assert set_code == "mh2"
    assert num == "138"
    assert "ragavan" in name.lower()

    # Pattern: Simple card name with marketplace fluff
    set_code2, num2, name2 = extract_magic_identifiers("Carta MTG Sheoldred, the Apocalypse Near Mint Spanish")
    assert "sheoldred" in name2.lower()
    assert "apocalypse" in name2.lower()


def test_resolve_scryfall_card_mock():
    """Verify resolve_scryfall_card delegates to ScryfallClient."""
    mock_client = MagicMock(spec=ScryfallClient)
    mock_client.get_card_by_set_number.return_value = {
        "id": "mock-uuid-1",
        "name": "The One Ring",
        "set": "ltr",
        "collector_number": "269",
        "cardmarket_id": 718300,
        "prices": {"eur": "95.00"},
        "purchase_uris": {"cardmarket": "https://www.cardmarket.com/en/Magic/Products?idProduct=718300"},
    }

    res = resolve_scryfall_card("The One Ring LTR 269", client=mock_client)
    assert res is not None
    assert res["idProduct"] == "718300"
    assert res["prices"]["avg"] == 95.0
    mock_client.get_card_by_set_number.assert_called_once_with("ltr", "269", lang="es")


def test_assign_scryfall_to_offers():
    """Verify assign_scryfall_to_offers populates OfferListing attributes."""
    mock_client = MagicMock(spec=ScryfallClient)
    mock_client.get_card_by_name.return_value = {
        "id": "mock-uuid-2",
        "name": "Sheoldred, the Apocalypse",
        "set": "dmu",
        "collector_number": "107",
        "cardmarket_id": 673200,
        "prices": {"eur": "75.00"},
        "purchase_uris": {"cardmarket": "https://www.cardmarket.com/en/Magic/Products?idProduct=673200"},
    }

    offers = [
        OfferListing(
            id="offer-1",
            title="Carta Magic Sheoldred, the Apocalypse",
            price=25.0,
            product_type="single_card",
            platform="wallapop",
            link="https://es.wallapop.com/item/1",
        ),
        OfferListing(
            id="offer-2",
            title="Lote 500 cartas Magic bulk",
            price=15.0,
            product_type="booster_box",  # non-single card
            platform="wallapop",
            link="https://es.wallapop.com/item/2",
        ),
    ]

    assigned = assign_scryfall_to_offers(offers, client=mock_client)
    assert assigned[0].cardmarket_product_id == "673200"
    assert assigned[0].cardmarket_avg_price == 75.0
    # Non-card offer should remain unassigned
    assert getattr(assigned[1], "cardmarket_product_id", None) is None


def test_enrich_magic_offers_in_db():
    """Verify database enrichment updates PlatformOffers, CardmarketProduct, and detects deals."""
    engine = get_engine("sqlite:///:memory:")
    from delta_cards.database import init_db
    init_db(engine)

    mock_client = MagicMock(spec=ScryfallClient)
    mock_client.get_card_by_name.return_value = {
        "id": "uuid-sheoldred",
        "name": "Sheoldred, the Apocalypse",
        "set": "dmu",
        "collector_number": "107",
        "cardmarket_id": 673200,
        "prices": {"eur": "80.00"},
        "purchase_uris": {"cardmarket": "https://www.cardmarket.com/es/Magic/Products?idProduct=673200"},
    }

    with Session(engine) as session:
        # Deal candidate (price 30 < market 80)
        session.add(
            PlatformOffers(
                id="deal_offer",
                title="Sheoldred, the Apocalypse",
                price=30.0,
                product_type="single_card",
                platform="wallapop",
                game="magic",
                link="https://es.wallapop.com/item/deal",
                assigned_state="pending",
                alert_status="none",
            )
        )
        session.commit()

    updated = enrich_magic_offers_in_db(engine=engine, client=mock_client)
    assert updated == 1

    with Session(engine) as session:
        db_offer = session.exec(select(PlatformOffers).where(PlatformOffers.id == "deal_offer")).first()
        assert db_offer.assigned_state == "assigned"
        assert db_offer.cardmarket_product_id == "673200"
        assert db_offer.alert_status == "pending"  # Deal detected!

        cm_prod = session.exec(select(CardmarketProduct).where(CardmarketProduct.id == "673200")).first()
        assert cm_prod is not None
        assert "Magic" in cm_prod.cardmarket_url

        cm_price = session.exec(select(CardmarketPrice).where(CardmarketPrice.cardmarket_product_id == "673200")).first()
        assert cm_price is not None
        assert cm_price.avg_price == 80.0


def test_extract_magic_listing_features():
    """Verify detailed version feature extraction for Magic listings."""
    # 1. Borderless + Foil + The One Ring
    f1 = extract_magic_listing_features("Carta Magic The One Ring Borderless Foil Near Mint")
    assert f1.candidate_name == "The One Ring"
    assert f1.is_borderless is True
    assert f1.is_foil is True
    assert f1.is_art_card is False

    # 2. Retro frame + Horizontes de Modern 2
    f2 = extract_magic_listing_features("Ragavan, Nimble Pilferer Marco Retro Foilada Horizontes de Modern 2")
    assert "Ragavan" in (f2.candidate_name or "")
    assert f2.is_retro is True
    assert f2.is_foil is True
    assert f2.set_code == "mh2"

    # 3. Showcase / Vitrina
    f3 = extract_magic_listing_features("Sheoldred, the Apocalypse Vitrina Dominaria United")
    assert "Sheoldred" in (f3.candidate_name or "")
    assert f3.is_showcase is True
    assert f3.set_code == "dmu"

    # 4. Extended Art + Non-foil
    f4 = extract_magic_listing_features("The One Ring Extended Art No Foil")
    assert f4.is_extended_art is True
    assert f4.is_foil is False

    # 5. Art card detection
    f5 = extract_magic_listing_features("Sol Ring Carta de Arte Sin Borde")
    assert f5.is_art_card is True

    # 6. Masterpiece Kaladesh Inventions
    f6 = extract_magic_listing_features("Sol Ring Masterpiece Kaladesh Inventions")
    assert f6.is_masterpiece is True
    assert f6.set_code == "mps"

    # 7. Serialized
    f7 = extract_magic_listing_features("The One Ring Serializada 001/500")
    assert f7.is_serialized is True


def test_finish_aware_pricing():
    """Verify that extract_scryfall_cardmarket_prices selects the correct EUR price based on finish."""
    card_data = {
        "cardmarket_id": 123456,
        "name": "Ragavan, Nimble Pilferer",
        "prices": {
            "eur": "35.00",
            "eur_foil": "75.00",
            "eur_etched": "55.00",
        },
    }

    # Explicit Foil: should select eur_foil
    res_foil = extract_scryfall_cardmarket_prices(card_data, is_foil=True)
    assert res_foil["prices"]["avg"] == 75.0
    assert res_foil["prices"]["eur_foil"] == 75.0

    # Explicit Non-foil: should select eur
    res_nonfoil = extract_scryfall_cardmarket_prices(card_data, is_foil=False)
    assert res_nonfoil["prices"]["avg"] == 35.0
    assert res_nonfoil["prices"]["eur"] == 35.0

    # Unspecified: should default to eur
    res_default = extract_scryfall_cardmarket_prices(card_data, is_foil=None)
    assert res_default["prices"]["avg"] == 35.0


def test_select_best_printing_the_one_ring():
    """Verify selection of the exact version of The One Ring based on treatments."""
    sample_prints = [
        {
            "id": "p-reg",
            "name": "The One Ring",
            "set": "ltr",
            "collector_number": "246",
            "border_color": "black",
            "frame_effects": ["legendary"],
            "promo_types": ["universesbeyond"],
            "cardmarket_id": 701676,
            "prices": {"eur": "93.68", "eur_foil": "114.43"},
            "released_at": "2023-06-23",
        },
        {
            "id": "p-ea",
            "name": "The One Ring",
            "set": "ltr",
            "collector_number": "380",
            "border_color": "black",
            "frame_effects": ["legendary", "extendedart"],
            "promo_types": ["boosterfun", "universesbeyond"],
            "cardmarket_id": 701771,
            "prices": {"eur": "130.64", "eur_foil": "387.50"},
            "released_at": "2023-06-23",
        },
        {
            "id": "p-borderless",
            "name": "The One Ring",
            "set": "ltr",
            "collector_number": "451",
            "border_color": "borderless",
            "frame_effects": ["legendary", "inverted"],
            "promo_types": ["universesbeyond", "boosterfun", "bundle"],
            "cardmarket_id": 716083,
            "prices": {"eur": "93.83", "eur_foil": "92.86"},
            "released_at": "2023-06-23",
        },
        {
            "id": "p-serial",
            "name": "The One Ring",
            "set": "ltr",
            "collector_number": "001",
            "border_color": "borderless",
            "frame_effects": ["legendary"],
            "promo_types": ["boosterfun", "serialized", "universesbeyond"],
            "cardmarket_id": 701764,
            "prices": {"eur": None, "eur_foil": None},
            "released_at": "2023-06-23",
        },
    ]

    # Case 1: Borderless listing -> should match p-borderless
    f_borderless = MagicListingFeatures(candidate_name="The One Ring", is_borderless=True)
    best_b = select_best_magic_printing(sample_prints, f_borderless)
    assert best_b is not None
    assert best_b["cardmarket_id"] == 716083
    assert best_b["collector_number"] == "451"

    # Case 2: Extended Art listing -> should match p-ea
    f_ea = MagicListingFeatures(candidate_name="The One Ring", is_extended_art=True)
    best_ea = select_best_magic_printing(sample_prints, f_ea)
    assert best_ea is not None
    assert best_ea["cardmarket_id"] == 701771
    assert best_ea["collector_number"] == "380"

    # Case 3: Regular / Unversioned listing -> MUST pick p-reg and NEVER p-serial
    f_reg = MagicListingFeatures(candidate_name="The One Ring")
    best_reg = select_best_magic_printing(sample_prints, f_reg)
    assert best_reg is not None
    assert best_reg["cardmarket_id"] == 701676
    assert best_reg["collector_number"] == "246"


def test_select_best_printing_ragavan_retro():
    """Verify that retro frame Ragavan correctly selects the retro printing."""
    sample_prints = [
        {
            "id": "p-mh2-reg",
            "name": "Ragavan, Nimble Pilferer",
            "set": "mh2",
            "collector_number": "138",
            "border_color": "black",
            "frame": "2015",
            "frame_effects": ["legendary"],
            "promo_types": [],
            "cardmarket_id": 566480,
            "prices": {"eur": "33.74", "eur_foil": "52.51"},
            "released_at": "2021-06-18",
        },
        {
            "id": "p-h2r-retro",
            "name": "Ragavan, Nimble Pilferer",
            "set": "h2r",
            "collector_number": "11",
            "border_color": "black",
            "frame": "1997",
            "frame_effects": [],
            "promo_types": ["boosterfun"],
            "cardmarket_id": 771287,
            "prices": {"eur": "44.06", "eur_foil": "87.74"},
            "released_at": "2021-06-18",
        },
    ]

    # Retro listing -> should pick p-h2r-retro
    f_retro = MagicListingFeatures(candidate_name="Ragavan, Nimble Pilferer", is_retro=True)
    best_retro = select_best_magic_printing(sample_prints, f_retro)
    assert best_retro is not None
    assert best_retro["cardmarket_id"] == 771287
    assert best_retro["collector_number"] == "11"

    # Normal listing -> should pick p-mh2-reg
    f_normal = MagicListingFeatures(candidate_name="Ragavan, Nimble Pilferer")
    best_normal = select_best_magic_printing(sample_prints, f_normal)
    assert best_normal is not None
    assert best_normal["cardmarket_id"] == 566480
    assert best_normal["collector_number"] == "138"


def test_resolve_scryfall_card_rejects_art_cards():
    """Verify that resolve_scryfall_card rejects art series / carta de arte listings."""
    mock_client = MagicMock(spec=ScryfallClient)
    res = resolve_scryfall_card("Sol Ring Carta de Arte", client=mock_client)
    assert res is None
    # Client should not even be queried
    mock_client.get_card_by_name.assert_not_called()


def test_resolve_scryfall_card_version_aware_e2e():
    """Verify that resolve_scryfall_card chooses the matching version from client printings."""
    mock_client = MagicMock(spec=ScryfallClient)
    mock_client.get_card_by_name.return_value = {
        "name": "The One Ring",
    }
    mock_client.get_card_prints.return_value = [
        {
            "id": "p-reg",
            "name": "The One Ring",
            "set": "ltr",
            "collector_number": "246",
            "border_color": "black",
            "frame_effects": ["legendary"],
            "promo_types": ["universesbeyond"],
            "cardmarket_id": 701676,
            "prices": {"eur": "93.68", "eur_foil": "114.43"},
        },
        {
            "id": "p-borderless",
            "name": "The One Ring",
            "set": "ltr",
            "collector_number": "451",
            "border_color": "borderless",
            "frame_effects": ["legendary", "inverted"],
            "promo_types": ["universesbeyond", "boosterfun", "bundle"],
            "cardmarket_id": 716083,
            "prices": {"eur": "93.83", "eur_foil": "92.86"},
        },
    ]

    # Call with borderless foil
    result = resolve_scryfall_card("The One Ring Sin Borde Foil", client=mock_client)
    assert result is not None
    assert result["idProduct"] == "716083"
    assert result["collector_number"] == "451"
    assert result["prices"]["avg"] == 92.86  # eur_foil picked because listing is foil


def test_scryfall_vector_cache_save_and_load():
    """Verify ScryfallVectorCache properly stores and retrieves vectors from SQLite and memory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_vectors.sqlite")
        cache = ScryfallVectorCache(db_path=db_path)

        # Create synthetic unit vector
        dummy_vec = np.zeros(384, dtype=np.float32)
        dummy_vec[0] = 1.0

        card_id = "test-card-uuid-1"
        cache.save_vector(
            scryfall_id=card_id,
            embedding=dummy_vec,
            name="Sol Ring",
            set_code="cmm",
            collector_number="400",
            image_url="https://cards.scryfall.io/test.jpg",
        )

        # 1. Retrieve from memory cache
        loaded_mem = cache.get_vector(card_id)
        assert loaded_mem is not None
        assert np.allclose(loaded_mem, dummy_vec)

        # 2. Re-open new cache instance to test persistent SQLite retrieval
        cache2 = ScryfallVectorCache(db_path=db_path)
        loaded_db = cache2.get_vector(card_id)
        assert loaded_db is not None
        assert np.allclose(loaded_db, dummy_vec)

        # 3. Cache miss returns None
        assert cache2.get_vector("non-existent-id") is None


def test_scryfall_vector_cache_get_or_compute():
    """Verify get_or_compute_vector computes on miss and uses cache on subsequent calls."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_vectors.sqlite")
        cache = ScryfallVectorCache(db_path=db_path)

        mock_embedder = MagicMock()
        dummy_vec = np.random.randn(384).astype(np.float32)
        dummy_vec /= np.linalg.norm(dummy_vec)
        mock_embedder.embed_image.return_value = dummy_vec

        card_data = {
            "id": "card-123",
            "name": "Black Lotus",
            "set": "lea",
            "collector_number": "232",
            "image_uris": {"normal": "https://cards.scryfall.io/lotus.jpg"},
        }

        # First call: computes via embedder
        vec1 = cache.get_or_compute_vector(card_data, mock_embedder)
        assert vec1 is not None
        assert np.allclose(vec1, dummy_vec)
        assert mock_embedder.embed_image.call_count == 1

        # Second call: uses cached vector without calling embedder again
        vec2 = cache.get_or_compute_vector(card_data, mock_embedder)
        assert vec2 is not None
        assert np.allclose(vec2, dummy_vec)
        assert mock_embedder.embed_image.call_count == 1


def test_scryfall_get_card_prints_for_card_expansion():
    """Verify that get_card_prints_for_card expands 1 card into all versions via oracle_id."""
    client = ScryfallClient()

    card = {
        "id": "card-one-ring-reg",
        "name": "The One Ring",
        "oracle_id": "oracle-uuid-ring",
        "set": "ltr",
        "collector_number": "246",
    }

    all_prints = [
        {"id": "card-one-ring-reg", "name": "The One Ring", "collector_number": "246"},
        {"id": "card-one-ring-borderless", "name": "The One Ring", "collector_number": "451"},
    ]

    with patch.object(client, "search_cards", return_value=all_prints) as mock_search:
        results = client.get_card_prints_for_card(card)
        assert len(results) == 2
        mock_search.assert_called_once_with('oracle_id:"oracle-uuid-ring" unique:prints')


def test_select_best_magic_printing_visual():
    """Verify that select_best_magic_printing_visual picks the printing with highest cosine similarity."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ScryfallVectorCache(db_path=os.path.join(tmpdir, "test.sqlite"))

        # Create vectors:
        # listing_vec is [1, 0, 0, ...]
        listing_vec = np.zeros(384, dtype=np.float32)
        listing_vec[0] = 1.0

        # version 1 (regular): [0, 1, 0, ...] -> dot product = 0.0
        vec_reg = np.zeros(384, dtype=np.float32)
        vec_reg[1] = 1.0

        # version 2 (showcase): [0.98, 0.2, 0, ...] -> dot product = 0.98 (winner!)
        vec_showcase = np.zeros(384, dtype=np.float32)
        vec_showcase[0] = 0.98
        vec_showcase[1] = 0.2
        vec_showcase /= np.linalg.norm(vec_showcase)

        printings = [
            {
                "id": "p-regular",
                "name": "Sheoldred",
                "collector_number": "107",
                "image_uris": {"normal": "https://cards.scryfall.io/reg.jpg"},
            },
            {
                "id": "p-showcase",
                "name": "Sheoldred",
                "collector_number": "315",
                "image_uris": {"normal": "https://cards.scryfall.io/showcase.jpg"},
            },
        ]

        cache.save_vector("p-regular", vec_reg)
        cache.save_vector("p-showcase", vec_showcase)

        mock_embedder = MagicMock()
        mock_embedder.embed_image.return_value = listing_vec

        winner, sim = select_best_magic_printing_visual(
            printings=printings,
            listing_image="https://marketplace.com/photo.jpg",
            embedder=mock_embedder,
            vector_cache=cache,
        )

        assert winner is not None
        assert winner["id"] == "p-showcase"
        assert sim > 0.90


def test_select_best_magic_printing_visual_tiebreaker():
    """Verify that when two printings have identical art/vectors, text features break the tie."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ScryfallVectorCache(db_path=os.path.join(tmpdir, "test.sqlite"))

        # Both versions have the identical card scan
        shared_vec = np.zeros(384, dtype=np.float32)
        shared_vec[0] = 1.0

        printings = [
            {
                "id": "p-nonfoil",
                "name": "Counterspell",
                "set": "mh2",
                "collector_number": "267",
                "finishes": ["nonfoil"],
                "image_uris": {"normal": "https://cards.scryfall.io/cs.jpg"},
                "released_at": "2021-06-18",
            },
            {
                "id": "p-foil",
                "name": "Counterspell",
                "set": "mh2",
                "collector_number": "267",
                "finishes": ["foil"],
                "image_uris": {"normal": "https://cards.scryfall.io/cs.jpg"},
                "released_at": "2021-06-18",
            },
        ]

        cache.save_vector("p-nonfoil", shared_vec)
        cache.save_vector("p-foil", shared_vec)

        mock_embedder = MagicMock()
        mock_embedder.embed_image.return_value = shared_vec

        # Listing specifies "Foil" -> features should choose p-foil
        features = MagicListingFeatures(candidate_name="Counterspell", is_foil=True)

        winner, sim = select_best_magic_printing_visual(
            printings=printings,
            listing_image="https://marketplace.com/photo.jpg",
            features=features,
            embedder=mock_embedder,
            vector_cache=cache,
        )

        assert winner is not None
        assert winner["id"] == "p-foil"
        assert sim == 1.0


def test_resolve_scryfall_card_expands_and_matches_visually():
    """Verify resolve_scryfall_card expands 1 card into versions and ranks with visual similarity."""
    mock_client = MagicMock(spec=ScryfallClient)
    # 1. Base lookup returns only 1 card
    mock_client.get_card_by_name.return_value = {
        "id": "p-default",
        "name": "The One Ring",
        "oracle_id": "oracle-ring",
    }
    # get_card_prints returns empty initially
    mock_client.get_card_prints.return_value = []

    # 2. Expansion returns all versions
    all_versions = [
        {
            "id": "p-default",
            "name": "The One Ring",
            "collector_number": "246",
            "cardmarket_id": 701676,
            "prices": {"eur": "90.00"},
            "image_uris": {"normal": "https://cards.scryfall.io/reg.jpg"},
        },
        {
            "id": "p-borderless",
            "name": "The One Ring",
            "collector_number": "451",
            "cardmarket_id": 716083,
            "prices": {"eur": "95.00"},
            "image_uris": {"normal": "https://cards.scryfall.io/borderless.jpg"},
        },
    ]
    mock_client.get_card_prints_for_card.return_value = all_versions

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ScryfallVectorCache(db_path=os.path.join(tmpdir, "test.sqlite"))

        # Setup vectors so that borderless is the visual winner
        listing_vec = np.zeros(384, dtype=np.float32)
        listing_vec[0] = 1.0

        vec_reg = np.zeros(384, dtype=np.float32)
        vec_reg[1] = 1.0

        vec_borderless = np.zeros(384, dtype=np.float32)
        vec_borderless[0] = 1.0

        cache.save_vector("p-default", vec_reg)
        cache.save_vector("p-borderless", vec_borderless)

        mock_embedder = MagicMock()
        mock_embedder.embed_image.return_value = listing_vec

        res = resolve_scryfall_card(
            title="The One Ring",
            client=mock_client,
            image_url="https://marketplace.com/ring_photo.jpg",
            vector_cache=cache,
            embedder=mock_embedder,
        )

        assert res is not None
        assert res["idProduct"] == "716083"  # Borderless chosen!
        assert res["collector_number"] == "451"
        assert res.get("visual_similarity") is not None
        assert res["visual_similarity"] == 1.0


def test_assign_scryfall_to_offers_with_image_url():
    """Verify assign_scryfall_to_offers passes image_url and sets visual similarity."""
    mock_client = MagicMock(spec=ScryfallClient)
    mock_client.get_card_by_name.return_value = {
        "id": "sheoldred-107",
        "name": "Sheoldred",
        "set": "dmu",
        "collector_number": "107",
        "cardmarket_id": 673200,
        "prices": {"eur": "70.00"},
        "image_uris": {"normal": "https://cards.scryfall.io/sheoldred.jpg"},
    }
    mock_client.get_card_prints.return_value = [mock_client.get_card_by_name.return_value]
    mock_client.get_card_prints_for_card.return_value = [mock_client.get_card_by_name.return_value]

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ScryfallVectorCache(db_path=os.path.join(tmpdir, "test.sqlite"))
        dummy_vec = np.zeros(384, dtype=np.float32)
        dummy_vec[0] = 1.0
        cache.save_vector("sheoldred-107", dummy_vec)

        mock_embedder = MagicMock()
        mock_embedder.embed_image.return_value = dummy_vec

        offers = [
            OfferListing(
                id="offer-img-1",
                title="Sheoldred the Apocalypse",
                price=35.0,
                product_type="single_card",
                platform="wallapop",
                link="https://es.wallapop.com/item/img-1",
                image_url="https://cdn.wallapop.com/images/sheoldred.jpg",
            )
        ]

        assigned = assign_scryfall_to_offers(
            offers,
            client=mock_client,
            vector_cache=cache,
            embedder=mock_embedder,
        )

        assert assigned[0].cardmarket_product_id == "673200"
        assert getattr(assigned[0], "_temp_visual_similarity", None) == 1.0


def test_real_dinov3_embedder_with_vector_cache():
    """Verify real DINOv3 embedder generates valid 384-dim embeddings and caches them."""
    from delta_cards.recognition.dinov3_embedder import DINOv3Embedder

    embedder = DINOv3Embedder()
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ScryfallVectorCache(db_path=os.path.join(tmpdir, "test.sqlite"))

        # Synthetic card images (e.g. 224x224 RGB image arrays)
        img1 = np.full((224, 224, 3), 128, dtype=np.uint8)
        img2 = np.full((224, 224, 3), 200, dtype=np.uint8)

        vec1 = embedder.embed_image(img1)
        assert vec1.shape == (384,)
        assert abs(np.linalg.norm(vec1) - 1.0) < 1e-4

        # Save into cache and verify
        cache.save_vector("real-test-1", vec1, name="Test Card")
        cached_vec = cache.get_vector("real-test-1")
        assert cached_vec is not None
        assert np.allclose(vec1, cached_vec)

        vec2 = embedder.embed_image(img2)
        # Cosine similarity between two unit vectors
        sim = float(np.dot(vec1, vec2))
        assert -1.0 <= sim <= 1.0


