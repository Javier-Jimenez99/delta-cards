"""Unit tests for Cardmarket catalog downloader and merger."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from delta_cards.catalog_downloader import (
    CARDMARKET_GAMES,
    _download_file,
    download_game_catalog,
    ensure_symlinks,
    merge_products_and_price_guide,
)


def test_merge_products_and_price_guide(tmp_path: Path):
    """Verify merging products JSON and price guide JSON into CSV."""
    prods_data = {
        "version": 1,
        "products": [
            {
                "idProduct": 1001,
                "name": "Test Booster Box",
                "idCategory": 53,
                "categoryName": "Pokémon Display",
                "idExpansion": 10,
                "idMetacard": 0,
                "dateAdded": "2024-01-01 00:00:00",
            },
            {
                "idProduct": 1002,
                "name": "Test Booster Pack",
                "idCategory": 52,
                "categoryName": "Pokémon Booster",
                "idExpansion": 10,
                "idMetacard": 0,
                "dateAdded": "2024-01-01 00:00:00",
            },
        ],
    }

    prices_data = {
        "version": 1,
        "priceGuides": [
            {
                "idProduct": 1001,
                "avg": 120.0,
                "low": 105.0,
                "trend": 115.0,
                "avg-holo": 130.0,
                "low-holo": 110.0,
                "trend-holo": 125.0,
            },
            {
                "idProduct": 1002,
                "avg": 4.5,
                "low": 3.8,
                "trend": 4.2,
            },
        ],
    }

    prod_file = tmp_path / "products.json"
    price_file = tmp_path / "prices.json"
    csv_file = tmp_path / "nonsingles.csv"

    prod_file.write_text(json.dumps(prods_data), encoding="utf-8")
    price_file.write_text(json.dumps(prices_data), encoding="utf-8")

    rows = merge_products_and_price_guide(prod_file, price_file, csv_file, game="pokemon")
    assert rows == 2
    assert csv_file.exists()

    content = csv_file.read_text(encoding="utf-8")
    assert "idProduct,name,idCategory" in content
    assert "Test Booster Box" in content
    assert "105.0" in content
    assert "Test Booster Pack" in content


def test_ensure_symlinks(tmp_path: Path):
    """Verify standard relative symlinks are correctly created."""
    game_dir = tmp_path / "magic"
    game_dir.mkdir(parents=True)

    (game_dir / "magic_keywords.json").write_text("{}", encoding="utf-8")
    (game_dir / "magic_nonsingles.csv").write_text("header\n", encoding="utf-8")
    (game_dir / "price_guide_magic.json").write_text("{}", encoding="utf-8")
    (game_dir / "products_nonsingles_magic.json").write_text("{}", encoding="utf-8")

    ensure_symlinks(game_dir, game="magic")

    assert (game_dir / "keywords.json").is_symlink()
    assert (game_dir / "nonsingles.csv").is_symlink()
    assert (game_dir / "price_guide.json").is_symlink()
    assert (game_dir / "products_nonsingles.json").is_symlink()


def test_download_file_validates_json(tmp_path: Path):
    """Verify that corrupt or invalid downloads are rejected and do not replace the target."""
    target = tmp_path / "test.json"
    target.write_text('{"existing": true}', encoding="utf-8")

    mock_resp = MagicMock()
    mock_resp.iter_content.return_value = [b"not valid json here"]
    mock_resp.raise_for_status = MagicMock()

    mock_session = MagicMock()
    mock_session.get.return_value = mock_resp

    with pytest.raises(ValueError, match="not valid JSON"):
        _download_file(
            url="https://example.com/bad.json",
            target_path=target,
            expected_top_key="priceGuides",
            session=mock_session,
        )

    # Existing file should not be modified
    assert json.loads(target.read_text(encoding="utf-8")) == {"existing": True}


def test_download_game_catalog_end_to_end(tmp_path: Path):
    """Verify download_game_catalog downloads both files and updates the CSV."""
    pg_data = {"version": 1, "priceGuides": [{"idProduct": 999, "avg": 50.0, "low": 40.0, "trend": 45.0}]}
    prod_data = {"version": 1, "products": [{"idProduct": 999, "name": "Secret Lair Test", "idCategory": 1}]}

    def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "priceGuide" in url:
            resp.iter_content.return_value = [json.dumps(pg_data).encode("utf-8")]
        else:
            resp.iter_content.return_value = [json.dumps(prod_data).encode("utf-8")]
        return resp

    mock_session = MagicMock()
    mock_session.get.side_effect = mock_get

    res = download_game_catalog(game="magic", base_dir=tmp_path, session=mock_session)

    assert res["game"] == "magic"
    assert res["csv_rows"] == 1
    assert Path(res["csv_path"]).exists()
    assert Path(res["price_guide_path"]).exists()
    assert Path(res["products_path"]).exists()
