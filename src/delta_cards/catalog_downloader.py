"""Cardmarket Catalog Downloader and Synchronizer for Pokémon and Magic: The Gathering.

Downloads official bulk catalog exports from Cardmarket's public S3 bucket:
  - Price Guides: https://downloads.s3.cardmarket.com/productCatalog/priceGuide/price_guide_{game_id}.json
  - Non-Singles:  https://downloads.s3.cardmarket.com/productCatalog/productList/products_nonsingles_{game_id}.json

Merges the non-singles products and latest price guides into the unified nonsingles.csv
catalogs used by the sealed product matcher (NonSinglesMatcher) and pricing services.
"""

import csv
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests

logger = logging.getLogger(__name__)

# Game configurations
# Cardmarket game IDs: 1 = Magic: The Gathering, 6 = Pokémon
CARDMARKET_GAMES: Dict[str, Dict[str, Any]] = {
    "magic": {
        "game_id": 1,
        "folder": "magic",
        "keywords_file": "magic_keywords.json",
        "csv_file": "magic_nonsingles.csv",
        "price_guide_file": "price_guide_magic.json",
        "products_file": "products_nonsingles_magic.json",
        "foil_suffix": "foil",
    },
    "pokemon": {
        "game_id": 6,
        "folder": "pokemon",
        "keywords_file": "pokemon_keywords.json",
        "csv_file": "pokemon_nonsingles.csv",
        "price_guide_file": "price_guide_pokemon.json",
        "products_file": "products_nonsingles_pokemon.json",
        "foil_suffix": "holo",
    },
}

S3_BASE_PRICE_GUIDE = "https://downloads.s3.cardmarket.com/productCatalog/priceGuide"
S3_BASE_PRODUCT_LIST = "https://downloads.s3.cardmarket.com/productCatalog/productList"

DEFAULT_USER_AGENT = "DeltaCards/1.0 (+https://github.com/Javier-Jimenez99/delta-cards)"


def get_default_catalogs_dir() -> Path:
    """Returns the default base directory for game catalogs."""
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / "data" / "catalogs"


def _download_file(
    url: str,
    target_path: Path,
    expected_top_key: str,
    session: Optional[requests.Session] = None,
    timeout: int = 120,
) -> int:
    """Streams a file from a URL to a temporary file, validates JSON, and atomically replaces target.

    Args:
        url: Remote file URL.
        target_path: Final destination path.
        expected_top_key: Key expected in top-level JSON dictionary (e.g. 'priceGuides' or 'products').
        session: Optional requests.Session instance.
        timeout: Download timeout in seconds.

    Returns:
        Number of bytes downloaded.

    Raises:
        ValueError: If response is not valid JSON or lacks the expected top key.
        requests.RequestException: If HTTP request fails.
    """
    tmp_path = target_path.with_suffix(".tmp")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    http = session or requests.Session()
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    logger.info("Downloading %s -> %s...", url, target_path.name)
    response = http.get(url, headers=headers, stream=True, timeout=timeout)
    response.raise_for_status()

    bytes_written = 0
    with open(tmp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                bytes_written += len(chunk)

    # Validate that downloaded file is valid JSON
    try:
        with open(tmp_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or expected_top_key not in data:
            raise ValueError(f"JSON from {url} is missing required top-level key '{expected_top_key}'")
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise ValueError(f"Downloaded file from {url} is not valid JSON: {exc}") from exc

    # Atomic replace
    os.replace(tmp_path, target_path)
    logger.info("✓ Saved %s (%.2f MB)", target_path.name, bytes_written / (1024 * 1024))
    return bytes_written


def merge_products_and_price_guide(
    products_path: Path,
    price_guide_path: Path,
    output_csv_path: Path,
    game: str,
) -> int:
    """Merges products JSON and price guide JSON into a unified non-singles CSV.

    Args:
        products_path: Path to products_nonsingles_{game}.json.
        price_guide_path: Path to price_guide_{game}.json.
        output_csv_path: Path to output {game}_nonsingles.csv.
        game: Game identifier ('magic' or 'pokemon').

    Returns:
        Total number of CSV rows written.
    """
    logger.info("Merging %s and %s into %s...", products_path.name, price_guide_path.name, output_csv_path.name)

    cfg = CARDMARKET_GAMES.get(game.lower())
    foil_sfx = cfg["foil_suffix"] if cfg else "foil"

    with open(products_path, "r", encoding="utf-8") as f:
        prod_data = json.load(f)
    products_list = prod_data.get("products", [])

    with open(price_guide_path, "r", encoding="utf-8") as f:
        price_data = json.load(f)
    price_guides = price_data.get("priceGuides", [])

    # Index price guides by idProduct
    prices_by_id: Dict[int, Dict[str, Any]] = {}
    for entry in price_guides:
        if isinstance(entry, dict) and entry.get("idProduct") is not None:
            try:
                pid = int(entry["idProduct"])
                prices_by_id[pid] = entry
            except (ValueError, TypeError):
                continue

    # Define standard columns
    columns = [
        "idProduct",
        "name",
        "idCategory",
        "categoryName",
        "idExpansion",
        "idMetacard",
        "dateAdded",
        "avg",
        "low",
        "trend",
        "avg1",
        "avg7",
        "avg30",
        f"avg-{foil_sfx}",
        f"low-{foil_sfx}",
        f"trend-{foil_sfx}",
        f"avg1-{foil_sfx}",
        f"avg7-{foil_sfx}",
        f"avg30-{foil_sfx}",
    ]

    tmp_csv = output_csv_path.with_suffix(".tmp")
    rows_written = 0

    with open(tmp_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for prod in products_list:
            if not isinstance(prod, dict) or prod.get("idProduct") is None:
                continue

            try:
                pid = int(prod["idProduct"])
            except (ValueError, TypeError):
                continue

            price_entry = prices_by_id.get(pid, {})

            row = {
                "idProduct": prod.get("idProduct", ""),
                "name": prod.get("name", ""),
                "idCategory": prod.get("idCategory", ""),
                "categoryName": prod.get("categoryName", ""),
                "idExpansion": prod.get("idExpansion", ""),
                "idMetacard": prod.get("idMetacard", "0"),
                "dateAdded": prod.get("dateAdded", ""),
                "avg": price_entry.get("avg", ""),
                "low": price_entry.get("low", ""),
                "trend": price_entry.get("trend", ""),
                "avg1": price_entry.get("avg1", ""),
                "avg7": price_entry.get("avg7", ""),
                "avg30": price_entry.get("avg30", ""),
                f"avg-{foil_sfx}": price_entry.get(f"avg-{foil_sfx}", ""),
                f"low-{foil_sfx}": price_entry.get(f"low-{foil_sfx}", ""),
                f"trend-{foil_sfx}": price_entry.get(f"trend-{foil_sfx}", ""),
                f"avg1-{foil_sfx}": price_entry.get(f"avg1-{foil_sfx}", ""),
                f"avg7-{foil_sfx}": price_entry.get(f"avg7-{foil_sfx}", ""),
                f"avg30-{foil_sfx}": price_entry.get(f"avg30-{foil_sfx}", ""),
            }
            writer.writerow(row)
            rows_written += 1

    os.replace(tmp_csv, output_csv_path)
    logger.info("✓ Wrote %d products to %s", rows_written, output_csv_path.name)
    return rows_written


def ensure_symlinks(game_dir: Path, game: str) -> None:
    """Ensures that standard relative symlinks exist in the game's catalog directory."""
    cfg = CARDMARKET_GAMES.get(game.lower())
    if not cfg:
        return

    symlink_map = {
        "price_guide.json": cfg["price_guide_file"],
        "products_nonsingles.json": cfg["products_file"],
        "nonsingles.csv": cfg["csv_file"],
        "keywords.json": cfg["keywords_file"],
    }

    for link_name, target_name in symlink_map.items():
        link_path = game_dir / link_name
        target_path = game_dir / target_name

        if not target_path.exists():
            continue

        # If link exists and is correct, do nothing
        if link_path.is_symlink():
            try:
                if os.readlink(link_path) == target_name:
                    continue
            except OSError:
                pass
            link_path.unlink()
        elif link_path.exists():
            # If a real regular file exists with identical name, don't clobber it
            continue

        try:
            link_path.symlink_to(target_name)
            logger.debug("Created symlink %s -> %s", link_name, target_name)
        except Exception as exc:
            logger.warning("Could not create symlink %s -> %s: %exc", link_name, target_name, exc)


def download_game_catalog(
    game: str,
    base_dir: Optional[Path] = None,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """Downloads Cardmarket price guide and product catalog for a specific game and updates CSV.

    Args:
        game: 'magic' or 'pokemon'.
        base_dir: Root directory for catalogs (defaults to data/catalogs).
        session: Optional requests.Session instance.

    Returns:
        Dict summarizing downloaded files and row counts.
    """
    g = game.lower().strip()
    if g not in CARDMARKET_GAMES:
        raise ValueError(f"Unsupported game '{game}'. Supported games: {list(CARDMARKET_GAMES.keys())}")

    cfg = CARDMARKET_GAMES[g]
    catalogs_dir = base_dir or get_default_catalogs_dir()
    game_dir = catalogs_dir / cfg["folder"]
    game_dir.mkdir(parents=True, exist_ok=True)

    game_id = cfg["game_id"]
    price_guide_url = f"{S3_BASE_PRICE_GUIDE}/price_guide_{game_id}.json"
    products_url = f"{S3_BASE_PRODUCT_LIST}/products_nonsingles_{game_id}.json"

    price_guide_path = game_dir / cfg["price_guide_file"]
    products_path = game_dir / cfg["products_file"]
    csv_path = game_dir / cfg["csv_file"]

    http = session or requests.Session()

    # 1. Download price guide
    pg_bytes = _download_file(
        url=price_guide_url,
        target_path=price_guide_path,
        expected_top_key="priceGuides",
        session=http,
    )

    # 2. Download non-singles product list
    prod_bytes = _download_file(
        url=products_url,
        target_path=products_path,
        expected_top_key="products",
        session=http,
    )

    # 3. Merge products and prices into unified CSV
    rows_written = merge_products_and_price_guide(
        products_path=products_path,
        price_guide_path=price_guide_path,
        output_csv_path=csv_path,
        game=g,
    )

    # 4. Ensure symlinks exist
    ensure_symlinks(game_dir, g)

    # 5. Invalidate in-memory caches if modules are imported
    try:
        from delta_cards.scryfall.price_guide import CardmarketPriceGuide
        guide_inst = CardmarketPriceGuide.get_instance(g)
        if hasattr(guide_inst, "reload"):
            guide_inst.reload()
    except Exception as exc:
        logger.debug("Could not reload CardmarketPriceGuide cache: %s", exc)

    try:
        from delta_cards.nonsingles.assigner import NonSinglesMatcher
        if g in NonSinglesMatcher._instances:
            del NonSinglesMatcher._instances[g]
    except Exception as exc:
        logger.debug("Could not clear NonSinglesMatcher cache: %s", exc)

    return {
        "game": g,
        "price_guide_path": str(price_guide_path),
        "price_guide_bytes": pg_bytes,
        "products_path": str(products_path),
        "products_bytes": prod_bytes,
        "csv_path": str(csv_path),
        "csv_rows": rows_written,
    }


def download_all_catalogs(
    games: Optional[List[str]] = None,
    base_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Downloads and updates catalogs for all specified games (default: both magic and pokemon)."""
    target_games = games or ["pokemon", "magic"]
    results = []

    with requests.Session() as session:
        for g in target_games:
            logger.info("=== Updating %s Catalog ===", g.upper())
            res = download_game_catalog(g, base_dir=base_dir, session=session)
            results.append(res)

    return results
