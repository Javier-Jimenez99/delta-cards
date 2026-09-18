"""Inspect sent and pending alerts for sealed products (Pokémon and Magic: The Gathering).

Queries PlatformOffers from SQLite database (read-only mode) and joins
with Cardmarket products and prices across both Pokémon and Magic catalogs.
"""

import argparse
import csv
from pathlib import Path
import sqlite3
from typing import Dict, Optional


def load_catalog(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """Loads Cardmarket non-singles CSV catalog mapping product ID to info."""
    catalog = {}
    if not csv_path.exists():
        return catalog
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pid = str(row.get("idProduct") or row.get("id") or "").strip()
                if pid:
                    catalog[pid] = {
                        "name": row.get("name", ""),
                        "expansion": row.get("expansion", "") or row.get("idExpansion", ""),
                        "category": row.get("categoryName", "") or row.get("category_name", ""),
                        "trend": row.get("trend", "") or row.get("trendPrice", ""),
                        "avg": row.get("avg", "") or row.get("avgPrice", ""),
                    }
            except (ValueError, KeyError):
                pass
    return catalog


def inspect(game_filter: Optional[str] = None, status_filter: str = "sent", limit: int = 100):
    con = sqlite3.connect("file:data/delta_cards.db?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    print("=== Database Summary ===")
    cur.execute("SELECT alert_status, count(*) FROM PlatformOffers GROUP BY alert_status;")
    status_counts = dict(cur.fetchall())
    print("Alert statuses in PlatformOffers:", status_counts)

    cur.execute("SELECT product_type, count(*) FROM PlatformOffers GROUP BY product_type;")
    pt_counts = dict(cur.fetchall())
    print("Product types in PlatformOffers:", pt_counts)

    cur.execute("SELECT COALESCE(game, 'unknown') as g, count(*) FROM PlatformOffers GROUP BY g;")
    game_counts = dict(cur.fetchall())
    print("Games in PlatformOffers:", game_counts)

    # 1. Load catalogs for both Pokémon and Magic
    project_root = Path(__file__).resolve().parent.parent

    pk_cat_candidates = [
        project_root / "data/catalogs/pokemon/nonsingles.csv",
        project_root / "data/catalogs/pokemon/pokemon_nonsingles.csv",
        project_root / "data/catalogs/pokemon_nonsingles.csv",
    ]
    pk_cat_path = next((p for p in pk_cat_candidates if p.exists()), pk_cat_candidates[0])
    pokemon_catalog = load_catalog(pk_cat_path)

    mtg_cat_candidates = [
        project_root / "data/catalogs/magic/nonsingles.csv",
        project_root / "data/catalogs/magic/magic_nonsingles.csv",
        project_root / "data/catalogs/magic_nonsingles.csv",
    ]
    mtg_cat_path = next((p for p in mtg_cat_candidates if p.exists()), mtg_cat_candidates[0])
    magic_catalog = load_catalog(mtg_cat_path)

    print(f"\nLoaded catalogs: Pokémon ({len(pokemon_catalog)} items), Magic ({len(magic_catalog)} items)")

    # 2. Query offers for sealed products
    query = """
        SELECT p.id, p.platform, p.title, p.price, p.product_type, COALESCE(p.game, 'pokemon') as game,
               p.cardmarket_product_id, p.tcgdex_id, p.assigned_state,
               p.alert_status, p.link, p.scraped_at, p.updated_at,
               pr.avg_price, pr.trend_price
        FROM PlatformOffers p
        LEFT JOIN CardmarketPrice pr ON p.cardmarket_product_id = pr.cardmarket_product_id
        WHERE p.product_type != 'single_card'
    """
    params = []

    if status_filter != "all":
        query += " AND p.alert_status = ?"
        params.append(status_filter)

    if game_filter:
        query += " AND LOWER(COALESCE(p.game, 'pokemon')) = ?"
        params.append(game_filter.lower())

    query += " ORDER BY p.updated_at DESC LIMIT ?"
    params.append(limit)

    cur.execute(query, params)
    rows = cur.fetchall()

    target_desc = f"status='{status_filter}'" if status_filter != "all" else "all statuses"
    game_desc = f"game='{game_filter}'" if game_filter else "all games"
    print(f"\n=== Sealed / Non-Single Offers ({target_desc}, {game_desc}, limit {limit}) ===")
    print(f"Found {len(rows)} matching offers\n")

    for r in rows:
        cm_id = str(r["cardmarket_product_id"] or "").strip()
        offer_game = (r["game"] or "pokemon").lower()
        active_cat = magic_catalog if offer_game in ("magic", "mtg") else pokemon_catalog

        cm_info = active_cat.get(cm_id, {})
        cm_title = cm_info.get("name", "Unknown (ID not in catalog)" if cm_id else "Not Assigned")
        cm_cat = cm_info.get("category", "")
        cm_exp = cm_info.get("expansion", "")

        trend_p = r["trend_price"] or cm_info.get("trend")
        avg_p = r["avg_price"] or cm_info.get("avg")

        print(f"[{r['platform'].upper()}] [{offer_game.upper()}] ID: {r['id']} | Type: {r['product_type']} | Price: {r['price']}€ | Alert: {r['alert_status']}")
        print(f"  Title: {r['title']}")
        print(f"  Matched Cardmarket: ID={cm_id} | Name: '{cm_title}' | Cat: '{cm_cat}'")
        if trend_p or avg_p:
            print(f"  Market Prices: Trend={trend_p}€ | Avg={avg_p}€")
        print(f"  Link: {r['link']}")
        print(f"  Updated: {r['updated_at']}")
        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="Audit and inspect sealed product alerts and assignments in SQLite database.")
    parser.add_argument("--game", choices=["pokemon", "magic"], default=None, help="Filter by game (pokemon or magic)")
    parser.add_argument("--status", choices=["sent", "pending", "none", "all"], default="sent", help="Alert status filter (default: sent)")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of offers to display (default: 100)")
    args = parser.parse_args()

    inspect(game_filter=args.game, status_filter=args.status, limit=args.limit)


if __name__ == "__main__":
    main()
