import sqlite3
import json

def inspect():
    con = sqlite3.connect("file:data/delta_cards.db?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    print("=== Table summary ===")
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]
    print("Tables:", tables)

    print("\n=== PlatformOffers alert_status distribution ===")
    cur.execute("SELECT alert_status, count(*) FROM PlatformOffers GROUP BY alert_status;")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]}")

    print("\n=== PlatformOffers product_type distribution ===")
    cur.execute("SELECT product_type, count(*) FROM PlatformOffers GROUP BY product_type;")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]}")

    print("\n=== PlatformOffers alert_status by product_type ===")
    cur.execute("SELECT product_type, alert_status, count(*) FROM PlatformOffers GROUP BY product_type, alert_status;")
    for row in cur.fetchall():
        print(f"  {row[0]} | {row[1]}: {row[2]}")

    cur.execute("PRAGMA table_info(PlatformOffers);")
    p_cols = [r[1] for r in cur.fetchall()]
    print("PlatformOffers cols:", p_cols)

    cur.execute("PRAGMA table_info(CardmarketProduct);")
    c_cols = [r[1] for r in cur.fetchall()]
    print("CardmarketProduct cols:", c_cols)

    import csv
    from pathlib import Path
    catalog_path = Path("data/catalogs/pokemon_nonsingles.csv")
    if not catalog_path.exists():
        catalog_path = Path("notebooks/data/non-singles/pokemon_nonsingles.csv")
    
    catalog_map = {}
    if catalog_path.exists():
        with open(catalog_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    pid = str(row.get("idProduct") or row.get("id") or "").strip()
                    if pid:
                        catalog_map[pid] = {
                            "name": row.get("name", ""),
                            "expansion": row.get("expansion", "") or row.get("idExpansion", ""),
                            "category": row.get("categoryName", "") or row.get("category_name", ""),
                        }
                except (ValueError, KeyError):
                    pass
        print(f"Loaded catalog with {len(catalog_map)} products from {catalog_path}")

    print("\n=== Sent alerts for sealed/boxes/packs/non_single ===")
    cur.execute("""
        SELECT p.id, p.platform, p.title, p.price, p.product_type, p.game,
               p.cardmarket_product_id, p.tcgdex_id, p.assigned_state,
               p.alert_status, p.link, p.scraped_at, p.updated_at,
               pr.avg_price, pr.trend_price
        FROM PlatformOffers p
        LEFT JOIN CardmarketPrice pr ON p.cardmarket_product_id = pr.cardmarket_product_id
        WHERE p.alert_status = 'sent' AND p.product_type != 'single_card'
        ORDER BY p.updated_at DESC
        LIMIT 200
    """)
    rows = cur.fetchall()
    print(f"Found {len(rows)} sent sealed/box/pack alerts")
    for r in rows:
        cm_id = str(r['cardmarket_product_id'] or '').strip()
        cm_info = catalog_map.get(cm_id, {})
        cm_title = cm_info.get("name", "Unknown")
        cm_exp = cm_info.get("expansion", "")
        cm_cat = cm_info.get("category", "")
        print(f"[{r['platform']}] ID: {r['id']} | Type: {r['product_type']} | Price: {r['price']}€ | State: {r['assigned_state']}")
        print(f"  Title: {r['title']}")
        print(f"  Matched CM: ID={cm_id} | Name: '{cm_title}' | Exp: '{cm_exp}' | Cat: '{cm_cat}'")
        print(f"  Prices: Avg={r['avg_price']}€ | Trend={r['trend_price']}€")
        print(f"  Link: {r['link']}")
        print(f"  Updated: {r['updated_at']}")
        print("-" * 60)

if __name__ == "__main__":
    inspect()
