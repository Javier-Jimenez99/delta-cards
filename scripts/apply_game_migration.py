"""Migration script to add game column to PlatformOffers and update alembic version."""
import sqlite3
import sys

def run():
    conn = sqlite3.connect("data/delta_cards.db", timeout=60.0)
    cursor = conn.cursor()
    
    # Check if game column already exists in PlatformOffers
    cursor.execute("PRAGMA table_info(PlatformOffers)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if "game" not in columns:
        print("Adding 'game' column to PlatformOffers...")
        cursor.execute("ALTER TABLE PlatformOffers ADD COLUMN game TEXT NOT NULL DEFAULT 'pokemon'")
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_PlatformOffers_game ON PlatformOffers(game)")
        print("Added 'game' column and index.")
    else:
        print("'game' column already exists.")

    # Tag existing magic offers
    cursor.execute("""
        UPDATE PlatformOffers 
        SET game = 'magic' 
        WHERE (
            LOWER(title) LIKE '%magic%' 
            OR LOWER(title) LIKE '%mtg%' 
            OR LOWER(search_profile_id) LIKE '%magic%'
            OR cardmarket_product_id IN (
                SELECT id FROM CardmarketProduct WHERE LOWER(cardmarket_url) LIKE '%/magic/%'
            )
        )
    """)
    print(f"Updated magic records: {cursor.rowcount}")

    # Ensure alembic_version is at 0007_add_game_column
    cursor.execute("SELECT version_num FROM alembic_version")
    row = cursor.fetchone()
    if row:
        cursor.execute("UPDATE alembic_version SET version_num = '0007_add_game_column'")
    else:
        cursor.execute("INSERT INTO alembic_version (version_num) VALUES ('0007_add_game_column')")
    
    conn.commit()
    print("Alembic version set to 0007_add_game_column.")
    
    # Verify
    cursor.execute("SELECT game, count(*) FROM PlatformOffers GROUP BY game")
    print("Offer counts by game:", cursor.fetchall())
    conn.close()

if __name__ == "__main__":
    run()
