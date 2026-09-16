"""Analyzes uniqueness of (Name, Card Number / Set) across TCGdex data."""

import sqlite3
from collections import defaultdict

DB_URI = "file:data/pokemon_cards_meta.sqlite?mode=ro"

def run_analysis():
    con = sqlite3.connect(DB_URI, uri=True)
    cur = con.cursor()

    # We will analyze for:
    # 1. English cards (lang='en')
    # 2. Spanish cards (lang='es')
    # 3. Distinct physical cards (tcgdex_id) with English names
    
    for target_lang in ["en", "es"]:
        print("=" * 80)
        print(f"ANALYSIS FOR LANGUAGE: {target_lang.upper()}")
        print("=" * 80)

        # Total cards
        cur.execute("SELECT COUNT(*) FROM card_metadata WHERE lang = ?", (target_lang,))
        total_records = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT tcgdex_id) FROM card_metadata WHERE lang = ?", (target_lang,))
        unique_tcgdex_ids = cur.fetchone()[0]
        print(f"Total cards in {target_lang}: {total_records} (Unique TCGdex IDs: {unique_tcgdex_ids})")

        # -------------------------------------------------------------
        # CASE 1: Key = (name, card_fraction) e.g. ('Celebi', '4/142')
        # -------------------------------------------------------------
        print("\n--- 1. Testing Key: (Name, Card Fraction) [e.g. 'Celebi', '4/142'] ---")
        cur.execute("""
            SELECT name, card_fraction, COUNT(DISTINCT tcgdex_id) as cnt, GROUP_CONCAT(set_name, ' | ') as sets, GROUP_CONCAT(tcgdex_id, ', ') as ids
            FROM card_metadata
            WHERE lang = ? AND card_fraction IS NOT NULL AND card_fraction != ''
            GROUP BY LOWER(TRIM(name)), card_fraction
            HAVING cnt > 1
            ORDER BY cnt DESC, name ASC
        """, (target_lang,))
        collisions_fraction = cur.fetchall()

        total_unique_keys_fraction = cur.execute("""
            SELECT COUNT(*) FROM (
                SELECT 1 FROM card_metadata
                WHERE lang = ? AND card_fraction IS NOT NULL AND card_fraction != ''
                GROUP BY LOWER(TRIM(name)), card_fraction
            )
        """, (target_lang,)).fetchone()[0]

        colliding_keys_count = len(collisions_fraction)
        cards_in_collision = sum(r[2] for r in collisions_fraction)
        
        print(f"Total unique (name, fraction) keys: {total_unique_keys_fraction}")
        print(f"Number of keys with collisions (>1 card): {colliding_keys_count} ({colliding_keys_count / max(1, total_unique_keys_fraction) * 100:.2f}%)")
        print(f"Cards involved in collision: {cards_in_collision} / {total_records} ({cards_in_collision / max(1, total_records) * 100:.2f}%)")
        print(f"Is (Name, card_fraction) a valid Primary Key? {'YES' if colliding_keys_count == 0 else 'NO'}")
        
        if collisions_fraction:
            print("\nTop 10 collisions for (Name, card_fraction):")
            for name, frac, cnt, sets, ids in collisions_fraction[:10]:
                print(f"  • '{name}' [{frac}] -> {cnt} cards across sets: {sets} (IDs: {ids})")

        # -------------------------------------------------------------
        # CASE 2: Key = (name, local_id) [e.g. 'Celebi', '4'] (without set total)
        # -------------------------------------------------------------
        print("\n--- 2. Testing Key: (Name, Local ID / Number) [e.g. 'Celebi', '4'] ---")
        cur.execute("""
            SELECT name, local_id, COUNT(DISTINCT tcgdex_id) as cnt, GROUP_CONCAT(set_name, ' | ') as sets
            FROM card_metadata
            WHERE lang = ? AND local_id IS NOT NULL AND local_id != ''
            GROUP BY LOWER(TRIM(name)), local_id
            HAVING cnt > 1
            ORDER BY cnt DESC, name ASC
        """, (target_lang,))
        collisions_local_id = cur.fetchall()

        total_unique_keys_local = cur.execute("""
            SELECT COUNT(*) FROM (
                SELECT 1 FROM card_metadata
                WHERE lang = ? AND local_id IS NOT NULL AND local_id != ''
                GROUP BY LOWER(TRIM(name)), local_id
            )
        """, (target_lang,)).fetchone()[0]

        colliding_local_count = len(collisions_local_id)
        cards_in_local_collision = sum(r[2] for r in collisions_local_id)

        print(f"Total unique (name, local_id) keys: {total_unique_keys_local}")
        print(f"Number of keys with collisions (>1 card): {colliding_local_count} ({colliding_local_count / max(1, total_unique_keys_local) * 100:.2f}%)")
        print(f"Cards involved in collision: {cards_in_local_collision} / {total_records} ({cards_in_local_collision / max(1, total_records) * 100:.2f}%)")
        print(f"Is (Name, local_id) a valid Primary Key? {'YES' if colliding_local_count == 0 else 'NO'}")

        if collisions_local_id:
            print("\nTop 10 collisions for (Name, local_id):")
            for name, lid, cnt, sets in collisions_local_id[:10]:
                short_sets = ' | '.join(sets.split(' | ')[:4]) + ('...' if len(sets.split(' | ')) > 4 else '')
                print(f"  • '{name}' [#{lid}] -> {cnt} cards across sets: {short_sets}")

        # -------------------------------------------------------------
        # CASE 3: Key = (name, local_id, set_id) [Within the same set!]
        # -------------------------------------------------------------
        print("\n--- 3. Testing Key: (Name, Local ID, Set ID) [Within the same set] ---")
        cur.execute("""
            SELECT name, local_id, set_id, set_name, COUNT(DISTINCT tcgdex_id) as cnt, GROUP_CONCAT(tcgdex_id, ', ') as ids
            FROM card_metadata
            WHERE lang = ? AND local_id IS NOT NULL AND local_id != ''
            GROUP BY LOWER(TRIM(name)), local_id, set_id
            HAVING cnt > 1
            ORDER BY cnt DESC
        """, (target_lang,))
        collisions_set = cur.fetchall()
        print(f"Collisions within the SAME set with identical (name, local_id): {len(collisions_set)}")
        for name, lid, sid, sname, cnt, ids in collisions_set[:10]:
            print(f"  • '{name}' [#{lid}] in set '{sname}' ({sid}) -> {cnt} cards (IDs: {ids})")

        # -------------------------------------------------------------
        # Specific check for Celebi
        # -------------------------------------------------------------
        print("\n--- Specific Check: Celebi Cards ---")
        cur.execute("""
            SELECT tcgdex_id, name, local_id, total_cards, card_fraction, set_name, set_id
            FROM card_metadata
            WHERE lang = ? AND name LIKE '%Celebi%'
            ORDER BY card_fraction, name
        """, (target_lang,))
        celebi_cards = cur.fetchall()
        print(f"Total Celebi cards found: {len(celebi_cards)}")
        celebi_by_fraction = defaultdict(list)
        celebi_by_num = defaultdict(list)
        for c in celebi_cards:
            tcg_id, cname, lid, tot, frac, sname, sid = c
            celebi_by_fraction[(cname.lower(), frac)].append((cname, lid, tot, frac, sname, sid, tcg_id))
            celebi_by_num[(cname.lower(), lid)].append((cname, lid, tot, frac, sname, sid, tcg_id))

        print("\nCelebi cards sharing (name, fraction):")
        collision_found = False
        for (cn, fr), cards in celebi_by_fraction.items():
            if len(cards) > 1:
                collision_found = True
                print(f"  Collision on '{cn}' [{fr}]:")
                for item in cards:
                    print(f"    - ID: {item[6]} | Set: {item[4]} ({item[5]}) | Name: {item[0]} | Number: {item[3]}")
        if not collision_found:
            print("  None! Every Celebi card in this language has a unique (Name, Fraction).")

        print("\nCelebi cards sharing (name, local_id):")
        for (cn, num), cards in sorted(celebi_by_num.items(), key=lambda x: -len(x[1])):
            if len(cards) > 1:
                print(f"  Number #{num} ('{cn}'): {len(cards)} cards:")
                for item in cards:
                    print(f"    - ID: {item[6]} | Set: {item[4]} ({item[5]}) | Fraction: {item[3]}")

if __name__ == "__main__":
    run_analysis()
