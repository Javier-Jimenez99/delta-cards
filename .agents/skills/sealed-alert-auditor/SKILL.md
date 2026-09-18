---
name: sealed-alert-auditor
description: >-
  Audit, diagnose, and fix Pokémon and Magic: The Gathering sealed product alerts (booster boxes, ETBs,
  bundles, booster packs, displays, collection boxes, commander decks). Inspect sent alerts from the
  database, identify matching errors and false positives, update extractor terms in data/catalogs/{game}/keywords.json,
  verify with tests, and reload services.
---

# Sealed Alert Auditor & Extractor Tuner (Pokémon & Magic)

This skill guides the auditing and tuning of the sealed product (non-singles) matching pipeline for both **Pokémon** and **Magic: The Gathering** (booster boxes, displays, booster packs, ETBs, bundles/fat packs, collection boxes, and commander decks). Use this workflow whenever investigating sent alerts, false positive deals, or updating keyword extraction rules.

---

## Architecture Overview

1. **Scraping**: Wallapop and Vinted scrapers fetch listings using `config/pokemon_cards_search.json` and `config/magic_cards_search.json` (search profile `pokemon_sealed` and `magic_sealed` with `product_type: "non_single"`).
2. **Catalog Organization**: Official Cardmarket catalogs are segregated by game under `data/catalogs/`:
   - **Pokémon**: `data/catalogs/pokemon/keywords.json`, `nonsingles.csv`, `price_guide.json`, `products_nonsingles.json`
   - **Magic**: `data/catalogs/magic/keywords.json`, `nonsingles.csv`, `price_guide.json`, `products_nonsingles.json`
3. **Entity Extraction**: `NonSinglesMatcher` in [`src/delta_cards/nonsingles/assigner.py`](file:///home/javij/delta-cards/src/delta_cards/nonsingles/assigner.py) runs FlashText (`KeywordProcessor(case_sensitive=False)`) over listing titles using canonical keywords from the game's `keywords.json`.
4. **Cardmarket Catalog & BM25**: Extracted entities are mapped to Cardmarket format categories (`FORMAT_TO_CATEGORIES` for Pokémon, `MAGIC_FORMAT_TO_CATEGORIES` for Magic) and scored with BM25 against the game's `nonsingles.csv`.
5. **Pricing & Alerts**: `delta-cards assign` checks Cardmarket trend/avg price against offer price. If a large discount is found, `alert_status` is set to `pending`, and `delta-cards alerts` dispatches Discord alerts (`alert_status = 'sent'`) routed to the appropriate webhook (`pokemon_booster_box_webhook`, `pokemon_booster_webhook`, `magic_sealed_webhook`, etc.).

---

## 1. Database Inspection Runbook

> [!IMPORTANT]
> The database (`data/delta_cards.db`) is continuously written to by background workers (`delta-cards-assign`, `delta-cards-alerts`).
> Always open SQLite in read-only URI mode (`file:data/delta_cards.db?mode=ro`) to prevent database lock contention.
> Run inspection using the pre-built CLI tool.

### Inspect Sent & Pending Sealed Alerts

Use `scripts/inspect_db_alerts.py` to inspect alerts filtered by game or status:

```bash
# View all sent sealed alerts (both games)
.venv/bin/python scripts/inspect_db_alerts.py --status sent

# View sent sealed alerts for Magic
.venv/bin/python scripts/inspect_db_alerts.py --game magic --status sent

# View pending or unalerted sealed offers for Pokémon
.venv/bin/python scripts/inspect_db_alerts.py --game pokemon --status pending

# Inspect recent sealed offers regardless of alert status
.venv/bin/python scripts/inspect_db_alerts.py --status all --limit 50
```

### Python Ad-Hoc Database Query Snippet

```python
import sqlite3, csv
from pathlib import Path

# 1. Load catalogs for both games
def load_cat(p):
    cat = {}
    if Path(p).exists():
        with open(p, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                pid = str(r.get("idProduct") or "").strip()
                if pid:
                    cat[pid] = r.get("name", "")
    return cat

pokemon_cat = load_cat("data/catalogs/pokemon/nonsingles.csv")
magic_cat = load_cat("data/catalogs/magic/nonsingles.csv")

# 2. Query sent alerts for non-singles
con = sqlite3.connect("file:data/delta_cards.db?mode=ro", uri=True)
con.row_factory = sqlite3.Row
cur = con.cursor()
cur.execute("""
    SELECT p.id, p.platform, p.title, p.price, p.product_type, COALESCE(p.game, 'pokemon') as game,
           p.cardmarket_product_id, pr.avg_price, pr.trend_price, p.link, p.updated_at
    FROM PlatformOffers p
    LEFT JOIN CardmarketPrice pr ON p.cardmarket_product_id = pr.cardmarket_product_id
    WHERE p.alert_status = 'sent' AND p.product_type != 'single_card'
    ORDER BY p.updated_at DESC
    LIMIT 50
""")

for r in cur.fetchall():
    game = r["game"].lower()
    cat = magic_cat if game in ("magic", "mtg") else pokemon_cat
    cm_name = cat.get(str(r["cardmarket_product_id"] or ""), "Unknown")
    print(f"[{r['platform']}] [{game.upper()}] {r['title']} ({r['price']}€)")
    print(f"  -> Matched: {cm_name} (ID: {r['cardmarket_product_id']}, Trend: {r['trend_price']}€)")
```

---

## 2. Taxonomy of Common Errors & Root Causes

When reviewing alerts, classify each false positive into one of these known patterns:

### A. 2–3 Letter Set Code Collisions
* **Symptom**: Common words in listing titles (e.g. Spanish preposition `"de"` matching German `"DE"`, or `"lot"` matching Pokémon *Lost Thunder*, or `"one"` matching MTG *Phyrexia: All Will Be One*) falsely trigger a set or language entity.
* **Cause**: `KeywordProcessor` is case-insensitive.
* **Fix**:
  - Never include 2-letter lowercase language prepositions (`DE`, `EN`, `SP`, `IT`, `FR`) in metadata language lists. Use `"GER"`, `"DEU"`, `"ESP"`, `"ENG"`, `"ITA"`, `"FRA"`.
  - Remove bare codes from `data/catalogs/{game}/keywords.json` if they collide with natural words (`THE`, `ONE`, `LOT`, `BOX`, `SET`).

### B. Overly Broad Format Keywords
* **Symptom**: A single pack or collection box matches a 36-pack booster box display.
* **Cause**: A generic word like `"Caja"` alone was mapped to `"Booster Box"`. In `assigner.py`, detecting `"Booster Box"` restricts allowed categories exclusively to `Pokémon Display` / `Magic Display`.
* **Fix**: Ensure bare `"Caja"` is not assigned to `"Booster Box"`. Keep multi-word phrases for displays: `"Booster Box"`, `"Display"`, `"Caja de sobres"`, `"Caja 36 sobres"`, `"Play Booster Box"`.

### C. Collector vs Regular Booster / Display Confusion (Magic)
* **Symptom**: A standard Play Booster Box (~130€) matches a Collector Booster Box (~350€), triggering a massive false deal alert.
* **Cause**: The word `"Collector"` was missed or not prioritized.
* **Fix**: Ensure `"Collector Booster Box"` and `"Collector Booster"` are explicitly mapped with synonyms like `"Caja Collector"`, `"Sobre Collector"`, `"De Coleccionista"`.

### D. Missing Regional & Multilingual Synonyms
* **Symptom**: Italian, French, or Spanish titles (e.g., `"Horizontes de Modern 3"`, `"Fuerzas Fantasmales"`, `"Scintille Folgoranti"`, `"Evoluzioni Prismatiche"`) fall back to default sets.
* **Fix**: Add regional spelling variations and translated set titles to `data/catalogs/{game}/keywords.json`.

### E. Non-Sealed Products (Cards, Empty Boxes, Lots, Merch)
* **Symptom**: Promo cards (`"Carta #Pokemon ... Promo ETB"`), empty boxes (`"caja vacia"`, `"scatola vuota"`), bulk lots (`"lote 500 cartas"`), or sleeves trigger alerts.
* **Fix**: Ensure these keywords exist in `"Excluded_NonSealed"` in `data/catalogs/{game}/keywords.json`. `assigner.py` rejects listings containing any `EXCLUSION_TAGS`.

---

## 3. Safe Keyword Updating Workflow

Follow this sequence whenever editing extractor keywords:

### Step 1: Backup
```bash
# For Pokémon:
cp data/catalogs/pokemon/pokemon_keywords.json data/catalogs/pokemon/pokemon_keywords.json.bak

# For Magic:
cp data/catalogs/magic/magic_keywords.json data/catalogs/magic/magic_keywords.json.bak
```

### Step 2: Edit `data/catalogs/{game}/keywords.json`
* Modify terms using Python or direct JSON edits.
* Keep entries clean and UTF-8 encoded (`ensure_ascii=False`).
* Maintain `Excluded_NonSealed` with negative keywords:
  ```json
  "Excluded_NonSealed": [
    "caja vacía", "caja vacia", "empty box", "lote", "bulk", "morralla",
    "cartas sueltas", "carta suelta", "single", "singles", "fundas",
    "sleeves", "carpeta", "binder", "sobres vacios", "solo caja"
  ]
  ```

### Step 3: Validate with Test Script
Verify candidate titles directly against `NonSinglesMatcher`:

```bash
# For Pokémon:
.venv/bin/python -c "
from delta_cards.nonsingles import NonSinglesMatcher
m = NonSinglesMatcher.get_instance('pokemon')
for t in ['Caja de sobres Pokémon Fusión Breve', 'Sobre Pokémon Tempestad Plateada']:
    ents, matches = m.search_product(t, top_k=1)
    res = f\"{matches[0]['name']} ({matches[0]['categoryName']})\" if matches else 'NO MATCH'
    print(f'{t} -> {res}')
"

# For Magic:
.venv/bin/python -c "
from delta_cards.nonsingles import NonSinglesMatcher
m = NonSinglesMatcher.get_instance('magic')
for t in ['Caja de sobres Bloomburrow Play booster', 'Modern Horizons 3 Gift Bundle sealed']:
    ents, matches = m.search_product(t, top_k=1)
    res = f\"{matches[0]['name']} ({matches[0]['categoryName']})\" if matches else 'NO MATCH'
    print(f'{t} -> {res}')
"
```

### Step 4: Run Automated Tests
```bash
.venv/bin/pytest tests/test_nonsingles_assigner.py tests/test_magic_nonsingles_assigner.py
```

### Step 5: Reload Background Services
Restart the workers so they load the updated JSON terms and catalog files:

```bash
sudo -n systemctl restart delta-cards-assign delta-cards-alerts
```

Check service status:
```bash
sudo -n systemctl status delta-cards-assign delta-cards-alerts --no-pager
```

---

## 4. Key Heuristics to Remember

1. **Strictly Sealed Products**: delta-cards non-singles assigner focuses strictly on sealed product (booster boxes, displays, ETBs, bundles, booster packs, collection boxes, commander decks). Never assign single cards, bulk lots, or empty boxes.
2. **No Blind Format Guessing**: If a listing title only has a format (e.g. `"Pokemon Booster Bundle Sealed"`, `"Magic Bundle precintado"`) without an identified set or character, `search_product` must return `[]` rather than guessing an arbitrary set.
3. **Category Lock Constraints**: `FORMAT_TO_CATEGORIES` (Pokémon) and `MAGIC_FORMAT_TO_CATEGORIES` (Magic) restrict BM25 candidates to specific Cardmarket categories (e.g. `Booster Box` $\rightarrow$ `Magic Display`).
