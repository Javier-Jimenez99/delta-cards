---
name: sealed-alert-auditor
description: >-
  Audit, diagnose, and fix Pokémon sealed product alerts (booster boxes, ETBs, booster packs, collection boxes).
  Inspect sent alerts from the database, identify matching errors and false positives, update extractor
  terms in pokemon_keywords.json, verify with tests, and reload services.
---

# Pokémon Sealed Alert Auditor & Extractor Tuner

This skill guides the auditing and tuning of the Pokémon sealed product matching pipeline (booster boxes, booster packs, ETBs, and collection boxes). Use this workflow whenever investigating sent alerts, false positive deals, or updating keyword extraction rules.

---

## Architecture Overview

1. **Scraping**: Wallapop and Vinted scrapers fetch listings using `extra_config/pokemon_sealed_search.json`.
2. **Entity Extraction**: `NonSinglesMatcher` in [`src/delta_cards/nonsingles/assigner.py`](file:///home/ubuntu/delta-cards/src/delta_cards/nonsingles/assigner.py) runs FlashText (`KeywordProcessor(case_sensitive=False)`) over listing titles using canonical keywords from [`data/catalogs/pokemon_keywords.json`](file:///home/ubuntu/delta-cards/data/catalogs/pokemon_keywords.json).
3. **Cardmarket Catalog & BM25**: Extracted entities are mapped to Cardmarket format categories (`FORMAT_TO_CATEGORIES`) and scored with BM25 against [`data/catalogs/pokemon_nonsingles.csv`](file:///home/ubuntu/delta-cards/data/catalogs/pokemon_nonsingles.csv).
4. **Pricing & Alerts**: `delta-cards assign` checks Cardmarket trend/avg price against offer price. If a large discount is found, `alert_status` is set to `pending`, and `delta-cards alerts` dispatches Discord alerts (`alert_status = 'sent'`).

---

## 1. Database Inspection Runbook

> [!IMPORTANT]
> The database (`data/delta_cards.db`) is continuously written to by background workers (`delta-cards-assign`, `delta-cards-alerts`).
> Always open SQLite in read-only URI mode (`file:data/delta_cards.db?mode=ro`) to prevent database lock contention.
> Avoid long multiline inline bash strings in terminal; run inspection via Python script.

### Inspect Sent Sealed Alerts

Use the pre-built script or run:

```bash
.venv/bin/python scripts/inspect_db_alerts.py
```

Or run a quick query with Python:

```python
import sqlite3, csv
from pathlib import Path

# 1. Load catalog names
catalog = {}
with open("data/catalogs/pokemon_nonsingles.csv", "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        pid = str(row.get("idProduct") or "").strip()
        if pid:
            catalog[pid] = row.get("name", "")

# 2. Query sent alerts for non-singles
con = sqlite3.connect("file:data/delta_cards.db?mode=ro", uri=True)
con.row_factory = sqlite3.Row
cur = con.cursor()
cur.execute("""
    SELECT p.id, p.platform, p.title, p.price, p.product_type,
           p.cardmarket_product_id, pr.avg_price, pr.trend_price, p.link, p.updated_at
    FROM PlatformOffers p
    LEFT JOIN CardmarketPrice pr ON p.cardmarket_product_id = pr.cardmarket_product_id
    WHERE p.alert_status = 'sent' AND p.product_type != 'single_card'
    ORDER BY p.updated_at DESC
    LIMIT 50
""")

for r in cur.fetchall():
    cm_name = catalog.get(str(r["cardmarket_product_id"] or ""), "Unknown")
    print(f"[{r['platform']}] {r['title']} ({r['price']}€)")
    print(f"  -> Matched: {cm_name} (ID: {r['cardmarket_product_id']}, Trend: {r['trend_price']}€)")
```

---

## 2. Taxonomy of Common Errors & Root Causes

When reviewing alerts, classify each false positive into one of these known patterns:

### A. 2–3 Letter Set Code Collisions
* **Symptom**: Common words in listings (e.g., French `"Lot de 9 Carte"`, or `"AR"`) match an expansion.
* **Cause**: `KeywordProcessor` is case-insensitive. Set codes like `"LOT"` (*Lost Thunder*) match the word `"lot"`.
* **Fix**: Remove the bare 3-letter code from `pokemon_keywords.json` if it collides with dictionary words in Spanish, French, English, or Italian.

### B. Overly Broad Format Keywords
* **Symptom**: Collection boxes or promo boxes (e.g. *Caja Colección Sylveon ex Jumbo*) match Japanese *Booster Boxes* (Displays).
* **Cause**: Words like `"Caja"`, `"Caja Pokémon"`, or `"Caja de"` mapped to `"Booster Box"`. In `assigner.py`, detecting `"Booster Box"` locks the allowed category to `Pokémon Display`, preventing the matcher from seeing the correct `Pokémon Box Set`.
* **Fix**: Remove bare `"Caja"`, `"Caja de"`, `"Caja Pokémon"` from `"Booster Box"`. Only keep explicit display phrases: `"Booster Box"`, `"Display"`, `"Caja de sobres"`, `"Caja 36 sobres"`, etc.

### C. Missing Regional & Multilingual Synonyms
* **Symptom**: Modern Italian/Dutch/French listings (e.g., `"ETB 30 Anniversario"`, `"30 jaar"`, `"Set Allenatore Fuoriclasse"`) fall back to vintage or expensive sets like *Celebrations ETB* (25th Anniversary, ~333€).
* **Cause**: Italian uses double *n* (`"Anniversario"`), ordinal symbols (`"30°"`, `"30^"`, `"30 esimo"`), or local format names (`"Set Allenatore Fuoriclasse"`). When the set entity is missed, the matcher only has the format entity and BM25 defaults to the highest-scoring Cardmarket product.
* **Fix**: Add regional spelling variations and official international terms to the set and format entities in `pokemon_keywords.json`.

### D. Vintage False Translations
* **Symptom**: Modern fan names or themed lots (e.g., Charcadet/Fire products with `"Fuegos Fantasmales"`) match a vintage 2014 set like *Phantom Forces* (~3,300€ ETB).
* **Cause**: Erroneous Spanish synonym added to a vintage set (official was *"Fuerzas Fantasmales"*).
* **Fix**: Remove the unofficial/colliding translation from the vintage set.

### E. Unregistered Modern Expansions
* **Symptom**: Unreleased or recent sets in local languages (*Scintille Folgoranti*, *Evoluzioni Prismatiche*, *Rivales Predestinado*, *Ascesa Eroica*, *White Flare*, *Black Bolt*, *Mega Evolution*) fall back to *151 Booster Bundle* or *Celebrations ETB*.
* **Fix**: Add the canonical expansion and all local language names to `pokemon_keywords.json`.

### F. Non-Sealed Products (Cards, Empty Boxes, Lots, Merch)
* **Symptom**: Promo cards (`"Carta #Pokemon ... Promo ETB"`), empty boxes (`"scatola con accessori"`), plush toys, or display frames trigger sealed alerts.
* **Fix**: Ensure these keywords exist in `"Excluded_NonSealed"` in `pokemon_keywords.json`. `assigner.py` rejects listings containing any `EXCLUSION_TAGS`.

---

## 3. Safe Keyword Updating Workflow

Follow this sequence whenever editing extractor keywords:

### Step 1: Backup
```bash
cp data/catalogs/pokemon_keywords.json data/catalogs/pokemon_keywords.json.bak
```

### Step 2: Edit `data/catalogs/pokemon_keywords.json`
* Modify terms using Python or JSON edits.
* Keep entries sorted, clean, and UTF-8 encoded (`ensure_ascii=False`).
* Maintain `Excluded_NonSealed` with negative keywords:
  ```json
  "Excluded_NonSealed": [
    "carta", "carte", "cards", "lote de cartas", "lot de cartes",
    "promo", "promos", "scatola con accessori", "scatola vuota",
    "caja vacia", "caja vacía", "taco cartas", "plush toy", "peluche",
    "marco expositor", "teca acrilico", "sobres sueltos", "solo caja",
    "solo fundas", "etb accesorios"
  ]
  ```

### Step 3: Validate with Test Script
Verify candidate titles directly against `NonSinglesMatcher`:

```bash
.venv/bin/python -c "
from delta_cards.nonsingles.assigner import NonSinglesMatcher
matcher = NonSinglesMatcher.get_instance()

test_titles = [
    'ETB 30 Anniversario Pokèmon ITA Sealed',
    'Caja Colección Sylveon ex Jumbo - Celebración 30 Aniversario',
    'Lot de 9 Carte Pokémon Ex',
    'Carta #Pokemon #Zorua di N SVP 189 - Promo ETB Avventure Insieme ITA Sealed',
    'Pokémon Scintille Folgoranti Booster Bundle 6 Buste ITA Box Sealed'
]

for t in test_titles:
    ents, matches = matcher.search_product(t, top_k=1)
    match_str = f\"{matches[0]['name']} ({matches[0]['trendPrice']}€)\" if matches else 'NO MATCH (Filtered)'
    print(f'{t[:50]}... -> {match_str}')
"
```

### Step 4: Run Automated Tests
```bash
.venv/bin/pytest tests/test_nonsingles_assigner.py
```

### Step 5: Reload Background Services
Restart the workers so they load the new JSON terms and code changes:

```bash
sudo -n systemctl restart delta-cards-assign delta-cards-alerts
```

Check health:
```bash
sudo -n systemctl status delta-cards-assign delta-cards-alerts --no-pager
```

---

## 4. Key Heuristics to Remember

1. **Only Sealed Products**: Delta-cards nonsingles assigner focuses strictly on sealed Pokémon material (booster boxes, displays, ETBs, booster packs, collection boxes, tins). Never assign single cards, bulk lots, or empty boxes.
2. **No Blind Format Guessing**: If a listing title only has a format (e.g., `"Pokemon Booster Bundle Sealed"`) without an identified set or character, `search_product` must return `[]` rather than guessing a default expensive set.
3. **Category Lock Awareness**: `FORMAT_TO_CATEGORIES` restricts BM25 to specific Cardmarket categories. Ensure format entities match the intended Cardmarket product category.
