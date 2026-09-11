# Delta Cards 🃏

A Python tool to scrape and monitor card offers on Wallapop, built using **WallaPy** and configured entirely through JSON.

Designed initially for **Pokémon cards**, with support for any card game or product type defined in the JSON configuration:
- **Booster Boxes / Displays**
- **Booster Packs**
- **Full Decks / Battle Decks**
- **Elite Trainer Boxes (ETB)**
- **Single Cards**

---

## 🚀 Features

- **Pure JSON Configuration**: All search queries, positive keywords, excluded negative keywords, price ranges, items limit, and geolocation coordinates are defined in JSON. **No hardcoded keywords or condition flags exist in the Python code.**
- **Modular Provider Architecture**: Wallapop integration is modularized in its own `wallapop/` module (`src/delta_cards/wallapop/`), making it simple to add more platforms in the future.
- **WallaPy Integration**: Uses WallaPy's Wallapop client with configurable delays, custom headers, and geolocation coordinates.
- **Clean Single Offers Export**: Exports structured offer metadata to both **JSON** and **CSV** files (`id`, `title`, `price`, `currency`, `product_type`, `location`, `link`, `image_url`, `published_date`, `seller_link`, etc.).
- **Rich CLI Interface**: Color-coded terminal progress and summary table showing prices and counts by product type.

---

## 📦 Installation & Setup

Delta Cards is managed with [`uv`](https://github.com/astral-sh/uv).

### ⚡ One-Command Automated Setup
Run the setup script to install all dependencies, download required AI models (ONNX YOLO detector and local Qwen2.5 GGUF model), configure `.env`, and initialize the database:

```bash
./setup.sh
```

Or run via Python directly:
```bash
uv sync
uv run python scripts/setup.py
```

### System Requirements
- **Python 3.12+** (handled automatically via `uv`)
- **Tesseract OCR**:
  - **Debian / Ubuntu**: `sudo apt-get install -y tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng`
  - **macOS**: `brew install tesseract tesseract-lang`

---

## ⚙️ Configuration (`config/pokemon_cards_search.json`)

You can customize searches by editing `config/pokemon_cards_search.json` or providing your own JSON file. Note that whether a search targets sealed products or single cards is defined entirely through the `keywords` list:

```json
{
  "name": "Pokemon Cards Search Config",
  "wallapop_settings": {
    "latitude": 40.4168,
    "longitude": -3.7038,
    "location_name": "Madrid, Spain",
    "order_by": "newest",
    "delay_between_requests_ms": 1500
  },
  "search_profiles": [
    {
      "id": "booster_box",
      "name": "Booster Box / Displays",
      "product_type": "booster_box",
      "query": [
        "pokemon booster box",
        "pokemon booster box caja",
        "pokemon box booster",
        "caja de sobres pokemon"
      ],
      "keywords": [
        "precintada",
        "precintado",
        "sellada",
        "sellado",
        "sin abrir",
        "sealed"
      ],
      "excluded_keywords": [
        "vacia",
        "vacio",
        "abierta",
        "abierto",
        "sin sobres",
        "codigo",
        "proxy",
        "falsa",
        "videojuego",
        "ps5",
        "switch"
      ],
      "min_price": 70.0,
      "max_price": null,
      "max_items": 50
    },
    {
      "id": "single_card",
      "name": "Single Cards",
      "product_type": "single_card",
      "query": "pokemon carta",
      "keywords": [
        "carta",
        "pokemon",
        "holo",
        "reverse",
        "secret",
        "ultra rare",
        "psa",
        "promo"
      ],
      "excluded_keywords": [
        "codigo",
        "code",
        "proxy",
        "falsa",
        "fake",
        "videojuego"
      ],
      "min_price": 1.0,
      "max_price": null,
      "max_items": 50
    }
  ],
  "output_settings": {
    "output_dir": "data/output",
    "save_json": true,
    "save_csv": true,
    "filename_prefix": "pokemon_offers"
  }
}
```

---

## 💻 CLI Usage

Delta Cards provides two decoupled commands that can run independently and concurrently, alongside a unified all-in-one command:

### 1. Scrape Offers (`delta-cards scrape`)
Fetches listings from Wallapop and stores them in the database and JSON/CSV output files. Fast execution with zero OCR/AI overhead:
```bash
# Run a single scrape cycle
uv run delta-cards scrape -p single_card

# Run scraper in continuous loop mode (e.g. polls every 5 minutes)
uv run delta-cards scrape --loop --interval 300
```
*(Also available as standalone binary: `uv run delta-cards-scrape`)*

### 2. Card Assigner & Deal Hunter (`delta-cards assign`)
Persistent consumer worker that stays **always on waiting for offers**. It continuously polls the database for new unassigned single cards, passes them through the AI recognition pipeline (YOLOv8 + OCR + LLM fallback), matches official TCGdex/Cardmarket IDs, checks Cardmarket prices, and sends Discord deal alerts:
```bash
# Starts the persistent worker (polls DB every 30 seconds by default):
uv run delta-cards assign --scrape-prices

# Customize polling interval (e.g. check every 15 seconds):
uv run delta-cards assign --interval 15 --scrape-prices

# Or run just once for the current batch and exit immediately:
uv run delta-cards assign --once --scrape-prices
```
*(Also available as standalone binary: `uv run delta-cards-assign`)*

### 3. Running Scraper and Assigner in Parallel
Because the Assigner is always on waiting for incoming offers, the Scraper can be triggered at any time (e.g. on demand or via a periodic cron/timer):
- **Worker (Always On)**: `uv run delta-cards assign --scrape-prices`
- **Scraper (Triggered on demand or via cron)**: `uv run delta-cards scrape -p single_card`

### 4. Unified All-in-One Pipeline (`delta-cards`)
Runs the complete sequential flow (scrape -> assign -> scrape prices -> alert deals):
```bash
uv run delta-cards
uv run delta-cards --limit 10 --scrape-prices
```

---

## 🐍 Python Library Usage

```python
from delta_cards import CardOffersScraper, load_config

# Load JSON configuration
config = load_config("config/pokemon_cards_search.json")

# Initialize and run scraper (from Wallapop package)
scraper = CardOffersScraper(config)
import asyncio
offers = asyncio.run(scraper.run())

# Save to JSON and CSV
scraper.save_results(offers)

for offer in offers[:5]:
    print(f"{offer.title} - {offer.price} {offer.currency} ({offer.link})")
```

---

## 🧪 Testing

Run automated tests:
```bash
uv run pytest
```

---

## 📖 Architecture & Recognition Pipeline

For an in-depth explanation of how single card visual detection, OCR, hybrid LLM extraction, TCGdex matching, and bounded multiprocessing work, see [docs/pipeline.md](docs/pipeline.md).

