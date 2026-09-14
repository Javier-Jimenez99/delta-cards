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

## 💻 CLI Usage & Architecture

Delta Cards is built as a decoupled pipeline with **three independent services** that communicate through a shared database, alongside a unified all-in-one command:

### 1. Scraper Service (`delta-cards scrape`)
Fetches listings from Wallapop and stores them in the database (`WallapopOffers` table with `assigned_state = 'pending'`) and optional JSON/CSV files. Runs fast with zero OCR/AI overhead:
```bash
# Run a single scrape cycle
uv run delta-cards scrape -p single_card

# Run scraper continuously in loop mode (e.g. polls every 5 minutes)
uv run delta-cards scrape --loop --interval 300
```
*(Also available as standalone binary: `uv run delta-cards-scrape`)*

### 2. Card Assigner & Pricing Service (`delta-cards assign`)
Persistent consumer worker that stays **always on waiting for offers**. It polls the database for unassigned listings, runs the vision & AI pipeline (YOLOv8 card detection, DINOv3 + USearch similarity search, RapidOCR, and fuzzy matching), retrieves live Cardmarket/TCGdex market prices (`--scrape-prices`), and flags potential deals (`is_deal = True`):
```bash
# Starts the continuous worker (polls DB every 30s by default):
uv run delta-cards assign --scrape-prices

# Customize polling interval (e.g. check every 15 seconds):
uv run delta-cards assign --interval 15 --scrape-prices

# Or process current unassigned batch once and exit:
uv run delta-cards assign --once --scrape-prices
```
*(Also available as standalone binary: `uv run delta-cards-assign`)*

### 3. Alert Dispatcher Service (`delta-cards alerts`)
Decoupled worker that scans the database for pending deal opportunities (`alert_status = 'pending'`). It passes candidates through an LLM verifier to prevent false positives (e.g., card lots, proxies, damaged cards) and dispatches rich notifications to your configured Discord webhook:
```bash
# Starts the continuous alert worker:
uv run delta-cards alerts

# Run once and exit:
uv run delta-cards alerts --once

# Send a test alert to verify Discord webhook connectivity:
uv run delta-cards alerts --test-discord
```
*(Also available as standalone binary: `uv run delta-cards-alert`)*

### 4. Running in the Background (Daemons via `nohup`)
To run all three services in the background without keeping terminals open:
```bash
nohup uv run delta-cards scrape --loop > scraper.log 2>&1 &
nohup uv run delta-cards assign --scrape-prices > assign.log 2>&1 &
nohup uv run delta-cards alerts > alerts.log 2>&1 &
```
Monitor them anytime with `tail -f <service>.log` or inspect running processes with `ps aux | grep delta-cards`.

### 5. Unified All-in-One Command (`delta-cards`)
Runs the complete sequential flow end-to-end (scrape -> assign -> price -> alert):
```bash
uv run delta-cards --scrape-prices
uv run delta-cards --limit 10 --scrape-prices
```

---

## ⚙️ Running as System Services (`systemd`)

For automated, persistent 24/7 background execution with auto-restart on crashes and system reboots, pre-configured `systemd` unit files are available in the [systemd/](systemd/) directory:

| Unit File | Type | Description |
|---|---|---|
| `delta-cards-scrape.service` | `oneshot` | Executes a single Wallapop scrape cycle |
| `delta-cards-scrape.timer` | `timer` | Triggers the scrape service hourly (at `:00`) |
| `delta-cards-assign.service` | `simple` | Continuous worker for AI recognition and pricing |
| `delta-cards-alerts.service` | `simple` | Continuous worker for deal verification & Discord alerts |

### Installation Steps

1. **Adjust user and paths** in `systemd/*.service` to match your deployment environment:
   ```ini
   User=your_username
   Group=your_username
   WorkingDirectory=/path/to/delta-cards
   Environment="PATH=/home/your_username/.local/bin:/usr/local/bin:/usr/bin:/bin"
   ExecStart=/home/your_username/.local/bin/uv run delta-cards ...
   ```

2. **Copy the unit files** to `/etc/systemd/system/`:
   ```bash
   sudo cp systemd/delta-cards-*.service /etc/systemd/system/
   sudo cp systemd/delta-cards-*.timer /etc/systemd/system/
   sudo systemctl daemon-reload
   ```

3. **Enable and start the services**:
   ```bash
   # Enable and start the hourly scraper timer
   sudo systemctl enable --now delta-cards-scrape.timer

   # Enable and start the recognition worker
   sudo systemctl enable --now delta-cards-assign.service

   # Enable and start the Discord alert worker
   sudo systemctl enable --now delta-cards-alerts.service
   ```

### Managing Systemd Services

- **Check status:**
  ```bash
  sudo systemctl status delta-cards-scrape.timer
  sudo systemctl status delta-cards-assign.service
  sudo systemctl status delta-cards-alerts.service
  ```

- **View live logs:**
  ```bash
  journalctl -u delta-cards-assign.service -f
  journalctl -u delta-cards-alerts.service -f
  journalctl -u delta-cards-scrape.service -n 50
  ```

- **Stop services:**
  ```bash
  sudo systemctl stop delta-cards-scrape.timer
  sudo systemctl stop delta-cards-assign.service
  sudo systemctl stop delta-cards-alerts.service
  ```

- **Restart a service:**
  ```bash
  sudo systemctl restart delta-cards-assign.service
  ```

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

