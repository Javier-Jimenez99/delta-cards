# ==============================================================================
# Delta Cards - Task Runner (just)
# Configuration and convenience recipes for managing services, scrapers,
# workers, evaluation tools, tests, and database migrations.
# ==============================================================================

set dotenv-load := true
set shell := ["bash", "-uc"]

# ------------------------------------------------------------------------------
# Default Recipe
# ------------------------------------------------------------------------------

# Show available commands
default:
    @just --list --unsorted

# ------------------------------------------------------------------------------
# Systemd Services Management
# ------------------------------------------------------------------------------

# Restart all background systemd services (assigner, alerts, scraper timer)
[group("services")]
restart-services:
    @echo "🔄 Restarting Delta Cards systemd services..."
    sudo systemctl restart delta-cards-assign.service delta-cards-alerts.service delta-cards-scrape.timer delta-cards-catalogs.timer
    @echo "✅ Services restarted successfully:"
    @sudo systemctl --no-pager status delta-cards-assign.service delta-cards-alerts.service delta-cards-scrape.timer delta-cards-catalogs.timer | grep -E "●|Active:"

alias restart := restart-services

# Show current status of all systemd services and timer
[group("services")]
status-services:
    sudo systemctl --no-pager status delta-cards-assign.service delta-cards-alerts.service delta-cards-scrape.timer delta-cards-catalogs.timer

alias status := status-services

# Stop all background systemd services
[group("services")]
stop-services:
    @echo "⏹️  Stopping Delta Cards systemd services..."
    sudo systemctl stop delta-cards-assign.service delta-cards-alerts.service delta-cards-scrape.timer
    @echo "✅ All Delta Cards services stopped."

alias stop := stop-services

# Start all background systemd services
[group("services")]
start-services:
    @echo "▶️  Starting Delta Cards systemd services..."
    sudo systemctl start delta-cards-assign.service delta-cards-alerts.service delta-cards-scrape.timer
    @echo "✅ All Delta Cards services started."

alias start := start-services

# Restart only the assigner service (AI card recognition and pricing)
[group("services")]
restart-assign:
    sudo systemctl restart delta-cards-assign.service
    @sudo systemctl --no-pager status delta-cards-assign.service | grep -E "●|Active:"

# Restart only the alerts service (Discord notification dispatcher)
[group("services")]
restart-alerts:
    sudo systemctl restart delta-cards-alerts.service
    @sudo systemctl --no-pager status delta-cards-alerts.service | grep -E "●|Active:"

# Restart only the scraper timer
[group("services")]
restart-scrape:
    sudo systemctl restart delta-cards-scrape.timer
    @sudo systemctl --no-pager status delta-cards-scrape.timer | grep -E "●|Active:"

# Tail live logs from all services or a specific service (e.g. `just logs` or `just logs assign`)
[group("services")]
logs service="all":
    #!/usr/bin/env bash
    if [ "{{service}}" = "assign" ]; then
        journalctl -u delta-cards-assign.service -f
    elif [ "{{service}}" = "alerts" ]; then
        journalctl -u delta-cards-alerts.service -f
    elif [ "{{service}}" = "scrape" ]; then
        journalctl -u delta-cards-scrape.service -n 50 -f
    else
        journalctl -u delta-cards-assign.service -u delta-cards-alerts.service -u delta-cards-scrape.service -f
    fi

# Tail live logs for the continuous card assigner worker
[group("services")]
logs-assign:
    journalctl -u delta-cards-assign.service -f

# Tail live logs for the continuous Discord alerts worker
[group("services")]
logs-alerts:
    journalctl -u delta-cards-alerts.service -f

# Tail recent logs from the scraper service
[group("services")]
logs-scrape:
    journalctl -u delta-cards-scrape.service -n 50 -f

# Copy systemd unit files from ./systemd to /etc/systemd/system and reload daemon
[group("services")]
install-services:
    @echo "Installing systemd units..."
    sudo cp systemd/delta-cards-*.service /etc/systemd/system/
    sudo cp systemd/delta-cards-*.timer /etc/systemd/system/
    sudo systemctl daemon-reload
    @echo "✅ Systemd units installed and daemon reloaded."

# Enable all systemd services and timer to launch at system boot
[group("services")]
enable-services:
    sudo systemctl enable delta-cards-assign.service delta-cards-alerts.service delta-cards-scrape.timer delta-cards-catalogs.timer

# Disable all systemd services and timer from auto-starting at boot
[group("services")]
disable-services:
    sudo systemctl disable delta-cards-assign.service delta-cards-alerts.service delta-cards-scrape.timer delta-cards-catalogs.timer

# ------------------------------------------------------------------------------
# Scraping Workflows
# ------------------------------------------------------------------------------

# Scrape marketplace card listings (platform: wallapop, vinted, or all)
[group("scraping")]
scrape platform="all" profile="single_card" limit="50" *args="":
    uv run delta-cards scrape --platform {{platform}} -p {{profile}} -l {{limit}} {{args}}

# Scrape Wallapop single cards only
[group("scraping")]
scrape-wallapop limit="50" *args="":
    uv run delta-cards scrape --platform wallapop -p single_card -l {{limit}} {{args}}

# Scrape Vinted single cards only
[group("scraping")]
scrape-vinted limit="50" *args="":
    uv run delta-cards scrape --platform vinted -p single_card -l {{limit}} {{args}}

# Scrape both platforms (Wallapop + Vinted) for single cards
[group("scraping")]
scrape-all limit="50" *args="":
    uv run delta-cards scrape --platform all -p single_card -l {{limit}} {{args}}

# Run scraper continuously in polling loop mode (e.g. interval=300 for 5 minutes)
[group("scraping")]
scrape-loop interval="300" platform="all" profile="single_card":
    uv run delta-cards scrape --platform {{platform}} -p {{profile}} --loop --interval {{interval}}

# ------------------------------------------------------------------------------
# AI Card Recognition & Assigner Worker
# ------------------------------------------------------------------------------

# Start persistent card recognition and pricing worker (polls DB every 30s)
[group("assigner")]
assign *args="--scrape-prices":
    uv run delta-cards assign {{args}}

# Process currently unassigned cards in database once and exit
[group("assigner")]
assign-once *args="":
    uv run delta-cards assign --once --scrape-prices {{args}}

# ------------------------------------------------------------------------------
# Discord Alerts Worker
# ------------------------------------------------------------------------------

# Start persistent deal verification and Discord alert dispatcher worker
[group("alerts")]
alerts *args="":
    uv run delta-cards alerts {{args}}

# Check pending deals once, verify with LLM, dispatch any alerts, and exit
[group("alerts")]
alerts-once:
    uv run delta-cards alerts --once

# Dispatch test deal alerts to Discord channels (channel: all, pokemon, magic, pokemon_cards, pokemon_boosters, pokemon_boxes, magic_cards, magic_boosters, magic_boxes)
[group("alerts")]
test-alert channel="all" platform="wallapop" *extra="":
    uv run python scripts/send_test_alert.py {{channel}} --platform {{platform}} {{extra}}

# Preview rendered alert text and fields on terminal without sending to Discord
[group("alerts")]
test-alert-dry channel="all" platform="wallapop":
    uv run python scripts/send_test_alert.py {{channel}} --platform {{platform}} --dry-run

# ------------------------------------------------------------------------------
# Full End-to-End Pipeline
# ------------------------------------------------------------------------------

# Run complete sequential flow: scrape -> assign -> price -> alert
[group("pipeline")]
pipeline limit="10" platform="all" profile="single_card" *args="":
    uv run delta-cards --platform {{platform}} -p {{profile}} -l {{limit}} --scrape-prices {{args}}

# ------------------------------------------------------------------------------
# Evaluation & Dataset Tools
# ------------------------------------------------------------------------------

# Launch Streamlit interactive evaluation app (auditing, visualization & benchmarking)
[group("evaluation")]
app:
    uv run streamlit run evaluation_app/app.py

alias eval-ui := app

# Run evaluation benchmark against ground truth dataset
[group("evaluation")]
eval limit="10" *args="":
    uv run python scripts/evaluate.py --limit {{limit}} {{args}}

# Run evaluation benchmark across the entire dataset without limit
[group("evaluation")]
eval-all *args="":
    uv run python scripts/evaluate.py {{args}}

# Rebuild evaluation dataset from scraped database listings
[group("evaluation")]
build-eval-dataset *args="":
    uv run python scripts/build_evaluation_dataset.py {{args}}

# Build offline USearch index and metadata from TCGdex SDK
[group("evaluation")]
build-index:
    uv run delta-cards build-index

# Incrementally check TCGdex catalog and index newly released cards
[group("evaluation")]
sync-catalog:
    uv run delta-cards sync-catalog

# ------------------------------------------------------------------------------
# Cardmarket Catalogs (Magic & Pokémon)
# ------------------------------------------------------------------------------

# Download official Cardmarket price guides & product lists, and rebuild nonsingles CSVs (game: all, magic, pokemon)
[group("catalogs")]
download-catalogs game="all":
    uv run delta-cards download-catalogs --game {{game}}

alias update-catalogs := download-catalogs

# ------------------------------------------------------------------------------
# Database Migrations (Alembic)
# ------------------------------------------------------------------------------

# Apply all pending database migrations
[group("database")]
db-migrate:
    uv run alembic upgrade head

# Generate a new autogenerated Alembic migration revision
[group("database")]
db-revision message:
    uv run alembic revision --autogenerate -m "{{message}}"

# Show current database revision
[group("database")]
db-current:
    uv run alembic current

# Show migration revision history
[group("database")]
db-history:
    uv run alembic history --verbose

# ------------------------------------------------------------------------------
# Setup, Dependencies & Testing
# ------------------------------------------------------------------------------

# Run complete one-command automated setup (system deps, Python env, models, DB)
[group("development")]
setup:
    ./setup.sh

# Synchronize Python virtual environment and dependencies using uv
[group("development")]
sync:
    uv sync

# Run the automated pytest test suite
[group("development")]
test *args="":
    uv run pytest {{args}}

# Run quick OCR sanity check test
[group("development")]
test-quick:
    uv run pytest tests/test_card_ocr.py -k test_card_ocr_language_detection -q
