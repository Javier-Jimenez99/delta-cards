"""Command-line interface for delta-cards."""

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Any, Dict

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from .config_loader import load_config
from .database import save_offers_to_db, get_engine, init_db
from .tcgdex import assign_tcgdex_to_offers, enrich_card_offers_in_db, is_card_offer
from .models import OfferListing
from .wallapop import CardOffersScraper


console = Console()
err_console = Console(stderr=True)
logger = logging.getLogger("delta_cards")


def build_parser() -> argparse.ArgumentParser:
    """Builds the main argument parser supporting subcommands and root-level arguments."""
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/pokemon_cards_search.json",
        help="Path to JSON configuration file (default: config/pokemon_cards_search.json)",
    )
    common_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging",
    )
    common_parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="Override database connection URL (default: DATABASE_URL env var)",
    )

    parser = argparse.ArgumentParser(
        description="Delta Cards: Pokemon card market scraper & Cardmarket deal detector.",
        parents=[common_parser],
    )

    # Root-level backward-compatible arguments (when no subcommand is passed)
    parser.add_argument("-o", "--output", type=str, default=None, help="Override output directory")
    parser.add_argument("-l", "--limit", type=int, default=None, help="Override max items to retrieve")
    parser.add_argument("--no-db", action="store_true", help="Skip saving scraped offers to database")
    parser.add_argument("-p", "--profile", nargs="+", default=None, help="One or more profile IDs to run")
    parser.add_argument("-t", "--type", nargs="+", default=None, help="One or more product types to run")
    parser.add_argument("--scrape-prices", action="store_true", help="Cardmarket prices are retrieved from TCGDex")
    parser.add_argument("--skip-tcgdex", action="store_true", help="Skip card recognition and assignment")
    parser.add_argument("--assign-existing-cards", action="store_true", help="Enrich unassigned cards in database")
    parser.add_argument("--no-discord", action="store_true", help="Skip sending Discord alert notifications")
    parser.add_argument("--discord-webhook", type=str, default=None, help="Override Discord webhook URL")
    parser.add_argument("--test-discord", action="store_true", help="Send a test deal alert to Discord and exit")
    parser.add_argument("--alert-all-deals", action="store_true", help="Send Discord alerts for all matching deals")

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. Scrape subcommand: purely Wallapop offer fetching -> DB/files
    scrape_parser = subparsers.add_parser(
        "scrape",
        parents=[common_parser],
        help="Scrape Wallapop offers and save to DB/files without running card recognition",
    )
    scrape_parser.add_argument("-o", "--output", type=str, default=None, help="Override output directory specified in config")
    scrape_parser.add_argument("-l", "--limit", type=int, default=None, help="Override max items to retrieve per search profile")
    scrape_parser.add_argument("-p", "--profile", nargs="+", default=None, help="One or more profile IDs to run (e.g. -p single_card)")
    scrape_parser.add_argument("-t", "--type", nargs="+", default=None, help="One or more product types to run (e.g. -t single_card booster_box)")
    scrape_parser.add_argument("--no-db", action="store_true", help="Skip saving scraped offers to the database")
    scrape_parser.add_argument("--loop", action="store_true", help="Run scraper continuously in a loop")
    scrape_parser.add_argument("--interval", type=int, default=300, help="Seconds between scrape cycles in loop mode (default: 300)")

    # 2. Assign subcommand: consumer worker reading unassigned offers from DB -> CardRecognitionPipeline
    assign_parser = subparsers.add_parser(
        "assign",
        parents=[common_parser],
        help="Run card recognition & assignment pipeline on unassigned DB offers",
    )
    assign_parser.add_argument("-l", "--limit", type=int, default=None, help="Max unassigned cards to process per cycle")
    assign_parser.add_argument("--scrape-prices", action="store_true", help="Cardmarket prices are retrieved from TCGDex")
    assign_parser.add_argument("--no-discord", action="store_true", help="Skip sending Discord alert notifications")
    assign_parser.add_argument("--discord-webhook", type=str, default=None, help="Override Discord webhook URL")
    assign_parser.add_argument("--test-discord", action="store_true", help="Send a test deal alert to Discord and exit")
    assign_parser.add_argument("--alert-all-deals", action="store_true", help="Send Discord alerts for all matching deals")
    assign_parser.add_argument("--once", action="store_true", help="Process existing unassigned cards once and exit (default: stays on waiting for new offers)")
    assign_parser.add_argument("--loop", action="store_true", default=True, help="Keep worker running continuously (default: True)")
    assign_parser.add_argument("--interval", type=int, default=30, help="Seconds between queue checks when waiting for offers (default: 30)")

    # 3. Alerts subcommand: decoupled worker reading pending deals -> LLM Verifier -> Discord Webhook
    alerts_parser = subparsers.add_parser(
        "alerts",
        parents=[common_parser],
        help="Run decoupled alert worker: checks pending deals, verifies with LLM, and dispatches Discord alerts periodically",
    )
    alerts_parser.add_argument("--interval", type=int, default=None, help="Seconds between alert dispatches (default: 1800s / 30min)")
    alerts_parser.add_argument("--once", action="store_true", help="Dispatch at most one verified alert and exit")
    alerts_parser.add_argument("--no-discord", action="store_true", help="Skip sending Discord alert notifications")
    alerts_parser.add_argument("--discord-webhook", type=str, default=None, help="Override Discord webhook URL")
    alerts_parser.add_argument("--test-discord", action="store_true", help="Send a test deal alert to Discord and exit")

    # 4. All subcommand: explicit full pipeline (scrape + assign + check deals)
    all_parser = subparsers.add_parser(
        "all",
        parents=[common_parser],
        help="Run complete pipeline: scrape Wallapop offers, assign cards, and check deals",
    )
    all_parser.add_argument("-o", "--output", type=str, default=None, help="Override output directory")
    all_parser.add_argument("-l", "--limit", type=int, default=None, help="Override max items to retrieve")
    all_parser.add_argument("-p", "--profile", nargs="+", default=None, help="One or more profile IDs to run")
    all_parser.add_argument("-t", "--type", nargs="+", default=None, help="One or more product types to run")
    all_parser.add_argument("--no-db", action="store_true", help="Skip saving scraped offers to database")
    all_parser.add_argument("--scrape-prices", action="store_true", help="Cardmarket prices are retrieved from TCGDex")
    all_parser.add_argument("--skip-tcgdex", action="store_true", help="Skip card recognition and assignment")
    all_parser.add_argument("--assign-existing-cards", action="store_true", help="Enrich unassigned cards in database")
    all_parser.add_argument("--no-discord", action="store_true", help="Skip sending Discord alert notifications")
    all_parser.add_argument("--discord-webhook", type=str, default=None, help="Override Discord webhook URL")
    all_parser.add_argument("--test-discord", action="store_true", help="Send a test deal alert to Discord and exit")
    all_parser.add_argument("--alert-all-deals", action="store_true", help="Send Discord alerts for all matching deals")

    # 4. build-index subcommand: build offline USearch index from TCGdex
    index_parser = subparsers.add_parser(
        "build-index",
        parents=[common_parser],
        help="Build offline USearch Mmap index and metadata from TCGdex SDK",
    )
    index_parser.add_argument("--sets", nargs="*", default=None, help="Set IDs to index (e.g. sv01 sv02 sv03)")
    index_parser.add_argument("--lang", default="en", help="Language code (e.g. es, en, fr, de) or 'all'")
    index_parser.add_argument("--limit", type=int, default=None, help="Limit total cards to index (for testing)")
    index_parser.add_argument("--index-path", default=None, help="Output .usearch index file (default: data/pokemon_cards_{lang}.usearch)")
    index_parser.add_argument("--cache-dir", default="data/images_cache", help="Image cache directory")
    index_parser.add_argument("--concurrency", type=int, default=8, help="Async download concurrency")

    # 5. export-models subcommand: export and quantize ONNX models
    export_parser = subparsers.add_parser(
        "export-models",
        parents=[common_parser],
        help="Export and quantize DINOv3 INT8 ONNX model",
    )
    export_parser.add_argument("--model", choices=["dinov3", "all"], default="all", help="Model to export")

    # 6. sync-catalog subcommand: check for and index newly released cards daily
    sync_parser = subparsers.add_parser(
        "sync-catalog",
        parents=[common_parser],
        help="Incrementally check TCGdex and index newly released cards",
    )
    sync_parser.add_argument("--daemon", action="store_true", help="Run continuously as a daily daemon")
    sync_parser.add_argument("--interval-hours", type=float, default=24.0, help="Sync interval in hours for daemon mode (default: 24)")
    sync_parser.add_argument("--lang", default="en", help="Language code (default: en)")
    sync_parser.add_argument("--index-path", default="data/pokemon_cards.usearch", help="Output .usearch index file")
    sync_parser.add_argument("--cache-dir", default="data/images_cache", help="Image cache directory")
    sync_parser.add_argument("--concurrency", type=int, default=16, help="Async download concurrency")

    return parser


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    parser = build_parser()
    parsed = parser.parse_args(args)
    if parsed.subcommand is None:
        parsed.subcommand = "all"
    return parsed


def setup_logging(verbose: bool) -> None:
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                show_path=False,
                rich_tracebacks=True,
                markup=True,
            )
        ],
    )
    if not verbose:
        for noisy in ["delta_cards", "wallapy", "httpx", "httpcore", "urllib3", "asyncio"]:
            logging.getLogger(noisy).setLevel(logging.WARNING)


def load_and_filter_config(args: argparse.Namespace):
    """Loads configuration file and applies CLI overrides and profile filters."""
    config_path = Path(args.config)
    if not config_path.is_file():
        project_root = Path(__file__).resolve().parent.parent.parent
        alt_path = project_root / args.config
        if alt_path.is_file():
            config_path = alt_path
        else:
            err_console.print(f"[bold red]Error:[/] Configuration file not found at '{args.config}'")
            sys.exit(1)

    try:
        config = load_config(config_path)
    except Exception as exc:
        err_console.print(f"[bold red]Error loading config:[/] {exc}")
        sys.exit(1)

    # Filter enabled profiles
    config.search_profiles = [p for p in config.search_profiles if getattr(p, "enabled", True)]

    # Filter by profile IDs if requested
    if getattr(args, "profile", None):
        selected_profiles = set(args.profile)
        config.search_profiles = [p for p in config.search_profiles if p.id in selected_profiles]
        if not config.search_profiles:
            err_console.print(f"[bold red]Error:[/] No profiles found matching: {', '.join(args.profile)}")
            sys.exit(1)

    # Filter by product types if requested
    if getattr(args, "type", None):
        selected_types = set(args.type)
        config.search_profiles = [p for p in config.search_profiles if p.product_type in selected_types]
        if not config.search_profiles:
            err_console.print(f"[bold red]Error:[/] No profiles found matching product type(s): {', '.join(args.type)}")
            sys.exit(1)

    # Apply overrides
    if getattr(args, "output", None):
        config.output_settings.output_dir = args.output
    if getattr(args, "limit", None) is not None:
        for profile in config.search_profiles:
            profile.max_items = args.limit

    return config


def handle_test_discord(args: argparse.Namespace) -> bool:
    """Executes Discord test notification if --test-discord is passed. Returns True if handled."""
    if not getattr(args, "test_discord", False):
        return False

    from .alerts import DiscordAlertService
    from .models import OfferListing
    from .settings import get_settings

    settings = get_settings()
    if getattr(args, "discord_webhook", None):
        settings = settings.model_copy(update={"discord_webhook_url": args.discord_webhook})

    webhook_url = settings.discord_webhook_url
    if not webhook_url:
        err_console.print(
            "[bold red]Error:[/] No DISCORD_WEBHOOK_URL found in .env or passed via --discord-webhook.\n"
            "Please configure DISCORD_WEBHOOK_URL in your .env file or pass --discord-webhook <url>."
        )
        sys.exit(1)

    console.print("[bold cyan]Testing Discord Alert Webhook...[/bold cyan]")
    masked = webhook_url[:35] + "..." if len(webhook_url) > 35 else webhook_url
    console.print(f"  • Webhook URL: [cyan]{masked}[/cyan]")
    console.print(f"  • Min saving %: [yellow]{settings.alert_min_saving_percentage}%[/yellow]")
    console.print(f"  • Min saving €: [yellow]{settings.alert_min_saving_absolute} €[/yellow]")
    console.print(f"  • Operator: [yellow]{settings.alert_condition_operator}[/yellow]")

    test_offer = OfferListing(
        id="test_deal_alert",
        title="Charizard Base Set Holo 4/102 (Test Alert)",
        price=45.0,
        product_type="single_card",
        location="Madrid, Spain",
        link="https://es.wallapop.com",
        image_url="https://images.pokemontcg.io/base1/4_hires.png",
        cardmarket_url="https://www.cardmarket.com/en/Pokemon/Products/Singles/Base-Set/Charizard",
        cardmarket_low_price=120.0,
        cardmarket_avg_price=140.0,
        cardmarket_price_spain=140.0,
        cardmarket_price_all=140.0,
    )

    service = DiscordAlertService(settings=settings)
    sent_count = service.process_and_alert_offers([test_offer])
    if sent_count > 0:
        console.print("[bold green]✓ Test Discord alert sent successfully! Check your Discord channel.[/bold green]")
    else:
        console.print("[bold red]✗ Failed to send Discord alert. Check logs or verify that DISCORD_ALERTS_ENABLED=true.[/bold red]")
    return True


def _create_scraper_progress_callback(status, db_save_enabled: bool, is_all: bool = False):
    def progress_callback(event: str, data: dict) -> None:
        if event == "profile_start":
            p_idx = data["index"]
            p_total = data["total"]
            p_name = data["profile_name"]
            n_queries = data["num_queries"]
            q_label = "query variation" if n_queries == 1 else "query variations"
            console.print(
                f"\n[bold cyan]▶ [{p_idx}/{p_total}][/bold cyan] [bold white]{p_name}[/bold white] [dim]({n_queries} {q_label})[/dim]"
            )
            status.update(f"[cyan]Searching {p_name}...[/cyan]")

        elif event == "query_start":
            q_idx = data["index"]
            q_total = data["total"]
            query = data["query"]
            status.update(
                f"[dim]Searching query [{q_idx}/{q_total}]:[/dim] [yellow]'{query}'[/yellow]..."
            )

        elif event == "query_done":
            query = data["query"]
            retrieved = data["retrieved"]
            kept = data["kept"]
            badge = f"[green]+{kept} valid[/green]" if kept > 0 else "[dim]0 valid[/dim]"
            db_note = " [dim](saved to DB)[/dim]" if (db_save_enabled and kept > 0) else ""
            console.print(
                f"   [dim]•[/dim] [white]'{query}'[/white] [dim]→[/dim] {retrieved} retrieved, {badge}{db_note}"
            )

        elif event == "profile_done":
            total_kept = data["total_kept"]
            retrieved_total = data["retrieved_total"]
            console.print(
                f"   [bold green]✓[/bold green] [dim]Profile completed:[/] [bold]{total_kept}[/bold] valid offers kept "
                f"[dim]({retrieved_total} total evaluated)[/dim]"
            )
            status.update("[cyan]Proceeding to next step...[/cyan]" if is_all else "[cyan]Saving scraped data...[/cyan]")
    return progress_callback


def _render_scraped_summary_table(
    offers: list,
    saved_files: dict,
    db_note: Optional[str] = None,
    extra_lines: Optional[List[str]] = None,
) -> None:
    if not offers:
        console.print("\n[yellow]No offers matched the configured search and filter criteria.[/yellow]")
        return

    table = Table(title=f"\nScraped Offers Summary ({len(offers)} unique items)", border_style="cyan")
    table.add_column("Product Type", style="bold green")
    table.add_column("Offers Found", justify="right", style="cyan")
    table.add_column("Min Price", justify="right", style="yellow")
    table.add_column("Max Price", justify="right", style="yellow")
    table.add_column("Avg Price", justify="right", style="yellow")

    by_type: Dict[str, list] = {}
    for offer in offers:
        by_type.setdefault(offer.product_type, []).append(offer)

    for ptype, items in sorted(by_type.items()):
        prices = [i.price for i in items]
        min_p = min(prices) if prices else 0.0
        max_p = max(prices) if prices else 0.0
        avg_p = sum(prices) / len(prices) if prices else 0.0
        table.add_row(
            ptype,
            str(len(items)),
            f"{min_p:.2f} €",
            f"{max_p:.2f} €",
            f"{avg_p:.2f} €",
        )

    console.print(table)

    console.print("\n[bold green]Saved Output Files & Database:[/bold green]")
    for fmt, pth in saved_files.items():
        console.print(f"  • [bold]{fmt.upper()}:[/bold] [cyan]{pth.resolve()}[/cyan]")
    if extra_lines:
        for line in extra_lines:
            console.print(line)
    if db_note:
        console.print(db_note)


def check_and_report_deals(engine, args: argparse.Namespace, live_offer_ids: Optional[List[str]] = None) -> None:
    """Queries DB for deals where Wallapop price < TCGDex reference price and prints summary table."""
    try:
        from .database.models import WallapopOffers, CardmarketPrice
        from sqlmodel import select, Session, and_, or_

        with Session(engine) as s:
            deals_stmt = (
                select(WallapopOffers, CardmarketPrice)
                .join(
                    CardmarketPrice,
                    and_(
                        WallapopOffers.cardmarket_product_id == CardmarketPrice.cardmarket_product_id,
                        WallapopOffers.language == CardmarketPrice.language,
                    ),
                )
                .where(
                    or_(
                        and_(CardmarketPrice.avg_price.is_not(None), WallapopOffers.price < CardmarketPrice.avg_price),
                        and_(CardmarketPrice.avg_price.is_(None), CardmarketPrice.min_price_all.is_not(None), WallapopOffers.price < CardmarketPrice.min_price_all),
                    )
                )
            )
            deals = s.exec(deals_stmt).all()
            if deals:
                live_ids = set(live_offer_ids) if live_offer_ids else set()
                deals_table = Table(
                    title=f"\n🔥 Deals Found (Wallapop < TCGDex Precio Medio): {len(deals)}",
                    border_style="bold green",
                )
                deals_table.add_column("Status", style="dim", justify="center")
                deals_table.add_column("Title", style="cyan")
                deals_table.add_column("Wallapop", justify="right", style="bold green")
                deals_table.add_column("Precio Medio", justify="right", style="yellow")
                deals_table.add_column("Savings", justify="right", style="bold magenta")
                deals_table.add_column("Link", style="cyan", overflow="fold", no_wrap=False)

                for off, pr in deals:
                    ref_p = pr.avg_price if pr.avg_price is not None else pr.min_price_all
                    diff = (ref_p or 0.0) - off.price
                    deal_link = (off.link or "").replace("https://wallapop.com/", "https://es.wallapop.com/")
                    status_tag = "[bold green]LIVE[/bold green]" if off.id in live_ids else "[dim]SAVED[/dim]"
                    deals_table.add_row(
                        status_tag,
                        off.title[:42],
                        f"{off.price:.2f} €",
                        f"{ref_p:.2f} €" if ref_p is not None else "N/A",
                        f"-{diff:.2f} €",
                        f"[link={deal_link}]{deal_link}[/link]",
                    )
                console.print(deals_table)

                console.print("\n[bold green]🔗 Clickable Deal Links:[/bold green]")
                for off, pr in deals:
                    ref_p = pr.avg_price if pr.avg_price is not None else pr.min_price_all
                    deal_link = (off.link or "").replace("https://wallapop.com/", "https://es.wallapop.com/")
                    tag = "[bold green][LIVE][/bold green]" if off.id in live_ids else "[dim][PREVIOUSLY SAVED][/dim]"
                    console.print(f"  • {tag} [bold]{off.title}[/bold] ({off.price:.2f} € vs {ref_p:.2f} €):")
                    console.print(f"    [cyan underline]{deal_link}[/cyan underline]")
            else:
                console.print("\n[dim]No offers found with Wallapop price below TCGDex reference price.[/dim]")
    except Exception as exc:
        logger.debug(f"Error checking deals: {exc}")


def dispatch_discord_alerts(engine, args: argparse.Namespace, offers: Optional[list] = None) -> None:
    """Dispatches Discord notifications for qualifying deals according to settings."""
    if getattr(args, "no_discord", False):
        return

    from .alerts import DiscordAlertService
    from .settings import get_settings

    app_settings = get_settings()
    if getattr(args, "discord_webhook", None):
        app_settings = app_settings.model_copy(update={"discord_webhook_url": args.discord_webhook})

    ref_prices = {}
    db_deals_to_alert = []
    saved_deals_count = 0
    try:
        from .database.models import CardmarketPrice, WallapopOffers
        from sqlmodel import select, Session, and_

        with Session(engine) as s:
            for row in s.exec(select(CardmarketPrice)).all():
                if row.cardmarket_product_id:
                    ref_prices[row.cardmarket_product_id] = {
                        "avg": row.avg_price,
                        "low": row.min_price_all,
                        "all": row.avg_price or row.min_price_all,
                    }

            from sqlmodel import or_
            deals_stmt = (
                select(WallapopOffers, CardmarketPrice)
                .join(
                    CardmarketPrice,
                    and_(
                        WallapopOffers.cardmarket_product_id == CardmarketPrice.cardmarket_product_id,
                        WallapopOffers.language == CardmarketPrice.language,
                    ),
                )
                .where(
                    or_(
                        and_(CardmarketPrice.avg_price.is_not(None), WallapopOffers.price < CardmarketPrice.avg_price),
                        and_(CardmarketPrice.avg_price.is_(None), CardmarketPrice.min_price_all.is_not(None), WallapopOffers.price < CardmarketPrice.min_price_all),
                    )
                )
            )
            all_deals = s.exec(deals_stmt).all()
            current_ids = {o.id for o in (offers or [])}
            saved_deals_count = sum(1 for off, _ in all_deals if off.id not in current_ids)

            include_saved = app_settings.debug or getattr(args, "alert_all_deals", False)
            if include_saved:
                from .models import OfferListing
                for off, pr in all_deals:
                    deal_item = OfferListing(
                        id=off.id,
                        title=off.title,
                        price=off.price,
                        currency=off.currency,
                        product_type=off.product_type,
                        location=off.location,
                        country=off.country,
                        description=off.description,
                        link=off.link,
                        image_url=off.image_url,
                        is_reserved=off.is_reserved,
                        shipping_available=off.shipping_available,
                        seller_user_id=off.seller_user_id,
                        seller_link=off.seller_link,
                        published_date=off.published_date,
                        search_profile_id=off.search_profile_id,
                        cardmarket_product_id=off.cardmarket_product_id,
                        cardmarket_url=(
                            f"https://www.cardmarket.com/es/Pokemon/Products?idProduct={off.cardmarket_product_id}"
                            if off.cardmarket_product_id
                            else None
                        ),
                        language=off.language,
                        cardmarket_low_price=pr.min_price_all,
                        cardmarket_avg_price=pr.avg_price,
                        cardmarket_price_all=pr.avg_price or pr.min_price_all,
                        tcgdex_id=off.tcgdex_id,
                    )
                    db_deals_to_alert.append(deal_item)
    except Exception:
        pass

    from .database.models import WallapopOffers

    def _normalize_offer(item: Any) -> OfferListing:
        if isinstance(item, WallapopOffers):
            cm_prod_id = item.cardmarket_product_id
            pr = ref_prices.get(cm_prod_id, {}) if cm_prod_id else {}
            avg_p = pr.get("avg") if isinstance(pr, dict) else getattr(item, "_temp_cm_avg_price", None)
            low_p = pr.get("low") or pr.get("all") if isinstance(pr, dict) else getattr(item, "_temp_cm_low_price", None)
            return OfferListing(
                id=item.id,
                title=item.title,
                price=item.price,
                currency=item.currency,
                product_type=item.product_type,
                location=item.location,
                country=item.country,
                description=item.description,
                link=item.link,
                image_url=item.image_url,
                is_reserved=item.is_reserved,
                shipping_available=item.shipping_available,
                seller_user_id=item.seller_user_id,
                seller_link=item.seller_link,
                published_date=item.published_date,
                search_profile_id=item.search_profile_id,
                cardmarket_product_id=cm_prod_id,
                cardmarket_url=(
                    getattr(item, "_temp_cm_url", None)
                    or (f"https://www.cardmarket.com/es/Pokemon/Products?idProduct={cm_prod_id}" if cm_prod_id else None)
                ),
                language=item.language,
                cardmarket_low_price=low_p,
                cardmarket_avg_price=avg_p,
                cardmarket_price_all=avg_p or low_p,
                tcgdex_id=item.tcgdex_id,
                assigned_state=item.assigned_state or "pending",
            )
        return item

    include_saved = app_settings.debug or getattr(args, "alert_all_deals", False)
    alert_svc = DiscordAlertService(settings=app_settings)
    raw_targets = db_deals_to_alert if include_saved and db_deals_to_alert else (offers or [])
    target_offers = [_normalize_offer(o) for o in raw_targets]
    alerts_sent = alert_svc.process_and_alert_offers(target_offers, reference_prices=ref_prices)
    if alerts_sent > 0:
        mode_desc = "including saved offers [DEBUG=true]" if app_settings.debug else ("all DB deals" if getattr(args, "alert_all_deals", False) else "live deals")
        console.print(f"  • [bold green]DISCORD ALERTS:[/] [bold]{alerts_sent}[/bold] alert notification(s) sent ({mode_desc})")
    elif app_settings.discord_webhook_url and app_settings.discord_alerts_enabled:
        note = f" ({saved_deals_count} existing saved deals skipped because DEBUG=false)" if saved_deals_count > 0 and not include_saved else ""
        console.print(f"  • [bold]DISCORD ALERTS:[/] [dim]Checked {len(target_offers)} live offers (0 met alert condition thresholds){note}[/dim]")


def run_scrape(args: argparse.Namespace, config) -> None:
    """Scrapes Wallapop offers and saves them to DB incrementally without running card recognition."""
    scraper = CardOffersScraper(config)

    console.print(
        Panel.fit(
            f"[bold cyan]{config.name}[/bold cyan]\n"
            f"[dim]{config.description or ''}[/dim]\n"
            f"[yellow]Location:[/] {config.wallapop_settings.location_name} "
            f"({config.wallapop_settings.latitude}, {config.wallapop_settings.longitude})\n"
            f"[yellow]Profiles:[/] {len(config.search_profiles)} configured\n"
            f"[yellow]Output Dir:[/] {config.output_settings.output_dir}\n"
            f"[yellow]Loop Mode:[/] {'Enabled (' + str(getattr(args, 'interval', 300)) + 's interval)' if getattr(args, 'loop', False) else 'Single run'}",
            title="🃏 Delta Cards Scraper (Wallapop Offers)",
            border_style="cyan",
        )
    )

    db_save_enabled = config.output_settings.save_db and not getattr(args, "no_db", False)
    if db_save_enabled:
        init_db(get_engine(getattr(args, "db_url", None)))

    while True:
        db_saved_total = 0
        db_error = None

        def handle_scraped_batch(batch: List[OfferListing]) -> None:
            nonlocal db_saved_total, db_error
            if db_save_enabled and batch:
                try:
                    saved = save_offers_to_db(batch, db_url=getattr(args, "db_url", None))
                    db_saved_total += saved
                except Exception as exc:
                    db_error = str(exc)
                    logger.warning(f"Error saving batch to DB: {exc}")

        with console.status("[bold cyan]Starting Wallapop scraper...[/bold cyan]", spinner="dots") as status:
            progress_cb = _create_scraper_progress_callback(status, db_save_enabled)
            offers = asyncio.run(
                scraper.run(
                    progress_callback=progress_cb,
                    on_batch=handle_scraped_batch,
                )
            )

            status.update("[bold cyan]Saving scraped data to files...[/bold cyan]")
            saved_files = scraper.save_results(offers)

        db_note = None
        if db_save_enabled:
            if db_saved_total > 0:
                db_note = f"  • [bold]DATABASE ('WallapopOffers'):[/bold] [green]{db_saved_total} offer(s) saved incrementally to database[/green]"
            elif db_error:
                db_note = f"  • [bold yellow]DATABASE ('WallapopOffers'):[/bold yellow] [dim]Skipped ({db_error})[/dim]"
            else:
                db_note = "  • [bold]DATABASE ('WallapopOffers'):[/bold] [dim]0 new offers to save[/dim]"

        _render_scraped_summary_table(offers, saved_files, db_note=db_note)

        if not getattr(args, "loop", False):
            break
        interval = getattr(args, "interval", 300)
        console.print(f"\n[dim]Sleeping {interval}s before next scrape cycle (Ctrl+C to stop)...[/dim]")
        time.sleep(interval)


def run_assign(args: argparse.Namespace, config) -> None:
    """Consumer worker: queries pending offers from DB, runs CardRecognitionPipeline, and checks deals."""
    if handle_test_discord(args):
        return

    from .database.models import WallapopOffers
    from .database import upsert_cardmarket_product, upsert_cardmarket_price
    from sqlmodel import Session, select
    from tqdm import tqdm

    from .database.connection import init_db
    engine = get_engine(getattr(args, "db_url", None))
    init_db(engine)

    is_continuous = not getattr(args, "once", False)
    interval = getattr(args, "interval", 30)

    console.print(
        Panel.fit(
            f"[bold cyan]Delta Cards Assigner[/bold cyan]\n"
            f"[dim]Card Recognition & Deal Evaluation Consumer Worker[/dim]\n"
            f"[yellow]Database:[/] {engine.url}\n"
            f"[yellow]Scrape Cardmarket Prices:[/] {bool(getattr(args, 'scrape_prices', False))}\n"
            f"[yellow]Worker Mode:[/] {'Always On (polling every ' + str(interval) + 's)' if is_continuous else 'Single run (--once)'}",
            title="🃏 Delta Cards Assigner",
            border_style="cyan",
        )
    )

    while True:
        with Session(engine) as session:
            stmt = select(WallapopOffers).where(
                WallapopOffers.assigned_state == "pending",
            )
            if getattr(args, "limit", None) is not None:
                stmt = stmt.limit(args.limit)
            raw_candidates = session.exec(stmt).all()

        if not raw_candidates:
            if is_continuous:
                console.print(f"[dim]Queue empty. Assigner waiting for new offers (polling every {interval}s, Ctrl+C to stop)...[/dim]")
                time.sleep(interval)
                continue
            else:
                console.print("[green]✓ No pending card offers found in database.[/green]")
                break

        unassigned_cards = []
        non_card_offers = []
        for r in raw_candidates:
            if is_card_offer(r.product_type, title=r.title, description=r.description):
                unassigned_cards.append(r)
            else:
                non_card_offers.append(r)

        # Non-card offers (booster boxes, packs, lots) cannot have cardmarket products assigned
        if non_card_offers:
            with Session(engine) as s:
                for item in non_card_offers:
                    db_item = s.exec(select(WallapopOffers).where(WallapopOffers.id == item.id)).first()
                    if db_item:
                        db_item.assigned_state = "not_assigned"
                        if getattr(db_item, "alert_status", None) != "sent":
                            db_item.alert_status = "none"
                        db_item.updated_at = datetime.now(timezone.utc)
                        s.add(db_item)
                s.commit()
            logger.info(f"Marked {len(non_card_offers)} non-card offer(s) as 'not_assigned'.")

        if not unassigned_cards:
            if is_continuous:
                console.print(f"[dim]No valid pending single cards in queue. Waiting {interval}s...[/dim]")
                time.sleep(interval)
                continue
            else:
                console.print("[green]✓ No valid pending single card offers to process.[/green]")
                break

        console.print(f"\n[bold cyan]Found {len(unassigned_cards)} pending card offer(s) to process.[/bold cyan]")

        matched_count = [0]
        pbar = tqdm(total=len(unassigned_cards), desc=f"Assigning cards (0/{len(unassigned_cards)} matched)", unit="card")

        def on_progress(index, total, offer):
            has_cm = bool(getattr(offer, "cardmarket_product_id", None))
            if has_cm or getattr(offer, "tcgdex_id", None):
                matched_count[0] += 1
            state = "assigned" if has_cm else "not_assigned"
            try:
                with Session(engine) as s:
                    db_item = s.exec(select(WallapopOffers).where(WallapopOffers.id == offer.id)).first()
                    low_p = getattr(offer, "_temp_cm_low_price", None) or getattr(offer, "cardmarket_low_price", None)
                    avg_p = getattr(offer, "_temp_cm_avg_price", None) or getattr(offer, "cardmarket_avg_price", None)
                    trend_p = getattr(offer, "_temp_cm_trend_price", None) or getattr(offer, "cardmarket_trend_price", None)

                    if getattr(offer, "cardmarket_product_id", None):
                        cm_id = offer.cardmarket_product_id
                        cm_url = (
                            getattr(offer, "_temp_cm_url", None)
                            or getattr(offer, "cardmarket_url", None)
                            or f"https://www.cardmarket.com/{offer.language or 'es'}/Pokemon/Products?idProduct={cm_id}"
                        )
                        upsert_cardmarket_product(product_id=cm_id, cardmarket_url=cm_url, session=s)
                        if low_p is not None or avg_p is not None or trend_p is not None:
                            upsert_cardmarket_price(
                                product_id=cm_id,
                                language=offer.language or "es",
                                min_price_all=low_p,
                                avg_price=avg_p,
                                trend_price=trend_p,
                                session=s,
                            )

                    if db_item:
                        db_item.assigned_state = state
                        db_item.updated_at = datetime.now(timezone.utc)
                        if getattr(offer, "tcgdex_id", None):
                            db_item.tcgdex_id = offer.tcgdex_id
                        if getattr(offer, "cardmarket_product_id", None):
                            db_item.cardmarket_product_id = offer.cardmarket_product_id
                        if getattr(offer, "language", None):
                            db_item.language = offer.language

                        # Evaluate pricing deal for decoupled alert_status:
                        ref_p = avg_p if avg_p is not None else low_p
                        if getattr(offer, "cardmarket_product_id", None) and ref_p is not None and ref_p > 0 and db_item.price < ref_p:
                            if getattr(db_item, "alert_status", None) != "sent":
                                db_item.alert_status = "pending"
                        else:
                            if getattr(db_item, "alert_status", None) != "sent":
                                db_item.alert_status = "none"
                        s.add(db_item)
                    s.commit()
            except Exception as exc:
                logger.debug(f"Error saving incremental match for {offer.id}: {exc}")
            pbar.set_description(f"Assigning cards ({matched_count[0]}/{len(unassigned_cards)} matched)")
            pbar.update(1)

        prev_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)
        assign_tcgdex_to_offers(unassigned_cards, progress_callback=on_progress, max_concurrency=4)
        logging.getLogger().setLevel(prev_level)
        pbar.close()

        # Reconcile any remaining cards in batch to ensure none stay 'pending'
        with Session(engine) as s:
            for o in unassigned_cards:
                db_item = s.exec(select(WallapopOffers).where(WallapopOffers.id == o.id)).first()
                if db_item:
                    has_id = bool(getattr(o, "cardmarket_product_id", None) or getattr(o, "tcgdex_id", None))
                    if db_item.assigned_state == "pending":
                        db_item.assigned_state = "assigned" if has_id else "not_assigned"
                        db_item.updated_at = datetime.now(timezone.utc)
                    if getattr(o, "tcgdex_id", None):
                        db_item.tcgdex_id = o.tcgdex_id
                    if getattr(o, "cardmarket_product_id", None):
                        db_item.cardmarket_product_id = o.cardmarket_product_id
                    if not getattr(o, "cardmarket_product_id", None) and getattr(db_item, "alert_status", None) != "sent":
                        db_item.alert_status = "none"
                    s.add(db_item)
            s.commit()

        console.print(f"[bold green]✓ Completed assignment batch: {matched_count[0]}/{len(unassigned_cards)} card(s) matched.[/bold green]")

        check_and_report_deals(engine, args, live_offer_ids=[o.id for o in unassigned_cards])

        if not is_continuous:
            break
        console.print(f"\n[dim]Assigner sleeping {interval}s before next queue check (Ctrl+C to stop)...[/dim]")
        time.sleep(interval)


def run_all(args: argparse.Namespace, config) -> None:
    """Runs full unified pipeline (scrape + assign + check deals) with backward compatibility."""
    console.print(
        Panel.fit(
            f"[bold cyan]{config.name}[/bold cyan]\n"
            f"[dim]{config.description or ''}[/dim]\n"
            f"[yellow]Location:[/] {config.wallapop_settings.location_name} "
            f"({config.wallapop_settings.latitude}, {config.wallapop_settings.longitude})\n"
            f"[yellow]Profiles:[/] {len(config.search_profiles)} configured\n"
            f"[yellow]Output Dir:[/] {config.output_settings.output_dir}",
            title="🃏 Delta Cards Scraper",
            border_style="cyan",
        )
    )

    if getattr(args, "assign_existing_cards", False):
        console.print("[bold cyan]Enriching unassigned card offers in database with TCGDex & Cardmarket IDs...[/bold cyan]")
        try:
            engine = get_engine(args.db_url)
            count = enrich_card_offers_in_db(engine=engine)
            console.print(f"[bold green]✓ Successfully matched and updated {count} card offer(s) in the database.[/bold green]")
        except Exception as exc:
            err_console.print(f"[bold red]Error updating database offers:[/] {exc}")
            sys.exit(1)

    if handle_test_discord(args):
        return

    scraper = CardOffersScraper(config)

    db_save_enabled = config.output_settings.save_db and not getattr(args, "no_db", False)
    if db_save_enabled:
        init_db(get_engine(args.db_url))

    def handle_scraped_batch(batch: List[OfferListing]) -> None:
        if db_save_enabled and batch:
            try:
                save_offers_to_db(batch, db_url=args.db_url)
            except Exception as exc:
                logger.warning(f"Error saving batch to DB: {exc}")

    with console.status("[bold cyan]Starting scraper...[/bold cyan]", spinner="dots") as status:
        progress_cb = _create_scraper_progress_callback(status, db_save_enabled, is_all=True)
        offers = asyncio.run(
            scraper.run(
                progress_callback=progress_cb,
                on_batch=handle_scraped_batch,
            )
        )

        card_offers = [o for o in offers if o.product_type == "single_card"]
        tcgdex_assigned_count = 0
        if not getattr(args, "skip_tcgdex", False) and card_offers:
            status.stop()

            already_in_db = set()
            if config.output_settings.save_db and not getattr(args, "no_db", False):
                try:
                    from .database.models import WallapopOffers
                    from sqlmodel import Session, select
                    eng = get_engine(args.db_url)
                    with Session(eng) as s:
                        card_ids = [o.id for o in card_offers]
                        for i in range(0, len(card_ids), 500):
                            chunk = card_ids[i:i + 500]
                            existing = s.exec(
                                select(
                                    WallapopOffers.id,
                                    WallapopOffers.tcgdex_id,
                                    WallapopOffers.cardmarket_product_id,
                                    WallapopOffers.language,
                                )
                                .where(WallapopOffers.id.in_(chunk))
                            ).all()
                            db_lookup = {row[0]: row for row in existing}
                            for o in card_offers:
                                if o.id in db_lookup:
                                    already_in_db.add(o.id)
                                    db_id, db_tcgdex, db_cm, db_lang = db_lookup[o.id]
                                    if db_tcgdex:
                                        o.tcgdex_id = db_tcgdex
                                    if db_cm:
                                        o.cardmarket_product_id = db_cm
                                    if db_lang:
                                        o.language = db_lang
                except Exception as exc:
                    logger.debug(f"Could not pre-check DB for existing card offers: {exc}")

            new_card_offers = [
                o for o in card_offers
                if o.id not in already_in_db and not o.tcgdex_id and not o.cardmarket_product_id
            ]
            already_count = len(card_offers) - len(new_card_offers)

            if already_count > 0:
                console.print(f"[cyan]ℹ {already_count}/{len(card_offers)} card offers already exist in DB (skipping card assignment).[/cyan]")

            if not new_card_offers:
                console.print("[green]✓ All card offers were already scraped previously.[/green]")
            else:
                from tqdm import tqdm

                matched = [0]
                pbar = tqdm(total=len(new_card_offers), desc=f"Assigning new cards (0/{len(new_card_offers)} matched)", unit="card")

                def assignment_progress(index, total, offer):
                    if getattr(offer, "cardmarket_product_id", None):
                        offer.assigned_state = "assigned"
                        matched[0] += 1
                    else:
                        offer.assigned_state = "not_assigned"
                    pbar.set_description(f"Assigning new cards ({matched[0]}/{len(new_card_offers)} matched)")
                    pbar.update(1)
                    if config.output_settings.save_db and not getattr(args, "no_db", False):
                        try:
                            save_offers_to_db([offer], db_url=args.db_url)
                        except Exception:
                            pass

                prev_level = logging.getLogger().level
                logging.getLogger().setLevel(logging.ERROR)
                assign_tcgdex_to_offers(new_card_offers, progress_callback=assignment_progress, max_concurrency=4)
                logging.getLogger().setLevel(prev_level)
                pbar.close()

            status.start()
            tcgdex_assigned_count = sum(1 for o in card_offers if o.tcgdex_id)

        status.update("[bold cyan]Saving scraped data...[/bold cyan]")
        saved_files = scraper.save_results(offers)

        db_saved_count = None
        db_error = None
        if config.output_settings.save_db and not getattr(args, "no_db", False) and offers:
            try:
                status.update("[bold cyan]Saving offers to database...[/bold cyan]")
                db_saved_count = save_offers_to_db(offers, db_url=args.db_url)
            except Exception as exc:
                db_error = str(exc)
    extra_lines = []
    if not getattr(args, "skip_tcgdex", False) and card_offers:
        extra_lines.append(f"  • [bold]TCGDEX / CARDMARKET ENRICHMENT:[/bold] [green]{tcgdex_assigned_count}/{len(card_offers)} card offer(s) identified[/green]")
    db_note = None
    if db_saved_count is not None:
        db_note = f"  • [bold]DATABASE ('WallapopOffers'):[/bold] [green]{db_saved_count} offer(s) saved/updated[/green]"
    elif db_error:
        db_note = f"  • [bold yellow]DATABASE ('WallapopOffers'):[/bold yellow] [dim]Skipped ({db_error})[/dim]"

    _render_scraped_summary_table(offers, saved_files, db_note=db_note, extra_lines=extra_lines)

    if config.output_settings.save_db and not getattr(args, "no_db", False):
        engine = get_engine(args.db_url)
        check_and_report_deals(engine, args, live_offer_ids=[o.id for o in offers])
        dispatch_discord_alerts(engine, args, offers=offers)


def run_build_index(args, config) -> None:
    """Builds offline USearch Mmap index from TCGdex SDK."""
    import logging
    logging.getLogger("delta_cards.recognition.index_builder").setLevel(logging.INFO)
    from delta_cards.recognition.index_builder import build_index

    languages = [args.lang]
    if args.lang.lower() == "all":
        languages = ["en", "es", "ja"]

    for lang in languages:
        target_index = args.index_path
        if not target_index:
            target_index = f"data/pokemon_cards_{lang}.usearch"
        elif len(languages) > 1:
            target_index = f"data/pokemon_cards_{lang}.usearch"

        console.print(Panel(
            f"[bold cyan]Building USearch Mmap Index from TCGdex[/bold cyan]\n"
            f"Sets: {args.sets or 'ALL'} | Language: {lang} | Output: {target_index}",
            border_style="cyan",
        ))
        asyncio.run(
            build_index(
                sets_filter=args.sets,
                language=lang,
                limit_cards=args.limit,
                index_path=target_index,
                cache_dir=args.cache_dir,
                concurrency=args.concurrency,
            )
        )
    console.print("[bold green]✓ Offline index construction complete![/bold green]")


def run_export_models(args, config) -> None:
    """Exports and quantizes models to ONNX INT8."""
    from delta_cards.recognition.dinov3_embedder import ensure_dinov3_onnx_model
    console.print(Panel("[bold cyan]Exporting and Quantizing DINOv3 ViT-Small INT8 ONNX[/bold cyan]", border_style="cyan"))
    pth = ensure_dinov3_onnx_model()
    console.print(f"[bold green]✓ DINOv3 INT8 ONNX ready at:[/bold green] [cyan]{pth}[/cyan]")


def run_sync_catalog(args, config) -> None:
    """Runs daily incremental catalog synchronization."""
    import logging
    logging.getLogger("delta_cards.tcgdex.catalog_sync").setLevel(logging.INFO)
    from delta_cards.tcgdex.catalog_sync import CatalogSyncService

    service = CatalogSyncService(
        index_path=args.index_path,
        cache_dir=args.cache_dir,
        language=args.lang,
        concurrency=args.concurrency,
    )

    if args.daemon:
        console.print(Panel(
            f"[bold cyan]Starting Daily Catalog Sync Daemon[/bold cyan]\n"
            f"Interval: Every {args.interval_hours} hours | Language: {args.lang} | Index: {args.index_path}",
            border_style="cyan",
        ))
        asyncio.run(service.run_daemon(interval_hours=args.interval_hours))
    else:
        console.print(Panel(
            f"[bold cyan]Running Incremental Catalog Sync (One-Shot)[/bold cyan]\n"
            f"Language: {args.lang} | Index: {args.index_path}",
            border_style="cyan",
        ))
        summary = asyncio.run(service.sync())
        if summary.updated:
            console.print(
                f"[bold green]✓ Sync complete! Added {summary.new_cards_added} new cards "
                f"across {summary.new_sets_detected} sets in {summary.elapsed_seconds}s.[/bold green]"
            )
        else:
            console.print(
                f"[bold green]✓ Catalog is completely up-to-date. (0 new cards, checked in {summary.elapsed_seconds}s)[/bold green]"
            )


def run_alerts(args: argparse.Namespace, config) -> None:
    """Decoupled alert worker: reads pending deals from DB, runs LLM verifier, and dispatches 1 alert every interval."""
    if handle_test_discord(args):
        return

    from .alerts import AlertWorkerService
    from .settings import get_settings
    from .database.connection import init_db

    engine = get_engine(getattr(args, "db_url", None))
    init_db(engine)

    app_settings = get_settings()
    if getattr(args, "discord_webhook", None):
        app_settings = app_settings.model_copy(update={"discord_webhook_url": args.discord_webhook})
    if getattr(args, "no_discord", False):
        app_settings = app_settings.model_copy(update={"discord_alerts_enabled": False})

    interval = getattr(args, "interval", None)
    if interval is None:
        interval = app_settings.alert_interval

    is_once = bool(getattr(args, "once", False))

    console.print(
        Panel.fit(
            f"[bold cyan]Delta Cards Alert Worker[/bold cyan]\n"
            f"[dim]Decoupled Periodic Alert Dispatcher & LLM Verifier[/dim]\n"
            f"[yellow]Database:[/] {engine.url}\n"
            f"[yellow]Discord Enabled:[/] {app_settings.discord_alerts_enabled}\n"
            f"[yellow]Interval:[/] {interval}s ({round(interval / 60, 1)} min)\n"
            f"[yellow]Worker Mode:[/] {'Single run (--once)' if is_once else 'Continuous Loop'}",
            title="🔔 Delta Cards Alert Worker",
            border_style="cyan",
        )
    )

    worker = AlertWorkerService(engine=engine, settings=app_settings)
    worker.run_loop(interval_seconds=interval, once=is_once)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if args.subcommand == "build-index":
        run_build_index(args, None)
    elif args.subcommand == "export-models":
        run_export_models(args, None)
    elif args.subcommand == "sync-catalog":
        run_sync_catalog(args, None)
    else:
        config = load_and_filter_config(args)
        if args.subcommand == "scrape":
            run_scrape(args, config)
        elif args.subcommand == "assign":
            run_assign(args, config)
        elif args.subcommand == "alerts":
            run_alerts(args, config)
        else:
            run_all(args, config)


def main_scrape() -> None:
    """Direct CLI entrypoint for 'delta-cards-scrape'."""
    if len(sys.argv) > 1 and sys.argv[1] != "scrape":
        sys.argv.insert(1, "scrape")
    elif len(sys.argv) == 1:
        sys.argv.append("scrape")
    main()


def main_assign() -> None:
    """Direct CLI entrypoint for 'delta-cards-assign'."""
    if len(sys.argv) > 1 and sys.argv[1] != "assign":
        sys.argv.insert(1, "assign")
    elif len(sys.argv) == 1:
        sys.argv.append("assign")
    main()


def main_alert() -> None:
    """Direct CLI entrypoint for 'delta-cards-alert' / 'delta-cards-alerts'."""
    if len(sys.argv) > 1 and sys.argv[1] not in ("alerts", "alert"):
        sys.argv.insert(1, "alerts")
    elif len(sys.argv) == 1:
        sys.argv.append("alerts")
    main()


if __name__ == "__main__":
    main()
