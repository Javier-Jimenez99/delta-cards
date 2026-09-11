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
    parser.add_argument("--scrape-prices", action="store_true", help="Scrape Cardmarket minimum prices")
    parser.add_argument("--scrape-offers", action="store_true", help="Scrape and save detailed Cardmarket seller offers")
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
    assign_parser.add_argument("--scrape-prices", action="store_true", help="Scrape Cardmarket minimum prices for matched cards")
    assign_parser.add_argument("--scrape-offers", action="store_true", help="Scrape and save detailed Cardmarket seller offers")
    assign_parser.add_argument("--no-discord", action="store_true", help="Skip sending Discord alert notifications")
    assign_parser.add_argument("--discord-webhook", type=str, default=None, help="Override Discord webhook URL")
    assign_parser.add_argument("--test-discord", action="store_true", help="Send a test deal alert to Discord and exit")
    assign_parser.add_argument("--alert-all-deals", action="store_true", help="Send Discord alerts for all matching deals")
    assign_parser.add_argument("--once", action="store_true", help="Process existing unassigned cards once and exit (default: stays on waiting for new offers)")
    assign_parser.add_argument("--loop", action="store_true", default=True, help="Keep worker running continuously (default: True)")
    assign_parser.add_argument("--interval", type=int, default=30, help="Seconds between queue checks when waiting for offers (default: 30)")

    # 3. All subcommand: explicit full pipeline (scrape + assign + check deals)
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
    all_parser.add_argument("--scrape-prices", action="store_true", help="Scrape Cardmarket minimum prices")
    all_parser.add_argument("--scrape-offers", action="store_true", help="Scrape and save detailed Cardmarket seller offers")
    all_parser.add_argument("--skip-tcgdex", action="store_true", help="Skip card recognition and assignment")
    all_parser.add_argument("--assign-existing-cards", action="store_true", help="Enrich unassigned cards in database")
    all_parser.add_argument("--no-discord", action="store_true", help="Skip sending Discord alert notifications")
    all_parser.add_argument("--discord-webhook", type=str, default=None, help="Override Discord webhook URL")
    all_parser.add_argument("--test-discord", action="store_true", help="Send a test deal alert to Discord and exit")
    all_parser.add_argument("--alert-all-deals", action="store_true", help="Send Discord alerts for all matching deals")

    # 4. Cardmarket offers inspection subcommand
    cm_parser = subparsers.add_parser(
        "cardmarket-offers",
        parents=[common_parser],
        help="Scrape and display specific Cardmarket seller offers for a product ID or URL",
    )
    cm_parser.add_argument("product_id", type=str, help="Cardmarket product ID (e.g. 483559) or product URL")
    cm_parser.add_argument("--language", "-L", type=str, default="es", help="Card language code (default: es)")
    cm_parser.add_argument("--spain-only", action="store_true", help="Only show offers from Spanish sellers")
    cm_parser.add_argument("-l", "--limit", type=int, default=20, help="Max offers to display (default: 20)")
    cm_parser.add_argument("--save-db", action="store_true", help="Save scraped offers to database")

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
        cardmarket_price_spain=140.0,
        cardmarket_price_all=120.0,
    )

    service = DiscordAlertService(settings=settings)
    sent_count = service.process_and_alert_offers([test_offer])
    if sent_count > 0:
        console.print("[bold green]✓ Test Discord alert sent successfully! Check your Discord channel.[/bold green]")
    else:
        console.print("[bold red]✗ Failed to send Discord alert. Check logs or verify that DISCORD_ALERTS_ENABLED=true.[/bold red]")
    return True


def scrape_cardmarket_prices_for_offers(offers: list, engine, args: argparse.Namespace, status=None) -> int:
    """Scrapes Cardmarket minimum prices and optionally specific offers for matched card offers."""
    scrape_prices_flag = getattr(args, "scrape_prices", False)
    scrape_offers_flag = getattr(args, "scrape_offers", False)
    if not (scrape_prices_flag or scrape_offers_flag) or not offers:
        return 0

    from .cardmarket import CardmarketScraper
    from .database import upsert_cardmarket_price, upsert_cardmarket_offers

    cm_scraper = CardmarketScraper()
    unique_pairs = list({
        (o.cardmarket_product_id, (getattr(o, "language", None) or "es").lower())
        for o in offers
        if getattr(o, "cardmarket_product_id", None)
    })
    cm_prices_scraped = 0
    if unique_pairs:
        msg = f"[bold cyan]Scraping Cardmarket data for {len(unique_pairs)} card product(s)...[/bold cyan]"
        if status:
            status.update(msg)
        else:
            console.print(msg)
        for pid, card_lang in unique_pairs:
            try:
                price_res = asyncio.run(cm_scraper.scrape_product_prices(pid, language=card_lang))
                if getattr(cm_scraper, "_warned_cloudflare", False):
                    console.print(
                        "  [yellow]⚠ Cardmarket web requests blocked by Cloudflare. "
                        "Skipping direct scraping; using reference prices.[/yellow]"
                    )
                    break
                if price_res.min_price_all is not None or price_res.min_price_spain is not None:
                    upsert_cardmarket_price(
                        product_id=price_res.cardmarket_product_id,
                        language=price_res.language,
                        min_price_all=price_res.min_price_all,
                        min_price_spain=price_res.min_price_spain,
                        engine=engine,
                    )
                    cm_prices_scraped += 1

                if scrape_offers_flag:
                    offers_res = asyncio.run(cm_scraper.scrape_product_offers(pid, language=card_lang))
                    if offers_res:
                        upsert_cardmarket_offers(offers_res, product_id=pid, engine=engine)
                        logger.info("Saved %d specific Cardmarket offers for product %s", len(offers_res), pid)
            except Exception as exc:
                logger.warning(f"Error scraping Cardmarket data for product {pid} ({card_lang}): {exc}")
    return cm_prices_scraped


def check_and_report_deals(engine, args: argparse.Namespace, live_offer_ids: Optional[List[str]] = None) -> None:
    """Queries DB for deals where Wallapop price < Cardmarket min price and prints summary table."""
    try:
        from .database.models import WallapopOffers, CardmarketPrice
        from sqlmodel import select, Session, and_

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
                    WallapopOffers.price < CardmarketPrice.min_price_all
                )
            )
            deals = s.exec(deals_stmt).all()
            if deals:
                live_ids = set(live_offer_ids) if live_offer_ids else set()
                deals_table = Table(
                    title=f"\n🔥 Deals Found (Wallapop < Cardmarket): {len(deals)}",
                    border_style="bold green",
                )
                deals_table.add_column("Status", style="dim", justify="center")
                deals_table.add_column("Title", style="cyan")
                deals_table.add_column("Wallapop", justify="right", style="bold green")
                deals_table.add_column("Min Cardmarket", justify="right", style="yellow")
                deals_table.add_column("Savings", justify="right", style="bold magenta")
                deals_table.add_column("Link", style="cyan", overflow="fold", no_wrap=False)

                for off, pr in deals:
                    diff = (pr.min_price_all or 0.0) - off.price
                    deal_link = (off.link or "").replace("https://wallapop.com/", "https://es.wallapop.com/")
                    status_tag = "[bold green]LIVE[/bold green]" if off.id in live_ids else "[dim]SAVED[/dim]"
                    deals_table.add_row(
                        status_tag,
                        off.title[:42],
                        f"{off.price:.2f} €",
                        f"{pr.min_price_all:.2f} €",
                        f"-{diff:.2f} €",
                        f"[link={deal_link}]{deal_link}[/link]",
                    )
                console.print(deals_table)

                console.print("\n[bold green]🔗 Clickable Deal Links:[/bold green]")
                for off, pr in deals:
                    deal_link = (off.link or "").replace("https://wallapop.com/", "https://es.wallapop.com/")
                    tag = "[bold green][LIVE][/bold green]" if off.id in live_ids else "[dim][PREVIOUSLY SAVED][/dim]"
                    console.print(f"  • {tag} [bold]{off.title}[/bold] ({off.price:.2f} € vs {pr.min_price_all:.2f} € CM):")
                    console.print(f"    [cyan underline]{deal_link}[/cyan underline]")
            else:
                console.print("\n[dim]No offers found with Wallapop price below Cardmarket minimum.[/dim]")
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
                        "all": row.min_price_all,
                        "spain": row.min_price_spain,
                    }

            deals_stmt = (
                select(WallapopOffers, CardmarketPrice)
                .join(
                    CardmarketPrice,
                    and_(
                        WallapopOffers.cardmarket_product_id == CardmarketPrice.cardmarket_product_id,
                        WallapopOffers.language == CardmarketPrice.language,
                    ),
                )
                .where(WallapopOffers.price < CardmarketPrice.min_price_all)
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
                        cardmarket_price_all=pr.min_price_all,
                        cardmarket_price_spain=pr.min_price_spain,
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
            low_p = pr.get("all") if isinstance(pr, dict) else getattr(item, "_temp_cm_low_price", None)
            spain_p = pr.get("spain") if isinstance(pr, dict) else None
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
                cardmarket_price_all=low_p,
                cardmarket_price_spain=spain_p,
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
                    status.update("[cyan]Saving scraped data...[/cyan]")

            offers = asyncio.run(
                scraper.run(
                    progress_callback=progress_callback,
                    on_batch=handle_scraped_batch,
                )
            )

            status.update("[bold cyan]Saving scraped data to files...[/bold cyan]")
            saved_files = scraper.save_results(offers)

        if not offers:
            console.print("\n[yellow]No offers matched the configured search and filter criteria.[/yellow]")
        else:
            table = Table(title=f"\nScraped Offers Summary ({len(offers)} unique items)", border_style="cyan")
            table.add_column("Product Type", style="bold green")
            table.add_column("Offers Found", justify="right", style="cyan")
            table.add_column("Min Price", justify="right", style="yellow")
            table.add_column("Max Price", justify="right", style="yellow")
            table.add_column("Avg Price", justify="right", style="yellow")

            by_type = {}
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
            if db_save_enabled:
                if db_saved_total > 0:
                    console.print(f"  • [bold]DATABASE ('WallapopOffers'):[/bold] [green]{db_saved_total} offer(s) saved incrementally to database[/green]")
                elif db_error:
                    console.print(f"  • [bold yellow]DATABASE ('WallapopOffers'):[/bold yellow] [dim]Skipped ({db_error})[/dim]")
                else:
                    console.print("  • [bold]DATABASE ('WallapopOffers'):[/bold] [dim]0 new offers to save[/dim]")

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
                    if db_item:
                        db_item.assigned_state = state
                        db_item.updated_at = datetime.now(timezone.utc)
                        if getattr(offer, "tcgdex_id", None):
                            db_item.tcgdex_id = offer.tcgdex_id
                        if getattr(offer, "cardmarket_product_id", None):
                            db_item.cardmarket_product_id = offer.cardmarket_product_id
                        if getattr(offer, "language", None):
                            db_item.language = offer.language
                        s.add(db_item)
                    if getattr(offer, "cardmarket_product_id", None):
                        cm_id = offer.cardmarket_product_id
                        cm_url = (
                            getattr(offer, "_temp_cm_url", None)
                            or getattr(offer, "cardmarket_url", None)
                            or f"https://www.cardmarket.com/{offer.language or 'es'}/Pokemon/Products?idProduct={cm_id}"
                        )
                        upsert_cardmarket_product(product_id=cm_id, cardmarket_url=cm_url, session=s)
                        low_p = getattr(offer, "_temp_cm_low_price", None) or getattr(offer, "cardmarket_low_price", None)
                        if low_p is not None:
                            upsert_cardmarket_price(product_id=cm_id, language=offer.language or "es", min_price_all=low_p, session=s)
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
                if db_item and db_item.assigned_state == "pending":
                    db_item.assigned_state = "assigned" if getattr(o, "cardmarket_product_id", None) else "not_assigned"
                    db_item.updated_at = datetime.now(timezone.utc)
                    s.add(db_item)
            s.commit()

        console.print(f"[bold green]✓ Completed assignment batch: {matched_count[0]}/{len(unassigned_cards)} card(s) matched.[/bold green]")

        if getattr(args, "scrape_prices", False):
            matched_offers = [o for o in unassigned_cards if getattr(o, "tcgdex_id", None)]
            cm_count = scrape_cardmarket_prices_for_offers(matched_offers, engine, args)
            if cm_count > 0:
                console.print(f"  • [bold]CARDMARKET PRICES:[/bold] [green]{cm_count} price(s) scraped and saved[/green]")

        check_and_report_deals(engine, args, live_offer_ids=[o.id for o in unassigned_cards])
        dispatch_discord_alerts(engine, args, offers=unassigned_cards)

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
                status.update("[cyan]Proceeding to next step...[/cyan]")

        offers = asyncio.run(
            scraper.run(
                progress_callback=progress_callback,
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

        # Scrape Cardmarket prices if requested
        cm_prices_scraped = 0
        if getattr(args, "scrape_prices", False) and card_offers:
            engine = get_engine(args.db_url)
            cm_prices_scraped = scrape_cardmarket_prices_for_offers(card_offers, engine, args, status=status)

    if not offers:
        console.print("\n[yellow]No offers matched the configured search and filter criteria.[/yellow]")
        return

    table = Table(title=f"\nScraped Offers Summary ({len(offers)} unique items)", border_style="cyan")
    table.add_column("Product Type", style="bold green")
    table.add_column("Offers Found", justify="right", style="cyan")
    table.add_column("Min Price", justify="right", style="yellow")
    table.add_column("Max Price", justify="right", style="yellow")
    table.add_column("Avg Price", justify="right", style="yellow")

    by_type = {}
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
    if not getattr(args, "skip_tcgdex", False) and card_offers:
        console.print(f"  • [bold]TCGDEX / CARDMARKET ENRICHMENT:[/bold] [green]{tcgdex_assigned_count}/{len(card_offers)} card offer(s) identified[/green]")
    if getattr(args, "scrape_prices", False) and card_offers:
        console.print(f"  • [bold]CARDMARKET PRICES:[/bold] [green]{cm_prices_scraped} card product price(s) scraped and saved[/green]")
    if db_saved_count is not None:
        console.print(f"  • [bold]DATABASE ('WallapopOffers'):[/bold] [green]{db_saved_count} offer(s) saved/updated[/green]")
    elif db_error:
        console.print(f"  • [bold yellow]DATABASE ('WallapopOffers'):[/bold yellow] [dim]Skipped ({db_error})[/dim]")

    if config.output_settings.save_db and not getattr(args, "no_db", False):
        engine = get_engine(args.db_url)
        check_and_report_deals(engine, args, live_offer_ids=[o.id for o in offers])
        dispatch_discord_alerts(engine, args, offers=offers)


def run_cardmarket_offers(args: argparse.Namespace) -> None:
    """Scrapes and renders individual Cardmarket seller offers for a given product."""
    from .cardmarket import CardmarketScraper
    from .database import get_engine, upsert_cardmarket_offers

    cm_scraper = CardmarketScraper()
    pid = args.product_id
    lang = getattr(args, "language", "es")
    spain_only = getattr(args, "spain_only", False)
    limit = getattr(args, "limit", 20)

    console.print(f"[bold cyan]Fetching Cardmarket offers for product: {pid} (language: {lang}, Spain only: {spain_only})...[/bold cyan]")
    try:
        offers = asyncio.run(
            cm_scraper.scrape_product_offers(
                product_id_or_url=pid,
                language=lang,
                spanish_sellers_only=spain_only,
                max_offers=limit,
            )
        )
    except Exception as exc:
        console.print(f"[bold red]Failed to scrape Cardmarket offers: {exc}[/bold red]")
        return

    if not offers:
        console.print(f"[yellow]No Cardmarket offers found for product {pid}.[/yellow]")
        return

    table = Table(
        title=f"\nCardmarket Offers for Product {pid} ({len(offers)} listings found)",
        border_style="cyan",
    )
    table.add_column("Seller", style="bold white")
    table.add_column("Country", style="green")
    table.add_column("Type", style="dim")
    table.add_column("Cond", justify="center", style="magenta")
    table.add_column("Lang", justify="center", style="cyan")
    table.add_column("Attrs", justify="center", style="yellow")
    table.add_column("Qty", justify="right", style="dim")
    table.add_column("Price", justify="right", style="bold yellow")
    table.add_column("Comment", style="italic dim", max_width=30)

    for o in offers:
        attrs = []
        if o.is_foil:
            attrs.append("Foil")
        if o.is_signed:
            attrs.append("Signed")
        if o.is_first_edition:
            attrs.append("1st")
        if o.is_playset:
            attrs.append("Playset")
        if o.is_reverse_holo:
            attrs.append("Reverse")
        attr_str = ", ".join(attrs) if attrs else "-"

        table.add_row(
            o.seller_name,
            o.seller_country or "-",
            o.seller_type or "-",
            o.condition or "-",
            (o.language or "-").upper(),
            attr_str,
            str(o.item_count),
            f"{o.price:.2f} €",
            (o.comment or "")[:30],
        )

    console.print(table)

    if getattr(args, "save_db", False):
        engine = get_engine(getattr(args, "db_url", None))
        saved = upsert_cardmarket_offers(offers, product_id=pid, engine=engine)
        console.print(f"\n[bold green]✓ Successfully saved {saved} offer(s) to database table 'CardmarketOffers'[/bold green]")


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    config = load_and_filter_config(args)

    if args.subcommand == "scrape":
        run_scrape(args, config)
    elif args.subcommand == "assign":
        run_assign(args, config)
    elif args.subcommand == "cardmarket-offers":
        run_cardmarket_offers(args)
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


if __name__ == "__main__":
    main()
