"""Command-line interface for delta-cards."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from .config_loader import load_config
from .database import save_offers_to_db
from .tcgdex import assign_tcgdex_to_offers, enrich_card_offers_in_db
from .wallapop import CardOffersScraper


console = Console()
err_console = Console(stderr=True)


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Wallapop for card offers (Pokemon sealed & single cards) using WallaPy."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/pokemon_cards_search.json",
        help="Path to JSON configuration file (default: config/pokemon_cards_search.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Override output directory specified in config",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="Override max items to retrieve per search profile (useful for quick testing)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip saving scraped offers to the PostgreSQL database",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="Override database connection URL (default: DATABASE_URL env var)",
    )
    parser.add_argument(
        "-p",
        "--profile",
        nargs="+",
        default=None,
        help="One or more profile IDs to run (e.g. -p single_card)",
    )
    parser.add_argument(
        "-t",
        "--type",
        nargs="+",
        default=None,
        help="One or more product types to run (e.g. -t single_card booster_box)",
    )
    parser.add_argument(
        "--scrape-prices",
        action="store_true",
        help="Scrape Cardmarket minimum prices (overall and Spanish sellers) for matched cards",
    )
    parser.add_argument(
        "--skip-tcgdex",
        action="store_true",
        help="Skip automatic assignment of TCGDex ID and Cardmarket Product ID to card offers",
    )
    parser.add_argument(
        "--assign-existing-cards",
        action="store_true",
        help="Enrich unassigned card offers in the database with TCGDex & Cardmarket IDs",
    )
    parser.add_argument(
        "--no-discord",
        action="store_true",
        help="Skip sending Discord alert notifications for scraped deals",
    )
    parser.add_argument(
        "--discord-webhook",
        type=str,
        default=None,
        help="Override Discord webhook URL specified in settings / .env",
    )
    parser.add_argument(
        "--test-discord",
        action="store_true",
        help="Send a test deal alert to Discord to verify the webhook connection and exit",
    )
    parser.add_argument(
        "--alert-all-deals",
        action="store_true",
        help="Send Discord alerts for all matching deals found in the database, including previously saved ones",
    )
    return parser.parse_args(args)




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


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    config_path = Path(args.config)
    if not config_path.is_file():
        # Try finding it relative to project root
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

    # Filter by enabled flag from config
    config.search_profiles = [p for p in config.search_profiles if getattr(p, "enabled", True)]

    # Filter search profiles if requested
    if args.profile:
        selected_profiles = set(args.profile)
        config.search_profiles = [p for p in config.search_profiles if p.id in selected_profiles]
        if not config.search_profiles:
            err_console.print(f"[bold red]Error:[/] No profiles found matching: {', '.join(args.profile)}")
            sys.exit(1)

    if args.type:
        selected_types = set(args.type)
        config.search_profiles = [p for p in config.search_profiles if p.product_type in selected_types]
        if not config.search_profiles:
            err_console.print(f"[bold red]Error:[/] No profiles found matching product type(s): {', '.join(args.type)}")
            sys.exit(1)

    # Apply CLI overrides
    if args.output:
        config.output_settings.output_dir = args.output
    if args.limit is not None:
        for profile in config.search_profiles:
            profile.max_items = args.limit

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

    if args.assign_existing_cards:
        console.print("[bold cyan]Enriching unassigned card offers in database with TCGDex & Cardmarket IDs...[/bold cyan]")
        try:
            from .database import get_engine
            engine = get_engine(args.db_url)
            count = enrich_card_offers_in_db(engine=engine)
            console.print(f"[bold green]✓ Successfully matched and updated {count} card offer(s) in the database.[/bold green]")
        except Exception as exc:
            err_console.print(f"[bold red]Error updating database offers:[/] {exc}")
            sys.exit(1)
    if args.test_discord:
        from .alerts import DiscordAlertService
        from .models import OfferListing
        from .settings import get_settings

        settings = get_settings()
        if args.discord_webhook:
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
        return

    scraper = CardOffersScraper(config)

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
                console.print(
                    f"   [dim]•[/dim] [white]'{query}'[/white] [dim]→[/dim] {retrieved} retrieved, {badge}"
                )

            elif event == "profile_done":
                total_kept = data["total_kept"]
                retrieved_total = data["retrieved_total"]
                console.print(
                    f"   [bold green]✓[/bold green] [dim]Profile completed:[/] [bold]{total_kept}[/bold] valid offers kept "
                    f"[dim]({retrieved_total} total evaluated)[/dim]"
                )
                status.update("[cyan]Proceeding to next step...[/cyan]")

        offers = asyncio.run(scraper.run(progress_callback=progress_callback))

        card_offers = [o for o in offers if o.product_type == "single_card"]
        tcgdex_assigned_count = 0
        if not args.skip_tcgdex and card_offers:
            status.stop()

            # Check database for offers that already exist
            already_in_db = set()
            if config.output_settings.save_db and not args.no_db:
                try:
                    from .database import get_engine, WallapopOffers
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

            # Offers already in DB or already assigned do NOT go into the assigning process
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
                    if getattr(offer, "tcgdex_id", None):
                        matched[0] += 1
                    pbar.set_description(f"Assigning new cards ({matched[0]}/{len(new_card_offers)} matched)")
                    pbar.update(1)
                    # Incrementally save this offer so progress is never lost if interrupted
                    if config.output_settings.save_db and not args.no_db:
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
        if config.output_settings.save_db and not args.no_db and offers:
            try:
                status.update("[bold cyan]Saving offers to PostgreSQL database...[/bold cyan]")
                db_saved_count = save_offers_to_db(offers, db_url=args.db_url)
            except Exception as exc:
                db_error = str(exc)

        # Optional: Scrape Cardmarket prices for matched cards
        cm_prices_scraped = 0
        if args.scrape_prices and card_offers:
            from .cardmarket import CardmarketScraper
            cm_scraper = CardmarketScraper()
            unique_pairs = list({(o.cardmarket_product_id, (o.language or "es").lower()) for o in card_offers if o.cardmarket_product_id})
            if unique_pairs:
                status.update(f"[bold cyan]Scraping Cardmarket prices for {len(unique_pairs)} card product(s)...[/bold cyan]")
                for pid, card_lang in unique_pairs:
                    try:
                        price_res = asyncio.run(cm_scraper.scrape_product_prices(pid, language=card_lang))
                        if config.output_settings.save_db and not args.no_db:
                            if price_res.min_price_all is not None or price_res.min_price_spain is not None:
                                from .database import upsert_cardmarket_price, get_engine
                                engine = get_engine(args.db_url)
                                upsert_cardmarket_price(
                                    product_id=price_res.cardmarket_product_id,
                                    language=price_res.language,
                                    min_price_all=price_res.min_price_all,
                                    min_price_spain=price_res.min_price_spain,
                                    engine=engine,
                                )
                                cm_prices_scraped += 1
                    except Exception as exc:
                        logging.getLogger("delta_cards").warning(f"Error scraping Cardmarket price for product {pid} ({card_lang}): {exc}")

    if not offers:
        console.print("\n[yellow]No offers matched the configured search and filter criteria.[/yellow]")
        return


    # Print summary table
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
    if not args.skip_tcgdex and card_offers:
        console.print(f"  • [bold]TCGDEX / CARDMARKET ENRICHMENT:[/bold] [green]{tcgdex_assigned_count}/{len(card_offers)} card offer(s) identified[/green]")
    if args.scrape_prices and card_offers:
        console.print(f"  • [bold]CARDMARKET PRICES:[/bold] [green]{cm_prices_scraped} card product price(s) scraped and saved[/green]")
    if db_saved_count is not None:
        console.print(f"  • [bold]DATABASE ('WallapopOffers'):[/bold] [green]{db_saved_count} offer(s) saved/updated[/green]")
    elif db_error:
        console.print(f"  • [bold yellow]DATABASE ('WallapopOffers'):[/bold yellow] [dim]Skipped ({db_error})[/dim]")

    # Check database for deals (Wallapop price < Cardmarket price)
    if config.output_settings.save_db and not args.no_db:
        try:
            from .database import get_engine
            from .database.models import WallapopOffers, CardmarketPrice
            from sqlmodel import select, Session, and_

            eng = get_engine(args.db_url)
            with Session(eng) as s:
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
                    current_offer_ids = {o.id for o in offers}
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
                        status_tag = "[bold green]LIVE[/bold green]" if off.id in current_offer_ids else "[dim]SAVED[/dim]"
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
                        tag = "[bold green][LIVE][/bold green]" if off.id in current_offer_ids else "[dim][PREVIOUSLY SAVED][/dim]"
                        console.print(f"  • {tag} [bold]{off.title}[/bold] ({off.price:.2f} € vs {pr.min_price_all:.2f} € CM):")
                        console.print(f"    [cyan underline]{deal_link}[/cyan underline]")
                else:
                    console.print("\n[dim]No offers found with Wallapop price below Cardmarket minimum.[/dim]")
        except Exception as exc:
            logging.getLogger("delta_cards").debug(f"Error checking deals: {exc}")

    # Dispatch Discord alerts if enabled
    if not args.no_discord:
        from .alerts import DiscordAlertService
        from .settings import get_settings

        app_settings = get_settings()
        if args.discord_webhook:
            app_settings = app_settings.model_copy(update={"discord_webhook_url": args.discord_webhook})

        ref_prices = {}
        db_deals_to_alert = []
        saved_deals_count = 0
        if config.output_settings.save_db and not args.no_db:
            try:
                from .database import get_engine
                from .database.models import CardmarketPrice, WallapopOffers
                from sqlmodel import select, Session, and_

                eng = get_engine(args.db_url)
                with Session(eng) as s:
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
                    current_ids = {o.id for o in offers}
                    saved_deals_count = sum(1 for off, _ in all_deals if off.id not in current_ids)

                    include_saved = app_settings.debug or args.alert_all_deals
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

        include_saved = app_settings.debug or args.alert_all_deals
        alert_svc = DiscordAlertService(settings=app_settings)
        target_offers = db_deals_to_alert if include_saved and db_deals_to_alert else offers
        alerts_sent = alert_svc.process_and_alert_offers(target_offers, reference_prices=ref_prices)
        if alerts_sent > 0:
            mode_desc = "including saved offers [DEBUG=true]" if app_settings.debug else ("all DB deals" if args.alert_all_deals else "live deals")
            console.print(f"  • [bold green]DISCORD ALERTS:[/] [bold]{alerts_sent}[/bold] alert notification(s) sent ({mode_desc})")
        elif app_settings.discord_webhook_url and app_settings.discord_alerts_enabled:
            note = f" ({saved_deals_count} existing saved deals skipped because DEBUG=false)" if saved_deals_count > 0 and not include_saved else ""
            console.print(f"  • [bold]DISCORD ALERTS:[/] [dim]Checked {len(target_offers)} live offers (0 met alert condition thresholds){note}[/dim]")



if __name__ == "__main__":
    main()

