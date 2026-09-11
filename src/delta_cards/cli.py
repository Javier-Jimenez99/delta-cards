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
            console.print(f"[bold red]Error:[/] Configuration file not found at '{args.config}'", file=sys.stderr)
            sys.exit(1)

    try:
        config = load_config(config_path)
    except Exception as exc:
        console.print(f"[bold red]Error loading config:[/] {exc}", file=sys.stderr)
        sys.exit(1)

    # Filter by enabled flag from config
    config.search_profiles = [p for p in config.search_profiles if getattr(p, "enabled", True)]

    # Filter search profiles if requested
    if args.profile:
        selected_profiles = set(args.profile)
        config.search_profiles = [p for p in config.search_profiles if p.id in selected_profiles]
        if not config.search_profiles:
            console.print(f"[bold red]Error:[/] No profiles found matching: {', '.join(args.profile)}", file=sys.stderr)
            sys.exit(1)

    if args.type:
        selected_types = set(args.type)
        config.search_profiles = [p for p in config.search_profiles if p.product_type in selected_types]
        if not config.search_profiles:
            console.print(f"[bold red]Error:[/] No profiles found matching product type(s): {', '.join(args.type)}", file=sys.stderr)
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
            console.print(f"[bold red]Error updating database offers:[/] {exc}", file=sys.stderr)
            sys.exit(1)
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
            status.update(f"[bold cyan]Assigning TCGDex & Cardmarket IDs to {len(card_offers)} card offers...[/bold cyan]")
            assign_tcgdex_to_offers(offers)
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
                    deals_table = Table(
                        title=f"\n🔥 Deals Found (Wallapop < Cardmarket): {len(deals)}",
                        border_style="bold green",
                    )
                    deals_table.add_column("Title", style="cyan")
                    deals_table.add_column("Wallapop", justify="right", style="bold green")
                    deals_table.add_column("Min Cardmarket", justify="right", style="yellow")
                    deals_table.add_column("Savings", justify="right", style="bold magenta")
                    deals_table.add_column("Link", style="dim")

                    for off, pr in deals:
                        diff = (pr.min_price_all or 0.0) - off.price
                        deals_table.add_row(
                            off.title[:45],
                            f"{off.price:.2f} €",
                            f"{pr.min_price_all:.2f} €",
                            f"-{diff:.2f} €",
                            off.link,
                        )
                    console.print(deals_table)
                else:
                    console.print("\n[dim]No offers found with Wallapop price below Cardmarket minimum.[/dim]")
        except Exception as exc:
            logging.getLogger("delta_cards").debug(f"Error checking deals: {exc}")



if __name__ == "__main__":
    main()

