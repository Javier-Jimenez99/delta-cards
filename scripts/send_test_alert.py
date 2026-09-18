#!/usr/bin/env python3
"""Script to dispatch test deal alerts to Discord channels."""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from delta_cards.alerts.service import (
    build_discord_embed,
    evaluate_alert_condition,
    send_discord_webhook,
    extract_alert_header,
)
from delta_cards.alerts.templates import render_alert_text, resolve_channel_key
from delta_cards.models import OfferListing
from delta_cards.settings import Settings, get_settings

console = Console()
err_console = Console(stderr=True)

CHANNELS_CONFIG = {
    "pokemon_cards": {
        "name": "Pokémon - Cartas Sueltas",
        "game": "pokemon",
        "product_type": "single_card",
        "webhook_attr": "pokemon_cards_webhook",
        "sample": {
            "wallapop": {
                "id": "test_pkmn_card_wp",
                "title": "Charizard VMAX Shiny Destinies SV107/SV122 (Alerta Prueba Wallapop)",
                "price": 60.0,
                "market_price": 140.0,
                "avg_price": 145.0,
                "low_price": 130.0,
                "link": "https://es.wallapop.com/item/charizard-vmax-shiny-sv107",
                "image_url": "https://images.pokemontcg.io/swsh45sv/SV107_hires.png",
                "cardmarket_product_id": "539281",
                "cardmarket_url": "https://www.cardmarket.com/es/Pokemon/Products/Singles/Shining-Fates/Charizard-VMAX-V2-SV107",
            },
            "vinted": {
                "id": "test_pkmn_card_vt",
                "title": "Pikachu Illustrator Promo Holo (Alerta Prueba Vinted)",
                "price": 45.0,
                "market_price": 110.0,
                "avg_price": 115.0,
                "low_price": 105.0,
                "link": "https://www.vinted.es/items/12345678-pikachu-promo",
                "image_url": "https://images.pokemontcg.io/swsh45sv/SV107_hires.png",
                "cardmarket_product_id": "539281",
                "cardmarket_url": "https://www.cardmarket.com/es/Pokemon/Products/Singles/Shining-Fates/Charizard-VMAX-V2-SV107",
            },
        },
    },
    "pokemon_boosters": {
        "name": "Pokémon - Sobres / Blisters",
        "game": "pokemon",
        "product_type": "booster_pack",
        "webhook_attr": "pokemon_booster_webhook",
        "sample": {
            "wallapop": {
                "id": "test_pkmn_booster_wp",
                "title": "Sobre 151 Scarlet & Violet Español Sellado (Alerta Prueba Wallapop)",
                "price": 4.50,
                "market_price": 8.50,
                "avg_price": 9.00,
                "low_price": 8.00,
                "link": "https://es.wallapop.com/item/sobre-pokemon-151-sellado",
                "image_url": "https://images.pokemontcg.io/sv3pt5/logo.png",
                "cardmarket_product_id": "733496",
                "cardmarket_url": "https://www.cardmarket.com/es/Pokemon/Products/Booster-Packs/Pokemon-151-Booster",
            },
            "vinted": {
                "id": "test_pkmn_booster_vt",
                "title": "Blister Pack Prismatic Evolutions Sellado (Alerta Prueba Vinted)",
                "price": 4.00,
                "market_price": 8.00,
                "avg_price": 8.50,
                "low_price": 7.50,
                "link": "https://www.vinted.es/items/87654321-sobre-prismatic-evolutions",
                "image_url": "https://images.pokemontcg.io/sv3pt5/logo.png",
                "cardmarket_product_id": "733496",
                "cardmarket_url": "https://www.cardmarket.com/es/Pokemon/Products/Booster-Packs/Pokemon-151-Booster",
            },
        },
    },
    "pokemon_boxes": {
        "name": "Pokémon - Cajas / Displays / ETBs",
        "game": "pokemon",
        "product_type": "etb",
        "webhook_attr": "pokemon_booster_box_webhook",
        "sample": {
            "wallapop": {
                "id": "test_pkmn_box_wp",
                "title": "Caja Elite Trainer Box (ETB) 151 Sellada Español (Alerta Prueba Wallapop)",
                "price": 45.0,
                "market_price": 90.0,
                "avg_price": 95.0,
                "low_price": 85.0,
                "link": "https://es.wallapop.com/item/etb-151-espanol-sellada",
                "image_url": "https://images.pokemontcg.io/sv3pt5/logo.png",
                "cardmarket_product_id": "733501",
                "cardmarket_url": "https://www.cardmarket.com/es/Pokemon/Products/Elite-Trainer-Boxes/Pokemon-151-Elite-Trainer-Box",
            },
            "vinted": {
                "id": "test_pkmn_box_vt",
                "title": "Booster Bundle Prismatic Evolutions 6 Sobres (Alerta Prueba Vinted)",
                "price": 28.0,
                "market_price": 55.0,
                "avg_price": 58.0,
                "low_price": 50.0,
                "link": "https://www.vinted.es/items/33445566-booster-bundle-pokemon",
                "image_url": "https://images.pokemontcg.io/sv3pt5/logo.png",
                "cardmarket_product_id": "733501",
                "cardmarket_url": "https://www.cardmarket.com/es/Pokemon/Products/Elite-Trainer-Boxes/Pokemon-151-Elite-Trainer-Box",
            },
        },
    },
    "magic_cards": {
        "name": "Magic - Cartas Sueltas",
        "game": "magic",
        "product_type": "single_card",
        "webhook_attr": "magic_cards_webhook",
        "sample": {
            "wallapop": {
                "id": "test_mtg_card_wp",
                "title": "The One Ring Foil Borderless 0269 (Alerta Prueba Wallapop)",
                "price": 35.0,
                "market_price": 105.0,
                "avg_price": 110.0,
                "low_price": 95.0,
                "link": "https://es.wallapop.com/item/the-one-ring-foil-0269",
                "image_url": "https://cards.scryfall.io/normal/front/d/5/d5806e68-1054-458e-866d-1f2470f682b2.jpg?1783916239",
                "cardmarket_product_id": "718588",
                "cardmarket_url": "https://www.cardmarket.com/es/Magic/Products/Singles/The-Lord-of-the-Rings-Tales-of-Middle-earth/The-One-Ring-V2",
            },
            "vinted": {
                "id": "test_mtg_card_vt",
                "title": "Sheoldred, the Apocalypse Foil (Alerta Prueba Vinted)",
                "price": 25.0,
                "market_price": 75.0,
                "avg_price": 82.0,
                "low_price": 70.0,
                "link": "https://www.vinted.es/items/44556677-sheoldred-foil",
                "image_url": "https://cards.scryfall.io/normal/front/d/6/d67be074-cdd4-41d9-ac89-0a0456c4e4b2.jpg?1783921327",
                "cardmarket_product_id": "671850",
                "cardmarket_url": "https://www.cardmarket.com/es/Magic/Products/Singles/Dominaria-United/Sheoldred-the-Apocalypse",
            },
        },
    },
    "magic_boosters": {
        "name": "Magic - Sobres / Blisters",
        "game": "magic",
        "product_type": "booster_pack",
        "webhook_attr": "magic_booster_webhook",
        "sample": {
            "wallapop": {
                "id": "test_mtg_booster_wp",
                "title": "Sobre Play Booster Modern Horizons 3 Español (Alerta Prueba Wallapop)",
                "price": 5.50,
                "market_price": 11.00,
                "avg_price": 11.50,
                "low_price": 10.00,
                "link": "https://es.wallapop.com/item/sobre-mh3-play-booster",
                "image_url": "https://www.magicbarcelona.net/wp-content/uploads/2024/04/MH3_PLAY_EN.png",
                "cardmarket_product_id": "766299",
                "cardmarket_url": "https://www.cardmarket.com/es/Magic/Products/Booster-Packs/Modern-Horizons-3-Play-Booster",
            },
            "vinted": {
                "id": "test_mtg_booster_vt",
                "title": "Sobre Collector Booster Bloomburrow Sellado (Alerta Prueba Vinted)",
                "price": 14.00,
                "market_price": 28.00,
                "avg_price": 29.00,
                "low_price": 25.00,
                "link": "https://www.vinted.es/items/55667788-bloomburrow-collector",
                "image_url": "https://www.magicbarcelona.net/wp-content/uploads/2024/06/BLB_CLT_EN.png",
                "cardmarket_product_id": "774902",
                "cardmarket_url": "https://www.cardmarket.com/es/Magic/Products/Booster-Packs/Bloomburrow-Collector-Booster",
            },
        },
    },
    "magic_boxes": {
        "name": "Magic - Cajas / Displays / Bundles",
        "game": "magic",
        "product_type": "booster_box",
        "webhook_attr": "magic_booster_box_webhook",
        "sample": {
            "wallapop": {
                "id": "test_mtg_box_wp",
                "title": "Booster Box Modern Horizons 3 Display Sellada (Alerta Prueba Wallapop)",
                "price": 175.0,
                "market_price": 260.0,
                "avg_price": 265.0,
                "low_price": 245.0,
                "link": "https://es.wallapop.com/item/display-mh3-booster-box-sellada",
                "image_url": "https://www.alcalacomics.com/wp-content/uploads/2024/05/195166253596-1.jpg",
                "cardmarket_product_id": "766295",
                "cardmarket_url": "https://www.cardmarket.com/es/Magic/Products/Booster-Boxes/Modern-Horizons-3-Play-Booster-Box",
            },
            "vinted": {
                "id": "test_mtg_box_vt",
                "title": "Bundle Bloomburrow Sellado (Alerta Prueba Vinted)",
                "price": 45.0,
                "market_price": 75.0,
                "avg_price": 78.0,
                "low_price": 70.0,
                "link": "https://www.vinted.es/items/66778899-bloomburrow-bundle",
                "image_url": "https://www.magicbarcelona.net/wp-content/uploads/2024/06/BLB_BNDL_EN.png",
                "cardmarket_product_id": "774900",
                "cardmarket_url": "https://www.cardmarket.com/es/Magic/Products/Bundles/Bloomburrow-Bundle",
            },
        },
    },
}

CHANNEL_ALIASES = {
    # Shortcuts
    "pkmn_cards": "pokemon_cards",
    "pkmn-cards": "pokemon_cards",
    "pokemon-cards": "pokemon_cards",
    "pkmn_boosters": "pokemon_boosters",
    "pkmn-boosters": "pokemon_boosters",
    "pokemon-boosters": "pokemon_boosters",
    "pkmn_boxes": "pokemon_boxes",
    "pkmn-boxes": "pokemon_boxes",
    "pokemon-boxes": "pokemon_boxes",
    "mtg_cards": "magic_cards",
    "mtg-cards": "magic_cards",
    "magic-cards": "magic_cards",
    "mtg_boosters": "magic_boosters",
    "mtg-boosters": "magic_boosters",
    "magic-boosters": "magic_boosters",
    "mtg_boxes": "magic_boxes",
    "mtg-boxes": "magic_boxes",
    "magic-boxes": "magic_boxes",
    "magic_sealed": "magic_boxes",
    "mtg_sealed": "magic_boxes",
    "magic-sealed": "magic_boxes",
}


def parse_target_channels(channel_arg: str) -> List[str]:
    """Resolves the user channel argument to a list of channel keys."""
    arg = channel_arg.strip().lower()
    if arg in ("all", "*"):
        return list(CHANNELS_CONFIG.keys())
    if arg in ("pokemon", "pkmn"):
        return ["pokemon_cards", "pokemon_boosters", "pokemon_boxes"]
    if arg in ("magic", "mtg"):
        return ["magic_cards", "magic_boosters", "magic_boxes"]

    resolved = CHANNEL_ALIASES.get(arg, arg)
    if resolved in CHANNELS_CONFIG:
        return [resolved]

    return []


def send_channel_test_alert(
    channel_key: str,
    platform: str,
    settings: Settings,
    dry_run: bool = False,
) -> bool:
    """Sends a single test alert to the designated channel webhook."""
    conf = CHANNELS_CONFIG[channel_key]
    webhook_url = getattr(settings, conf["webhook_attr"], None)

    # Fallback to general webhook if specific one is not set
    if not webhook_url:
        if "pokemon" in channel_key:
            webhook_url = settings.pokemon_cards_webhook or settings.discord_webhook_url
        else:
            webhook_url = settings.magic_cards_webhook or settings.discord_webhook_url

    sample_data = conf["sample"].get(platform, conf["sample"]["wallapop"])

    offer = OfferListing(
        id=sample_data["id"],
        title=sample_data["title"],
        price=sample_data["price"],
        game=conf["game"],
        product_type=conf["product_type"],
        platform=platform,
        location="Madrid, Spain",
        link=sample_data["link"],
        image_url=sample_data.get("image_url"),
        cardmarket_product_id=sample_data.get("cardmarket_product_id"),
        cardmarket_url=sample_data.get("cardmarket_url"),
        cardmarket_low_price=sample_data.get("low_price"),
        cardmarket_avg_price=sample_data.get("avg_price"),
    )

    cond = evaluate_alert_condition(
        offer_price=offer.price,
        market_price=sample_data["market_price"],
        settings=settings,
    )

    # Render alert message using the channel template
    msg_text = render_alert_text(
        offer=offer,
        result=cond,
        channel_key=channel_key,
        emoji=settings.alert_wild_offer_emoji,
        templates_dir=Path(settings.alert_templates_dir),
    )

    header = extract_alert_header(msg_text)
    embed = build_discord_embed(offer, cond, description=msg_text)

    console.print(f"\n[bold cyan]▶ Canal:[/] [bold]{conf['name']}[/] ([dim]{channel_key}[/dim])")
    console.print(f"  • Plataforma: [yellow]{platform.capitalize()}[/yellow]")
    console.print(f"  • Oferta: [green]{offer.title}[/green]")
    console.print(f"  • Precio: [bold]{offer.price:.2f} €[/bold] (Mercado: {sample_data['market_price']:.2f} € | Descuento: -{cond.savings_percentage:.1f}%)")

    if not webhook_url:
        err_console.print(f"  [bold red]✗ Error:[/] No hay webhook configurado en .env para {conf['webhook_attr'].upper()}. Saltando.")
        return False

    masked = webhook_url[:40] + "..." if len(webhook_url) > 40 else webhook_url
    console.print(f"  • Webhook URL: [dim]{masked}[/dim]")

    if dry_run:
        console.print("  [bold magenta]• Modo dry-run activado (no se envía a Discord).[/bold magenta]")
        console.print(f"  [bold cyan]Cabecera (fuera del bloque verde, encima):[/bold cyan] {header}")
        console.print(f"  [bold cyan]Título (title):[/bold cyan] {embed.get('title')}")
        if embed.get("thumbnail"):
            console.print(f"  [bold cyan]Miniatura (thumbnail):[/bold cyan] {embed['thumbnail']['url']}")
        console.print("  [bold cyan]Columnas (fields inline):[/bold cyan]")
        for f in embed.get("fields", []):
            inline_str = " [dim](inline: True)[/dim]" if f.get("inline") else ""
            console.print(f"    • [bold]{f['name']}:[/bold] {f['value']}{inline_str}")
        return True

    ok = send_discord_webhook(
        webhook_url=webhook_url,
        embeds=[embed],
        content=header,
        delay_between_offers=0,
    )

    if ok:
        console.print("  [bold green]✓ Alerta de prueba enviada con éxito.[/bold green]")
        return True
    else:
        err_console.print("  [bold red]✗ Error al enviar el webhook a Discord.[/bold red]")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Envía alertas de prueba a los canales de Discord (Pokémon y Magic)."
    )
    parser.add_argument(
        "channel",
        nargs="?",
        default="all",
        help=(
            "Canal de destino: 'all' (todos), 'pokemon', 'magic', "
            "'pokemon_cards', 'pokemon_boosters', 'pokemon_boxes', "
            "'magic_cards', 'magic_boosters', 'magic_boxes'."
        ),
    )
    parser.add_argument(
        "--platform",
        "-p",
        choices=["wallapop", "vinted"],
        default="wallapop",
        help="Plataforma de prueba a simular (wallapop o vinted). Por defecto: wallapop.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra el contenido de la alerta sin enviarla a Discord.",
    )

    args = parser.parse_args()
    target_channels = parse_target_channels(args.channel)

    if not target_channels:
        err_console.print(
            f"[bold red]Canal desconocido:[/] '{args.channel}'.\n"
            f"Opciones válidas: all, pokemon, magic, pokemon_cards, pokemon_boosters, pokemon_boxes, magic_cards, magic_boosters, magic_boxes."
        )
        sys.exit(1)

    settings = get_settings()

    console.print(
        f"[bold blue]═══════════════════════════════════════════════════════════[/bold blue]\n"
        f"[bold white]  Delta Cards - Envío de Alertas de Prueba a Discord [/bold white]\n"
        f"[bold blue]═══════════════════════════════════════════════════════════[/bold blue]"
    )
    console.print(f"Canales seleccionados ({len(target_channels)}): [cyan]{', '.join(target_channels)}[/cyan]\n")

    success_count = 0
    total = len(target_channels)

    for ch in target_channels:
        if send_channel_test_alert(ch, args.platform, settings, dry_run=args.dry_run):
            success_count += 1

    console.print(f"\n[bold]Resumen:[/] {success_count}/{total} alertas procesadas correctamente.")
    if success_count < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
