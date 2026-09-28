"""Channel-based alert message template manager with dynamic placeholders.

Extracts alert message content into separate text files per Discord channel,
allowing custom wording and formatting with placeholders ({title}, {price},
{market_price}, {discount_percentage}, {url}, etc.).
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
import re
import string
from typing import Any, Dict, Optional

from ..models import OfferListing

logger = logging.getLogger(__name__)

# Default message templates if files are missing or unreadable
POKEMON_ALERT_TEMPLATE: str = (
    "<:pokeball_light:1548052080357412995> ¡Ha aparecido una oferta salvaje!\n"
    "**[Oferta: {title}]({url})**\n\n"
    "{platform_emoji} **{platform}:** {price}\n"
    "<:cardmarket:1548103777750487056> **Precio medio:** {avg_price}\n"
    "<:cardmarket:1548103777750487056> **Precio mínimo:** {low_price}\n"
    "📦 **Tipo:** {product_type}\n"
    "📍 **Ubicación:** {location}\n"
    "🔗 **Enlaces:** [Oferta en {platform}]({url}) • [Producto en Cardmarket]({cardmarket_url})"
)

MAGIC_ALERT_TEMPLATE: str = (
    "🔮 ¡Ha aparecido una oferta salvaje de Magic!\n"
    "**[Oferta: {title}]({url})**\n\n"
    "{platform_emoji} **{platform}:** {price}\n"
    "<:cardmarket:1548103777750487056> **Precio medio:** {avg_price}\n"
    "<:cardmarket:1548103777750487056> **Precio mínimo:** {low_price}\n"
    "📦 **Tipo:** {product_type}\n"
    "📍 **Ubicación:** {location}\n"
    "🔗 **Enlaces:** [Oferta en {platform}]({url}) • [Producto en Cardmarket]({cardmarket_url})"
)

DEFAULT_ALERT_TEMPLATE: str = (
    "✨ ¡Ha aparecido una oferta salvaje!\n"
    "**[Oferta: {title}]({url})**\n\n"
    "{platform_emoji} **{platform}:** {price}\n"
    "<:cardmarket:1548103777750487056> **Precio medio:** {avg_price}\n"
    "<:cardmarket:1548103777750487056> **Precio mínimo:** {low_price}\n"
    "📦 **Tipo:** {product_type}\n"
    "📍 **Ubicación:** {location}\n"
    "🔗 **Enlaces:** [Oferta en {platform}]({url}) • [Producto en Cardmarket]({cardmarket_url})"
)

FALLBACK_TEMPLATES: Dict[str, str] = {
    "pokemon_cards": POKEMON_ALERT_TEMPLATE,
    "pokemon_boosters": POKEMON_ALERT_TEMPLATE,
    "pokemon_boxes": POKEMON_ALERT_TEMPLATE,
    "magic_cards": MAGIC_ALERT_TEMPLATE,
    "magic_boosters": MAGIC_ALERT_TEMPLATE,
    "magic_boxes": MAGIC_ALERT_TEMPLATE,
    "magic_sealed": MAGIC_ALERT_TEMPLATE,
    "default": DEFAULT_ALERT_TEMPLATE,
}


def resolve_channel_key(game: Optional[str] = None, product_type: Optional[str] = None) -> str:
    """Deterministically resolves the alert channel identifier based on game and product type."""
    g = str(game or "pokemon").lower().strip()
    pt = str(product_type or "").lower().strip()

    if g in ("magic", "mtg"):
        if pt in ("booster_pack", "booster", "blister", "pack", "sobre"):
            return "magic_boosters"
        if pt in (
            "booster_box",
            "display",
            "etb",
            "elite_trainer_box",
            "caja",
            "box",
            "tin",
            "box_set",
            "deck",
            "bundle",
            "booster_bundle",
            "non_single",
            "sealed",
        ):
            return "magic_boxes"
        return "magic_cards"

    if g in ("pokemon", "pkmn"):
        if pt in ("booster_pack", "booster", "blister", "pack", "sobre"):
            return "pokemon_boosters"

        if pt in (
            "booster_box",
            "display",
            "etb",
            "elite_trainer_box",
            "caja",
            "box",
            "tin",
            "box_set",
            "deck",
            "bundle",
            "booster_bundle",
            "non_single",
            "sealed",
        ):
            return "pokemon_boxes"

        return "pokemon_cards"

    return "default"


class SafeTemplateFormatter(string.Formatter):
    """String formatter that retains missing {placeholders} instead of raising KeyError."""

    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            if key in kwargs:
                return kwargs[key]
            return f"{{{key}}}"
        return super().get_value(key, args, kwargs)


class AlertTemplateManager:
    """Loads and renders channel alert text templates from disk with dynamic placeholders."""

    _instance: Optional["AlertTemplateManager"] = None

    def __init__(self, templates_dir: Optional[Path] = None):
        if templates_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            templates_dir = project_root / "config/templates/alerts"
        self.templates_dir = Path(templates_dir)
        self.formatter = SafeTemplateFormatter()
        self._template_cache: Dict[str, str] = {}
        self._cache_mtimes: Dict[str, float] = {}

    @classmethod
    def get_instance(cls, templates_dir: Optional[Path] = None) -> "AlertTemplateManager":
        """Returns singleton instance of AlertTemplateManager."""
        if templates_dir is not None:
            if cls._instance is None or cls._instance.templates_dir != Path(templates_dir):
                return cls(templates_dir=templates_dir)
        if cls._instance is None:
            cls._instance = cls(templates_dir=templates_dir)
        return cls._instance

    def load_template(self, channel_key: str) -> str:
        """Loads template text for a channel from disk, falling back to default or built-in templates."""
        template_file = self.templates_dir / f"{channel_key}.txt"

        if template_file.exists():
            try:
                mtime = template_file.stat().st_mtime
                if channel_key not in self._template_cache or self._cache_mtimes.get(channel_key) != mtime:
                    with open(template_file, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    self._template_cache[channel_key] = text
                    self._cache_mtimes[channel_key] = mtime
                return self._template_cache[channel_key]
            except Exception as exc:
                logger.warning(f"Error reading template file {template_file}: {exc}")

        # Fallback to default.txt
        default_file = self.templates_dir / "default.txt"
        if default_file.exists():
            try:
                with open(default_file, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                pass

        return FALLBACK_TEMPLATES.get(channel_key, FALLBACK_TEMPLATES["default"])

    def build_placeholders(
        self,
        offer: OfferListing,
        result: Optional[Any] = None,
        llm_verification: Optional[Any] = None,
        emoji: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Constructs the dictionary of placeholder variables for template substitution."""
        # Platform
        is_vinted = (getattr(offer, "platform", None) == "vinted") or ("vinted" in (offer.link or "").lower())
        platform_name = "Vinted" if is_vinted else "Wallapop"
        platform_emoji = "<:vinted:1549420019518013470>" if is_vinted else "<:wallapop:1548103689477038080>"

        # URL
        raw_url = offer.link or ""
        if not is_vinted:
            offer_url = raw_url.replace("https://wallapop.com/", "https://es.wallapop.com/")
        else:
            offer_url = raw_url

        # Game
        game_raw = (getattr(offer, "game", None) or "pokemon").lower()
        is_magic = game_raw in ("magic", "mtg")
        game_name = "Magic" if is_magic else "Pokémon"

        # Cardmarket URL
        cm_id = str(offer.cardmarket_product_id or "").strip()
        cm_url = offer.cardmarket_url
        if not cm_url and cm_id:
            game_category = "Magic" if is_magic else "Pokemon"
            cm_url = f"https://www.cardmarket.com/es/{game_category}/Products?idProduct={cm_id}"

        # Pricing
        price_val = getattr(offer, "price", None)
        price_str = f"{price_val:.2f} €" if price_val is not None else "N/A"

        trend_p = getattr(offer, "cardmarket_trend_price", None)
        avg_p = getattr(offer, "cardmarket_avg_price", None)
        low_p = getattr(offer, "cardmarket_low_price", None)

        # Primary reference market price: prioritize trend price, fallback to avg_price or result.market_price
        ref_market_p = trend_p if trend_p is not None else avg_p
        if ref_market_p is None and result:
            ref_market_p = getattr(result, "market_price", None)

        # For display: avg_price displays the market reference price
        display_avg = ref_market_p
        display_low = low_p

        # Enforce that minimum price is never greater than average price
        if display_low is not None and display_avg is not None and display_low > display_avg:
            display_low, display_avg = min(display_low, display_avg), max(display_low, display_avg)

        avg_str = f"{display_avg:.2f} €" if display_avg is not None else "N/A"
        low_str = f"{display_low:.2f} €" if display_low is not None else "N/A"
        trend_str = f"{trend_p:.2f} €" if trend_p is not None else "N/A"
        market_str = f"{ref_market_p:.2f} €" if ref_market_p is not None else avg_str

        # Savings / Discounts
        savings_pct = getattr(result, "savings_percentage", None) if result else None
        savings_abs = getattr(result, "savings_absolute", None) if result else None

        if savings_pct is None and price_val is not None and ref_market_p is not None and ref_market_p > 0:
            savings_pct = max(0.0, ((ref_market_p - price_val) / ref_market_p) * 100.0)
        if savings_abs is None and price_val is not None and ref_market_p is not None:
            savings_abs = max(0.0, ref_market_p - price_val)

        pct_num_str = f"{savings_pct:.1f}" if savings_pct is not None else "N/A"
        pct_formatted_str = f"{savings_pct:.1f}%" if savings_pct is not None else "N/A"
        abs_str = f"{savings_abs:.2f} €" if savings_abs is not None else "N/A"

        # Product Type Translation
        pt_raw = str(getattr(offer, "product_type", "") or "").lower().strip()
        type_labels = {
            "single_card": "Carta",
            "card": "Carta",
            "booster_pack": "Sobre",
            "booster": "Sobre",
            "blister": "Blister",
            "booster_box": "Booster Box / Display",
            "etb": "Elite Trainer Box",
            "tin": "Lata",
            "box_set": "Caja Colección",
            "deck": "Mazo",
            "bundle": "Bundle / Fat Pack",
            "non_single": "Producto Sellado",
        }
        product_type_label = type_labels.get(pt_raw, pt_raw.replace("_", " ").title() if pt_raw else "Producto")

        # Emoji
        if not emoji:
            emoji = "🔮" if is_magic else "<:pokeball_light:1548052080357412995>"

        # LLM Verification
        llm_str = "N/A"
        if llm_verification:
            score = getattr(llm_verification, "confidence_score", None)
            if score is None and isinstance(llm_verification, dict):
                score = llm_verification.get("confidence_score")
            confirmed = getattr(llm_verification, "confirmed_count", None)
            if confirmed is None and isinstance(llm_verification, dict):
                confirmed = llm_verification.get("confirmed_count")
            total = getattr(llm_verification, "total_questions", 20)
            if total is None and isinstance(llm_verification, dict):
                total = llm_verification.get("total_questions", 20)

            if confirmed is not None and total and total > 0:
                pass_pct = (confirmed / total) * 100.0
            elif score is not None:
                pass_pct = float(score)
            else:
                pass_pct = 0.0

            if pass_pct >= 67.0:
                llm_str = f"{round(pass_pct)}% ✅"
            elif pass_pct <= 33.0:
                llm_str = f"{round(pass_pct)}% ❌"
            else:
                llm_str = f"{round(pass_pct)}% ⚠️"

        return {
            "title": offer.title or "",
            "price": price_str,
            "raw_price": price_val if price_val is not None else 0.0,
            "currency": offer.currency or "EUR",
            "market_price": market_str,
            "avg_price": avg_str,
            "trend_price": trend_str,
            "low_price": low_str,
            "discount_percentage": pct_num_str,
            "discount_percentage_formatted": pct_formatted_str,
            "saving_percentage": pct_num_str,
            "saving_percentage_formatted": pct_formatted_str,
            "discount_amount": abs_str,
            "saving_amount": abs_str,
            "platform": platform_name,
            "platform_emoji": platform_emoji,
            "product_type": product_type_label,
            "location": offer.location or "N/A",
            "url": offer_url,
            "link": offer_url,
            "cardmarket_url": cm_url or "",
            "cardmarket_id": cm_id,
            "cardmarket_emoji": "<:cardmarket:1548103777750487056>",
            "cardmarket_link": f"[Producto en Cardmarket]({cm_url})" if cm_url else "",
            "offer_link": f"[Oferta en {platform_name}]({offer_url})",
            "links": f"[Oferta en {platform_name}]({offer_url})" + (f" • [Producto en Cardmarket]({cm_url})" if cm_url else ""),
            "game": game_name,
            "emoji": emoji,
            "llm_verification": llm_str,
        }

    def render(
        self,
        channel_key: str,
        offer: OfferListing,
        result: Optional[Any] = None,
        llm_verification: Optional[Any] = None,
        emoji: Optional[str] = None,
    ) -> str:
        """Renders the alert template for the specified channel with offer variables."""
        template_text = self.load_template(channel_key)
        variables = self.build_placeholders(
            offer=offer,
            result=result,
            llm_verification=llm_verification,
            emoji=emoji,
        )

        try:
            return self.formatter.format(template_text, **variables)
        except Exception as exc:
            logger.warning(f"Error formatting template '{channel_key}': {exc}. Falling back to regex.")
            rendered = template_text
            for k, v in variables.items():
                rendered = re.sub(r"\{" + re.escape(k) + r"\}", str(v), rendered)
            return rendered


def render_alert_text(
    offer: OfferListing,
    result: Optional[Any] = None,
    channel_key: Optional[str] = None,
    llm_verification: Optional[Any] = None,
    emoji: Optional[str] = None,
    templates_dir: Optional[Path] = None,
) -> str:
    """Convenience helper to render alert message text for an offer and channel."""
    manager = AlertTemplateManager.get_instance(templates_dir=templates_dir)
    if not channel_key:
        channel_key = resolve_channel_key(
            game=getattr(offer, "game", None),
            product_type=getattr(offer, "product_type", None),
        )
    return manager.render(
        channel_key=channel_key,
        offer=offer,
        result=result,
        llm_verification=llm_verification,
        emoji=emoji,
    )
