"""Playwright browser simulation fetcher for bypassing Cloudflare bot challenges."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from .proxy_rotator import ProxyRotator

logger = logging.getLogger(__name__)

# Ensure user local libraries (~/.local/lib) are loaded for browser dependencies
_LOCAL_LIB = Path.home() / ".local" / "lib"
if _LOCAL_LIB.is_dir():
    _cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if str(_LOCAL_LIB) not in _cur_ld:
        os.environ["LD_LIBRARY_PATH"] = f"{_LOCAL_LIB}:{_cur_ld}" if _cur_ld else str(_LOCAL_LIB)

CLOUDFLARE_CHALLENGE_TITLES = [
    "just a moment...",
    "attention required! | cloudflare",
    "ddos protection by cloudflare",
    "cloudflare",
]

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


class PlaywrightFetchError(Exception):
    """Raised when Playwright browser simulation fails to retrieve page content."""
    pass


class PlaywrightFetcher:
    """Simulates a realistic browser using Playwright to bypass Cloudflare challenges."""

    def __init__(
        self,
        headless: bool = True,
        timeout: float = 30.0,
        user_agent: str = DEFAULT_USER_AGENT,
        ws_endpoint: Optional[str] = None,
        cdp_url: Optional[str] = None,
        browser_type: str = "firefox",
    ):
        """Initializes the Playwright fetcher.

        Args:
            headless: Whether to run the browser in headless mode.
            timeout: Default timeout in seconds for page navigation and challenge resolution.
            user_agent: Realistic user agent string.
            ws_endpoint: Optional WebSocket endpoint for remote browser (e.g. ws://localhost:3000).
            cdp_url: Optional Chrome DevTools Protocol URL (e.g. http://localhost:9222).
            browser_type: Default browser engine ('firefox' or 'chromium').
        """
        self.headless = headless
        self.timeout = timeout
        self.user_agent = user_agent
        self.ws_endpoint = ws_endpoint or os.environ.get("PLAYWRIGHT_WS_ENDPOINT")
        self.cdp_url = cdp_url or os.environ.get("PLAYWRIGHT_CDP_URL")
        self.browser_type = browser_type or os.environ.get("PLAYWRIGHT_BROWSER", "firefox")

    async def is_available(self) -> bool:
        """Checks if Playwright and browser dependencies are ready to launch."""
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                if self.ws_endpoint:
                    b = await p.chromium.connect(self.ws_endpoint, timeout=3000)
                    await b.close()
                    return True
                if self.cdp_url:
                    b = await p.chromium.connect_over_cdp(self.cdp_url, timeout=3000)
                    await b.close()
                    return True
                b = await p.firefox.launch(headless=True)
                await b.close()
                return True
        except Exception:
            return False

    async def fetch_html(
        self,
        url: str,
        proxy: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Navigates to URL using a headless/simulated browser and returns rendered HTML.

        Args:
            url: The Cardmarket URL to fetch.
            proxy: Optional proxy URL string (e.g. 'http://user:pass@host:port').
            timeout: Override timeout in seconds.

        Returns:
            The rendered HTML content.

        Raises:
            PlaywrightFetchError: If browser fails to load or challenge cannot be passed.
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise PlaywrightFetchError(
                "Playwright is not installed. Run 'uv add playwright && uv run playwright install chromium'."
            ) from exc

        effective_timeout = timeout or self.timeout
        timeout_ms = int(effective_timeout * 1000)

        playwright_proxy = ProxyRotator.to_playwright_proxy(proxy) if proxy else None
        if playwright_proxy:
            logger.info("Launching Playwright browser with proxy: %s", playwright_proxy.get("server"))
        else:
            logger.info("Launching Playwright browser (direct connection)...")

        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-infobars",
            "--window-size=1920,1080",
            "--disable-dev-shm-usage",
        ]

        async with async_playwright() as p:
            browser = None
            try:
                if self.ws_endpoint:
                    browser = await p.chromium.connect(self.ws_endpoint, timeout=timeout_ms)
                elif self.cdp_url:
                    browser = await p.chromium.connect_over_cdp(self.cdp_url, timeout=timeout_ms)
                else:
                    # Prefer Firefox as it has zero missing system dependencies on this platform
                    if self.browser_type == "firefox":
                        try:
                            browser = await p.firefox.launch(
                                headless=self.headless,
                                proxy=playwright_proxy,  # type: ignore
                            )
                        except Exception as f_err:
                            logger.debug("Firefox launch failed: %s; falling back to chromium", f_err)
                            browser = await p.chromium.launch(
                                headless=self.headless,
                                args=launch_args,
                                proxy=playwright_proxy,  # type: ignore
                            )
                    else:
                        try:
                            browser = await p.chromium.launch(
                                headless=self.headless,
                                args=launch_args,
                                proxy=playwright_proxy,  # type: ignore
                            )
                        except Exception as c_err:
                            logger.debug("Chromium launch failed: %s; falling back to firefox", c_err)
                            browser = await p.firefox.launch(
                                headless=self.headless,
                                proxy=playwright_proxy,  # type: ignore
                            )

                context = await browser.new_context(
                    user_agent=self.user_agent,
                    viewport={"width": 1920, "height": 1080},
                    locale="es-ES,es;q=0.9,en;q=0.8",
                    timezone_id="Europe/Madrid",
                )

                page = await context.new_page()
                page.set_default_timeout(timeout_ms)

                logger.debug("Navigating with Playwright to: %s", url)
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

                # Check if Cloudflare challenge is active
                page_title = (await page.title()).strip().lower()
                is_challenged = any(ch in page_title for ch in CLOUDFLARE_CHALLENGE_TITLES)

                if is_challenged:
                    logger.info("Cloudflare challenge detected in browser ('%s'). Waiting for resolution...", page_title)
                    start_time = asyncio.get_event_loop().time()
                    resolved = False

                    # Wait for challenge to resolve or article elements to appear
                    while (asyncio.get_event_loop().time() - start_time) < effective_timeout:
                        await asyncio.sleep(1.0)

                        # Attempt to click Turnstile checkbox if present in any frame
                        for f in page.frames:
                            try:
                                for selector in [
                                    "input[type='checkbox']",
                                    ".ctp-checkbox-label",
                                    "#challenge-stage",
                                    "label.ctp-checkbox-label",
                                ]:
                                    box = await f.query_selector(selector)
                                    if box:
                                        await box.click(timeout=1000)
                                        logger.debug("Attempted click on Turnstile checkbox.")
                                        break
                            except Exception:
                                pass

                        current_title = (await page.title()).strip().lower()
                        if not any(ch in current_title for ch in CLOUDFLARE_CHALLENGE_TITLES):
                            resolved = True
                            logger.info("Cloudflare challenge resolved ('%s')!", current_title)
                            break

                        # Check if article content has appeared in DOM despite title
                        has_articles = await page.evaluate(
                            "() => !!document.querySelector('.article-row, .table-body, .articles-table')"
                        )
                        if has_articles:
                            resolved = True
                            logger.info("Cardmarket listings found in DOM.")
                            break

                    if not resolved:
                        raise PlaywrightFetchError(
                            f"Cloudflare challenge did not resolve within {effective_timeout:.1f}s (title: '{page_title}')"
                        )

                # Wait a brief moment for dynamic hydration/rendering of article listings
                try:
                    await page.wait_for_selector(".article-row, .table-body, .articles-table, .alert-info", timeout=5000)
                except Exception:
                    pass

                content = await page.content()
                logger.info("Playwright successfully fetched %d characters from %s", len(content), url)
                return content

            finally:
                if browser is not None:
                    await browser.close()
