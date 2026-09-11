"""Proxy rotation and health tracking for Cardmarket scraping."""

import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ProxyRotator:
    """Manages a pool of HTTP/HTTPS/SOCKS proxies with rotation and failure tracking.

    Supports round-robin and random rotation strategies, tracks temporary
    failures and cooldown periods, and formats proxy entries for curl_cffi and Playwright.
    """

    def __init__(
        self,
        proxies: Optional[Union[str, List[str]]] = None,
        proxies_file: Optional[Union[str, Path]] = None,
        strategy: Literal["round_robin", "random"] = "round_robin",
        max_failures: int = 3,
        cooldown_seconds: float = 300.0,
    ):
        """Initializes the proxy rotator.

        Args:
            proxies: Comma-separated proxy URLs or a list of proxy URLs.
            proxies_file: Path to a file containing one proxy URL per line.
            strategy: Rotation strategy ('round_robin' or 'random').
            max_failures: Maximum consecutive failures before cooling down a proxy.
            cooldown_seconds: Duration in seconds to cool down a failed proxy.
        """
        self.strategy = strategy
        self.max_failures = max_failures
        self.cooldown_seconds = cooldown_seconds

        self._proxies: List[str] = []
        self._index: int = 0
        self._failure_counts: Dict[str, int] = {}
        self._cooldown_until: Dict[str, float] = {}

        if proxies:
            if isinstance(proxies, str):
                self.add_proxies([p.strip() for p in proxies.split(",") if p.strip()])
            else:
                self.add_proxies(proxies)

        if proxies_file:
            self.load_from_file(proxies_file)

    def add_proxies(self, proxies: List[str]) -> None:
        """Adds proxies to the pool, ignoring duplicates and empty entries."""
        for p in proxies:
            cleaned = p.strip()
            if cleaned and cleaned not in self._proxies:
                self._proxies.append(cleaned)
                self._failure_counts[cleaned] = 0
        if proxies:
            logger.info("Added %d proxies to rotator (total active: %d)", len(proxies), len(self._proxies))

    def load_from_file(self, filepath: Union[str, Path]) -> int:
        """Loads proxy URLs from a plain text file (one URL per line).

        Args:
            filepath: Path to the proxy list file.

        Returns:
            Number of new proxies added.
        """
        p_path = Path(filepath)
        if not p_path.is_file():
            logger.warning("Proxy file not found: %s", filepath)
            return 0

        initial_len = len(self._proxies)
        with open(p_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        self.add_proxies(lines)
        added = len(self._proxies) - initial_len
        logger.info("Loaded %d proxies from file: %s", added, filepath)
        return added

    def has_proxies(self) -> bool:
        """Returns True if any proxies are configured."""
        return len(self._proxies) > 0

    @property
    def total_count(self) -> int:
        """Returns total registered proxies count."""
        return len(self._proxies)

    @property
    def active_count(self) -> int:
        """Returns count of proxies currently not in cooldown."""
        now = time.time()
        return sum(1 for p in self._proxies if self._cooldown_until.get(p, 0) <= now)

    def _get_available_pool(self) -> List[str]:
        """Returns proxies that are currently healthy or whose cooldown has expired."""
        now = time.time()
        available = []
        for p in self._proxies:
            cooldown = self._cooldown_until.get(p, 0)
            if cooldown <= now:
                available.append(p)
        # If all proxies are currently in cooldown, reset cooldown to avoid total starvation
        if not available and self._proxies:
            logger.warning("All %d proxies are in cooldown; resetting cooldowns.", len(self._proxies))
            self._cooldown_until.clear()
            available = list(self._proxies)
        return available

    def get_proxy(self) -> Optional[str]:
        """Returns the next available proxy URL based on the rotation strategy."""
        pool = self._get_available_pool()
        if not pool:
            return None

        if self.strategy == "random":
            return random.choice(pool)

        # Round-robin across active pool
        selected = pool[self._index % len(pool)]
        self._index = (self._index + 1) % len(pool)
        return selected

    def report_success(self, proxy: Optional[str]) -> None:
        """Records a successful request for a proxy, resetting its error count."""
        if not proxy or proxy not in self._proxies:
            return
        self._failure_counts[proxy] = 0
        self._cooldown_until.pop(proxy, None)
        logger.debug("Proxy %s succeeded; error count reset.", proxy)

    def report_failure(self, proxy: Optional[str], exc: Optional[Exception] = None) -> None:
        """Records a failed request for a proxy and initiates cooldown if threshold reached."""
        if not proxy or proxy not in self._proxies:
            return

        cnt = self._failure_counts.get(proxy, 0) + 1
        self._failure_counts[proxy] = cnt
        logger.warning(
            "Proxy %s failed (%d/%d consecutive errors): %s",
            proxy,
            cnt,
            self.max_failures,
            exc or "Blocked or connection error",
        )

        if cnt >= self.max_failures:
            cooldown_until = time.time() + self.cooldown_seconds
            self._cooldown_until[proxy] = cooldown_until
            logger.warning(
                "Proxy %s exceeded failure threshold (%d); entering cooldown for %ds.",
                proxy,
                self.max_failures,
                int(self.cooldown_seconds),
            )

    @staticmethod
    def to_playwright_proxy(proxy_url: Optional[str]) -> Optional[Dict[str, str]]:
        """Converts a standard proxy URL (http://user:pass@host:port) to Playwright proxy dict.

        Args:
            proxy_url: Raw proxy URL.

        Returns:
            Dict suitable for Playwright's `proxy` option, or None.
        """
        if not proxy_url:
            return None

        parsed = urlparse(proxy_url)
        server = f"{parsed.scheme}://{parsed.hostname}"
        if parsed.port:
            server += f":{parsed.port}"

        result: Dict[str, str] = {"server": server}
        if parsed.username:
            result["username"] = parsed.username
        if parsed.password:
            result["password"] = parsed.password
        return result
