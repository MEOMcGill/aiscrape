"""
Shared camoufox launch + Firefox-cookie warming.

Both the Google AI Mode scraper and the chatbot web-UI scrapers drive
camoufox (stealth Firefox) via Playwright, loading a saved logged-in session
(`storage_state`) and optionally overlaying fresh cookies read straight from
the real local Firefox profile. This module holds the construction those
scrapers share; each keeps its own lifecycle (reusable class vs. adapter).
"""

from __future__ import annotations

from camoufox.async_api import AsyncCamoufox

from aiscrape.firefox_cookies import default_firefox_profile, read_cookies


def new_camoufox(
    headless: bool = True,
    locale: str | None = None,
    geoip: bool = False,
    os: list[str] | None = None,
    proxy: dict | None = None,
) -> AsyncCamoufox:
    """A configured (but not-yet-started) AsyncCamoufox context manager.

    Only non-None fingerprint options are passed through so callers that want
    camoufox's defaults (the chatbot scrapers) and callers that pin a locale /
    geoip / OS spoof (the Google scraper) share one construction path. `proxy`
    is a Playwright proxy dict (``{"server": ..., "username"?: ..., "password"?:
    ...}``), as carried by a pooled account.
    """
    kwargs: dict = {"headless": headless}
    if locale is not None:
        kwargs["locale"] = locale
    if geoip:
        kwargs["geoip"] = True
    if os is not None:
        kwargs["os"] = os
    if proxy is not None:
        kwargs["proxy"] = proxy
    return AsyncCamoufox(**kwargs)


async def warm_from_firefox(context, host_like: str | None = "%google.%") -> int:
    """Inject cookies from the real Firefox profile into a Playwright context.

    Keeps the scraping session "warm" with real day-to-day usage behind it.
    Returns the number of cookies added (0 if no profile / no matching cookies).
    """
    profile = default_firefox_profile()
    if profile is None:
        return 0
    cookies = read_cookies(profile, host_like)
    if not cookies:
        return 0
    await context.add_cookies(cookies)
    return len(cookies)
