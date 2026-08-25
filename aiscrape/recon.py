"""
Reconnaissance: drive camoufox to a Google search that should trigger an AI
Overview, dismiss consent, expand the overview, then dump the rendered HTML and
a screenshot so we can identify robust selectors for the real scraper.

Paths resolve against the current working directory.

Usage:
  python -m aiscrape.recon "how does photosynthesis work"
"""

import asyncio
import sys
from pathlib import Path
from urllib.parse import quote_plus

from aiscrape.browser import new_camoufox

OUT = Path("results/recon")
AUTH_FILE = Path("auth/google.json")


async def dismiss_consent(page) -> None:
    """Click any Google cookie-consent 'Accept/Reject all' button if present."""
    for sel in (
        'button:has-text("Accept all")',
        'button:has-text("Reject all")',
        'button[aria-label*="Accept all" i]',
        'button[aria-label*="Reject all" i]',
        '#L2AGLb',  # Google's consent "I agree" button id
    ):
        try:
            btn = page.locator(sel).first
            if await btn.count() > 0 and await btn.is_visible():
                await btn.click(timeout=3_000)
                await page.wait_for_timeout(1_000)
                return
        except Exception:
            pass


async def main(query: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    url = f"https://www.google.com/search?q={quote_plus(query)}&hl=en&gl=ca"

    storage = str(AUTH_FILE) if AUTH_FILE.exists() else None
    async with new_camoufox(
        headless=True,
        locale="en-CA",
        geoip=True,
        os=["windows", "macos"],
    ) as browser:
        context = await browser.new_context(storage_state=storage)
        page = await context.new_page()
        await page.goto(url, timeout=60_000)
        await page.wait_for_load_state("domcontentloaded", timeout=30_000)
        await dismiss_consent(page)
        # AI Overview streams in asynchronously; give it time.
        await page.wait_for_timeout(6_000)

        # Try to expand "Show more" on the overview, if present.
        for sel in (
            'div[aria-label*="Show more" i]',
            'button:has-text("Show more")',
            'div[role="button"]:has-text("Show more")',
        ):
            try:
                btn = page.locator(sel).first
                if await btn.count() > 0 and await btn.is_visible():
                    await btn.click(timeout=3_000)
                    await page.wait_for_timeout(2_500)
                    break
            except Exception:
                pass

        html = await page.content()
        (OUT / "page.html").write_text(html, encoding="utf-8")
        await page.screenshot(path=str(OUT / "page.png"), full_page=True)

        # Quick signal: does the word "AI Overview" appear?
        body = (await page.inner_text("body")).lower()
        print("len(html):", len(html))
        print("contains 'ai overview':", "ai overview" in body)
        print("contains captcha/unusual traffic:",
              "unusual traffic" in body or "recaptcha" in html.lower())
        print(f"wrote {OUT/'page.html'} and {OUT/'page.png'}")


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "how does photosynthesis work"
    asyncio.run(main(q))
