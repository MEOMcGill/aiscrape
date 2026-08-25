"""
Save a logged-in browser session for a target platform to `auth/<platform>.json`.

Opens a headed camoufox window so you can confirm you're logged in (and log in
manually if not), then saves the session's cookies/storage so the scrapers can
load it via `storage_state`.

By default we first import that platform's cookies straight from your real
Firefox profile (see `firefox_cookies.py`) -- that session already has real
day-to-day usage behind it, which usually skips the manual login and looks more
legitimate (this matters most for Google, which aggressively CAPTCHA-blocks
fresh/automated browsers on Search). Each platform's cookies are matched by host
(`aiscrape.chatbots.PLATFORM_COOKIE_HOSTS` for the chatbots; meta also pulls
Facebook/Instagram, since its login is federated). AIOverviewScraper also
re-warms Google cookies on every scrape, so as long as you keep using the
product normally in Firefox the saved session stays valid.

Google additionally gets the anti-bot fingerprint (en-CA locale + geoip) the
scraper uses, and its confirm step opens a live search so you can check Search
isn't blocked.

Usage:
  DISPLAY=:1 python -m aiscrape.auth --platform google
  DISPLAY=:1 python -m aiscrape.auth --platform google --no-firefox-cookies
  python -m aiscrape.auth --platform claude    # or chatgpt / gemini / meta / characterai

`auth/` resolves against the current working directory, so run this from the
repo whose session you want to save.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from aiscrape.browser import new_camoufox, warm_from_firefox
from aiscrape.chatbots import PLATFORM_URLS as CHATBOT_URLS
from aiscrape.chatbots import warm_chatbot_from_firefox

# The URL each platform opens for the manual login/confirm step. The four
# chatbot URLs are reused from aiscrape.chatbots so there's one source of
# truth; Google opens a live search so you can confirm Search isn't blocked.
PLATFORM_URLS: dict[str, str] = {
    "google": "https://www.google.com/search?q=test",
    **CHATBOT_URLS,
    "characterai": "https://character.ai",
}


async def _prompt(message: str) -> None:
    """Block on stdin without freezing the asyncio event loop."""
    await asyncio.get_event_loop().run_in_executor(None, input, message)


async def save_auth(
    platform: str,
    use_firefox_cookies: bool = True,
    auth_dir: Path = Path("auth"),
) -> Path:
    auth_dir = Path(auth_dir)
    auth_dir.mkdir(parents=True, exist_ok=True)
    auth_file = auth_dir / f"{platform}.json"
    url = PLATFORM_URLS[platform]
    is_google = platform == "google"

    # Google gets the anti-bot fingerprint (locale/geoip) that the scraper uses.
    cm = new_camoufox(
        headless=False,
        locale="en-CA" if is_google else None,
        geoip=is_google,
    )
    async with cm as browser:
        context = await browser.new_context()

        # Warm the login window with cookies from your real Firefox profile so
        # you're likely already logged in (works for every platform, not just
        # Google -- the same trick keeps scrapes' sessions valid).
        if use_firefox_cookies:
            if is_google:
                n = await warm_from_firefox(context, "%google.%")
            elif platform == "characterai":
                n = await warm_from_firefox(context, "%character.ai")
            else:
                n = await warm_chatbot_from_firefox(context, platform)
            if n:
                print(f"Imported {n} {platform} cookies from your Firefox "
                      f"profile -- you should already be logged in below.")
            else:
                print(f"No {platform} cookies found in your Firefox profile "
                      f"(or no profile found) -- log in manually below.")

        page = await context.new_page()
        await page.goto(url)

        if is_google:
            msg = ("Check the window: logged in, and Search working (no CAPTCHA "
                   "/ 'unusual traffic')? Log in manually there if not, then "
                   "press Enter to save... ")
        else:
            msg = f"Logged in to {platform}? Press Enter to save session and exit... "
        await _prompt(msg)

        await context.storage_state(path=str(auth_file))
        await context.close()

    print(f"Saved session to {auth_file}")
    return auth_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=list(PLATFORM_URLS),
                        help="Platform to authenticate")
    parser.add_argument("--no-firefox-cookies", action="store_true",
                        help="Skip importing cookies from your real Firefox "
                             "profile; log in fresh instead.")
    args = parser.parse_args()
    asyncio.run(save_auth(args.platform,
                          use_firefox_cookies=not args.no_firefox_cookies))


if __name__ == "__main__":
    main()
