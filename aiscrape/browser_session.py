"""A camoufox browser bound to one pooled account's session.

`AIBrowserSession` launches camoufox with an account's `storage_state` (+ proxy,
+ fingerprint), warms it with fresh Firefox cookies for the provider, and runs
provider-appropriate scrapes against it:

    google                     -> run_ai_mode / run_search  (AIOverviewResult / SearchResult)
    chatgpt/claude/gemini/meta -> run_chat                  (ChatResult)

Each chat task runs on a *fresh* conversation, so it's an independent unit of
work the WorkerPool can hand to any worker. After each scrape the refreshed
cookies are written back to the pool so a rotated session stays current.
"""

from __future__ import annotations

from typing import Optional

from camoufox.async_api import AsyncCamoufox
from playwright.async_api import BrowserContext, Page

from .account import Account
from .accounts_pool import AccountsPool
from .browser import new_camoufox, warm_from_firefox
from .chatbots import (
    PLATFORM_URLS,
    SEND_FNS,
    detect_l2_classifier,
    detect_rate_limit,
    pre_send_turn_count,
    read_served_model,
    select_claude_model,
    start_fresh_conversation,
    warm_chatbot_from_firefox,
)
from .exceptions import SessionExpiredError
from .google_aimode import AIOverviewResult, AIOverviewScraper, SearchResult
from .logger import logger
from .models import ChatResult, now_iso

# Providers whose scrape is a chatbot turn (vs. Google Search's AI Mode).
CHATBOT_PROVIDERS = frozenset({"chatgpt", "claude", "gemini", "meta"})


async def warm_session_from_firefox(context: BrowserContext, provider: str) -> int:
    """Overlay this provider's fresh Firefox cookies onto a context.

    Google/Gemini ride the Google cookie jar; the other chatbots use their own
    hosts (and Meta additionally pulls Facebook/Instagram) — dispatch to the
    matching warm helper. Returns how many cookies were added."""
    if provider == "google":
        return await warm_from_firefox(context, "%google.%")
    return await warm_chatbot_from_firefox(context, provider)


class AIBrowserSession:
    """A live camoufox session for one account of one provider."""

    def __init__(
        self,
        account: Account,
        provider: str,
        pool: AccountsPool | None = None,
        headless: bool = True,
        settle_ms: int = 9_000,
        refresh_from_firefox: bool = True,
        debug: bool = False,
    ):
        self.account = account
        self.provider = provider
        self.pool = pool
        self.headless = headless
        self.settle_ms = settle_ms
        self.refresh_from_firefox = refresh_from_firefox
        self.debug = debug

        self._cm: Optional[AsyncCamoufox] = None
        self._browser = None
        self._context: Optional[BrowserContext] = None

    @property
    def context(self) -> BrowserContext:
        if self._context is None:
            raise RuntimeError("AIBrowserSession not initialised")
        return self._context

    async def __aenter__(self) -> "AIBrowserSession":
        await self.initialize()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def initialize(self) -> None:
        # Google Search CAPTCHA-blocks fresh automated browsers, so it gets the
        # same anti-bot fingerprint the standalone scraper uses (en-CA + geoip +
        # OS spoof); the chatbots don't need it.
        is_google = self.provider == "google"
        self._cm = new_camoufox(
            headless=self.headless,
            locale=self.account.locale or ("en-CA" if is_google else None),
            geoip=is_google,
            os=["windows", "macos"] if is_google else None,
            proxy=self.account.proxy_dict,
        )
        self._browser = await self._cm.__aenter__()
        self._context = await self._browser.new_context(
            storage_state=self.account.storage_state or None
        )

        if self.refresh_from_firefox:
            n = await warm_session_from_firefox(self._context, self.provider)
            logger.debug(
                f"[{self.provider}] warmed {self.account.label} with {n} "
                f"Firefox cookie(s)"
            )
        logger.info(f"[{self.provider}] session ready for {self.account.label}")

    async def close(self) -> None:
        try:
            if self._context:
                await self._context.close()
        finally:
            if self._cm:
                await self._cm.__aexit__(None, None, None)
            self._context = None
            self._browser = None
            self._cm = None

    async def save_session(self) -> None:
        """Persist the context's current cookies/origins back to the pool so a
        session the product rotated mid-scrape stays current."""
        if self.pool is None or self._context is None:
            return
        state = await self._context.storage_state()
        await self.pool.update_storage_state(self.account.label, state)

    # ── Google AI Mode / Search ──────────────────────────────────────────────

    def _google_scraper(self) -> AIOverviewScraper:
        # Drive the extraction logic against our own (account-bound) context.
        return AIOverviewScraper(
            locale=self.account.locale or "en-CA",
            settle_ms=self.settle_ms,
            debug=self.debug,
            context=self.context,
        )

    async def run_ai_mode(self, prompt: str) -> AIOverviewResult:
        result = await self._google_scraper().search(prompt)
        if result.blocked:
            # Google's CAPTCHA / 'unusual traffic' wall means this session is no
            # longer trusted — surface it so the Worker can mark it inactive.
            raise SessionExpiredError(
                f"Google blocked {self.account.label} (CAPTCHA / unusual traffic)"
            )
        return result

    async def run_search(self, prompt: str, top_n: int = 10) -> SearchResult:
        result = await self._google_scraper().search_normal(prompt, top_n=top_n)
        if result.blocked:
            raise SessionExpiredError(
                f"Google blocked {self.account.label} (CAPTCHA / unusual traffic)"
            )
        return result

    # ── Chatbot turn ─────────────────────────────────────────────────────────

    async def run_chat(self, prompt: str, model: str | None = None) -> ChatResult:
        if self.provider not in CHATBOT_PROVIDERS:
            raise ValueError(f"run_chat not supported for provider {self.provider!r}")
        scraped_at = now_iso()
        page: Page = await self.context.new_page()
        try:
            await start_fresh_conversation(page, PLATFORM_URLS[self.provider])
            if model and self.provider == "claude":
                await select_claude_model(page, model)

            pre = await pre_send_turn_count(page, self.provider)
            response = await SEND_FNS[self.provider](page, prompt)

            rate_limited = await detect_rate_limit(page, self.provider)
            l2_block = await detect_l2_classifier(page, self.provider, pre)
            served = await read_served_model(page, self.provider)

            return ChatResult(
                provider=self.provider,
                prompt=prompt,
                response=response,
                served_model=served,
                requested_model=model,
                rate_limited=rate_limited,
                l2_block=l2_block,
                scraped_at=scraped_at,
            )
        finally:
            await page.close()
