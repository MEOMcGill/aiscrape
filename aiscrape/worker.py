"""Worker: acquires one account, runs tasks against it, rotates on trouble.

Adapts igscrape's Worker to the AI-product signals the scrapers already detect:
a rate-limit banner locks the account out for a cooldown and rotates; a CAPTCHA
/ expired-session wall marks it inactive and rotates; an application-layer (L2)
safety block is a valid per-conversation outcome and is returned as-is (the
account is fine for other prompts).
"""

import asyncio
from typing import Optional, Union

from .account import Account
from .accounts_pool import AccountsPool
from .browser_session import AIBrowserSession
from .exceptions import NoAccountError, RateLimitError, SessionExpiredError
from .google_aimode import AIOverviewResult, SearchResult
from .logger import logger
from .models import ChatResult, Task

TaskResult = Union[AIOverviewResult, SearchResult, ChatResult]

# Rotation policy: rest after this many queries, then a short cooldown, so no
# single session stays continuously hot.
QUERIES_PER_REST = 50
REST_MINUTES = 5
# How long to lock an account out after it trips a rate-limit banner.
RATE_LIMIT_MINUTES = 30
MAX_RETRIES = 3


class Worker:
    """Runs tasks for one acquired account until rotation or shutdown."""

    def __init__(
        self,
        id: str,
        pool: AccountsPool,
        provider: str,
        queries_per_rest: int = QUERIES_PER_REST,
        headless: bool = True,
        settle_ms: int = 9_000,
        refresh_from_firefox: bool = True,
    ):
        self.id = id
        self.pool = pool
        self.provider = provider
        self.queries_per_rest = queries_per_rest
        self.headless = headless
        self.settle_ms = settle_ms
        self.refresh_from_firefox = refresh_from_firefox

        self.current_account: Optional[Account] = None
        self.queries_run: int = 0
        self._initialized = False
        # Persistent session, reused across tasks for the lifetime of the current
        # account; recreated on rotation / crash.
        self.session: Optional[AIBrowserSession] = None

    @classmethod
    async def create(cls, id: str, pool: AccountsPool, provider: str, **kwargs) -> "Worker":
        instance = cls(id=id, pool=pool, provider=provider, **kwargs)
        if not await instance.initialize():
            raise NoAccountError(f"Worker {id}: no account available")
        return instance

    async def __aenter__(self):
        if not self._initialized and not await self.initialize():
            raise NoAccountError(f"Worker {self.id}: no account available")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False

    async def initialize(self) -> bool:
        account = await self.pool.get_available()
        if not account:
            return False
        self.current_account = account
        self.queries_run = 0
        self._initialized = True
        logger.info(f"Worker {self.id} acquired {self.provider} account {account.label}")
        return True

    async def close(self):
        await self._close_session()
        if self.current_account:
            await self.pool.release_account(self.current_account.label)
            self.current_account = None
        self.queries_run = 0
        self._initialized = False

    async def _ensure_session(self) -> AIBrowserSession:
        if self.session is None:
            self.session = AIBrowserSession(
                account=self.current_account,
                provider=self.provider,
                pool=self.pool,
                headless=self.headless,
                settle_ms=self.settle_ms,
                refresh_from_firefox=self.refresh_from_firefox,
            )
            await self.session.initialize()
        return self.session

    async def _close_session(self):
        if self.session is not None:
            try:
                await self.session.close()
            except Exception:
                pass
            self.session = None

    async def _dispatch(self, session: AIBrowserSession, task: Task) -> TaskResult:
        p = task.payload
        if task.endpoint == "ai_mode":
            return await session.run_ai_mode(p["prompt"])
        if task.endpoint == "search":
            return await session.run_search(p["prompt"], top_n=p.get("top_n", 10))
        if task.endpoint == "chat":
            return await session.run_chat(p["prompt"], model=p.get("model"))
        raise ValueError(f"Unsupported endpoint {task.endpoint!r}")

    async def execute_task(self, task: Task) -> TaskResult:
        """Run one task, applying the rate-limit / expired-session policy."""
        if self.queries_run >= self.queries_per_rest:
            logger.info(
                f"Worker {self.id}: ran {self.queries_run} queries, "
                f"rotating {self.current_account.label}"
            )
            await self.rotate_account()

        for attempt in range(MAX_RETRIES):
            try:
                session = await self._ensure_session()
                result = await self._dispatch(session, task)

                # A rate-limit banner is not real output: lock this account for a
                # cooldown, rotate, and retry the SAME task on another account.
                if isinstance(result, ChatResult) and result.rate_limited:
                    logger.warning(
                        f"Worker {self.id}: {self.current_account.label} rate-limited, "
                        f"attempt {attempt + 1}/{MAX_RETRIES}"
                    )
                    await self.pool.lock_until(
                        self.current_account.label,
                        f"datetime('now', '+{RATE_LIMIT_MINUTES} minutes')",
                    )
                    await self.rotate_account()
                    await asyncio.sleep(2)
                    continue

                # Success (including a valid L2 block for chat, or a no-overview
                # Google result): persist the refreshed session and count it.
                await session.save_session()
                await self.pool.update_query_count(
                    self.current_account.label, task.endpoint
                )
                self.queries_run += 1
                return result

            except RateLimitError as e:
                logger.warning(f"Worker {self.id}: rate limited: {e}")
                await self.pool.lock_until(
                    self.current_account.label,
                    f"datetime('now', '+{RATE_LIMIT_MINUTES} minutes')",
                )
                await self.rotate_account()
                await asyncio.sleep(2)
            except SessionExpiredError as e:
                logger.warning(f"Worker {self.id}: session expired: {e}")
                await self.pool.mark_inactive(self.current_account.label, str(e))
                await self.rotate_account()

        raise RuntimeError(
            f"Worker {self.id}: failed to execute task after {MAX_RETRIES} retries"
        )

    async def rotate_account(self):
        """Release the current account with a short cooldown, then acquire the
        next available one (which may be the same account once its cooldown
        expires, in a single-account pool)."""
        await self._close_session()
        if self.current_account:
            await self.pool.lock_until(
                self.current_account.label,
                f"datetime('now', '+{REST_MINUTES} minutes')",
            )
            await self.pool.release_account(self.current_account.label)
            logger.info(
                f"Worker {self.id} released {self.current_account.label} "
                f"({REST_MINUTES}-minute cooldown)"
            )
            self.current_account = None

        self.queries_run = 0
        self._initialized = False

        account = await self.pool.get_available_or_wait()
        if not account:
            raise NoAccountError(f"Worker {self.id}: no account available for rotation")
        self.current_account = account
        self._initialized = True
        logger.info(f"Worker {self.id} acquired {self.provider} account {account.label}")
