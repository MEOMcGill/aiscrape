"""Async pooling and CAPTCHA rotation over a farm of Android phones.

The surface scrapers (`phone_farm.PhoneFarmAIOverviewScraper` for Google AI Mode
and plain search, `phone_chatgpt.PhoneChatGPTScraper` for ChatGPT) each drive one
handset. This adds what you need to drive *many* of them for a long batch:

- **Every surface off one phone.** A `PhoneBackend` opens one Chrome session per
  handset and runs all three scrapers over it (`search`, `search_normal`, `chat`),
  because the CDP tunnel binds a port that a second session would collide with.
- **Async wrapping.** The phone scrapers are synchronous (subprocess + websocket);
  each call runs in a worker thread via `asyncio.to_thread`, so async runners keep
  their structure.
- **Serial rotation when a phone is walled.** Like the Google account pool rotates
  to the next account when one is walled, this rotates to the next phone serial
  when a scrape comes back `blocked` — Google's "unusual traffic", or ChatGPT's
  verification wall — re-trying the same prompt. When every phone is walled it
  raises `PhoneFarmExhausted`, which callers catch to stop and save (unreached
  prompts resume next run).
- **A shared allocator**, so concurrent workers never drive the same handset and a
  walled phone's replacement comes from the same global pool rather than a
  per-worker slice.
- **Proactive, no-fault rotation and inter-ask pacing, on by default.** A worker
  also hands its phone back after `rest_after_asks` asks even when nothing is
  wrong (`PhoneBackend._rest`), and sleeps a jittered
  `ask_delay_min_s`..`ask_delay_max_s` between asks. Both are
  opt-OUT (pass `None`/`0`): a farm-wide safety default beats a caller that
  forgets to set it, or loses the setting to something as mundane as a git
  branch switch reverting an uncommitted CLI default (this is not hypothetical
  -- it is why these became library defaults instead of a runner's).

Note: the farm's phones typically share one WiFi egress IP, and Google's burst
limit is largely IP-level — rotating phones spreads the *per-account* block, not
the *per-IP* one. Repeated elicitation sweeps saw the runway before a wall shrink
cycle over cycle at a 2s/never-rotate pace, and a 1400+ query sweep then completed
with zero walls at a flat 10s gap with rotation on. DEFAULT_ASK_DELAY_MIN_S/MAX_S
jitter around that rather than sitting on it: a constant period is its own bot tell.

This lived in the ai-overview study repo (`experiments/phone_backend.py`) until
the query-bank runner needed it here; that module is now a thin Hydra-shaped
adapter over this one.
"""

from __future__ import annotations

import asyncio
import random
import time

from aiscrape.logger import logger
from aiscrape.phone_chatgpt import PhoneChatGPTScraper
from aiscrape.phone_chatgpt import DEFAULT_ANSWER_TIMEOUT_S as DEFAULT_CHAT_TIMEOUT_S
from aiscrape.phone_chatgpt import DEFAULT_SETTLE_MS as DEFAULT_CHAT_SETTLE_MS
from aiscrape.phone_chatgpt import DEFAULT_WARMUP_S as DEFAULT_CHAT_WARMUP_S
from aiscrape.phone_farm import (
    DEFAULT_SEARCH_SETTLE_MS,
    PhoneChromeSession,
    GoogleSignedOutError,
    PhoneFarmAIOverviewScraper,
    UnusableResultError,
    list_serials,
)

DEFAULT_CDP_PORT = 9222

# How many AI Mode asks in a row may come back empty on one handset before it is
# treated as walled and rotated away from.
#
# Rotating only on `blocked` is not enough: Google can stop answering partway through
# a session without ever showing a CAPTCHA, leaving `blocked` false, no rotation
# firing, and one phone absorbing a whole run while the rest sit idle. An unanswered
# ask is not proof of a bad phone -- but three in a row is the same evidence a CAPTCHA
# would have given, arriving quietly.
DRY_STREAK_BEFORE_ROTATE = 3

# How many *different* prompts must come back empty on a handset that has never held
# a ChatGPT conversation before it is left out of chat for the rest of the run. Two,
# not one: a good phone does drop the occasional ask, and being wrong here costs the
# farm a chat handset for the whole run.
CHAT_STRIKES_BEFORE_EXCLUDING = 2
# How many handsets one ChatGPT prompt may be moved across before its empty answer is
# taken at face value and stored. Without it, a prompt ChatGPT genuinely will not
# answer would walk the farm collecting a strike on every phone.
MAX_HANDSETS_PER_CHAT_PROMPT = 3

# How long a walled handset is left alone before the farm will try it again. A
# CAPTCHA is a property of the session and it decays; treating it as permanent is
# what turns "one phone tripped a wall" into "this run is over" on a small farm.
# Only reached for when nothing else is free, so a healthy farm never re-tests.
# 0 disables revival, restoring the walled-for-the-session behaviour.
DEFAULT_WALL_COOLDOWN_S = 30 * 60

# Proactive rotation every 5 asks, and a jittered 5-10s between them. Against the live
# farm (2026-08) a flat 2s gap with stay-until-walled saw the runway before a CAPTCHA
# shrink cycle over cycle, while rotation at a flat 10s carried a 1400+ query sweep
# with zero walls. The range brackets that 10s rather than centring on it (mean ~7.5s),
# trading a little of the proven margin for jitter, since a constant period is itself a
# bot tell -- widen it if walls reappear.
DEFAULT_REST_AFTER_ASKS = 5
DEFAULT_ASK_DELAY_MIN_S = 5.0
DEFAULT_ASK_DELAY_MAX_S = 10.0


class PhoneFarmExhausted(RuntimeError):
    """Every phone in the farm is walled (CAPTCHA), and none has cooled down yet."""


class SerialPool:
    """Shared allocator over the farm's serials.

    A phone is *checked out* while a worker holds it, *walled* once it trips a
    CAPTCHA, and returned to the free list when a worker closes it cleanly.

    A wall is timed rather than permanent: once every other handset is spoken for,
    one whose wall has aged past `wall_cooldown_s` is offered again. Walls decay,
    and on a farm of a few phones treating them as final ends the run the first
    time they all trip -- with the bank unfinished and no way back inside the run.
    Revival is a last resort, so a farm with anything free rotates exactly as before.
    """

    def __init__(self, serials: list[str], *,
                 wall_cooldown_s: float = DEFAULT_WALL_COOLDOWN_S):
        self._free = list(serials)
        self._wall_cooldown_s = max(0.0, float(wall_cooldown_s))
        # serial -> when it was walled, on the monotonic clock so a system clock
        # change mid-run cannot make a wall look hours old.
        self._walled: dict[str, float] = {}
        self._in_use: set[str] = set()
        # Handsets whose Google web session has already been probed this run, so a
        # phone that rests and is picked up again is not probed over and over.
        self._verified: set[str] = set()
        # Whether a handset can hold a ChatGPT conversation. Kept apart from walling:
        # a phone that cannot chat usually still serves Google, and walling it would
        # cost the farm a Google handset to fix a ChatGPT problem.
        self._chatgpt_ok: dict[str, bool] = {}
        # Distinct prompts that have come back empty on a handset that has never
        # answered one, keyed by serial.
        self._chat_strikes: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()

    @property
    def total(self) -> int:
        return len(self._free) + len(self._in_use) + len(self._walled)

    def snapshot(self) -> str:
        return (f"{len(self._free)} free / {len(self._in_use)} in use / "
                f"{len(self._walled)} walled")

    def _revive_walled(self) -> list[str]:
        """Move handsets whose wall has aged out back to the free list.

        Caller holds the lock. Oldest wall first, so the phone with the best chance
        of having recovered is the one tried.
        """
        if not self._wall_cooldown_s:
            return []
        now = time.monotonic()
        due = sorted(
            (t, serial) for serial, t in self._walled.items()
            if now - t >= self._wall_cooldown_s
        )
        for _, serial in due:
            del self._walled[serial]
            self._free.append(serial)
        return [serial for _, serial in due]

    async def acquire(self) -> str:
        async with self._lock:
            if not self._free:
                # Last resort: a healthy farm never gets here, so this cannot
                # disturb the ordinary rotation.
                revived = self._revive_walled()
                if revived:
                    logger.info(
                        f"[phone] wall cooldown elapsed on {', '.join(revived)}; "
                        f"trying {'them' if len(revived) > 1 else 'it'} again "
                        f"({self.snapshot()})")
            if not self._free:
                raise PhoneFarmExhausted(
                    f"no phone available ({self.snapshot()}"
                    + (f", none walled longer than {self._wall_cooldown_s:.0f}s"
                       if self._walled and self._wall_cooldown_s else "")
                    + ")")
            serial = self._free.pop(0)
            self._in_use.add(serial)
            return serial

    async def wall(self, serial: str) -> None:
        async with self._lock:
            self._in_use.discard(serial)
            # Re-walling restarts the clock, which is what a handset that came back
            # and tripped again has earned.
            self._walled[serial] = time.monotonic()

    async def chatgpt_state(self, serial: str) -> bool | None:
        """False once this handset is out for chat, True once it has answered one."""
        async with self._lock:
            return self._chatgpt_ok.get(serial)

    async def set_chatgpt_ok(self, serial: str, ok: bool) -> None:
        async with self._lock:
            self._chatgpt_ok[serial] = ok

    async def note_chat_answer(self, serial: str) -> None:
        """This handset held a conversation, so it is a working one."""
        async with self._lock:
            self._chatgpt_ok[serial] = True
            self._chat_strikes.pop(serial, None)

    async def note_chat_failure(self, serial: str, prompt: str) -> int:
        """Record a failed ask, returning how many *distinct* prompts have failed here.

        Distinct on purpose. One prompt failing across many handsets says something
        about the prompt; one handset failing many prompts says something about the
        handset, and only the second is grounds for dropping it.
        """
        async with self._lock:
            self._chat_strikes.setdefault(serial, set()).add(prompt)
            return len(self._chat_strikes[serial])

    def chatgpt_snapshot(self) -> str:
        ok = sum(1 for v in self._chatgpt_ok.values() if v)
        bad = sum(1 for v in self._chatgpt_ok.values() if not v)
        return f"{ok} can chat / {bad} cannot / {self.total - ok - bad} unproven"

    async def mark_verified(self, serial: str) -> None:
        """Note that this handset passed its sign-in probe for the rest of the run."""
        async with self._lock:
            self._verified.add(serial)

    async def is_verified(self, serial: str) -> bool:
        async with self._lock:
            return serial in self._verified

    async def release(self, serial: str) -> None:
        async with self._lock:
            self._in_use.discard(serial)
            if serial not in self._walled:
                self._free.append(serial)


class PhoneBackend:
    """Async, serial-rotating adapter over the scrapers that drive one phone.

    Holds one open handset at a time; opening one sets up its wake + CDP tunnel
    (~a few seconds), so we stay on one phone rather than rotating every prompt --
    until it is walled (`_rotate`), or until `rest_after_asks` asks have passed on
    it with nothing wrong at all (`_rest`), whichever comes first. Both `rest_after_asks`
    and the jittered inter-ask pace are on by default (see DEFAULT_REST_AFTER_ASKS /
    DEFAULT_ASK_DELAY_MIN_S / DEFAULT_ASK_DELAY_MAX_S) -- a caller has to opt OUT, not
    in, of gentle pacing on a farm whose phones share one egress IP. This is the only
    place asks are paced; runners do not add a sleep of their own on top.

    A phone is one `PhoneChromeSession` with every surface scraper sharing it —
    Google (`search`, `search_normal`) and ChatGPT (`chat`). Sharing is not an
    optimisation: the tunnel binds a TCP port on this machine and an `adb forward`
    on the farm host, so a second session for the same handset would collide with
    the first on both.
    """

    def __init__(self, pool: SerialPool, *, ssh_host: str | None = None,
                 adb: str | None = None, cdp_port: int = DEFAULT_CDP_PORT,
                 settle_ms: int = 9_000, answer_timeout_s: int = 60,
                 sleep_on_exit: bool = True, debug: bool = False,
                 search_settle_ms: int = DEFAULT_SEARCH_SETTLE_MS,
                 chat_settle_ms: int = DEFAULT_CHAT_SETTLE_MS,
                 chat_answer_timeout_s: int = DEFAULT_CHAT_TIMEOUT_S,
                 chat_warmup_s: float = DEFAULT_CHAT_WARMUP_S,
                 chatgpt_login: bool = True,
                 signin_precheck: bool = True,
                 chatgpt_probe: bool = True,
                 rest_after_asks: int | None = DEFAULT_REST_AFTER_ASKS,
                 ask_delay_min_s: float = DEFAULT_ASK_DELAY_MIN_S,
                 ask_delay_max_s: float = DEFAULT_ASK_DELAY_MAX_S,
                 label: str = "phone"):
        self._pool = pool
        self._signin_precheck = signin_precheck
        self._chatgpt_probe = chatgpt_probe
        # Proactive, no-fault rotation: after this many asks, voluntarily hand the
        # phone back (see `_rest`) even though nothing went wrong. On by default --
        # pass None (or 0) to opt out and keep the old stay-until-walled behaviour.
        self._rest_after = rest_after_asks
        self._asks_on_current = 0
        # Jittered gap between asks issued by this worker, on any phone. Also on by
        # default: repeated elicitation sweeps saw the runway before a CAPTCHA wall
        # shrink cycle over cycle (580 -> 211 -> 114 queries) at a flat 2s gap,
        # consistent with request density -- not just count -- feeding the farm's
        # shared-IP rate limit. Pass 0 for both to disable.
        lo = max(0.0, float(ask_delay_min_s or 0.0))
        hi = max(0.0, float(ask_delay_max_s or 0.0))
        # Tolerate a swapped pair rather than sampling an empty range: getting the
        # order wrong should still pace, not silently stop pacing.
        self._ask_delay = (min(lo, hi), max(lo, hi))
        # Each concurrent worker needs its own CDP port: the tunnel binds
        # <port> locally *and* `adb forward tcp:<port>` on the farm host, so two
        # workers on 9222 would collide on both ends.
        self._session_kw = dict(ssh_host=ssh_host, adb=adb, cdp_port=cdp_port,
                                sleep_on_exit=sleep_on_exit, debug=debug)
        self._google_kw = dict(settle_ms=settle_ms, answer_timeout_s=answer_timeout_s,
                               search_settle_ms=search_settle_ms, debug=debug)
        self._chat_kw = dict(settle_ms=chat_settle_ms,
                             answer_timeout_s=chat_answer_timeout_s,
                             warmup_s=chat_warmup_s, debug=debug)
        self._chatgpt_login = chatgpt_login
        self._label = label
        self._session: PhoneChromeSession | None = None
        self._scraper: PhoneFarmAIOverviewScraper | None = None
        self._chatgpt: PhoneChatGPTScraper | None = None
        self._serial: str | None = None
        # Consecutive asks on the current phone that produced no answer, per surface.
        # Kept apart because the surfaces wall independently: Google going quiet says
        # nothing about whether OpenAI is still answering this handset, and pooling
        # the two would rotate a phone that is only half-blocked.
        self._dry: dict[str, int] = {}

    @property
    def label(self) -> str:
        return self._label

    @property
    def current_serial(self) -> str | None:
        return self._serial

    async def __aenter__(self) -> "PhoneBackend":
        await self._open()
        return self

    async def __aexit__(self, *exc) -> None:
        await self._close()

    async def _open(self) -> None:
        """Take phones from the pool until one actually starts up.

        A handset can be adb-authorized yet unusable — most often Chrome runs but
        never opens `chrome_devtools_remote`, which needs Chrome's first run
        completed on the device itself. Such a phone is walled rather than handed
        back, so the next worker doesn't trip over it too.
        """
        while True:
            serial = await self._pool.acquire()   # raises PhoneFarmExhausted
            session = PhoneChromeSession(serial, **self._session_kw)
            try:
                await asyncio.to_thread(session.__enter__)
            except Exception as e:   # noqa: BLE001 — any startup failure disqualifies it
                logger.warning(f"[{self._label}] {serial} unusable "
                               f"({type(e).__name__}: {str(e)[:100]}); skipping")
                await self._pool.wall(serial)
                continue
            self._serial, self._session = serial, session
            self._scraper = PhoneFarmAIOverviewScraper(session=session, **self._google_kw)
            self._chatgpt = PhoneChatGPTScraper(session=session, **self._chat_kw)
            # Before this handset serves anything: is Chrome still signed in to
            # Google? A signed-out phone answers in full prose and cites nothing, so
            # its rows read as successes -- worth one page load per phone per run to
            # find out up front rather than a run's worth of sourceless answers.
            if self._signin_precheck and not await self._pool.is_verified(serial):
                try:
                    await asyncio.to_thread(self._scraper.assert_google_signed_in)
                except GoogleSignedOutError as e:
                    logger.error(f"[{self._label}] {e}")
                    await asyncio.to_thread(session.__exit__, None, None, None)
                    self._serial = self._session = None
                    self._scraper = self._chatgpt = None
                    await self._pool.wall(serial)
                    continue
                except Exception as e:   # noqa: BLE001 — a probe must not cost a phone
                    logger.warning(f"[{self._label}] sign-in probe on {serial} failed "
                                   f"({type(e).__name__}: {str(e)[:80]}); using it anyway")
                await self._pool.mark_verified(serial)
            self._dry = {}
            self._asks_on_current = 0
            logger.info(f"[{self._label}] using {serial} ({self._pool.snapshot()})")
            # Clear whatever this handset was still carrying before asking anything.
            # Asks close their own tabs now, but phones that predate that hold
            # hundreds, and a long tab list is exactly what makes the right tab hard
            # to find -- doubly so for ChatGPT, whose tab choice costs a CDP call per
            # open tab. Best-effort: a phone that will not close tabs still scrapes.
            for scraper in (self._scraper, self._chatgpt):
                try:
                    await asyncio.to_thread(scraper.clear_tab_backlog)
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"[{self._label}] tab cleanup on {serial} failed: {e}")
            if self._chatgpt_login:
                await self._sign_in_chatgpt(serial)
            return

    async def _sign_in_chatgpt(self, serial: str) -> None:
        """Make sure ChatGPT is signed in on this phone, best-effort.

        Cheap when it already is (one page read), and it usually is: the session
        persists in Chrome, so this is the second run onwards doing nothing. When it
        is not, `log_in` signs in with the handset's own Google account -- which is
        what keeps ChatGPT answering, since the wall that stops these phones is a
        check on *anonymous* visitors.

        A failure is logged and the phone is still used: anonymous asks work when the
        phone is not walled, and half a run is better than a handset dropped over a
        login. It does not wall the phone either -- that is for what the asks
        themselves report.
        """
        try:
            await asyncio.to_thread(self._chatgpt.ensure_logged_in)
        except Exception as e:  # noqa: BLE001 — never fail a run over the login
            logger.warning(f"[{self._label}] chatgpt login on {serial} failed "
                           f"({type(e).__name__}: {str(e)[:120]}); asking anonymously")

    async def _close(self) -> None:
        if self._session is not None:
            await asyncio.to_thread(self._session.__exit__, None, None, None)
            self._session = None
        self._scraper = self._chatgpt = None
        if self._serial is not None:
            await self._pool.release(self._serial)
            self._serial = None

    async def _rotate(self, reason: str = "CAPTCHA") -> None:
        """Wall the current phone and take the next free one from the pool."""
        if self._serial is not None:
            await self._pool.wall(self._serial)
            logger.warning(f"[{self._label}] {self._serial} walled ({reason}); "
                           f"rotating ({self._pool.snapshot()})")
            if self._session is not None:
                await asyncio.to_thread(self._session.__exit__, None, None, None)
                self._session = None
            self._scraper = self._chatgpt = None
            self._serial = None
        await self._open()   # raises PhoneFarmExhausted if none left

    async def _rest(self) -> None:
        """Voluntarily give up the current phone and pick up a fresh one.

        Unlike `_rotate`, this does NOT wall the serial -- it has done nothing
        wrong, it has just done its share for now. `_close` releases it back to
        the pool's free list (FIFO), so with more phones in the farm than
        concurrent workers this naturally round-robins the whole set: the phone
        that has rested longest is always the next one picked up. That spreads
        request volume over many distinct sessions instead of one phone
        absorbing a long unbroken run of queries, which is a more bot-like
        traffic pattern even when the total request rate to the shared IP is
        unchanged.
        """
        if self._serial is not None:
            logger.info(f"[{self._label}] {self._serial} resting after "
                       f"{self._asks_on_current} asks ({self._pool.snapshot()})")
            await self._close()
        await self._open()

    async def _ask_rotating(self, surface: str, call, answered, *, key: str = ""):
        """Run one ask, rotating handsets past a wall or a run of empty answers.

        Shared by every surface because the rotation rules are about the *phone*:
        a blocked ask means this handset is walled and the prompt should be tried on
        the next one, and a run of empty answers is the same evidence arriving
        quietly (see DRY_STREAK_BEFORE_ROTATE). `answered` says what counts as an
        answer for this surface, since the result types differ.
        """
        retried = False
        reasked = False
        moved = 0
        while True:
            if self._session is None:
                await self._open()
            elif self._rest_after and self._asks_on_current >= self._rest_after:
                await self._rest()
            # After the handset has been settled, not before the loop: every path
            # back to the top of this loop can have changed phones -- the rest above,
            # a rotate past a wall, a chat verdict moving the prompt on -- and the
            # ask below goes to whichever one we now hold. Checking only on the way
            # in let the proactive rest hand a fresh prompt to a handset already
            # ruled out of chat, once every `rest_after_asks`.
            if surface == "chatgpt" and self._chatgpt_probe:
                await self._skip_chat_incapable()
            self._asks_on_current += 1
            try:
                result = await asyncio.to_thread(call)
            except UnusableResultError as e:
                # The page rendered but cannot be trusted, so it is not written
                # down. Whether the handset is to blame decides what happens next.
                logger.error(f"[{self._label}] {e}")
                self._dry[surface] = 0
                if e.walls_handset:
                    await self._rotate(reason=type(e).__name__)
                    continue
                if reasked:
                    # A second handset saw the same thing, so the page was honest.
                    logger.warning(f"[{self._label}] two handsets agree; storing it")
                    return e.result
                reasked = True
                await self._rest()   # a different phone, but this one did no wrong
                continue
            if self._ask_delay[1]:
                await asyncio.sleep(random.uniform(*self._ask_delay))
            if result.blocked:
                self._dry[surface] = 0
                await self._rotate()   # try the same prompt on the next phone
                continue
            if answered(result):
                self._dry[surface] = 0
                if surface == "chatgpt" and self._chatgpt_probe:
                    # It held a conversation, so it is a working chat handset and its
                    # empty asks from here are the model's, not the phone's.
                    await self._pool.note_chat_answer(self._serial)
                return result
            # An unproven chat handset is judged by this ask, since it is the first
            # real one it has had; see `_chat_failure_verdict`.
            if surface == "chatgpt" and self._chatgpt_probe \
                    and moved < MAX_HANDSETS_PER_CHAT_PROMPT \
                    and await self._chat_failure_verdict(key) == "elsewhere":
                self._dry[surface] = 0
                moved += 1
                await self._rest()   # not walled: it still serves Google
                continue
            # An ask that produced nothing. One is ordinary; a run of them means this
            # handset has stopped answering, which Google does WITHOUT ever showing a
            # CAPTCHA. Rotate and re-ask once, so a genuinely unanswerable prompt
            # cannot walk the whole farm.
            self._dry[surface] = self._dry.get(surface, 0) + 1
            if self._dry[surface] < DRY_STREAK_BEFORE_ROTATE or retried:
                return result
            logger.warning(f"[{self._label}] {self._dry[surface]} {surface} asks in a row "
                           f"with no answer on {self._serial}; rotating (no wall seen)")
            self._dry[surface] = 0
            retried = True
            await self._rotate()

    async def search(self, prompt: str):
        """AI Mode answer for `prompt`, rotating past a CAPTCHA or a dry streak."""
        return await self._ask_rotating(
            "ai_mode",
            lambda: self._scraper.search(prompt),
            lambda r: r.has_overview,
        )

    async def chat(self, prompt: str):
        """ChatGPT's answer to `prompt`, rotating past a wall or a dry streak.

        Rotates on the same evidence as `search`, on the ChatGPT versions of it: the
        verification wall and the anonymous usage cap both arrive as `blocked`. The
        walls are independent of Google's, which is why the dry streak is counted
        per surface -- a phone Google has stopped answering usually still chats.

        Handsets already out for chat are skipped inside `_ask_rotating`, which is
        the only place that knows which one is about to be asked.
        """
        return await self._ask_rotating(
            "chatgpt",
            lambda: self._chatgpt.ask(prompt),
            lambda r: bool(r.response),
            key=prompt,
        )

    async def _skip_chat_incapable(self) -> None:
        """Move off a handset already known not to hold ChatGPT conversations."""
        if not self._chatgpt_probe:
            return
        for _ in range(max(1, self._pool.total)):
            if self._session is None:
                await self._open()
            if await self._pool.chatgpt_state(self._serial) is not False:
                return
            logger.info(f"[{self._label}] {self._serial} is out for chat "
                        f"({self._pool.chatgpt_snapshot()}); trying another handset")
            await self._rest()
        raise PhoneFarmExhausted(
            f"no handset in the farm can hold a ChatGPT conversation "
            f"({self._pool.chatgpt_snapshot()})")

    async def _chat_failure_verdict(self, prompt: str) -> str:
        """What to do about a ChatGPT ask that came back empty: store it, or re-ask.

        The handset's first real ask is its capability test -- there is no throwaway
        ask, so finding out costs nothing that was not being asked anyway. Until a
        handset has answered one, its empty asks are treated as being about the phone:
        they are re-asked elsewhere rather than written down as answers the model did
        not give. Once it has answered, it has proved itself and its empty asks are
        the model's, so they store as before.
        """
        serial = self._serial
        strikes = await self._pool.note_chat_failure(serial, prompt)
        if await self._pool.chatgpt_state(serial):
            return "store"
        if strikes >= CHAT_STRIKES_BEFORE_EXCLUDING:
            await self._pool.set_chatgpt_ok(serial, False)
            logger.error(
                f"[{self._label}] {serial}: {strikes} different prompts have come "
                f"back empty and it has never answered one, so it is out of ChatGPT "
                f"asks for this run ({self._pool.chatgpt_snapshot()}). Its Google "
                f"asks are unaffected.")
        return "elsewhere"

    async def search_normal(self, prompt: str, top_n: int = 10):
        """Google's normal top results for `prompt`, rotating phones past any CAPTCHA.

        Rotates like `search` rather than returning the blocked result: a wall is a
        property of the phone, not of the query, and the two surfaces share one
        handset -- so a caller collecting both would otherwise get an answer for AI
        Mode and a silent blank for the web results off the same walled phone.

        No dry-streak rotation, unlike `search`: `search_normal` cannot come back
        empty any more. A page that ranks nothing is raised as an
        `EmptySearchResultsError` and re-asked on another handset, so whatever
        arrives here is either real results or an emptiness two phones agreed on.
        """
        return await self._ask_rotating(
            "search",
            lambda: self._scraper.search_normal(prompt, top_n),
            lambda r: True,
        )


def discover_serials(*, serials: list[str] | None = None,
                     exclude: list[str] | None = None,
                     ssh_host: str | None = None,
                     adb: str | None = None) -> list[str]:
    """The farm's serials: `serials` if pinned, else live from `adb devices`.

    `exclude` drops handsets by serial afterwards, either way. For a phone that is
    known bad and cannot be fixed yet -- signed out of Google, borrowed by another
    project -- this keeps it out of the farm without editing the pinned list or
    unplugging it, and without spending an ask per run rediscovering the problem.
    """
    if serials:
        found = list(serials)
    else:
        found = list_serials(ssh_host=ssh_host, adb=adb)
        if found:
            logger.info(f"[phone] discovered {len(found)} phone(s) from adb: {found}")
    if not found:
        raise PhoneFarmExhausted(
            "no ready phones found via `adb devices` — check the farm is "
            "reachable and phones are authorized (`adb devices` shows 'device')")
    if exclude:
        dropped = [s for s in found if s in set(exclude)]
        found = [s for s in found if s not in set(exclude)]
        if dropped:
            logger.warning(f"[phone] excluding {len(dropped)} phone(s) by config: {dropped}")
        if not found:
            raise PhoneFarmExhausted(
                f"every phone in the farm is in exclude_serials ({dropped}); "
                "there is nothing left to scrape with")
    return found


def open_phone_backend(*, serials: list[str] | None = None,
                       exclude_serials: list[str] | None = None,
                       cdp_port: int = DEFAULT_CDP_PORT,
                       wall_cooldown_s: float = DEFAULT_WALL_COOLDOWN_S,
                       **kwargs) -> PhoneBackend:
    """One phone at a time, rotating through the farm on CAPTCHA."""
    pool = SerialPool(discover_serials(serials=serials, exclude=exclude_serials,
                                       ssh_host=kwargs.get("ssh_host"),
                                       adb=kwargs.get("adb")),
                      wall_cooldown_s=wall_cooldown_s)
    return PhoneBackend(pool, cdp_port=cdp_port, label="phone", **kwargs)


def open_phone_workers(n: int, *, serials: list[str] | None = None,
                       exclude_serials: list[str] | None = None,
                       cdp_port: int = DEFAULT_CDP_PORT,
                       wall_cooldown_s: float = DEFAULT_WALL_COOLDOWN_S,
                       **kwargs) -> tuple[list[PhoneBackend], SerialPool]:
    """`n` concurrent phone workers sharing one SerialPool.

    Each gets its own CDP port (base, base+1, …) because the tunnel binds that
    port on both this machine and the farm host. `n` is clamped to the number of
    phones actually available — asking for more workers than handsets would just
    starve the extras at startup.

    NOTE: the phones usually share one WiFi egress IP and Google's burst limit is
    largely IP-level, so `n` concurrent workers multiply the request rate from a
    single IP. Raising this makes a CAPTCHA *more* likely, not less; it buys
    wall-clock, not headroom.
    """
    found = discover_serials(serials=serials, exclude=exclude_serials,
                             ssh_host=kwargs.get("ssh_host"), adb=kwargs.get("adb"))
    n = max(1, min(n, len(found)))
    pool = SerialPool(found, wall_cooldown_s=wall_cooldown_s)
    workers = [
        PhoneBackend(pool, cdp_port=cdp_port + i, label=f"phone{i + 1}", **kwargs)
        for i in range(n)
    ]
    return workers, pool
