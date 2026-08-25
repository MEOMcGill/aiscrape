"""Google AI Mode scraper that drives a real Android phone over adb.

A second way to get a Google **AI Mode** answer + its cited sources, parallel to
`google_aimode.AIOverviewScraper` (camoufox on the desktop). Instead of a stealth
browser it drives **Chrome on a physical Android phone** through `adb`:

    launch Chrome at the `udm=50` search URL  →  wait for the answer to render  →
    read the rendered DOM over the Chrome DevTools Protocol (CDP)  →  parse the
    same `AIOverviewResult` shape the desktop scraper returns.

Why a phone: the handset is a real logged-in Google account on a residential
mobile IP, which sails past the "unusual traffic" CAPTCHA that walls the desktop
scraper after a burst. The trade-off is throughput — see the notes below.

The phones usually hang off a remote host (e.g. a Windows laptop reachable over
SSH); set `ssh_host` and every `adb` call is prefixed with that one SSH hop. Runs
against a local adb too (`ssh_host=None`).

CDP-over-adb has three non-obvious gotchas, all handled here:
  1. `adb forward` is torn down when the ssh session that created it exits, so the
     forward is established *inside* the same long-lived ssh process that holds the
     `-L` tunnel open.
  2. the tunnel's remote target must be `127.0.0.1`, not `localhost` (which can
     resolve to `::1` while adb binds IPv4).
  3. Chrome rejects the CDP websocket with 403 unless the `Origin` header is
     omitted (`suppress_origin=True`) or it is launched with `--remote-allow-origins`.

Extraction reuses the desktop parser's building blocks (`#cnt` container,
`_real_url`/`_domain`/`_clean_title`), so rows join cleanly with desktop rows —
only `surface` differs ("phone_farm" vs "ai_mode").

The phone itself — waking it, starting Chrome, holding the tunnel, driving tabs —
lives in `PhoneChromeSession`, separately from the Google scraping above it, because
a handset also serves ChatGPT (`phone_chatgpt`). One session per phone, shared by
every surface: the tunnel binds a port on both ends that a second one would collide
with. `PhoneFarmAIOverviewScraper` opens its own session when handed a serial, and
takes an existing one when handed a `session=`.

Throughput: all handsets on one WiFi share a single egress IP, and Google's burst
limit is largely IP-level, so rotating phones spreads the *per-account* load but
not the *per-IP* load. Keep concurrency modest unless the phones have their own
SIMs/proxies.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from urllib.parse import parse_qs, quote_plus, urlparse

from websocket import create_connection

from .google_aimode import (
    AIOverviewResult,
    Reference,
    SearchResult,
    _clean_title,
    _domain,
    _real_url,
    _LEADING_CHROME,
    _TRAILING_CHROME_RE,
)
from .logger import logger
from .models import now_iso

AI_MODE_URL = "https://www.google.com/search?q={q}&udm=50"
# The same page WITHOUT `udm=50`: Google's ordinary ranked web results. What Google
# surfaces to a user, as opposed to what the AI Mode answer chooses to cite -- the
# point of collecting both from one phone is that they are then comparable.
SEARCH_URL = "https://www.google.com/search?q={q}"
# Sign-in probe. A Google page that needs an account: a handset with a live web
# session stays on it, one without is bounced away. Cheaper and far less ambiguous
# than reading a search page -- it spends no search quota, and the signed-out marker
# on a results page lives in the header, outside the container the extractor reads
# (the Google homepage renders identically either way, so that is no use either).
#
# The test is that we STAYED, not where we were sent: Google varies the destination
# -- the sign-in form, or the "About Google Account" marketing page -- and a list of
# bounce targets goes stale without saying so, which is the failure mode this whole
# change exists to stop.
SIGNIN_PROBE_URL = "https://myaccount.google.com/"
SIGNIN_PROBE_HOST = "myaccount.google.com"
CHROME_PKG = "com.android.chrome"
DEFAULT_CDP_PORT = 9222

# Plain results are static HTML -- nothing streams in -- so the AI Mode settle (9s,
# sized for a multi-paragraph answer arriving token by token) is dead time here. This
# is just "has the page navigated and painted".
DEFAULT_SEARCH_SETTLE_MS = 2_500

# What AI Mode puts in the answer container when it declined to generate one. It looks
# like prose, is long enough to pass the readiness bar, and carries citations, so
# without this check it is stored as a real answer. Not a CAPTCHA: the session is
# still good, the ask just produced nothing, so it counts as a failed ask rather than
# a block.
#
# Matched on "...wasn't generated" rather than the "Something went wrong" these
# messages open with, because that opener is a phrase an answer could plausibly use
# ("...because something went wrong with the fiscal transfers") and discarding a real
# answer is the worse error. Apostrophes are normalised first: Google has served both
# the straight and the typographic form.
_AI_MODE_ERRORS = ("an ai response wasn't generated",
                   "the content wasn't generated")


def _looks_like_ai_refusal(text: str) -> bool:
    return any(m in text.lower().replace("’", "'") for m in _AI_MODE_ERRORS)

# Extra trailing markers the mobile AI Mode surface appends after the answer.
_MOBILE_TRAILING = ("Ask anything", "AI Mode response is ready",
                    "AI responses may include mistakes")

# On mobile the bottom "sources" strip renders *inside* #cnt as a run of source
# cards, each led by a snippet line like "Jun 12, 2026 — <excerpt>…". That date
# form never starts a line of the answer prose, so it marks where the answer ends.
_SOURCE_CARD_DATE_RE = re.compile(
    r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* "
    r"\d{1,2},? \d{4}\s*[—–-]")
_SENTENCE_END = (".", "!", "?", ":", '"', "”", ")")
# The sources-count chip ("10 sites") that leads the rendered answer.
_SITES_CHIP_RE = re.compile(r"^\d+\s+sites?$", re.IGNORECASE)

# One CDP round-trip: return the answer container's text + every anchor's title
# candidates + a slice of the container HTML. Google's own chrome links are left
# in and filtered out in Python (needs urlparse).
_EXTRACT_JS = r"""
(() => {
  const cnt = document.querySelector('#cnt')
           || document.querySelector('#main')
           || document.body;
  if (!cnt) return JSON.stringify({found: false});
  const anchors = [...cnt.querySelectorAll('a[href]')]
    .filter(a => /^https?:/.test(a.href))
    .map(a => ({
      href: a.href,
      text: (a.innerText || '').trim(),
      aria: (a.getAttribute('aria-label') || '').trim(),
    }));
  return JSON.stringify({
    found: true,
    url: location.href,
    text: cnt.innerText || '',
    html: (cnt.innerHTML || '').slice(0, 200000),
    anchors,
  });
})()
"""


# One CDP round-trip for a plain results page. Mirrors the desktop path's
# `a:has(h3)` inside `#rso` (`#search` is the wider wrapper): the title link is what
# positively identifies a *result*, as opposed to nav chrome or a "People also ask"
# entry.
#
# Every external anchor is returned with a `heading` flag rather than filtered in JS,
# so Python can fall back to un-headed anchors and SAY it did. Mobile Google is
# re-skinned often, and a selector that quietly matches nothing would otherwise look
# exactly like a query with no results.
_SEARCH_EXTRACT_JS = r"""
(() => {
  const region = document.querySelector('#rso')
              || document.querySelector('#search')
              || document.querySelector('#main')
              || document.body;
  if (!region) return JSON.stringify({found: false});
  const anchors = [...region.querySelectorAll('a[href]')]
    .filter(a => /^https?:/.test(a.href))
    .map(a => {
      const h = a.querySelector('h3, [role="heading"]');
      const heading = (h && h.innerText || '').trim();
      return {
        href: a.href,
        text: heading || (a.innerText || '').trim(),
        aria: (a.getAttribute('aria-label') || '').trim(),
        heading: !!heading,
      };
    });
  return JSON.stringify({
    found: true,
    url: location.href,
    // Body text, not the region's: on a CAPTCHA page there is no results region at
    // all, and this is what the block check reads.
    text: (document.body.innerText || '').slice(0, 4000),
    anchors,
  });
})()
"""


class PhoneFarmError(RuntimeError):
    """adb / CDP transport failure while driving a phone."""


class UnusableResultError(RuntimeError):
    """The page rendered, but what came back must not be stored as a result.

    Raised instead of returning an empty result, because the two are not the same
    thing downstream: an empty result is written as a row saying Google had no
    answer, while raising puts the prompt on another handset. Anything that makes a
    *parse* fail belongs here; a query Google genuinely has nothing for does not.

    `walls_handset` says whether the evidence accuses the phone. When it does, the
    pool takes that phone out of rotation; when it only says the page was odd, the
    pool re-asks elsewhere and believes the second handset.
    """

    walls_handset = True

    def __init__(self, message: str, result=None):
        super().__init__(message)
        # The result that was withheld, so a caller that decides the page was
        # honest after all can still store it rather than re-ask forever.
        self.result = result


class GoogleSignedOutError(UnusableResultError):
    """Chrome on this handset has no Google session, so no source URL survives."""


class MissingCitationsError(UnusableResultError):
    """The answer says how many sites it cites and none of them could be parsed."""


class EmptySearchResultsError(UnusableResultError):
    """A plain search page that ranked nothing at all.

    Google effectively always ranks something for a natural-language query, so an
    empty page is a parse failure, a consent wall or a re-skin far more often than
    it is the truth about the query. Not proof against the handset though -- unlike
    the two above, nothing here points at the phone -- so this one re-asks
    elsewhere instead of walling it, and a second handset agreeing settles it.
    """

    walls_handset = False


# ── adb transport (local or over one SSH hop) ────────────────────────────────

@dataclass
class _Adb:
    """Runs adb, either locally or by prefixing one SSH hop to a remote host."""
    adb: str = "adb"
    ssh_host: str | None = None
    timeout: int = 60

    def _run(self, cmd: str) -> subprocess.CompletedProcess:
        if self.ssh_host:
            return subprocess.run(["ssh", self.ssh_host, cmd], capture_output=True,
                                  text=True, timeout=self.timeout)
        # The LOCAL shell, not bash. On the farm laptop `bash` is WSL's bash, and
        # running adb there fails twice over: WSL eats the backslashes, so
        # `C:\platform-tools\adb.exe` arrives as `C:platform-toolsadb.exe`, and even
        # with the path fixed WSL has no USB access -- which is the whole reason the
        # scrape runs natively on Windows.
        # shell=True gets cmd.exe there and /bin/sh on POSIX; every command built here
        # is plain POSIX-compatible quoting, and `shell()` already documents relying on
        # cmd.exe keeping `&` literal inside double quotes.
        return subprocess.run(cmd, capture_output=True, text=True, shell=True,
                              timeout=self.timeout)

    def shell(self, serial: str, script: str) -> str:
        """Run `adb -s <serial> shell "<script>"`.

        `script` runs in the phone's own sh (so `;`-chaining works); wrap URLs in
        single quotes inside it so `&` survives. cmd.exe on a Windows ssh host
        keeps `&` literal because the whole thing is double-quoted.
        """
        out = self._run(f'{self.adb} -s {serial} shell "{script}"')
        return out.stdout

    def raw(self, serial: str, args: str) -> str:
        out = self._run(f"{self.adb} -s {serial} {args}")
        return out.stdout

    def devices(self) -> list[tuple[str, str]]:
        """`adb devices` as (serial, state) pairs, e.g. ("R58MEXAMPLE", "device").

        States other than "device" mean the handset is attached but not drivable:
        "unauthorized" (the on-screen "Allow USB debugging" prompt hasn't been
        accepted), "offline", "no permissions".
        """
        out = self._run(f"{self.adb} devices")
        pairs = []
        for line in out.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("List of devices") or line.startswith("*"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
        return pairs


def list_serials(*, ssh_host: str | None = None, adb: str | None = None,
                 timeout: int = 60) -> list[str]:
    """Serials of the phones that are actually drivable right now.

    Only handsets in adb state "device" are returned — an "unauthorized" phone
    needs its on-screen USB-debugging prompt accepted before it can be used, so
    it is silently invisible to a run rather than failing one mid-sweep.
    """
    adb_t = _Adb(adb=adb or os.getenv("AISCRAPE_ADB", "adb"),
                 ssh_host=ssh_host or os.getenv("AISCRAPE_PHONE_SSH") or None,
                 timeout=timeout)
    pairs = adb_t.devices()
    ready = [s for s, state in pairs if state == "device"]
    if not ready and pairs:
        states = ", ".join(f"{s}={st}" for s, st in pairs)
        logger.warning(f"[phone] no drivable phones; adb reports: {states}")
    return ready


# ── CDP tunnel: adb forward + (optional) SSH -L, kept alive together ──────────

class _CdpTunnel:
    """Expose the phone's Chrome DevTools endpoint at http://127.0.0.1:<port>.

    Over SSH, one process both adds the `adb forward` and holds the `-L` tunnel
    open (a keepalive), because the forward dies with its creating session.
    """

    def __init__(self, adb: _Adb, serial: str, port: int = DEFAULT_CDP_PORT):
        self._adb = adb
        self._serial = serial
        self._port = port
        self._proc: subprocess.Popen | None = None

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    def __enter__(self) -> "_CdpTunnel":
        fwd = (f"{self._adb.adb} -s {self._serial} forward tcp:{self._port} "
               f"localabstract:chrome_devtools_remote")
        if self._adb.ssh_host:
            # Windows keepalive; killed when we terminate the ssh process.
            keepalive = "ping -n 100000 127.0.0.1 >NUL"
            self._proc = subprocess.Popen(
                ["ssh", "-L", f"{self._port}:127.0.0.1:{self._port}",
                 self._adb.ssh_host, f"{fwd} && {keepalive}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # Through _Adb, so the local case uses the platform's own shell -- `bash`
            # here is WSL on the farm laptop, which cannot reach USB. Same reason as
            # _Adb._run; this was a second copy of that call.
            done = self._adb._run(fwd)
            if done.returncode != 0:
                raise PhoneFarmError(
                    f"`adb forward` failed for {self._serial} (rc {done.returncode}): "
                    f"{(done.stderr or done.stdout or '').strip()[:200]}")
        self._wait_ready()
        return self

    def __exit__(self, *exc) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        else:
            self._adb.raw(self._serial, f"forward --remove tcp:{self._port}")

    def _wait_ready(self, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            try:
                self._get("/json/version")
                return
            except Exception as e:  # noqa: BLE001 — retry any transport hiccup
                last = e
                time.sleep(0.7)
        raise PhoneFarmError(f"CDP endpoint never came up on {self.base}: {last}")

    def _get(self, path: str) -> dict | list:
        with urllib.request.urlopen(self.base + path, timeout=8) as r:
            return json.loads(r.read().decode())

    def pages(self) -> list[dict]:
        data = self._get("/json/list")
        return [p for p in data if p.get("type") == "page"]

    def activate(self, target_id: str) -> bool:
        """Bring one tab to the foreground. False if Chrome refused or it was gone."""
        try:
            with urllib.request.urlopen(f"{self.base}/json/activate/{target_id}",
                                        timeout=8) as r:
                r.read()
            return True
        except Exception:  # noqa: BLE001 — best-effort, like `close`
            return False

    def close(self, target_id: str) -> bool:
        """Close one tab by CDP target id. False if Chrome refused or it was gone.

        Never raises: a tab that cannot be closed is untidy, not a failed scrape, and
        this runs in the teardown of every ask.
        """
        try:
            with urllib.request.urlopen(f"{self.base}/json/close/{target_id}",
                                        timeout=8) as r:
                r.read()
            return True
        except Exception:  # noqa: BLE001 — closing is best-effort by design
            return False

    def close_tabs_matching(self, needle: str, keep: int = 1) -> int:
        """Close every open tab whose URL contains `needle`, leaving `keep` alone.

        For clearing a backlog: the phones in the farm had accumulated hundreds of
        AI Mode tabs, one per ask ever made, which is what made tab choice ambiguous
        (see `_ai_mode_page`). Leaves one tab behind so Chrome is not left with zero
        and reopening does not race the next `am start`.
        """
        try:
            targets = [p for p in self.pages() if needle in p.get("url", "")]
        except Exception:  # noqa: BLE001
            return 0
        closed = 0
        for page in targets[keep:]:
            if page.get("id") and self.close(page["id"]):
                closed += 1
        return closed

    def close_google_tabs(self, keep: int = 1) -> int:
        return self.close_tabs_matching("google.com/search", keep=keep)


def _cdp_calls(ws_url: str, calls: list[tuple[str, dict]], timeout: float = 25.0) -> list[dict]:
    """Send a sequence of CDP commands over one socket and return their results.

    One socket for the lot because the tunnel runs over adb and (usually) an ssh hop:
    typing eight characters as eight connections would spend most of its time in
    handshakes.
    """
    ws = create_connection(ws_url.replace("localhost", "127.0.0.1"),
                           timeout=timeout, suppress_origin=True)
    try:
        results = []
        for n, (method, params) in enumerate(calls, start=1):
            ws.send(json.dumps({"id": n, "method": method, "params": params}))
            while True:
                msg = json.loads(ws.recv())
                if msg.get("id") == n:
                    results.append(msg.get("result", {}))
                    break
        return results
    finally:
        ws.close()


def _cdp_evaluate(ws_url: str, expression: str, timeout: float = 25.0):
    """Open a CDP websocket, run one Runtime.evaluate, return its JS value."""
    ws = create_connection(ws_url.replace("localhost", "127.0.0.1"),
                           timeout=timeout, suppress_origin=True)
    try:
        ws.send(json.dumps({
            "id": 1, "method": "Runtime.evaluate",
            "params": {"expression": expression, "returnByValue": True,
                       "awaitPromise": True},
        }))
        while True:
            msg = json.loads(ws.recv())
            if msg.get("id") == 1:
                result = msg.get("result", {})
                if "exceptionDetails" in result:
                    raise PhoneFarmError(f"CDP eval failed: {result['exceptionDetails']}")
                return result.get("result", {}).get("value")
    finally:
        ws.close()


# ── one phone's Chrome, shared by every surface scraped on it ────────────────

class PhoneChromeSession:
    """Chrome on one handset: wake it, start it, hold the CDP tunnel, drive tabs.

    Everything here is about the *phone*, not about what is being scraped on it,
    and it is a separate object because a handset now serves more than one surface
    -- Google AI Mode and plain search (`PhoneFarmAIOverviewScraper`) and ChatGPT
    (`phone_chatgpt.PhoneChatGPTScraper`). The tunnel binds one TCP port on this
    machine *and* one `adb forward` on the farm host, so two scrapers each opening
    their own session for one phone would collide on both ends; they take one of
    these instead and share the tunnel, the wake and the Chrome start.

    Use as a context manager. A scraper handed a session it did not create leaves
    the lifecycle to whoever did::

        with PhoneChromeSession("R58MEXAMPLE", ssh_host="phone-farm") as phone:
            google = PhoneFarmAIOverviewScraper(session=phone)
            chatgpt = PhoneChatGPTScraper(session=phone)
    """

    def __init__(self, serial: str, *, ssh_host: str | None = None,
                 adb: str | None = None, cdp_port: int = DEFAULT_CDP_PORT,
                 keep_awake: bool = True, sleep_on_exit: bool = False,
                 debug: bool = False):
        self.serial = serial
        self.adb = _Adb(
            adb=adb or os.getenv("AISCRAPE_ADB", "adb"),
            ssh_host=ssh_host or os.getenv("AISCRAPE_PHONE_SSH") or None,
        )
        self.cdp_port = cdp_port
        self.keep_awake = keep_awake
        self.sleep_on_exit = sleep_on_exit
        self.debug = debug
        self._tunnel: _CdpTunnel | None = None
        self._screen_size: tuple[int, int] | None = None

    # -- lifecycle --------------------------------------------------------------

    def __enter__(self) -> "PhoneChromeSession":
        if self.keep_awake:
            self.wake()
        self._ensure_chrome()
        self._tunnel = _CdpTunnel(self.adb, self.serial, self.cdp_port)
        self._tunnel.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._tunnel is not None:
            self._tunnel.__exit__(*exc)
            self._tunnel = None
        if self.sleep_on_exit:
            self.adb.shell(self.serial,
                           "svc power stayon false; input keyevent KEYCODE_SLEEP")

    @property
    def open(self) -> bool:
        return self._tunnel is not None

    def require_open(self) -> _CdpTunnel:
        if self._tunnel is None:
            raise PhoneFarmError(
                f"the CDP tunnel to {self.serial} is not open — use the phone "
                "session (or the scraper that owns it) as a context manager")
        return self._tunnel

    def wake(self) -> None:
        """Wake the screen and keep it awake (Samsung AOD won't render otherwise)."""
        self.adb.shell(
            self.serial,
            "svc power stayon true; input keyevent KEYCODE_WAKEUP; "
            "input keyevent KEYCODE_HOME")

    def _ensure_chrome(self, timeout: float = 40.0) -> None:
        """Start Chrome and wait for its DevTools socket to exist.

        The tunnel forwards to `localabstract:chrome_devtools_remote`, which only
        exists *while Chrome is running*. On a phone that has been sitting idle
        Chrome is dead, so `adb forward` still succeeds but every connection to it
        is refused — surfacing as "CDP endpoint never came up". Launching Chrome
        on about:blank first (rather than straight at the query) keeps this step
        independent of what we're about to scrape.
        """
        deadline = time.monotonic() + timeout
        launched = False
        while time.monotonic() < deadline:
            sockets = self.adb.shell(
                self.serial, "cat /proc/net/unix | grep chrome_devtools_remote")
            if "chrome_devtools_remote" in sockets:
                return
            if not launched:
                self.launch("about:blank")
                launched = True
            time.sleep(1.0)
        raise PhoneFarmError(
            f"Chrome's DevTools socket never appeared on {self.serial} after "
            f"{timeout:.0f}s — is Chrome installed and USB debugging authorized?")

    # -- driving Chrome ---------------------------------------------------------

    def launch(self, url: str) -> None:
        """Open `url` in Chrome on the phone (single-quoted so `&` survives)."""
        self.adb.shell(
            self.serial,
            f"am start -a android.intent.action.VIEW -d '{url}' {CHROME_PKG}")

    def pages(self) -> list[dict]:
        """Every open tab, or [] if CDP could not be read this moment."""
        try:
            return self.require_open().pages()
        except Exception as e:  # noqa: BLE001 — a poll failing is not a scrape failing
            if self.debug:
                logger.debug(f"[{self.serial}] /json/list failed: {e}")
            return []

    def evaluate(self, page: dict, expression: str) -> dict | None:
        """Run one JS expression in `page` and return its decoded JSON value.

        None on any transport failure, so a caller polling a page it just launched
        treats "the tab is not answering yet" the same as "not ready yet".
        """
        if not page or "webSocketDebuggerUrl" not in page:
            return None
        try:
            value = _cdp_evaluate(page["webSocketDebuggerUrl"], expression)
        except Exception as e:  # noqa: BLE001
            if self.debug:
                logger.debug(f"[{self.serial}] CDP eval failed: {e}")
            return None
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # Every extractor here returns JSON.stringify(...), so this is a page that
            # answered with something else entirely. Treated as "not ready", like any
            # other unreadable read, rather than taking down the scrape it was polling.
            if self.debug:
                logger.debug(f"[{self.serial}] CDP eval returned non-JSON: {value[:120]!r}")
            return None

    def activate(self, page: dict) -> bool:
        """Bring `page` to the foreground, and give Chrome a beat to wake it.

        Android Chrome freezes background tabs: their JS stops running, so a click
        dispatched into one does nothing and the page sits at whatever it had
        rendered. Anything that drives a page rather than just reading it — the
        ChatGPT login flow, which walks across several tabs — has to foreground it
        first. Reading is unaffected: a frozen tab still answers `Runtime.evaluate`.
        """
        if self._tunnel is None or not page or not page.get("id"):
            return False
        done = self._tunnel.activate(page["id"])
        if done:
            time.sleep(1.5)
        return done

    def screen_state(self) -> str:
        """"ON_UNLOCKED" / "ON_LOCKED" / "OFF", or "" if the phone did not say.

        Read out of `dumpsys nfc`, which reports it in one line on these Samsungs.
        """
        match = re.search(r"mScreenState=(\S+)",
                          self.adb.shell(self.serial, "dumpsys nfc | grep mScreenState"))
        return match.group(1) if match else ""

    def unlock(self, tries: int = 3) -> bool:
        """Wake the phone and swipe past a swipe-only lock screen. True if unlocked.

        Matters only where something is *typed*: CDP reaches Chrome through the lock
        screen quite happily, so scraping never noticed, but `adb input` goes to
        whatever is actually on the display — so on a locked phone the keystrokes
        land on the lock screen and the form they were meant for stays empty. That
        is what made the ChatGPT signup fail on one handset and work on another.

        Returns False for a phone locked with a PIN or pattern, which nothing here
        can answer; that needs a person, once.
        """
        for _ in range(tries):
            if "UNLOCKED" in self.screen_state():
                return True
            width, height = self.screen_size()
            self.adb.shell(
                self.serial,
                f"input keyevent KEYCODE_WAKEUP; "
                f"input swipe {width // 2} {int(height * 0.8)} {width // 2} {int(height * 0.2)}")
            time.sleep(1.5)
        return "UNLOCKED" in self.screen_state()

    def foreground_chrome(self) -> None:
        """Put Chrome back in front of whatever is covering it.

        Android freezes a browser tab that is not on screen, so *anything* over
        Chrome — the notification shade left pulled down, the launcher after a HOME
        keypress — stops its JS and quietly turns every click into a no-op. That is
        not hypothetical: two handsets sat at the Google account chooser doing
        nothing at all, with the shade open over Chrome the whole time.

        HOME first because the shade has to be dismissed before anything can be
        raised; then Chrome's launcher activity, which brings it forward without
        touching which tab is open.
        """
        self.adb.shell(
            self.serial,
            "input keyevent KEYCODE_HOME; "
            "am start -a android.intent.action.MAIN -c android.intent.category.LAUNCHER "
            f"-n {CHROME_PKG}/com.google.android.apps.chrome.Main")
        time.sleep(1.5)

    def focused_app(self) -> str:
        """The window with focus, e.g. "com.android.chrome/...ChromeTabbedActivity"."""
        match = re.search(r"mCurrentFocus=\S+ \S+ (\S+)}",
                          self.adb.shell(self.serial, "dumpsys window | grep -m1 mCurrentFocus"))
        return match.group(1) if match else ""

    def screen_size(self, default: tuple[int, int] = (1080, 2340)) -> tuple[int, int]:
        """The display in pixels, for swipes. Cached; falls back to a common size."""
        if self._screen_size is None:
            match = re.search(r"(\d+)x(\d+)", self.adb.shell(self.serial, "wm size"))
            self._screen_size = ((int(match.group(1)), int(match.group(2)))
                                 if match else default)
        return self._screen_size

    def insert_text(self, page: dict, text: str) -> bool:
        """Type `text` into whatever has focus in `page`, as an IME would.

        For fields where setting `value` from JS is not enough: OpenAI's signup
        rejects a name whose React state never saw real input, however right the DOM
        value looks. `Input.insertText` is a browser-level edit, so React sees what
        it would see from a person.

        Deliberately not `adb shell input text`, which was the first thing tried
        here: those keystrokes go to whatever is on the *display*, so they land on
        the lock screen of a locked phone (and, on one handset, went nowhere even
        unlocked). Going through CDP keeps typing a property of the tab rather than
        of what the screen happens to be showing.
        """
        try:
            _cdp_calls(page["webSocketDebuggerUrl"], [("Input.insertText", {"text": text})])
            return True
        except Exception as e:  # noqa: BLE001
            if self.debug:
                logger.debug(f"[{self.serial}] insertText failed: {e}")
            return False

    def type_keys(self, page: dict, text: str) -> bool:
        """Send `text` to `page` one keystroke at a time.

        The slower sibling of `insert_text`, for widgets that listen for keys rather
        than for edits -- the react-aria date field on ChatGPT's signup takes its
        digits this way and ignores an inserted string.
        """
        calls = []
        for ch in text:
            key = {"key": ch, "text": ch, "windowsVirtualKeyCode": ord(ch.upper())}
            calls.append(("Input.dispatchKeyEvent", {"type": "keyDown", **key}))
            calls.append(("Input.dispatchKeyEvent", {"type": "keyUp", **key}))
        try:
            _cdp_calls(page["webSocketDebuggerUrl"], calls)
            return True
        except Exception as e:  # noqa: BLE001
            if self.debug:
                logger.debug(f"[{self.serial}] key events failed: {e}")
            return False

    def google_accounts(self) -> list[str]:
        """The Google accounts signed in on this phone, from `adb dumpsys account`.

        Which is how the ChatGPT login knows who to sign in as without being told:
        each handset carries its own Google account.
        """
        out = self.adb.shell(self.serial, "dumpsys account | grep com.google")
        found, seen = [], set()
        for match in re.finditer(r"name=([^,}]+), type=com\.google\b", out):
            email = match.group(1).strip()
            if "@" in email and email not in seen:
                seen.add(email)
                found.append(email)
        return found

    def close_tab(self, target_id: str) -> bool:
        if not target_id or self._tunnel is None:
            return False
        return self._tunnel.close(target_id)

    def close_tabs_matching(self, needle: str, *, keep: int = 1) -> int:
        """Close open tabs whose URL contains `needle`, leaving `keep` behind."""
        if self._tunnel is None:
            return 0
        return self._tunnel.close_tabs_matching(needle, keep=keep)


# ── the scraper ──────────────────────────────────────────────────────────────

class PhoneFarmAIOverviewScraper:
    """Scrape Google AI Mode by driving Chrome on one Android phone over adb+CDP.

    Args:
        serial:       the phone's adb serial (from `adb devices`). Optional when
                      `session` is given, which already names a handset.
        session:      an open `PhoneChromeSession` to scrape on, instead of opening
                      one for `serial`. Pass this when something else is already
                      driving the phone (the ChatGPT scraper, say) so both share the
                      one tunnel; whoever created the session also closes it.
        ssh_host:     SSH host the phone hangs off (None → local adb). Falls back
                      to env `AISCRAPE_PHONE_SSH`.
        adb:          path to the adb binary on that host. Falls back to env
                      `AISCRAPE_ADB`, else "adb". (On a Windows farm host:
                      ``C:\\platform-tools\\adb.exe``.)
        cdp_port:     local+remote TCP port for the DevTools forward/tunnel.
        settle_ms:    minimum wait after launching the query before reading.
        answer_timeout_s: give up waiting for the answer to finish after this.
        keep_awake:   set `svc power stayon true` on enter (phones sit in Samsung
                      always-on-display and won't render otherwise).
        sleep_on_exit: put the screen back to sleep and drop stayon on exit.

    Use as a context manager; reuse one instance for many prompts on one phone::

        with PhoneFarmAIOverviewScraper("R58MEXAMPLE", ssh_host="phone-farm",
                                        adb=r"C:\\platform-tools\\adb.exe") as s:
            r = s.search("what are the economic consequences of ...")
    """

    def __init__(self, serial: str | None = None, *, session: PhoneChromeSession | None = None,
                 ssh_host: str | None = None,
                 adb: str | None = None, cdp_port: int = DEFAULT_CDP_PORT,
                 settle_ms: int = 9_000, answer_timeout_s: int = 60,
                 keep_awake: bool = True, sleep_on_exit: bool = False,
                 debug: bool = False,
                 search_settle_ms: int = DEFAULT_SEARCH_SETTLE_MS):
        if session is None:
            if not serial:
                raise ValueError("pass a phone serial, or a PhoneChromeSession to scrape on")
            session = PhoneChromeSession(
                serial, ssh_host=ssh_host, adb=adb, cdp_port=cdp_port,
                keep_awake=keep_awake, sleep_on_exit=sleep_on_exit, debug=debug)
            self._owns_session = True
        else:
            # Somebody else's phone: they opened it and they close it. This is how one
            # handset serves AI Mode and ChatGPT off a single tunnel.
            self._owns_session = False
        self.session = session
        self.settle_ms = settle_ms
        self.search_settle_ms = search_settle_ms
        self.answer_timeout_s = answer_timeout_s
        self.debug = debug
        # CDP target id of the tab the current ask read, so it can be closed on the
        # way out. Set by the extractors, consumed by `_close_last_tab`.
        self._last_target_id = ""

    @property
    def serial(self) -> str:
        return self.session.serial

    # -- lifecycle --------------------------------------------------------------

    def __enter__(self) -> "PhoneFarmAIOverviewScraper":
        if self._owns_session:
            self.session.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._owns_session:
            self.session.__exit__(*exc)

    def wake(self) -> None:
        """Wake the screen and keep it awake (Samsung AOD won't render otherwise)."""
        self.session.wake()

    # -- scraping ---------------------------------------------------------------

    def search(self, prompt: str) -> AIOverviewResult:
        """Submit one prompt to AI Mode on the phone, return the parsed answer."""
        self.session.require_open()
        scraped_at = now_iso()
        self.session.launch(AI_MODE_URL.format(q=quote_plus(prompt)))

        self._last_target_id = ""
        try:
            data = self._await_answer(prompt)
            if data is None:
                return AIOverviewResult(query=prompt, has_overview=False, blocked=False,
                                        note="no AI Mode page/answer found",
                                        surface="phone_farm", scraped_at=scraped_at,
                                        serial=self.serial)

            text = data.get("text", "") or ""
            low = text.lower()
            if "unusual traffic" in low or "/sorry/" in data.get("url", ""):
                return AIOverviewResult(query=prompt, has_overview=False, blocked=True,
                                        note="captcha / unusual traffic",
                                        surface="phone_farm", scraped_at=scraped_at,
                                        serial=self.serial)
            # Google declined to answer. Reported as a failed ask, not stored as one:
            # the text is long and cited enough to look like prose otherwise.
            if _looks_like_ai_refusal(text):
                return AIOverviewResult(query=prompt, has_overview=False, blocked=False,
                                        note="ai mode declined to generate an answer",
                                        surface="phone_farm", scraped_at=scraped_at,
                                        serial=self.serial)

            overview_text = _clean_overview_text(text, prompt)
            references = _parse_references(data.get("anchors", []))
            if not references:
                _check_signed_in(data.get("anchors", []), text, self.serial)
                # Google says how many sites it used; nothing parsed out of the page
                # means the citations were missed, not that the answer had none. The
                # prose still renders, so this would otherwise store as a success.
                cited = _cited_site_count(text)
                if cited:
                    raise MissingCitationsError(
                        f"{self.serial}: the answer cites {cited} site(s) but none "
                        f"could be parsed from the page -- storing it would record "
                        f"an answer with no sources. Re-check the citation selectors "
                        f"in _EXTRACT_JS."
                    )
            return AIOverviewResult(
                query=prompt,
                has_overview=bool(overview_text),
                overview_text=overview_text,
                references=references,
                container_html=data.get("html", ""),
                surface="phone_farm",
                scraped_at=scraped_at,
                serial=self.serial,
            )
        finally:
            # One tab per ask, closed as we leave, however we leave. Chrome keeps tabs
            # across sessions, so without this every ask ever made is still open: the
            # farm's phones were carrying hundreds, which is what made `_ai_mode_page`
            # ambiguous in the first place. In the `finally` so a timeout or a raise
            # does not leak the tab that caused it.
            self._close_last_tab()

    def assert_google_signed_in(self, settle_s: float = 9.0) -> None:
        """Raise `GoogleSignedOutError` unless Chrome here has a Google session.

        Loads a page that needs an account and watches where its tab settles: a
        signed-in handset stays on it, a signed-out one is bounced away. Meant to run
        once per handset per run, before it serves any ask -- a phone whose links all
        come back as opaque redirects produces rows that look like answers with no
        sources, and the point is to find that out before collecting a run's worth.

        Checking the Android account instead would not catch it: the device account
        can be present and correct while Chrome's *web* session is gone, which is
        exactly the state this is here to find.
        """
        self.session.require_open()
        # Any tab already parked on the probe host would answer for the new one, and
        # a stale one proves nothing about the session now.
        for page in self.session.pages():
            if SIGNIN_PROBE_HOST in (page.get("url") or ""):
                self.session.close_tab(page.get("id") or "")
        before = {p.get("id") for p in self.session.pages()}
        self.session.launch(SIGNIN_PROBE_URL)

        # Poll for the whole window rather than reading once: the bounce is a
        # redirect, so the first URL seen is often still the one we asked for.
        probe_id, url = "", ""
        deadline = time.monotonic() + settle_s
        while time.monotonic() < deadline:
            time.sleep(1.0)
            for page in self.session.pages():
                pid = page.get("id")
                if pid and pid not in before and "google" in (page.get("url") or ""):
                    probe_id, url = pid, page.get("url") or ""
        try:
            if not url:
                # The probe never opened a tab. That says nothing about the session,
                # and costing the farm a phone over it would be worse than the risk.
                logger.warning(f"[{self.serial}] sign-in probe opened no Google tab; "
                               "treating the handset as usable")
                return
            if SIGNIN_PROBE_HOST in url:
                return
            raise GoogleSignedOutError(
                f"{self.serial}: the sign-in probe left {SIGNIN_PROBE_HOST} for "
                f"{url[:90]}, so Chrome on this handset has no Google session. Its "
                f"results would carry no source URLs at all -- sign it back in, or "
                f"put the serial in phone.exclude_serials."
            )
        finally:
            if probe_id:
                self.session.close_tab(probe_id)

    def search_many(self, prompts: list[str]) -> list[AIOverviewResult]:
        return [self.search(p) for p in prompts]

    def search_normal(self, prompt: str, top_n: int = 10) -> SearchResult:
        """Google's ordinary ranked web results for `prompt`, on the same phone.

        The plain search page (no `udm=50`), so this is what Google surfaces to a
        user rather than what the AI Mode answer cites. Same handset, same egress IP
        and same session as `search`, which is what makes the two comparable for one
        query -- and also what makes them share a CAPTCHA budget.
        """
        self.session.require_open()
        scraped_at = now_iso()
        self.session.launch(SEARCH_URL.format(q=quote_plus(prompt)))

        self._last_target_id = ""
        try:
            data = self._await_results(prompt)
            if data is None:
                return SearchResult(query=prompt, note="no plain search page found",
                                    scraped_at=scraped_at, serial=self.serial)

            text = data.get("text", "") or ""
            if "unusual traffic" in text.lower() or "/sorry/" in data.get("url", ""):
                return SearchResult(query=prompt, blocked=True,
                                    note="captcha / unusual traffic",
                                    scraped_at=scraped_at, serial=self.serial)

            results, note = _parse_search_results(data.get("anchors", []), top_n)
            if not results:
                # A signed-out session produces this exact shape on every query.
                _check_signed_in(data.get("anchors", []), text, self.serial)
                # Nothing accuses the handset, but nothing ranked either, and for
                # these queries that is far likelier to be a parse failure than the
                # truth. Hand it up rather than writing it down; the pool re-asks on
                # another phone and stores this if that one agrees.
                raise EmptySearchResultsError(
                    f"{self.serial}: the results page ranked nothing for "
                    f"{prompt!r}{' (' + note + ')' if note else ''}",
                    result=SearchResult(
                        query=prompt, results=results,
                        note=note or "No search results extracted for this query.",
                        scraped_at=scraped_at, serial=self.serial),
                )
            return SearchResult(query=prompt, results=results, note=note,
                                scraped_at=scraped_at, serial=self.serial)
        finally:
            self._close_last_tab()

    def clear_tab_backlog(self, keep: int = 1) -> int:
        """Close the Google tabs already open on this phone, returning how many.

        For phones that predate tab-closing. Worth calling once per session rather
        than per ask: a shorter tab list also means a faster `/json/list` on every
        poll, and CDP returns every tab on every call.
        """
        self.session.require_open()
        closed = self.session.close_tabs_matching("google.com/search", keep=keep)
        if closed:
            logger.info(f"[{self.serial}] closed {closed} leftover Google tab(s)")
        return closed

    # -- internals --------------------------------------------------------------

    def _close_last_tab(self) -> None:
        target, self._last_target_id = self._last_target_id, ""
        self.session.close_tab(target)

    def _await_answer(self, prompt: str) -> dict | None:
        """Poll CDP until the answer stops growing (or timeout). Returns the last
        extraction dict, or None if no matching page ever appeared."""
        time.sleep(self.settle_ms / 1000.0)
        deadline = time.monotonic() + self.answer_timeout_s
        last_len, stable, latest = -1, 0, None
        while time.monotonic() < deadline:
            data = self._extract_once(prompt)
            if data and data.get("found"):
                latest = data
                anchors = data.get("anchors", [])
                cur = len(data.get("text", ""))
                # ready once there is real prose past the query echo + ≥1 source,
                # and the text length has held steady across two reads.
                body = _clean_overview_text(data.get("text", ""), prompt)
                if len(body) > 200 and anchors:
                    stable = stable + 1 if cur == last_len else 0
                    last_len = cur
                    if stable >= 1:
                        return data
                else:
                    last_len = cur
            time.sleep(1.5)
        if latest is None and self.debug:
            logger.warning(f"[phone_farm] no AI Mode page for {prompt!r}")
        return latest

    def _extract_once(self, prompt: str) -> dict | None:
        page = _ai_mode_page(self.session.pages(), prompt)
        if page is None:
            return None
        # Remembered so the ask can close its own tab afterwards; see `search`.
        self._last_target_id = page.get("id") or ""
        return self.session.evaluate(page, _EXTRACT_JS)

    def _await_results(self, prompt: str) -> dict | None:
        """Poll CDP until the plain results page for `prompt` has results.

        No stability check, unlike `_await_answer`: these results are static HTML, so
        the first non-empty read is the whole thing. What this waits for is the
        navigation landing and painting.
        """
        time.sleep(self.search_settle_ms / 1000.0)
        deadline = time.monotonic() + self.answer_timeout_s
        latest = None
        while time.monotonic() < deadline:
            data = self._extract_search_once(prompt)
            if data and data.get("found"):
                latest = data
                # A CAPTCHA page is a final answer, not something to keep waiting on.
                if data.get("anchors") or "unusual traffic" in (data.get("text") or "").lower():
                    return data
            time.sleep(1.0)
        if latest is None and self.debug:
            logger.warning(f"[phone_farm] no plain search page for {prompt!r}")
        return latest

    def _extract_search_once(self, prompt: str) -> dict | None:
        page = _plain_search_page(self.session.pages(), prompt)
        if page is None:
            return None
        self._last_target_id = page.get("id") or ""
        return self.session.evaluate(page, _SEARCH_EXTRACT_JS)


# ── parsing (shared shape with google_aimode) ────────────────────────────────

def _clean_overview_text(text: str, prompt: str) -> str:
    """Trim the mobile AI Mode container text down to just the answer prose.

    The `#cnt` innerText leads with nav tabs + a query echo and trails with the
    "Ask anything" box / disclaimer. Cut everything up to the last echo of the
    query, then drop leading chrome lines and stop at the first trailing marker.
    """
    idx = text.rfind(prompt)
    if idx != -1:
        text = text[idx + len(prompt):]
    lines = text.splitlines()
    while lines and (not lines[0].strip()
                     or lines[0].strip() in _LEADING_CHROME
                     or _SITES_CHIP_RE.match(lines[0].strip())):
        lines.pop(0)
    kept: list[str] = []
    for i, ln in enumerate(lines):
        if any(m in ln for m in _TRAILING_CHROME_RE) \
           or any(m in ln for m in _MOBILE_TRAILING):
            break
        # First source card: the date-snippet line, preceded by the card's title.
        # Drop that title too (it won't end like a sentence).
        if _SOURCE_CARD_DATE_RE.match(ln.strip()):
            if kept and not kept[-1].rstrip().endswith(_SENTENCE_END):
                kept.pop()
            break
        kept.append(ln)
    return "\n".join(kept).strip()


# Hosts that are Google's own furniture rather than a search result. Matched on the
# domain OR any subdomain of it, because `_domain` only strips `www.` -- an exact-match
# list lets `x.gstatic.com` through as though it were a ranked result.
_GOOGLE_OWNED = ("gstatic.com", "googleusercontent.com", "googleapis.com", "youtube.com")


def _is_google_owned(domain: str) -> bool:
    """Google's own property, at any ccTLD or subdomain.

    Google's search domains are `google.<tld>` or `<sub>.google.<tld>`, so the test is
    whether "google" is one of the domain's *labels* -- a substring test would also
    swallow `notgoogle.org` and `googlefightclub.com`, which are somebody's real sites.
    """
    labels = domain.split(".")
    if "google" in labels:
        return True
    return any(domain == d or domain.endswith("." + d) for d in _GOOGLE_OWNED)


def _ai_mode_page(pages: list[dict], prompt: str) -> dict | None:
    """The open tab showing the AI Mode answer for `prompt`, or None.

    The counterpart of `_plain_search_page`, and it exists for the same reason: one
    Chrome serves every query on a phone, so "the `udm=50` tab" is not a unique thing.
    Taking the first `udm=50` tab in CDP's list with no check of which query it
    belongs to is fine on a clean phone and wrong on a real one: a handset that has
    accumulated leftover AI Mode tabs answers a large share of asks with an empty.

    A stale tab could not corrupt an answer (`_clean_overview_text` cuts at the echo of
    *this* prompt and finds none in someone else's page), so the damage was empties
    rather than wrong data. Requiring the `q` match turns "read the wrong tab and
    report nothing" into "wait for the right tab", and closing tabs after use keeps
    the list short enough that the match is found quickly.
    """
    wanted = prompt.strip()
    for page in pages:
        url = page.get("url", "")
        if "udm=50" not in url:
            continue
        try:
            query = parse_qs(urlparse(url).query)
        except ValueError:
            continue
        if (query.get("q") or [""])[0].strip() == wanted:
            return page
    return None


def _plain_search_page(pages: list[dict], prompt: str) -> dict | None:
    """The open tab showing plain results for `prompt`, or None.

    Two filters, both load-bearing because one Chrome is reused for every query on a
    phone: `udm=50` tabs are excluded, or an AI Mode tab left over from the same query
    would be scraped as though it were the web results; and the `q` param must equal
    this prompt, or the *previous* query's results page would answer for this one.
    Compared through parse_qs so `+` and `%20` encodings do not matter.

    Falling back to any plain search tab would defeat the point, so there is no
    fallback: no matching tab means keep polling, then report nothing found.
    """
    wanted = prompt.strip()
    for page in pages:
        url = page.get("url", "")
        if "google.com/search" not in url or "udm=50" in url:
            continue
        try:
            query = parse_qs(urlparse(url).query)
        except ValueError:
            continue
        if (query.get("q") or [""])[0].strip() == wanted:
            return page
    return None


# Google hands a signed-out session its result and citation links wrapped in an
# opaque redirect -- `/goto?url=<blob>`, where the blob is encrypted rather than the
# percent-encoded target `/url?q=` carries. Nothing client-side recovers the URL, so
# every such anchor parses as google.com and is dropped as Google's own chrome,
# leaving a page that looks exactly like a query with no results.
_WRAPPED_REDIRECT_PATHS = ("/goto", "/url", "/aclk")
# Below this it is ordinary page furniture; at or above it, it is the session.
MIN_WRAPPED_LINKS_FOR_SIGNED_OUT = 3


def _is_wrapped_redirect(href: str) -> bool:
    """A Google redirect whose target `_real_url` cannot recover."""
    dom = _domain(href)
    if not dom or not _is_google_owned(dom):
        return False
    if urlparse(href).path not in _WRAPPED_REDIRECT_PATHS:
        return False
    return _is_google_owned(_domain(_real_url(href)))


def _looks_signed_out(text: str) -> bool:
    """The `Sign in` affordance the header shows only with no Google session."""
    return any(ln.strip() == "Sign in" for ln in text.splitlines()[:12])


def _cited_site_count(text: str) -> int:
    """The `N sites` chip AI Mode renders above an answer, or 0 when absent.

    Google's own count of what it cited, which makes it the one thing on the page
    that can contradict an empty reference list.
    """
    for ln in text.splitlines()[:40]:
        m = re.match(r"^(\d+)\s+sites?$", ln.strip(), re.IGNORECASE)
        if m:
            return int(m.group(1))
    return 0


def _check_signed_in(anchors: list[dict], text: str, serial: str) -> None:
    """Raise if this page's links are opaque redirects rather than real URLs."""
    wrapped = sum(1 for a in anchors if _is_wrapped_redirect(a.get("href", "")))
    if wrapped < MIN_WRAPPED_LINKS_FOR_SIGNED_OUT:
        return
    hint = " and the page still offers a 'Sign in' link" if _looks_signed_out(text) else ""
    raise GoogleSignedOutError(
        f"{serial}: {wrapped} link(s) came back as opaque Google redirects "
        f"(/goto?url=...){hint}, so no source URL can be recovered. Chrome on this "
        f"handset is signed out of Google -- sign it back in."
    )


def _parse_search_results(anchors: list[dict], top_n: int) -> tuple[list[Reference], str]:
    """Ranked results → (Reference list in rank order, note).

    DOM order is Google's rank order, so first-seen wins on a duplicate URL: a site
    appearing twice keeps its better position.

    Anchors carrying a heading are the result title links. If none do, every external
    anchor is used instead and the note records that -- mobile Google gets re-skinned
    and a selector matching nothing would otherwise be indistinguishable from a query
    that genuinely returned nothing.
    """
    headed = [a for a in anchors if a.get("heading")]
    note = ""
    if not headed and anchors:
        headed = anchors
        note = ("no result-title headings matched; fell back to every external link, "
                "so this row's results may include page furniture (re-check the "
                "selectors in _SEARCH_EXTRACT_JS)")

    by_url: dict[str, Reference] = {}
    order: list[str] = []
    for a in headed:
        if len(order) >= top_n:
            break
        url = _real_url(a.get("href", ""))
        dom = _domain(url)
        if not url.startswith("http") or not dom or _is_google_owned(dom):
            continue
        if url in by_url:
            continue
        title = _clean_title(a.get("text", ""), a.get("aria", ""), dom)
        by_url[url] = Reference(title=title or dom, url=url, domain=dom)
        order.append(url)
    return [by_url[u] for u in order], note


def _parse_references(anchors: list[dict]) -> list[Reference]:
    """Dedupe external citation links → Reference(title, url, domain).

    Mobile citation anchors mostly lack `data-ved`, and a source's title may live
    on the link text of one anchor and the `aria-label` of a duplicate, so we
    dedupe by unwrapped URL and keep the longest title seen.
    """
    by_url: dict[str, Reference] = {}
    order: list[str] = []
    for a in anchors:
        url = _real_url(a.get("href", ""))
        dom = _domain(url)
        if not dom or dom.startswith("google.") or "google.com" in dom \
           or dom in ("gstatic.com", "googleusercontent.com"):
            continue
        title = _clean_title(a.get("text", ""), a.get("aria", ""), dom)
        if url in by_url:
            if len(title) > len(by_url[url].title):
                by_url[url].title = title
        else:
            by_url[url] = Reference(title=title, url=url, domain=dom)
            order.append(url)
    return [by_url[u] for u in order]


# ── CLI: python -m aiscrape.phone_farm ────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scrape Google AI Mode by driving a phone over adb + CDP.")
    ap.add_argument("prompts", nargs="*", help="prompt(s) to submit")
    ap.add_argument("--serial", required=True, help="adb serial of the phone")
    ap.add_argument("--ssh-host", default=None,
                    help="SSH host the phone hangs off (env AISCRAPE_PHONE_SSH)")
    ap.add_argument("--adb", default=None,
                    help="adb binary path on that host (env AISCRAPE_ADB)")
    ap.add_argument("--cdp-port", type=int, default=DEFAULT_CDP_PORT)
    ap.add_argument("--settle-ms", type=int, default=9_000)
    ap.add_argument("--sleep-on-exit", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    prompts = args.prompts or ["what are the economic consequences of "
                               "how does photosynthesis work"]
    with PhoneFarmAIOverviewScraper(
            args.serial, ssh_host=args.ssh_host, adb=args.adb,
            cdp_port=args.cdp_port, settle_ms=args.settle_ms,
            sleep_on_exit=args.sleep_on_exit, debug=args.debug) as s:
        for p in prompts:
            r = s.search(p)
            print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
