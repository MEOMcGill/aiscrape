"""
Scrape Google's AI answer for a search query, via AI Mode (`udm=50`).

Submit a prompt, get back the AI-generated answer text plus the web
references (sources) it cites. Built on camoufox + playwright, loading a
logged-in Google session (see `aiscrape.auth`) so Google Search doesn't
CAPTCHA-block the browser.

Uses AI Mode rather than the plain AI Overview snippet that used to render
atop regular search results: verified live (2026-07), regular AI Overview
simply omits its box for sensitive queries (personalized voting
recommendations, location-dependent procedural questions) -- a silent gap.
AI Mode applies the same underlying guardrail but always writes a full
answer, including an explicit refusal ("As an AI, I cannot recommend a
specific party...") when it declines -- so scraping never leaves a missing
row. `AIOverviewResult.surface` records this ("ai_mode") so any pre-existing
data scraped from the old plain-panel approach (which has no `surface`
column at all) stays distinguishable from rows scraped after this switch.

Public API
----------
    from aiscrape import scrape_ai_overview, AIOverviewScraper

    # one-shot
    result = scrape_ai_overview("how does photosynthesis work")
    print(result.overview_text)
    for ref in result.references:
        print(ref.title, ref.url)

    # many prompts, reusing one browser
    async with AIOverviewScraper() as s:
        for r in await s.search_many(prompts):
            ...

Each result is an `AIOverviewResult` with: query, has_overview, overview_text,
references (list of `Reference`), blocked (CAPTCHA/consent wall), and the raw
container HTML when debug dumping is on.

Paths (`auth/google.json`, `results/debug/`) resolve against the current
working directory, so run from the repo whose session you want to use.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus, urlparse, parse_qs, unquote

from playwright.async_api import Page

from aiscrape.browser import new_camoufox, warm_from_firefox

AUTH_FILE = Path("auth/google.json")
DEBUG_DIR = Path("results/debug")


@dataclass
class Reference:
    """A single web source cited by the AI answer."""
    title: str
    url: str
    domain: str


@dataclass
class SearchResult:
    """Google's normal (non-AI) top web results for a query.

    A plain Google Search (no `udm=50`) returns a ranked list of web results;
    `results` holds them in rank order (results[0] is the top hit). These are
    what Google *surfaces to the user*, distinct from the sources the AI Mode
    answer chooses to *cite* — the point of capturing both is to compare them.
    """
    query: str
    results: list[Reference] = field(default_factory=list)
    blocked: bool = False
    note: str = ""
    # UTC ISO-8601 timestamp of when the query was issued.
    scraped_at: str = ""
    # Which handset answered, when a phone did. Empty for the desktop scraper.
    # Worth carrying because results can differ by phone -- different Google
    # account, and a phone Google has started throttling looks exactly like a
    # query with nothing to say unless you can group by handset.
    serial: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AIOverviewResult:
    query: str
    has_overview: bool
    overview_text: str = ""
    references: list[Reference] = field(default_factory=list)
    blocked: bool = False
    note: str = ""
    container_html: str = ""
    # Which Google surface produced this row. Always "ai_mode" now; rows
    # scraped before this switch (the plain AI Overview panel) predate this
    # field and simply won't have it, which is itself the distinguishing signal.
    surface: str = "ai_mode"
    # UTC ISO-8601 timestamp of when the Google query was issued.
    scraped_at: str = ""
    # Which handset answered, when a phone did. See SearchResult.serial.
    serial: str = ""
    # Always None: AI Mode names no model build anywhere in the page or its RPCs.
    # Its picker offers only a tier ("Fast", or "Pro" via `arv=1`), not a version.
    served_model: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ── URL / consent helpers ────────────────────────────────────────────────────

def _real_url(href: str) -> str:
    """Unwrap a Google redirect URL (/url?q=... or /url?url=...) to the target."""
    if not href:
        return ""
    if href.startswith("/url") or "google.com/url" in href:
        q = parse_qs(urlparse(href).query)
        for key in ("q", "url"):
            if key in q and q[key]:
                return unquote(q[key][0])
    return href


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


async def _dismiss_consent(page: Page) -> None:
    """Click any Google cookie-consent button if the consent wall is shown."""
    for sel in (
        "#L2AGLb",                                  # "I agree" / "Accept all"
        'button:has-text("Accept all")',
        'button:has-text("Reject all")',
        'button[aria-label*="Accept all" i]',
        'button[aria-label*="Reject all" i]',
    ):
        try:
            btn = page.locator(sel).first
            if await btn.count() > 0 and await btn.is_visible():
                await btn.click(timeout=3_000)
                await page.wait_for_timeout(800)
                return
        except Exception:
            pass


async def _is_blocked(page: Page) -> bool:
    """True if Google served its 'unusual traffic' / CAPTCHA interstitial."""
    try:
        body = (await page.inner_text("body")).lower()
    except Exception:
        return False
    return "unusual traffic" in body or "/sorry/" in page.url


# ── AI Mode location + extraction ────────────────────────────────────────────

# The whole AI Mode conversation turn (query echo + answer + inline citations
# + bottom "N sites" strip) renders inside a container with the stable id
# "#cnt". Verified against live DOM (2026-07).

# Leading UI chrome (nav tabs + the "You said:" query-echo marker) and the
# trailing disclaimer / sources-toggle text we strip from the answer.
_LEADING_CHROME = ("AI Mode", "All", "Images", "Videos", "News", "More",
                   "Search Results", "You said:")
_TRAILING_CHROME_RE = (
    "AI can make mistakes",          # "AI can make mistakes, so double-check..."
    "Generative AI is experimental",
)


async def _find_overview_container(page: Page):
    """Return a Locator for the AI Mode answer block, or None if absent."""
    cand = page.locator("#cnt").first
    try:
        if await cand.count() > 0 and len((await cand.inner_text()).strip()) > 80:
            return cand
    except Exception:
        pass
    return None


def _clean_title(text: str, aria: str, dom: str) -> str:
    """Best title for a source link: prefer link text, fall back to aria-label."""
    title = " ".join((text or "").split())
    if not title:
        title = " ".join((aria or "").split())
        # aria-labels end with " Opens in new tab."
        title = title.replace("Opens in new tab.", "").strip().rstrip(".").strip()
    return (title or dom)[:300]


async def _extract_references(container, page: Page) -> list[Reference]:
    """External source links cited by the answer, deduped by URL.

    The same source appears multiple times (inline citation chip + bottom
    sources list); we keep the variant with the most informative title.
    """
    refs: dict[str, Reference] = {}
    if container is None:
        return []
    try:
        # Positively select the citation anchors rather than grabbing every
        # link and blacklisting Google's own domains. Verified against the live
        # DOM (2026-07): both citation surfaces -- the inline citation chips
        # woven into the prose and the bottom "sites" carousel -- render their
        # source links as `<a target="_blank" data-ved=...>`. `target="_blank"`
        # excludes the in-place nav tabs (All / Images / Maps / Flights ...);
        # `data-ved` (Google's result-logging stamp) excludes the static UI
        # chrome that opens in a new tab but carries no ved -- the footer
        # disclaimer links (Privacy Policy, Terms of Service) and the "delete
        # this link" legal-removal links. What's left is exactly the cited
        # sources, so a source hosted on a Google-owned property (youtube.com,
        # scholar.google.com, ...) is now kept on its merits instead of being
        # collateral of a domain blacklist.
        anchors = container.locator('a[href][data-ved][target="_blank"]')
        n = await anchors.count()
    except Exception:
        return []
    for i in range(min(n, 80)):
        a = anchors.nth(i)
        try:
            href = await a.get_attribute("href") or ""
            url = _real_url(href)
            dom = _domain(url)
            if not url.startswith("http") or not dom:
                continue
            title = _clean_title(
                await a.inner_text(), await a.get_attribute("aria-label") or "", dom
            )
            existing = refs.get(url)
            if existing is None or (len(title) > len(existing.title) and title != dom):
                refs[url] = Reference(title=title, url=url, domain=dom)
        except Exception:
            continue
    return list(refs.values())


async def _extract_overview_text(container, prompt: str) -> str:
    """The answer prose, with chrome, the echoed query, disclaimer, and the
    trailing sources strip trimmed off.

    Uses the live (visibility-respecting) innerText. Google interleaves inline
    citation labels (a source name after a sentence) into the rendered prose;
    those remain embedded, faithful to what's shown. The bottom sources strip
    ("N sites" + individual source cards) is popped off the tail — the
    structured source list is returned separately as `references`.
    """
    if container is None:
        return ""
    txt = (await container.inner_text()).strip()
    lines = [ln.rstrip() for ln in txt.splitlines()]
    # Drop leading chrome lines.
    while lines and lines[0].strip() in _LEADING_CHROME:
        lines.pop(0)
    # The line right after the "You said:" chrome is the echoed query.
    if lines and lines[0].strip().lower() == prompt.strip().lower():
        lines.pop(0)
    # Cut at the trailing disclaimer if present -- everything after it is the
    # "N sites" summary + individual source cards.
    for i, ln in enumerate(lines):
        if any(marker in ln for marker in _TRAILING_CHROME_RE):
            lines = lines[:i]
            break
    # Strip any trailing UI chrome / blanks left over.
    chrome = {"show all", "show more", "show less", "view all", ""}
    while lines and lines[-1].strip().lower() in chrome:
        lines.pop()
    return "\n".join(lines).strip()


# ── Normal (non-AI) search results extraction ────────────────────────────────

async def _extract_search_results(page: Page, top_n: int) -> list[Reference]:
    """The ranked web results Google shows on a plain (non-AI) results page.

    Result titles render as `<h3>` inside an anchor, within the results
    region `#rso` (`#search` is the wider wrapper). Selecting `a:has(h3)` picks
    the result title links positively; we unwrap Google redirect hrefs, drop
    Google's own internal links (nav, "People also ask", image/video packs that
    point back into google.*), dedupe by URL keeping first (best-ranked)
    appearance, and keep the top `top_n`. The list stays in DOM order, which is
    Google's rank order.

    Selectors here are as fragile as the AI Mode ones — re-inspect with
    `python -m aiscrape.probe "<query>"` if extraction degrades.
    """
    anchors = None
    for region in ("#rso", "#search"):
        try:
            cand = page.locator(f'{region} a:has(h3)')
            if await cand.count() > 0:
                anchors = cand
                break
        except Exception:
            continue
    if anchors is None:
        return []

    results: dict[str, Reference] = {}
    try:
        n = await anchors.count()
    except Exception:
        return []
    for i in range(min(n, 60)):
        if len(results) >= top_n:
            break
        a = anchors.nth(i)
        try:
            href = await a.get_attribute("href") or ""
            url = _real_url(href)
            dom = _domain(url)
            if not url.startswith("http") or not dom or "google." in dom:
                continue
            if url in results:
                continue
            title = " ".join((await a.locator("h3").first.inner_text()).split())
            results[url] = Reference(title=title or dom, url=url, domain=dom)
        except Exception:
            continue
    return list(results.values())


# ── Scraper ──────────────────────────────────────────────────────────────────

class AIOverviewScraper:
    """Reusable camoufox-backed Google AI Mode scraper.

    Loads a logged-in Google session so Search isn't CAPTCHA-blocked, then (by
    default) overlays fresh Google cookies read straight from the local Firefox
    profile -- as long as you keep using Google normally in Firefox, this keeps
    the scraping session "warm" with real usage history behind it without needing
    to periodically re-run the auth saver. Use as an async context manager to
    reuse one browser across many prompts.

    Session source (first that's set wins):
      - `storage_state`: a Playwright storage_state dict (e.g. from an
        `aiscrape.accounts_pool` account), or
      - `auth_file`: the on-disk ``auth/google.json`` (the default).

    Pass an existing Playwright `context` to drive the scraper against a browser
    someone else owns (the `WorkerPool` does this to reuse one pooled account's
    session across many prompts). When `context` is given the scraper neither
    launches nor closes a browser, and skips Firefox warming (the owner handles
    it) unless `refresh_from_firefox` is set.
    """

    def __init__(
        self,
        headless: bool = True,
        auth_file: Path = AUTH_FILE,
        locale: str = "en-CA",
        gl: str = "ca",
        hl: str = "en",
        settle_ms: int = 9_000,
        debug: bool = False,
        refresh_from_firefox: bool = True,
        storage_state: dict | None = None,
        context=None,
    ):
        self.headless = headless
        self.auth_file = Path(auth_file)
        self.locale = locale
        self.gl = gl
        self.hl = hl
        self.settle_ms = settle_ms
        self.debug = debug
        self.storage_state = storage_state
        self._cm = None
        self._browser = None
        self._context = context
        self._owns_browser = context is None
        # An externally-supplied context is warmed by its owner; only warm here
        # when we created the browser ourselves (or the caller explicitly asks).
        self.refresh_from_firefox = refresh_from_firefox if self._owns_browser else False

    async def __aenter__(self) -> "AIOverviewScraper":
        if not self._owns_browser:
            # Driven against a caller-owned context; nothing to launch.
            return self

        storage = self.storage_state
        if storage is None and self.auth_file.exists():
            storage = str(self.auth_file)
        self._cm = new_camoufox(
            headless=self.headless,
            locale=self.locale,
            geoip=True,
            os=["windows", "macos"],
        )
        self._browser = await self._cm.__aenter__()
        self._context = await self._browser.new_context(storage_state=storage)

        if self.refresh_from_firefox:
            await warm_from_firefox(self._context, "%google.%")

        return self

    async def __aexit__(self, *exc) -> None:
        if not self._owns_browser:
            return
        try:
            if self._context:
                await self._context.close()
        finally:
            if self._cm:
                await self._cm.__aexit__(*exc)

    async def search(self, prompt: str) -> AIOverviewResult:
        page = await self._context.new_page()
        try:
            return await self._search_on_page(page, prompt)
        finally:
            await page.close()

    async def _search_on_page(self, page: Page, prompt: str) -> AIOverviewResult:
        # Stamp the moment the query is issued (UTC), carried onto every result.
        scraped_at = datetime.now(timezone.utc).isoformat()
        url = (
            f"https://www.google.com/search?q={quote_plus(prompt)}"
            f"&hl={self.hl}&gl={self.gl}&udm=50"
        )
        await page.goto(url, timeout=60_000)
        await page.wait_for_load_state("domcontentloaded", timeout=30_000)
        await _dismiss_consent(page)

        if await _is_blocked(page):
            return AIOverviewResult(
                query=prompt, has_overview=False, blocked=True,
                note="Google served a CAPTCHA / 'unusual traffic' page. "
                     "Refresh the session with `aiscrape.auth`.",
                scraped_at=scraped_at,
            )

        # AI Mode streams its answer in over several seconds -- longer than a
        # plain AI Overview snippet since it writes full paragraphs.
        await page.wait_for_timeout(self.settle_ms)
        container = await _find_overview_container(page)
        if container is None:
            # Give it one more beat; some queries are slow to generate.
            await page.wait_for_timeout(3_000)
            container = await _find_overview_container(page)

        if container is None:
            return AIOverviewResult(
                query=prompt, has_overview=False,
                note="No AI Mode answer rendered for this query.",
                scraped_at=scraped_at,
            )

        refs = await _extract_references(container, page)
        text = await _extract_overview_text(container, prompt)
        if not text:
            return AIOverviewResult(
                query=prompt, has_overview=False,
                note="AI Mode container found but no answer text extracted.",
                scraped_at=scraped_at,
            )

        html = ""
        if self.debug:
            html = await container.inner_html()
            DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            stem = quote_plus(prompt)[:60]
            (DEBUG_DIR / f"{stem}.html").write_text(html, encoding="utf-8")
            await page.screenshot(
                path=str(DEBUG_DIR / f"{stem}.png"), full_page=True
            )

        return AIOverviewResult(
            query=prompt,
            has_overview=True,
            overview_text=text,
            references=refs,
            container_html=html,
            scraped_at=scraped_at,
        )

    async def search_many(self, prompts: list[str]) -> list[AIOverviewResult]:
        """Search prompts sequentially on one browser (gentle on rate limits)."""
        out = []
        for p in prompts:
            out.append(await self.search(p))
            await asyncio.sleep(2)
        return out

    async def search_normal(self, prompt: str, top_n: int = 10) -> SearchResult:
        """Fetch Google's normal (non-AI) top web results for a query.

        A separate plain-search page load (no `udm=50`) from `search()`, on the
        same warmed session. Returns the top `top_n` results in rank order.
        """
        page = await self._context.new_page()
        try:
            return await self._normal_search_on_page(page, prompt, top_n)
        finally:
            await page.close()

    async def _normal_search_on_page(self, page: Page, prompt: str,
                                     top_n: int) -> SearchResult:
        scraped_at = datetime.now(timezone.utc).isoformat()
        url = (
            f"https://www.google.com/search?q={quote_plus(prompt)}"
            f"&hl={self.hl}&gl={self.gl}"
        )
        await page.goto(url, timeout=60_000)
        await page.wait_for_load_state("domcontentloaded", timeout=30_000)
        await _dismiss_consent(page)

        if await _is_blocked(page):
            return SearchResult(
                query=prompt, blocked=True,
                note="Google served a CAPTCHA / 'unusual traffic' page. "
                     "Refresh the session with `aiscrape.auth`.",
                scraped_at=scraped_at,
            )

        # These results are static HTML (no streaming), but give the results
        # region a beat to render before extracting.
        try:
            await page.wait_for_selector("#search, #rso", timeout=15_000)
        except Exception:
            pass
        await page.wait_for_timeout(1_500)

        results = await _extract_search_results(page, top_n)
        note = "" if results else "No search results extracted for this query."
        return SearchResult(
            query=prompt, results=results, note=note, scraped_at=scraped_at,
        )


# ── Convenience one-shot wrappers ────────────────────────────────────────────

async def scrape_ai_overview_async(
    prompt: str, headless: bool = True, debug: bool = False
) -> AIOverviewResult:
    async with AIOverviewScraper(headless=headless, debug=debug) as s:
        return await s.search(prompt)


def scrape_ai_overview(
    prompt: str, headless: bool = True, debug: bool = False
) -> AIOverviewResult:
    """Blocking one-shot: submit a prompt, get the AI answer + references."""
    return asyncio.run(
        scrape_ai_overview_async(prompt, headless=headless, debug=debug)
    )


async def _cli(prompts: list[str], headless: bool, debug: bool, out: Path | None):
    async with AIOverviewScraper(headless=headless, debug=debug) as s:
        results = await s.search_many(prompts) if len(prompts) > 1 \
            else [await s.search(prompts[0])]
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        print(f"Wrote {len(results)} result(s) to {out}")
    else:
        print(json.dumps(
            [r.to_dict() for r in results] if len(results) > 1
            else results[0].to_dict(),
            indent=2, ensure_ascii=False,
        ))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Scrape Google's AI Mode answer")
    parser.add_argument("prompts", nargs="*", help="Search prompt(s) / question(s)")
    parser.add_argument("--from-file", type=Path,
                        help="Read prompts from a file, one per line")
    parser.add_argument("--out", type=Path,
                        help="Write results as JSONL to this path instead of stdout")
    parser.add_argument("--headed", action="store_true", help="Show the browser")
    parser.add_argument("--debug", action="store_true",
                        help="Dump container HTML + screenshot to results/debug")
    args = parser.parse_args()

    prompts = list(args.prompts)
    if args.from_file:
        prompts += [ln.strip() for ln in args.from_file.read_text().splitlines()
                    if ln.strip()]
    if not prompts:
        parser.error("provide at least one prompt, or --from-file")

    asyncio.run(_cli(prompts, not args.headed, args.debug, args.out))


if __name__ == "__main__":
    main()
