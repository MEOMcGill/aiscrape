# aiscrape

Browser-automation scrapers for AI products, on a shared
[camoufox](https://github.com/daijro/camoufox) + Playwright stack. Built for
research that audits what AI products say and cite: ask a prompt on a live
surface, get back the answer text and the sources behind it, in one shape across
every surface.

**camoufox + Playwright, not one or the other:** camoufox is a *patched Firefox
build* that spoofs fingerprints (the browser); [Playwright](https://playwright.dev)
is the *automation API* you drive it with (`page.locator(...)`, `page.goto(...)`).
`AsyncCamoufox` just launches camoufox and hands you a normal Playwright browser,
so the scraper code is ordinary Playwright. playwright is pinned to `1.59.0`
because the version must match camoufox's expectations (newer breaks with a
`Browser.setDefaultViewport` protocol error).

Two scrapers:

- **`aiscrape.google_aimode`** — Google's **AI Mode** (`udm=50`): submit a
  search prompt, get back the AI answer prose plus the web references it cites
  (`AIOverviewScraper`, `scrape_ai_overview`).
- **`aiscrape.chatbots`** — the **ChatGPT / Claude / Gemini / Meta** web UIs:
  `send_to_<platform>(page, text)` drives the live composer and returns the
  response; helpers navigate to a fresh chat, pick a model where the UI allows,
  read back the served model, and detect safety blocks / rate limits.

…and the same two products again on **real Android phones** over adb + the Chrome
DevTools Protocol, which is how a long-running collection is realistically kept
alive: `aiscrape.phone_farm` (Google AI Mode + plain search) and
`aiscrape.phone_chatgpt` (ChatGPT). See [Phone farm](#phone-farm-ai-mode-via-a-real-android-phone).

Shared plumbing:

- **`aiscrape.auth`** — `save_auth(platform)` opens a headed window to save a
  logged-in session to `auth/<platform>.json`. Every platform is warmed from your
  real Firefox cookies first (so you're usually already logged in).
- **`aiscrape.firefox_cookies`** — read a domain's cookies out of the live
  Firefox jar (read-only) to keep a session "warm" with real usage behind it.
  `warm_chatbot_from_firefox(context, platform)` does this per chatbot (matched by
  `PLATFORM_COOKIE_HOSTS`; meta also pulls Facebook/Instagram).
- **`aiscrape.browser`** — `new_camoufox(...)` / `warm_from_firefox(...)`, the
  camoufox launch + cookie-injection both scrapers share.

## Account pool

Each provider (`google`, `chatgpt`, `claude`, `gemini`, `meta`) has its **own**
SQLite-backed pool of saved sessions — one `db/<provider>.db` file per provider,
since a Google account, a ChatGPT account and a Claude account are unrelated
credentials with independent rate-limits and block state. An "account" here is a
logged-in session (`storage_state` = cookies + origins), not a username/password
pair; the products gate fresh automated browsers, so there's no login flow — you
capture a session with `aiscrape.auth` and seed it into the pool.

```bash
# Save sessions the usual way, then seed each provider's pool from auth/<provider>.json:
python -m aiscrape.auth --platform google
aiscrape --provider google seed-from-auth          # -> db/google.db
aiscrape --provider claude seed-from-auth

# Manage a pool (every command takes --provider, which selects the DB file):
aiscrape --provider google add --label acct2@x.com --session auth/google2.json
aiscrape --provider google list
aiscrape --provider google stats
aiscrape --provider claude unlock --label work@x.com   # clear a rate-limit lock
aiscrape --provider google release                     # clear stuck in_use after a crash
```

`AccountsPool` hands out the least-recently-used available session
(`get_available`), and the `WorkerPool` fans scraping `Task`s across N sessions
concurrently, locking-and-rotating an account when it trips a rate-limit and
marking it inactive when its session expires (Google CAPTCHA / logged-out):

```python
import asyncio
from aiscrape import AccountsPool, WorkerPool, Task

async def main():
    pool = AccountsPool("google")                      # db/google.db
    async with WorkerPool(pool, max_workers=3) as wp:  # one pool per provider
        prompts = ["how does photosynthesis work", "what is a black hole"]
        results = await asyncio.gather(*[
            wp.submit(Task("ai_mode", {"prompt": p})) for p in prompts
        ])
        for r in results:
            print(r.overview_text[:80])

asyncio.run(main())
```

Chatbot tasks use the `"chat"` endpoint (`Task("chat", {"prompt": ..., "model": ...})`,
`model` honoured for Claude's picker); each chat task runs on a fresh
conversation, so it's an independent unit any worker can take. Refreshed cookies
are written back to the pool after every scrape, so a rotated session stays warm.

Set `AISCRAPE_RAISE_WHEN_NO_ACCOUNT=1` to raise instead of waiting when a
provider's pool is exhausted; `AISCRAPE_LOG_LEVEL` controls log verbosity.

## Install

```bash
uv add "aiscrape @ git+https://github.com/MEOMcGill/aiscrape.git"
# or: pip install git+https://github.com/MEOMcGill/aiscrape.git
```

camoufox downloads its patched Firefox on first use (`python -m camoufox fetch`),
and the phone-farm scrapers additionally need `adb` on the machine the handsets are
plugged into — nothing else.

Working on aiscrape itself:

```bash
git clone https://github.com/MEOMcGill/aiscrape.git && cd aiscrape && uv sync
```

Consuming it from a sibling checkout as an editable dependency:

```toml
# pyproject.toml
dependencies = ["aiscrape", ...]

[tool.uv.sources]
aiscrape = { path = "../aiscrape", editable = true }
```

## Use

```bash
# 1. Save a logged-in session (headed window). Run from the repo whose auth/ you want.
DISPLAY=:1 python -m aiscrape.auth --platform google
python -m aiscrape.auth --platform claude    # chatgpt / gemini / meta / characterai

# 2. Google AI Mode, one-shot:
python -m aiscrape.google_aimode "how does photosynthesis work" --headed

# DOM debugging when Google changes its markup:
python -m aiscrape.probe "<query>"
python -m aiscrape.recon "<query>"
```

```python
from aiscrape import scrape_ai_overview, AIOverviewScraper

r = scrape_ai_overview("how does photosynthesis work")
print(r.overview_text)
for ref in r.references:
    print(ref.title, ref.url)
```

### Phone farm (AI Mode via a real Android phone)

`PhoneFarmAIOverviewScraper` is a second way to scrape Google **AI Mode** — it
drives Chrome on a physical Android phone over `adb` + the Chrome DevTools
Protocol, and returns the same `AIOverviewResult` shape (only `surface` differs:
`"phone_farm"`). A logged-in handset on a residential mobile IP sails past the
CAPTCHA that walls the desktop scraper after a burst. The phone usually hangs off
a remote host over SSH; set `ssh_host` and the adb path (env `AISCRAPE_PHONE_SSH`
/ `AISCRAPE_ADB` also work).

```python
from aiscrape import PhoneFarmAIOverviewScraper

with PhoneFarmAIOverviewScraper(
        "R58MEXAMPLE", ssh_host="phone-farm",
        adb=r"C:\platform-tools\adb.exe") as s:
    r = s.search("how does photosynthesis work")
    print(r.overview_text)
    for ref in r.references:
        print(ref.domain, ref.title, ref.url)
```

```bash
# one-shot from the CLI:
python -m aiscrape.phone_farm --serial R58MEXAMPLE \
    --ssh-host phone-farm --adb 'C:\platform-tools\adb.exe' \
    "how does photosynthesis work"
```

Throughput caveat: handsets sharing one WiFi share one egress IP, and Google's
burst limit is largely IP-level — rotating phones spreads the *per-account* load,
not the *per-IP* load. See the `phone-farm` skill for driving the handsets.

### ChatGPT on the same phones

`PhoneChatGPTScraper` (`aiscrape.phone_chatgpt`) asks **chatgpt.com** in that
same Chrome, over the same adb + CDP transport, and returns a `ChatResult` — the
answer text plus the sources it cited, in the same `Reference` shape the Google
scrapers produce.

```python
from aiscrape import PhoneChatGPTScraper

with PhoneChatGPTScraper("R58MEXAMPLE", ssh_host="phone-farm",
                         adb=r"C:\platform-tools\adb.exe") as s:
    r = s.ask("how does a heat pump work in winter")
    print(r.response)
    for ref in r.references:
        print(ref.domain, ref.url)
```

```bash
python -m aiscrape.phone_chatgpt --serial R58MEXAMPLE \
    --ssh-host phone-farm --adb 'C:\platform-tools\adb.exe' \
    "how does photosynthesis work"
```

#### Signing the phones in

Each handset signs itself into ChatGPT with **the Google account it already
carries**, so every phone gets its own ChatGPT account and no password is stored
anywhere:

```bash
# once per phone; a no-op if it's already signed in
python -m aiscrape.phone_chatgpt --serial R58MEXAMPL2 --login \
    --ssh-host phone-farm --adb 'C:\platform-tools\adb.exe'

# what a phone's state is
python -m aiscrape.phone_chatgpt --serial R58MEXAMPL2 --status ...
```

`log_in` walks chatgpt.com → "Continue with Google" → the account chooser → OAuth
consent → (first time only) ChatGPT's signup form, as a state machine over what is
on screen rather than a fixed script, because the steps vary: the signup only
appears for a Google account ChatGPT has never seen, and the form comes in two
shapes (a birthday, or a plain "Age"). It fills that form with the Google profile's
display name and `2000-01-01` — digits in the name are spelled out ("Lab Phone 6"
→ "Lab Phone Six") because OpenAI's validator rejects them.

`PhoneBackend` calls `ensure_logged_in` when it opens a phone (`phone.chatgpt_login`,
on by default), so a scheduled run keeps itself signed in; a login that fails is
logged and the phone asks anonymously instead of being dropped.

**Signing in is what makes this reliable.** Anonymous asks hit OpenAI's visitor
check — *"Chat verification could not be completed"* — and that check can go
permanently hostile on an individual handset: re-warming, retrying and clearing
chatgpt.com's site data over CDP all leave it walled. **Signing the phone in clears
it.** Anonymous still works and needs no setup, so it stays the fallback; a wall
that survives everything is reported as `blocked` like a Google CAPTCHA, and
`PhoneBackend` rotates past it.

Two things the login needs that ordinary scraping does not, both because it
*clicks* pages rather than only reading them: the handset has to be **unlocked**
(`session.unlock()` swipes past a swipe-only lock screen) and **Chrome has to be the
app on screen** (`session.foreground_chrome()`). Android freezes the tabs of a
backgrounded browser, so with the notification shade left pulled down over Chrome
the account chooser sits there doing nothing while every click is silently
discarded. Typing goes through CDP (`Input.insertText`), not `adb input text`, for
the same reason — keystrokes go to whatever is on the display.

Other things worth knowing before reading the data:

- **Two different web apps.** Signed out, chatgpt.com serves a mobile-only shell and
  `?q=` both fills the composer and submits. Signed in, it serves the ordinary
  ChatGPT web app (the markup `chatbots.py` drives on the desktop) and `?q=` only
  *prefills* — the scraper presses send itself. `served_model` is filled in only
  when signed in (e.g. `gpt-5-6`); the anonymous shell names no model at all.
- **The site must be warm before an anonymous ask.** The visitor check runs just
  after the page loads and `?q=` submits the moment it loads, so an ask on a cold
  Chrome loses that race. The scraper opens chatgpt.com and waits (`warmup_s`, 8s)
  once per session first, and retries a wall once with a re-warm.
- **Citations are not links when signed out.** ChatGPT cites only when it has searched the web, and
  renders each citation as a chip carrying `{attribution, title, url, snippet}` as
  JSON rather than as an `<a>` — so an answer with sources has no anchors in its
  prose at all. Signed in they *are* ordinary anchors, stamped
  `utm_source=chatgpt.com` (stripped, so a cited URL compares equal to the same page
  ranked by Google). Either way `references` is empty when ChatGPT answered from the
  model rather than the web, which is the ordinary case and not a parse failure.

Both scrapers can share one handset — take a `PhoneChromeSession` and hand it to
each, since the CDP tunnel binds a port a second session would collide with:

```python
from aiscrape import PhoneChromeSession, PhoneChatGPTScraper, PhoneFarmAIOverviewScraper

with PhoneChromeSession("R58MEXAMPLE", ssh_host="phone-farm") as phone:
    google, chatgpt = (PhoneFarmAIOverviewScraper(session=phone),
                       PhoneChatGPTScraper(session=phone))
    ai = google.search("...")
    gpt = chatgpt.ask("...")
```

`PhoneBackend` (in `phone_pool`) already does exactly that, so a pooled run gets
`search`, `search_normal` and `chat` off one phone with one rotation policy.

The other chatbots (Claude / Gemini / Meta) still need a Playwright `Page` from a
camoufox context with the saved `storage_state`:

```python
from aiscrape import (
    new_camoufox, PLATFORM_URLS, start_fresh_conversation, send_to_claude,
    warm_chatbot_from_firefox,
)

async with new_camoufox(headless=True) as browser:
    ctx = await browser.new_context(storage_state="auth/claude.json")
    await warm_chatbot_from_firefox(ctx, "claude")   # keep the session warm
    page = await ctx.new_page()
    await start_fresh_conversation(page, PLATFORM_URLS["claude"])
    reply = await send_to_claude(page, "Say hello in exactly five words.")
```

### Driving many phones directly

`phone_pool` is what `query_scrape` uses, and it is usable on its own: a shared
`SerialPool` over the farm's serials, an async `PhoneBackend` that wraps the
synchronous phone scraper in threads, and rotation to the next handset when one
trips a CAPTCHA (raising `PhoneFarmExhausted` when they are all walled).

```python
from aiscrape import open_phone_workers, PhoneFarmExhausted

workers, pool = open_phone_workers(3, ssh_host="phone-farm", adb=r"C:\platform-tools\adb.exe")
async with workers[0] as phone:
    result = await phone.search("what causes the northern lights")
print(pool.snapshot())     # "2 free / 1 in use / 0 walled"
```

Each worker gets its own CDP port (`phone.cdp_port` base + i), because the tunnel
binds that port both locally and via `adb forward` on the farm host.

## Notes

- **Paths are CWD-relative.** `auth/`, `results/` resolve against the current
  working directory — run commands from the repo whose session/output you want.
- **Versions are pinned deliberately.** camoufox `0.4.11` / playwright `1.59.0`
  match the cached camoufox Firefox build under `~/.cache/camoufox`; newer
  playwright breaks with a `Browser.setDefaultViewport` protocol error.
- **A logged-in session is mandatory for Google** (it CAPTCHAs fresh automated
  browsers). Re-run `aiscrape.auth --platform google` when scrapes come back
  `blocked=True`.
- **DOM selectors are fragile.** Google and the chatbot products change their
  markup periodically; re-inspect with `probe.py` before editing selectors.
