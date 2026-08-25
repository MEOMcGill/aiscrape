# aiscrape

Browser-automation scrapers for AI products. Built for auditing AI products: send a prompt to multiple products, get back the answer text and the sources behind it.

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

