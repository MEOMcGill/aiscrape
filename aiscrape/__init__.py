"""aiscrape — browser-automation scrapers for AI products.

Two scrapers on a shared camoufox + Playwright stack:

- Google **AI Mode** (`google_aimode`): submit a search prompt, get the AI
  answer text + the web references it cites.
- The **chatbot web UIs** (`chatbots`): ChatGPT / Claude / Gemini / Meta —
  send a message to the live product, read the response, detect blocks.

Shared plumbing: `auth.save_auth` (save a logged-in session), `firefox_cookies`
(warm a session from the real Firefox profile), `browser` (camoufox launch).

**Account pool.** Each provider has its own SQLite-backed pool of saved
sessions (`AccountsPool`, one `db/<provider>.db` file per provider) and an
asyncio `WorkerPool` that fans scraping `Task`s out across those sessions,
locking/rotating an account when it hits a rate-limit or its session expires.
See `aiscrape.cli` (`aiscrape --provider ... `) to manage accounts.
"""

from aiscrape.account import Account
from aiscrape.accounts_pool import AccountsPool, default_db_file
from aiscrape.auth import save_auth
from aiscrape.browser import new_camoufox, warm_from_firefox
from aiscrape.browser_session import AIBrowserSession, warm_session_from_firefox
from aiscrape.exceptions import (
    AccountBlockedError,
    AIScraperError,
    NoAccountError,
    RateLimitError,
    SessionExpiredError,
)
from aiscrape.models import ChatResult, Task
from aiscrape.utils import PROVIDERS
from aiscrape.worker import Worker
from aiscrape.worker_pool import WorkerPool
from aiscrape.chatbots import (
    CLAUDE_MODELS,
    META_L2_TEXT_RE,
    PLATFORM_COOKIE_HOSTS,
    PLATFORM_URLS,
    SEND_FNS,
    detect_l2_classifier,
    detect_rate_limit,
    pre_send_turn_count,
    read_served_model,
    select_claude_model,
    send_to_chatgpt,
    send_to_claude,
    send_to_gemini,
    send_to_meta,
    start_fresh_conversation,
    tag_l2_block_turns,
    warm_chatbot_from_firefox,
)
from aiscrape.firefox_cookies import default_firefox_profile, read_cookies
from aiscrape.google_aimode import (
    AIOverviewResult,
    AIOverviewScraper,
    Reference,
    SearchResult,
    scrape_ai_overview,
    scrape_ai_overview_async,
)
from aiscrape.phone_chatgpt import PhoneChatGPTScraper
from aiscrape.phone_farm import (
    PhoneChromeSession,
    PhoneFarmAIOverviewScraper,
    PhoneFarmError,
    list_serials,
)
from aiscrape.phone_pool import (
    PhoneBackend,
    PhoneFarmExhausted,
    SerialPool,
    discover_serials,
    open_phone_backend,
    open_phone_workers,
)

__all__ = [
    # google AI mode
    "AIOverviewScraper",
    "AIOverviewResult",
    "Reference",
    "SearchResult",
    "scrape_ai_overview",
    "scrape_ai_overview_async",
    # phone farm (Google AI Mode + ChatGPT via a real Android phone over adb + CDP)
    "PhoneChromeSession",
    "PhoneFarmAIOverviewScraper",
    "PhoneChatGPTScraper",
    "PhoneFarmError",
    "list_serials",
    # phone farm pooling / CAPTCHA rotation across many handsets
    "PhoneBackend",
    "PhoneFarmExhausted",
    "SerialPool",
    "discover_serials",
    "open_phone_backend",
    "open_phone_workers",
    # chatbots
    "PLATFORM_URLS",
    "PLATFORM_COOKIE_HOSTS",
    "warm_chatbot_from_firefox",
    "SEND_FNS",
    "send_to_chatgpt",
    "send_to_claude",
    "send_to_gemini",
    "send_to_meta",
    "start_fresh_conversation",
    "select_claude_model",
    "read_served_model",
    "detect_rate_limit",
    "detect_l2_classifier",
    "pre_send_turn_count",
    "META_L2_TEXT_RE",
    "tag_l2_block_turns",
    "CLAUDE_MODELS",
    # auth / cookies / browser
    "save_auth",
    "new_camoufox",
    "warm_from_firefox",
    "default_firefox_profile",
    "read_cookies",
    # account pool
    "Account",
    "AccountsPool",
    "default_db_file",
    "AIBrowserSession",
    "warm_session_from_firefox",
    "Worker",
    "WorkerPool",
    "Task",
    "ChatResult",
    "PROVIDERS",
    "AIScraperError",
    "NoAccountError",
    "RateLimitError",
    "AccountBlockedError",
    "SessionExpiredError",
]
