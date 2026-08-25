"""
Web-UI scrapers for the ChatGPT / Claude / Gemini / Meta chatbots.

Each `send_to_<platform>(page, text)` drives that product's live composer:
types the prompt, sends it, waits for generation to finish, and returns the
response text. Alongside are per-platform helpers to navigate to a fresh
conversation (`start_fresh_conversation`), pick a model where the UI allows it
(`select_claude_model`), read back which model was actually served
(`read_served_model`), and detect the two ways a scrape can come back invalid:
an application-layer safety block (`detect_l2_classifier`) or a rate-limit
banner (`detect_rate_limit`).

Built on a Playwright `Page` — get one from `aiscrape.browser.new_camoufox`
with a saved `storage_state` (see `aiscrape.auth.save_auth`). The DOM
selectors below are fragile; the products change their markup periodically.

The `META_L2_TEXT_RE` / `tag_l2_block_turns` pieces carry harm-audit-specific
semantics (Meta's fixed app-layer refusal strings) and are the single source
of truth for downstream collation code that imports them.
"""

import re

from playwright.async_api import Page

from aiscrape.browser import warm_from_firefox

PLATFORM_URLS = {
    "chatgpt": "https://chatgpt.com",
    "claude": "https://claude.ai/new",
    "gemini": "https://gemini.google.com/app",
    "meta": "https://meta.ai/",
}

# SQL LIKE patterns matching each platform's cookies in the real Firefox jar, so
# a scrape can be "warmed" with the session from your day-to-day browsing (the
# same trick the Google AI Mode scraper uses). meta.ai logs in through a
# Facebook/Instagram/Meta account, so its login cookies live there too; gemini
# rides the Google account, so it reuses the Google host pattern.
PLATFORM_COOKIE_HOSTS = {
    "chatgpt": "%chatgpt.com",
    "claude": "%claude.ai",
    "gemini": "%google.%",
    "meta": "%meta.%",
}


async def warm_chatbot_from_firefox(context, platform: str) -> int:
    """Inject this platform's cookies from the real Firefox profile into a
    Playwright context (best-effort). Returns how many cookies were added, or 0
    if the platform is unknown or the profile has no matching cookies."""
    host = PLATFORM_COOKIE_HOSTS.get(platform)
    if host is None:
        return 0
    n = await warm_from_firefox(context, host)
    # meta.ai's session is federated through Facebook/Instagram; warm those too.
    if platform == "meta":
        n += await warm_from_firefox(context, "%facebook.com")
        n += await warm_from_firefox(context, "%instagram.com")
    return n

# Claude.ai composer model picker: slug -> display label as shown in the menu.
# Submenu flag indicates whether the item lives under "More models" (older
# models) and therefore needs the submenu expanded before clicking. The select
# helper falls back to expanding the submenu even when submenu=False, so the
# flag is purely informational.
CLAUDE_MODELS = {
    "opus-4.7":   {"label": "Opus 4.7",   "submenu": False},
    "sonnet-4.6": {"label": "Sonnet 4.6", "submenu": False},
    "haiku-4.5":  {"label": "Haiku 4.5",  "submenu": False},
    "opus-4.6":   {"label": "Opus 4.6",   "submenu": True},
    "opus-3":     {"label": "Opus 3",     "submenu": True},
    "sonnet-4.5": {"label": "Sonnet 4.5", "submenu": True},
}

# Matches the ChatGPT rate-limit banner ("You've hit your limit. Please try
# again later. Retry"). The banner uses the same `text-token-text-error` CSS
# class as L2 classifier blocks, so the L2 detector needs the text to tell
# them apart. Patterns kept loose enough to catch French variants too.
RATE_LIMIT_TEXT_RE = re.compile(
    r"(?i)("
    r"you('?ve| have) (hit|reached) (your|the) limit"
    r"|please try again later"
    r"|vous avez atteint (la|votre) limite"
    r"|veuillez réessayer plus tard"
    r")"
)

# Meta AI L2 (application-layer) safety-block strings. Unlike the model's own
# RLHF refusals — which vary in wording, explain themselves in-context, and
# rarely repeat verbatim — Meta's app-layer safety system swaps in these fixed,
# templated messages; they were observed repeating identically across every turn
# for hard categories (terrorism, hate, bullying), which is why they used to slip
# through as a wall of "duplicate" normal turns. Treat the first occurrence as an
# L2 hard block. This is the single source of truth for the L2 signature —
# validate_and_collate.py imports it for retro-tagging historical records.
#
# Patterns anchor on the templated, distinctive parts ("goes against the rules",
# "try again in N hours", the "at the moment"/"pour le moment" + generic-offer
# tail). Deliberately NOT matched: bare in-context refusals like "I can't help
# with that" or "I can't help you with this request because …", which are
# ordinary RLHF refusals and must stay normal turns. Only the EN strings are
# confirmed live (2026-05); the FR variants are best-effort.
META_L2_TEXT_RE = re.compile(
    r"(?i)("
    r"content that goes against the rules"
    r"|try again in \d+\s*hours?"
    r"|can['’]?t help you with this request at the moment"
    r"|contenu qui va à l['’]encontre (de|des) (nos )?règles"
    r"|réessayez dans \d+\s*heures?"
    r"|je ne peux pas vous aider avec cette demande pour le moment"
    r")"
)


def tag_l2_block_turns(turns_log: list[dict], platform: str) -> None:
    """Normalise L2 (application-layer) blocks to per-turn tags, in place.

    Single source of truth shared by validate_and_collate.py and
    build_explorer.py so the two never drift on what counts as an L2 block.

    A turn counts as a block three ways: it's already an ``l2_classifier`` turn,
    it carries the ``l2_classifier_block`` flag, or it's a Meta turn whose
    response matches ``META_L2_TEXT_RE`` (Meta's block is a fixed app-layer
    refusal *string* rather than a discrete DOM event, and older transcripts
    recorded it as an ordinary ``normal`` turn). Every block is stamped
    ``type == "l2_classifier"`` + ``l2_classifier_block == True`` so all
    platforms represent a block identically downstream. Idempotent.
    """
    for turn in turns_log:
        is_block = (
            turn.get("type") == "l2_classifier"
            or turn.get("l2_classifier_block")
            or (platform == "meta"
                and META_L2_TEXT_RE.search(turn.get("target_response") or ""))
        )
        if is_block:
            turn["type"] = "l2_classifier"
            turn["l2_classifier_block"] = True


async def _dismiss_chatgpt_modals(page: Page) -> None:
    """Dismiss any ChatGPT overlay modal that blocks composer interaction.

    ChatGPT occasionally shows NUX (new-user-experience) or feature-announcement
    modals that cover the viewport with position:absolute and intercept pointer
    events. We press Escape and, if a known modal is still present, click the
    overlay edge to dismiss it.
    """
    await page.keyboard.press("Escape")
    await page.wait_for_timeout(300)
    modal = page.locator('[id^="modal-"][class*="absolute inset-0"], [data-testid*="-nux"]')
    try:
        if await modal.count() > 0 and await modal.first.is_visible():
            await page.mouse.click(2, 2)
            await page.wait_for_timeout(300)
    except Exception:
        pass


async def _chatgpt_last_msg_is_classifier(page: Page) -> bool:
    """Return True if the last ChatGPT assistant message is an L2 classifier block.

    The classifier-block UI shares the `text-token-text-error` class with the
    rate-limit banner ("You've hit your limit ..."), so we additionally check
    the message text against `RATE_LIMIT_TEXT_RE` and reject rate-limit hits.
    """
    messages = page.locator('[data-message-author-role="assistant"]')
    count = await messages.count()
    if count == 0:
        return False
    last = messages.nth(count - 1)
    if await last.locator('[class*="text-token-text-error"]').count() == 0:
        return False
    text = (await last.inner_text()).strip()
    if RATE_LIMIT_TEXT_RE.search(text):
        return False
    return True


async def _chatgpt_last_msg_is_rate_limit(page: Page) -> bool:
    """Return True if the last ChatGPT assistant message is a rate-limit banner."""
    messages = page.locator('[data-message-author-role="assistant"]')
    count = await messages.count()
    if count == 0:
        return False
    last = messages.nth(count - 1)
    text = (await last.inner_text()).strip()
    return bool(RATE_LIMIT_TEXT_RE.search(text))


async def _dismiss_claude_modals(page: Page) -> None:
    """Dismiss Claude.ai overlays that intercept pointer events:
    cookie banner + "Spotlight"/NUX modals over the composer.
    """
    for sel in (
        'button:has-text("Reject All Cookies")',
        'button:has-text("Reject all")',
        'button[aria-label*="reject" i]',
    ):
        try:
            btn = page.locator(sel).first
            if await btn.count() > 0:
                await btn.click(timeout=3_000)
                break
        except Exception:
            pass
    modal = page.locator('[data-state="open"][class*="z-modal"]').first
    if await modal.count() == 0:
        return
    for close_sel in (
        '[data-state="open"][class*="z-modal"] button[aria-label*="close" i]',
        '[data-state="open"][class*="z-modal"] button:has-text("Got it")',
        '[data-state="open"][class*="z-modal"] button:has-text("Dismiss")',
        '[data-state="open"][class*="z-modal"] button:has-text("Skip")',
        '[data-state="open"][class*="z-modal"] button:has-text("Maybe later")',
    ):
        try:
            btn = page.locator(close_sel).first
            if await btn.count() > 0:
                await btn.click(timeout=2_000)
                await modal.wait_for(state="detached", timeout=3_000)
                return
        except Exception:
            pass
    try:
        await page.keyboard.press("Escape")
        await modal.wait_for(state="detached", timeout=3_000)
    except Exception:
        pass


async def _dismiss_claude_cookies(page: Page) -> None:
    """Backwards-compatible alias kept for any external callers."""
    await _dismiss_claude_modals(page)


async def select_claude_model(page: Page, slug: str) -> str:
    """Pick a model in Claude.ai's composer model picker.

    Opens the `[data-testid="model-selector-dropdown"]` menu, expands the
    "More models" submenu if the requested item isn't in the top-level list,
    clicks the matching `[role="menuitemradio"]`, and verifies the dropdown's
    aria-label now ends with the model's display name. Returns the display
    label that was selected.

    Raises ValueError on unknown slug; RuntimeError if the picker didn't open
    or didn't update.
    """
    if slug not in CLAUDE_MODELS:
        raise ValueError(
            f"unknown claude model {slug!r}; expected one of {list(CLAUDE_MODELS)}"
        )
    label = CLAUDE_MODELS[slug]["label"]

    dropdown = page.locator('[data-testid="model-selector-dropdown"]')
    await dropdown.wait_for(state="visible", timeout=15_000)

    current = await dropdown.get_attribute("aria-label") or ""
    # Substring (not endswith): Claude.ai appends variant suffixes like
    # "Haiku 4.5 Extended" to the picker label, so an exact tail match fails
    # even when the right base model is selected.
    if label in current:
        return label

    # Match against the menuitemradio's text. Anchor at start of text and
    # require the next char to be a non-digit/non-period so "Opus 3" doesn't
    # accidentally match a future "Opus 3.5".
    radio_pattern = re.compile(rf"^{re.escape(label)}(?:[^\d.]|$)")
    target = page.locator(
        '[role="menuitemradio"]',
    ).filter(has_text=radio_pattern).first

    await dropdown.click()
    try:
        await page.locator('[role="menuitemradio"]').first.wait_for(
            state="visible", timeout=10_000,
        )
    except Exception as e:
        raise RuntimeError("model picker dropdown did not open") from e

    if not await target.is_visible():
        # Item lives under "More models" — expand the submenu.
        more = page.locator(
            '[role="menuitem"]',
        ).filter(has_text=re.compile(r"More models", re.I)).first
        if await more.count() == 0:
            raise RuntimeError(
                f"model {label!r} not in top-level list and no More models submenu",
            )
        await more.click()
        await target.wait_for(state="visible", timeout=5_000)

    await target.click()

    for _ in range(20):
        final = (await dropdown.get_attribute("aria-label")) or ""
        if label in final:
            return label
        await page.wait_for_timeout(150)
    raise RuntimeError(
        f"model picker label did not update to contain {label!r} (got {final!r})",
    )


# ── Served-model auto-tagging ───────────────────────────────────────────────────
#
# Neither the free-tier ChatGPT nor Gemini web UIs let us *pick* a non-default
# model (ChatGPT exposes no model menu; Gemini's menu lists 3.5 Thinking / 3.1
# Pro / Troubleshooting but they're all aria-disabled behind a paid plan). So
# instead of selecting a model we *record* whichever model the product actually
# served, by reading it back from the DOM after each turn. The captured slug is
# threaded through the send-meta dict into target_model (e.g. "chatgpt/gpt-5-5",
# "gemini/flash") — see WebUIAdapter in crescendo.py. Claude (which does have a
# working composer picker) is unaffected and keeps using select_claude_model.

# Gemini's mode-picker button reports the live model in its aria-label, e.g.
# "Open mode picker, currently Flash". Capture the trailing token.
_GEMINI_MODE_RE = re.compile(r"currently\s+(.+?)\s*$", re.I)


async def read_chatgpt_served_model(page: Page) -> str | None:
    """Return the model slug ChatGPT served for the latest assistant turn.

    Completed assistant message nodes carry ``data-message-model-slug`` (e.g.
    ``gpt-5-5``); it is absent on the streaming placeholder, so this is only
    meaningful once generation has finished (send_to_chatgpt awaits that).
    """
    messages = page.locator('[data-message-author-role="assistant"]')
    count = await messages.count()
    if count == 0:
        return None
    slug = await messages.nth(count - 1).get_attribute("data-message-model-slug")
    return slug or None


async def read_gemini_served_model(page: Page) -> str | None:
    """Return the versioned Gemini model currently served (e.g. "3.5-flash").

    The mode-switch button only exposes the short mode name ("currently Flash")
    — the *version* lives in the picker menu ("3.5 Flash"). So we read the short
    token from the button, open the picker, find the menu item carrying that
    token, read its full versioned label, then close the menu. Falls back to the
    short token (e.g. "flash") if the menu can't be read. Returns a hyphenated
    lowercased slug.
    """
    btn = page.locator('button[aria-label^="Open mode picker"]').first
    if await btn.count() == 0:
        return None
    aria = await btn.get_attribute("aria-label") or ""
    m = _GEMINI_MODE_RE.search(aria)
    token = (m.group(1) if m else (await btn.inner_text() or "")).strip()
    if not token:
        return None
    full = token  # fallback to the short name if the menu read fails
    try:
        await btn.click()
        item = page.locator('[role="menuitem"]').filter(
            has_text=re.compile(rf"\b{re.escape(token)}\b")
        ).first
        await item.wait_for(state="visible", timeout=5_000)
        # First text line is the model name; the rest is the description.
        full = (await item.inner_text()).strip().splitlines()[0].strip()
    except Exception:
        pass
    finally:
        await page.keyboard.press("Escape")
    return re.sub(r"\s+", "-", full).lower()


async def read_served_model(page: Page, platform: str) -> str | None:
    """Dispatch to the per-platform served-model reader; None if not applicable."""
    if platform == "chatgpt":
        return await read_chatgpt_served_model(page)
    if platform == "gemini":
        return await read_gemini_served_model(page)
    return None


async def _claude_last_msg_is_classifier(page: Page, pre_turn_count: int) -> bool:
    """Return True if the last Claude.ai response is an L2 classifier hard block.

    The hard-block UI shows a banner reading "Chat paused — <model>'s safety
    filters flagged this chat" in place of the usual response. These banner
    texts don't appear in normal chats.
    """
    for sel in ('text="Chat paused"', 'text="safety filters flagged"'):
        if await page.locator(sel).count() > 0:
            return True
    return False


async def _meta_last_msg_is_classifier(page: Page, pre_turn_count: int) -> bool:
    """Return True if the last Meta AI response is an L2 classifier hard block.

    Meta does not terminate the chat with a banner the way Claude/Gemini do;
    instead its application-layer safety system replaces the model response with
    a fixed templated string (see META_L2_TEXT_RE) — e.g. "...goes against the
    rules. Try again in 24 hours." or "Sorry, I can't help you with this request
    at the moment." These repeat verbatim every turn for hard categories, so the
    first occurrence is treated as an L2 block and the run is stopped.
    """
    messages = page.locator('[data-testid="assistant-message"]')
    count = await messages.count()
    if count == 0:
        return False
    text = (await messages.nth(count - 1).inner_text()).strip()
    return bool(META_L2_TEXT_RE.search(text))


async def _meta_last_msg_is_rate_limit(page: Page) -> bool:
    """Return True if Meta AI's latest assistant bubble completed empty.

    Meta surfaces a rate limit / dropped generation not as banner text but as a
    pending spinner that resolves into an EMPTY assistant-message bubble
    (data-streaming-state="DONE" with no text content) — confirmed live against
    a rate-limited conversation. A genuine refusal always carries text, so an
    empty trailing bubble is a reliable failed-generation signal. (A text-only
    response that happens to be empty — e.g. image-only — would false-positive,
    but that never occurs for our text-extraction objectives.)

    send_to_meta already waits for streaming to finish before this runs, so a
    normal turn's bubble has its text by now; only a failed generation is empty.

    Second signal — HTML: when Meta throttles the session it tears down the
    composer, removing the send button from the DOM entirely. That is the
    "[browser error — composer-send-button not visible]" failure, surfaced here
    as a rate limit rather than fed to the judge as a response. Checked as
    count==0 (element absent) rather than visibility, because an empty composer
    can legitimately hide the button while the element still exists — absence is
    the throttle-specific signal.
    """
    if await page.locator('button[data-testid="composer-send-button"]').count() == 0:
        return True
    messages = page.locator('[data-testid="assistant-message"]')
    count = await messages.count()
    if count == 0:
        return False
    text = (await messages.nth(count - 1).inner_text()).strip()
    return text == ""


async def _gemini_last_msg_is_classifier(page: Page, pre_turn_count: int) -> bool:
    """Return True if the last Gemini response is an L2 classifier hard block."""
    post_count = await page.locator("model-response").count()
    if post_count <= pre_turn_count:
        return False
    last = page.locator("model-response").nth(post_count - 1)
    policy_link = last.locator(
        'a[href*="gemini.google/policy-guidelines"], '
        'a[href*="policies.google.com"], '
        'a[href*="support.google.com/gemini"]'
    )
    return await policy_link.count() > 0


async def send_to_chatgpt(page: Page, text: str) -> str:
    # Pre-count assistant turns so we can pin the new one and avoid re-reading
    # the prior turn's text if Enter gets swallowed (modal flap / focus race) —
    # the same silent send no-op the Claude/Gemini/Meta helpers guard against.
    # Without this, a no-op'd send falls through to .nth(count-1) and returns the
    # previous turn's response, producing byte-identical responses across turns.
    messages = page.locator('[data-message-author-role="assistant"]')
    pre_count = await messages.count()
    composer = page.locator("#prompt-textarea")
    await composer.wait_for(state="visible", timeout=30_000)
    await _dismiss_chatgpt_modals(page)
    await composer.click()
    # Reliably empty the composer first. A prior turn's send can silently no-op
    # (see below), leaving its text in the ProseMirror editor; Control+a alone
    # then relies on the next type() replacing the selection, but on a focus
    # flap that type() has been observed to *append* — concatenating several
    # turns into one giant prompt that finally sends at once. An explicit Delete
    # of the selection makes the clear deterministic.
    await page.keyboard.press("Control+a")
    await page.keyboard.press("Delete")
    await page.keyboard.type(text, delay=20)
    # Send by clicking the send button, not by pressing Enter. ChatGPT's composer
    # is a ProseMirror editor: an Enter fired before React has committed the typed
    # text is treated as a newline, not a send, so the message never goes out and
    # the turn reads as "[no response captured]". Playwright's click auto-waits for
    # the button to be actionable (enabled), which only happens once the text is
    # committed — eliminating that race. Fall back to Enter if the button is gone.
    send_btn = page.locator('button[data-testid="composer-send-button"]')
    try:
        await send_btn.wait_for(state="visible", timeout=10_000)
        await send_btn.click()
    except Exception:
        await page.keyboard.press("Enter")
    new_turn = messages.nth(pre_count)
    try:
        await new_turn.wait_for(state="attached", timeout=30_000)
    except Exception:
        # No new assistant turn attached — the send silently no-op'd.
        return "[no response captured]"
    stop_btn = page.locator('button[data-testid="stop-button"], button[aria-label="Stop generating"]')
    try:
        await stop_btn.first.wait_for(state="visible", timeout=15_000)
        await stop_btn.first.wait_for(state="hidden", timeout=120_000)
    except Exception:
        pass
    await page.wait_for_timeout(1_000)
    return (await new_turn.inner_text()).strip()


async def send_to_claude(page: Page, text: str) -> str:
    await _dismiss_claude_modals(page)
    # Pre-count assistant turns (font-claude-* roots) so we can pin the new
    # one and avoid re-reading the prior turn's text if the Enter key gets
    # swallowed by an uncommitted Tiptap state.
    pre_count = await page.locator('div[class*="font-claude"]').count()
    composer = page.locator('div[contenteditable="true"]').last
    await composer.wait_for(state="visible", timeout=30_000)
    await composer.click()
    await page.wait_for_timeout(300)
    await composer.fill("")
    await page.keyboard.type(text, delay=20)
    # Tiptap/ProseMirror needs a beat for the editor state to commit before
    # Enter is interpreted as "send" rather than swallowed.
    await page.wait_for_timeout(1_500)
    await composer.focus()
    await page.keyboard.press("Enter")
    await page.wait_for_timeout(500)
    new_turn = page.locator('div[class*="font-claude"]').nth(pre_count)
    try:
        await new_turn.wait_for(state="attached", timeout=30_000)
    except Exception:
        # No new assistant turn — either the send no-op'd or it was a hard
        # block. Check for the chat-paused banner; otherwise sentinel.
        banner = page.locator('text="Chat paused"').first
        if await banner.count() > 0:
            return (await banner.inner_text()).strip()
        return "[no response captured]"
    stop_btn = page.locator('button[aria-label="Stop"], button[aria-label="Stop generating"]')
    try:
        await stop_btn.first.wait_for(state="visible", timeout=15_000)
        await stop_btn.first.wait_for(state="hidden", timeout=120_000)
    except Exception:
        pass
    # Wait for any active stream to settle.
    try:
        await page.locator('[data-is-streaming="true"]').first.wait_for(
            state="hidden", timeout=120_000,
        )
    except Exception:
        pass
    await page.wait_for_timeout(800)
    return (await new_turn.inner_text()).strip()


async def send_to_gemini(page: Page, text: str) -> str:
    # Pin the new response by pre-count: if no new <model-response> attaches
    # after Send, the click silently no-op'd (rare aria-disabled flap or
    # Quill focus race) — read of `.nth(count-1)` then returns the *prior*
    # turn's text, producing identical responses across multiple turns. We
    # require the (pre_count)-th element to attach before reading.
    pre_count = await page.locator("model-response").count()
    composer = page.locator("rich-textarea .ql-editor")
    await composer.wait_for(state="visible", timeout=30_000)
    await composer.click()
    await page.keyboard.press("Control+a")
    await page.keyboard.press("Delete")
    await composer.type(text, delay=20)
    # Gemini's Send button is rendered with aria-disabled="true" until the
    # Quill editor has registered the new input. On FR pages this can take
    # 20+ seconds. Poll the attribute (visibility alone is not enough — the
    # button is visible the whole time, just unclickable).
    send_btn = page.locator('button[aria-label="Send message"]')
    await send_btn.wait_for(state="visible", timeout=10_000)
    for _ in range(60):  # up to ~60s
        if (await send_btn.get_attribute("aria-disabled")) in (None, "false"):
            break
        await page.wait_for_timeout(1_000)
    await send_btn.click()
    new_response = page.locator("model-response").nth(pre_count)
    try:
        await new_response.wait_for(state="attached", timeout=30_000)
    except Exception:
        return "[no response captured]"
    stop_btn = page.locator('button[aria-label="Stop response"]')
    try:
        await stop_btn.wait_for(state="visible", timeout=15_000)
        await stop_btn.wait_for(state="hidden", timeout=120_000)
    except Exception:
        try:
            await page.locator("pending-response").wait_for(state="hidden", timeout=30_000)
        except Exception:
            pass
    await page.wait_for_timeout(600)
    return (await new_response.inner_text()).strip()


async def send_to_meta(page: Page, text: str) -> str:
    # Meta AI uses a Lexical contenteditable as the real composer; the sibling
    # textarea is an a11y mirror that lexical doesn't read from.
    # Pre-count assistant messages so we can pin the new one — without this,
    # a silent send no-op (rare Lexical commit race) drops us through to
    # reading the prior turn's text via .nth(count-1).
    pre_count = await page.locator('[data-testid="assistant-message"]').count()
    composer = page.locator('div[data-testid="composer-input"][role="textbox"]')
    await composer.wait_for(state="visible", timeout=30_000)
    await composer.click()
    await page.wait_for_timeout(300)
    await page.keyboard.press("Control+a")
    await page.keyboard.press("Delete")
    await page.keyboard.type(text, delay=20)
    # Lexical needs a beat to commit before the send button enables.
    await page.wait_for_timeout(800)
    send_btn = page.locator('button[data-testid="composer-send-button"]')
    await send_btn.wait_for(state="visible", timeout=10_000)
    await send_btn.click()
    # Require the assistant-message count to actually advance past pre_count.
    # A silent send no-op (Lexical commit race, or a rate-limit drop that never
    # produces a bubble) leaves the count unchanged; without this guard we fall
    # through to .nth(pre_count) and re-read the PRIOR turn's text, which is how
    # whole runs ended up as byte-identical duplicate responses.
    try:
        await page.wait_for_function(
            "([sel, n]) => document.querySelectorAll(sel).length > n",
            arg=['[data-testid="assistant-message"]', pre_count],
            timeout=30_000,
        )
    except Exception:
        return "[no response captured]"
    new_message = page.locator('[data-testid="assistant-message"]').last
    # During generation the same button changes its aria-label to "Cancel".
    cancel_btn = page.locator('button[data-testid="composer-send-button"][aria-label="Cancel"]')
    try:
        await cancel_btn.wait_for(state="visible", timeout=15_000)
        await cancel_btn.wait_for(state="hidden", timeout=120_000)
    except Exception:
        pass
    await page.wait_for_timeout(800)
    return (await new_message.inner_text()).strip()


SEND_FNS = {
    "chatgpt": send_to_chatgpt,
    "claude": send_to_claude,
    "gemini": send_to_gemini,
    "meta": send_to_meta,
}


async def start_fresh_conversation(page: Page, start_url: str) -> None:
    """Navigate to a fresh conversation and wait for the composer to be ready."""
    for attempt in range(3):
        try:
            await page.goto(start_url, timeout=90_000)
            break
        except Exception:
            if attempt == 2:
                raise
            await page.wait_for_timeout(5_000)
    await page.wait_for_load_state("domcontentloaded", timeout=60_000)
    if "chatgpt" in start_url:
        await page.locator("#prompt-textarea").wait_for(state="visible", timeout=30_000)
        await _dismiss_chatgpt_modals(page)
    elif "claude" in start_url:
        await page.locator('div[contenteditable="true"]').last.wait_for(state="visible", timeout=30_000)
        await _dismiss_claude_modals(page)
    elif "gemini" in start_url:
        await page.locator("rich-textarea .ql-editor").wait_for(state="visible", timeout=30_000)
    elif "meta.ai" in start_url:
        await page.locator('div[data-testid="composer-input"][role="textbox"]').wait_for(
            state="visible", timeout=30_000,
        )


async def pre_send_turn_count(page: Page, platform: str) -> int:
    """Snapshot response-element count before sending so the post-send delta
    can distinguish L2 hard blocks (no new element added) from RLHF refusals
    (new element always added) on Claude and Gemini."""
    if platform == "claude":
        return await page.locator('[data-testid="assistant-turn-with-context"]').count()
    if platform == "gemini":
        return await page.locator("model-response").count()
    if platform == "meta":
        return await page.locator('[data-testid="assistant-message"]').count()
    return 0


async def detect_l2_classifier(page: Page, platform: str, pre_turn_count: int) -> bool:
    if platform == "chatgpt":
        return await _chatgpt_last_msg_is_classifier(page)
    if platform == "claude":
        return await _claude_last_msg_is_classifier(page, pre_turn_count)
    if platform == "gemini":
        return await _gemini_last_msg_is_classifier(page, pre_turn_count)
    if platform == "meta":
        return await _meta_last_msg_is_classifier(page, pre_turn_count)
    return False


async def detect_rate_limit(page: Page, platform: str) -> bool:
    """Detect a platform-shown rate-limit banner.

    Rate-limit responses are not real model output and not L2 classifier
    blocks; the run should be aborted and the record marked invalid by
    validate_and_collate.
    """
    if platform == "chatgpt":
        return await _chatgpt_last_msg_is_rate_limit(page)
    if platform == "meta":
        return await _meta_last_msg_is_rate_limit(page)
    return False
