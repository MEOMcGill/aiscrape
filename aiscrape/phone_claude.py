"""Claude scraper that drives claude.ai in Chrome on a real Android phone.

The Claude counterpart of `phone_chatgpt.PhoneChatGPTScraper`, on the same handsets
and transport (`PhoneChromeSession`: adb + the Chrome DevTools Protocol):

    launch claude.ai/new?q=<prompt>  →  press send  →  poll the transcript until the
    turn stops streaming  →  read the finished turn from the conversation JSON the web
    app itself loads  →  return a `ChatResult`.

claude.ai has no anonymous mode, so every phone needs an account. `log_in` makes one
from the Google account the handset carries, the same way ChatGPT's does.

The page is only used to find this ask's tab and to tell when the turn is done. The
answer, model and citations come from `/api/organizations/<org>/chat_conversations/
<id>`, because the page folds a citation's extra sources behind "+N" and names the
model only as a display label ("Sonnet 5.5 Medium").
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from urllib.parse import quote

from .google_aimode import DEBUG_DIR, Reference, _domain, _real_url
from .logger import logger
from .models import ChatResult, now_iso
from .phone_chatgpt import (DEFAULT_BIRTHDAY, _CLICK_BY_TEXT_JS, _CLICK_SELECTOR_JS,
                            _local_part, _normalise)
from .phone_farm import DEFAULT_CDP_PORT, PhoneChromeSession

# `q` only fills the composer; the ask still has to press send.
CLAUDE_ASK_URL = "https://claude.ai/new?q={q}"
CLAUDE_HOME_URL = "https://claude.ai/new"
CLAUDE_LOGIN_URL = "https://claude.ai/login"
CLAUDE_HOST = "claude.ai"

DEFAULT_SETTLE_MS = 6_000
# Longer than ChatGPT's: a turn that searches the web runs several searches first.
DEFAULT_ANSWER_TIMEOUT_S = 180

LOGIN_TIMEOUT_S = 300

# Chrome's native "Continue as <name>" sheet for Google sign-in (FedCM).
_SHEET_CONTINUE = r'resource-id="com\.android\.chrome:id/account_selection_continue_btn"'
_SHEET_NAME = r'resource-id="com\.android\.chrome:id/title"'
_GOOGLE_BUTTON = r'text="Continue with Google"'
_NO_THANKS = r'text="No thanks"[^>]*package="com\.android\.chrome"'

# The answer is replaced by a banner, so these are matched on the page, not the turn.
_LIMIT_RE = re.compile(r"(you('|’)?ve (hit|reached) your (usage|message) limit"
                       r"|out of free messages|usage limit reached)", re.IGNORECASE)
_PAUSED_RE = re.compile(r"\bchat paused\b|safety filters flagged", re.IGNORECASE)

# The screen-reader lead-in on the page's copy of a turn.
_TURN_PREFIX_RE = re.compile(r"^Claude responded:\s*", re.IGNORECASE)

# One read of the page: which ask this tab holds and whether its turn is done.
_CHAT_EXTRACT_JS = r"""
(() => {
  const users = [...document.querySelectorAll('[data-testid="user-message"]')];
  const bots = [...document.querySelectorAll('[data-testid="assistant-message"]')];
  const last = bots.length ? bots[bots.length - 1] : null;
  const composer = document.querySelector('[data-testid="chat-input"]');
  const streaming = !!document.querySelector('[data-is-streaming="true"]')
    || [...document.querySelectorAll('button')].some(b =>
         /^stop( response| generating)?$/i.test((b.getAttribute('aria-label') || '').trim()));
  const anchors = last ? [...last.querySelectorAll('a[href]')]
    .filter(a => /^https?:/.test(a.href))
    .map(a => ({href: a.href, text: (a.innerText || '').trim()})) : [];
  return JSON.stringify({
    url: location.href,
    signedIn: !!composer && !/\/login|\/onboarding/.test(location.pathname),
    prompt: users.length ? (users[0].innerText || '').trim() : '',
    composerText: composer ? (composer.innerText || '').trim() : '',
    nUser: users.length,
    nAssistant: bots.length,
    streaming,
    text: last ? (last.innerText || '').trim() : '',
    anchors,
    bodyText: (document.body.innerText || '').slice(0, 4000),
  });
})()
"""

_SEND_JS = r"""
(() => {
  const b = document.querySelector('[data-testid="chat-input-send"]');
  if (!b || b.disabled) return JSON.stringify({sent: false});
  b.click();
  return JSON.stringify({sent: true});
})()
"""

# The org a conversation lives in: the active one when the cookie names it.
_ORG_JS = r"""
  const orgs = await fetch('/api/organizations', {credentials: 'include'}).then(r => r.json());
  const active = (document.cookie.match(/(?:^|; )lastActiveOrg=([^;]+)/) || [])[1];
  const org = (orgs.find(o => o.uuid === active)
               || orgs.find(o => (o.capabilities || []).includes('chat')) || orgs[0] || {}).uuid;
"""

# The finished turn, from the JSON the app renders it from. The answer is the prose
# after the last tool call; text before it is narration ("I'll look that up").
_CONVERSATION_JS = r"""
(async () => {
  const conv = (location.pathname.match(/\/chat\/([0-9a-f-]{36})/) || [])[1];
  if (!conv) return JSON.stringify({error: 'no conversation id in ' + location.pathname});
  try {
""" + _ORG_JS + r"""
    const r = await fetch(`/api/organizations/${org}/chat_conversations/${conv}`
                          + '?tree=True&rendering_mode=messages&render_all_tools=true',
                          {credentials: 'include'});
    if (!r.ok) return JSON.stringify({error: 'conversation ' + r.status});
    const j = await r.json();
    const msgs = j.chat_messages || [];
    const human = msgs.find(m => m.sender === 'human');
    const bot = msgs.filter(m => m.sender === 'assistant').pop();
    const blocks = (bot && bot.content) || [];
    let lastTool = -1;
    blocks.forEach((b, i) => { if (b.type === 'tool_use' || b.type === 'tool_result') lastTool = i; });
    let answer = blocks.slice(lastTool + 1).filter(b => b.type === 'text');
    if (!answer.length) answer = blocks.filter(b => b.type === 'text');
    const citations = [];
    for (const b of answer) {
      for (const c of b.citations || []) {
        const sources = (c.sources && c.sources.length) ? c.sources : [c];
        for (const s of sources) if (s && s.url) citations.push({url: s.url, title: s.title || ''});
      }
    }
    return JSON.stringify({
      model: j.model || '',
      settings: j.settings || {},
      prompt: human ? (human.content || []).filter(b => b.type === 'text')
                        .map(b => b.text).join('\n').trim() : '',
      answer: answer.map(b => b.text || '').join('\n\n').trim(),
      stopReason: (bot && bot.stop_reason) || '',
      searches: blocks.filter(b => b.type === 'tool_use' && b.name === 'web_search')
                      .map(b => (b.input || {}).query || ''),
      citations,
    });
  } catch (e) { return JSON.stringify({error: String(e)}); }
})()
"""

# `enabled_melange` is Settings → Memory → "Generate memory from chats"; unset means off.
_MEMORY_JS = r"""
(async () => {
  try {
    const r = await fetch('/api/account', {credentials: 'include'});
    if (!r.ok) return JSON.stringify({error: 'account ' + r.status});
    const s = (await r.json()).settings || {};
    return JSON.stringify({memory: s.enabled_melange === true});
  } catch (e) { return JSON.stringify({error: String(e)}); }
})()
"""

_ACCEPT_COOKIES_JS = r"""
(() => {
  const b = document.querySelector('[data-testid="consent-accept"]');
  if (b) b.click();
  return JSON.stringify({clicked: !!b});
})()
"""

# ── signing up with the phone's own Google account ──────────────────────────
#
# Order of a first sign-in: Google sheet → terms → birthday → plan → app promo →
# training opt-in → name → role → /new. A returning account skips straight to /new.

_LOGIN_STEP_ORDER = {
    "error": 1, "waiting": 2, "start": 3, "choose_account": 4, "consent": 5,
    "terms": 6, "birthday": 7, "plan": 8, "app": 9, "training": 10, "name": 11,
    "role": 12, "done": 13,
}

_LOGIN_STATE_JS = r"""
(() => {
  const body = document.body.innerText || '';
  const txt = e => ((e.innerText || '') + ' ' + (e.getAttribute('aria-label') || '')).trim();
  const has = re => [...document.querySelectorAll('button, a[href], [role="button"]')]
    .some(e => re.test(txt(e)));
  return JSON.stringify({
    url: location.href,
    composer: !!document.querySelector('[data-testid="chat-input"]'),
    googleButton: !!document.querySelector('[data-testid="login-with-google"]'),
    accounts: [...document.querySelectorAll('[data-identifier]')]
      .map(e => e.getAttribute('data-identifier')),
    consent: /accounts\.google\.com/.test(location.host) && has(/^continue$/i),
    terms: /consumer terms/i.test(body) && !!document.querySelector('input[type=checkbox]'),
    birthday: !!document.querySelector('[role="spinbutton"][aria-label^="day" i]'),
    plan: has(/^use claude for free$/i),
    app: /get the app/i.test(body) && has(/^skip$/i),
    training: /before your first chat/i.test(body),
    name: /what.s your name/i.test(body) && !!document.querySelector('input[type=text]'),
    role: has(/^set up later$/i),
    error: /error logging you in/i.test(body),
    body: body.slice(0, 300),
  });
})()
"""

# Tick the two required boxes; leave the promotional-email one alone.
_TERMS_JS = r"""
(() => {
  for (const box of document.querySelectorAll('input[type=checkbox]')) {
    const label = ((box.closest('label') || box.parentElement.parentElement).innerText || '');
    if (/consumer terms|privacy policy/i.test(label) && !box.checked) box.click();
  }
  const go = document.querySelector('[data-testid="continue"]');
  if (go) go.click();
  return JSON.stringify({clicked: !!go});
})()
"""

# "Help improve our AI models" — the only checkbox on that screen. `want` is JSON.
_TRAINING_JS = r"""
(() => {
  const want = %s;
  const box = document.querySelector('input[type=checkbox]');
  if (box && box.checked !== want) box.click();
  return JSON.stringify({checked: box ? box.checked : null});
})()
"""

_FOCUS_JS = r"""
(() => {
  const el = document.querySelector(%s);
  if (!el) return JSON.stringify({ok: false});
  el.focus();
  return JSON.stringify({ok: true});
})()
"""


class ClaudeLoginError(RuntimeError):
    """The Google sign-up/sign-in flow could not be completed on this phone."""


class ClaudeSignedOutError(RuntimeError):
    """claude.ai is signed out on this phone, so it cannot be asked anything."""


class PhoneClaudeScraper:
    """Ask Claude on one Android phone, over adb + CDP.

    Args mirror `PhoneChatGPTScraper`: pass a `serial` (and `ssh_host`/`adb`) to
    open a session, or an open `session` to share one with the other scrapers.

    `memory_enabled` on each result is read from the account, never changed.

        with PhoneClaudeScraper("R58MEXAMPLE", ssh_host="phone-farm") as s:
            print(s.ask("how does a heat pump work").response)
    """

    def __init__(self, serial: str | None = None, *,
                 session: PhoneChromeSession | None = None,
                 ssh_host: str | None = None, adb: str | None = None,
                 cdp_port: int = DEFAULT_CDP_PORT,
                 settle_ms: int = DEFAULT_SETTLE_MS,
                 answer_timeout_s: int = DEFAULT_ANSWER_TIMEOUT_S,
                 keep_awake: bool = True, sleep_on_exit: bool = False,
                 debug: bool = False):
        if session is None:
            if not serial:
                raise ValueError("pass a phone serial, or a PhoneChromeSession to ask on")
            session = PhoneChromeSession(
                serial, ssh_host=ssh_host, adb=adb, cdp_port=cdp_port,
                keep_awake=keep_awake, sleep_on_exit=sleep_on_exit, debug=debug)
            self._owns_session = True
        else:
            self._owns_session = False
        self.session = session
        self.settle_ms = settle_ms
        self.answer_timeout_s = answer_timeout_s
        self.debug = debug
        self._prepared = False
        self.memory_enabled: bool | None = None
        self._last_target_id = ""
        self._last_page: dict | None = None

    @property
    def serial(self) -> str:
        return self.session.serial

    def __enter__(self) -> "PhoneClaudeScraper":
        if self._owns_session:
            self.session.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._owns_session:
            self.session.__exit__(*exc)

    # -- asking -----------------------------------------------------------------

    def ask(self, prompt: str) -> ChatResult:
        """Put one prompt to Claude on the phone, in a fresh conversation.

        Raises ClaudeSignedOutError on a phone with no claude.ai session.
        """
        self.session.require_open()
        self.prepare()
        self.session.ensure_visible()
        scraped_at = now_iso()
        self.session.launch(CLAUDE_ASK_URL.format(q=quote(prompt, safe="")))
        self._last_target_id, self._last_page = "", None
        try:
            data = self._await_answer(prompt)
            if data is None:
                return self._result(prompt, scraped_at,
                                    note="no claude conversation found for this ask")
            if not data.get("signedIn") and not data.get("nUser"):
                raise ClaudeSignedOutError(f"{self.serial}: claude.ai is signed out")

            page_answer = _strip_turn_prefix(data.get("text", ""))
            # The answer is quoted out of the page text so it cannot match a banner.
            chrome = (data.get("bodyText") or "").replace(page_answer[:3000], "")
            if not page_answer and _LIMIT_RE.search(chrome):
                return self._result(prompt, scraped_at, blocked=True, rate_limited=True,
                                    note="claude usage limit")
            l2 = bool(_PAUSED_RE.search(chrome))

            conv = self.session.evaluate(self._last_page, _CONVERSATION_JS, timeout=45) or {}
            if conv.get("error") or not conv:
                gap = f"conversation JSON unreadable ({conv.get('error', 'no response')})"
                self._snapshot(prompt, data, [gap])
                return self._result(
                    prompt, scraped_at, response=page_answer, l2_block=l2,
                    memory_enabled=self.memory_enabled,
                    references=_parse_citations([], data.get("anchors", [])),
                    note=gap + "; answer and sources read off the page")

            answer = conv.get("answer", "")
            notes = []
            if l2:
                notes.append("claude paused the chat (safety filter)")
            if not answer:
                notes.append("the answer turn rendered empty")
            if conv.get("searches"):
                notes.append(f"searched the web {len(conv['searches'])}x")
            return self._result(
                prompt, scraped_at, response=answer,
                served_model=conv.get("model") or None,
                memory_enabled=self.memory_enabled, l2_block=l2,
                references=_parse_citations(conv.get("citations", []), []),
                note="; ".join(notes),
            )
        finally:
            self._close_last_tab()

    def ask_many(self, prompts: list[str]) -> list[ChatResult]:
        return [self.ask(p) for p in prompts]

    def prepare(self, force: bool = False) -> None:
        """Open claude.ai once per session: accept cookies, read the memory setting.

        Raises ClaudeSignedOutError if the phone has no session.
        """
        if self._prepared and not force:
            return
        self.session.unlock()
        self.session.foreground_chrome()
        self.session.launch(CLAUDE_HOME_URL)
        deadline = time.monotonic() + 45
        refused = ""
        while time.monotonic() < deadline:
            time.sleep(2.0)
            page = _any_claude_page(self.session.pages())
            state = self.session.evaluate(page, _LOGIN_STATE_JS) if page else None
            if not state:
                continue
            if state.get("googleButton"):
                self.session.close_tab(page.get("id") or "")
                raise ClaudeSignedOutError(f"{self.serial}: claude.ai is signed out")
            if not state.get("composer"):
                continue
            # A signed-out /new paints the composer before redirecting to /login, so
            # only an account the API will return counts as signed in.
            mem = self.session.evaluate(page, _MEMORY_JS) or {}
            if "memory" not in mem:
                refused = mem.get("error", "no response")
                continue
            self.session.evaluate(page, _ACCEPT_COOKIES_JS)
            self.memory_enabled = mem["memory"]
            self.session.close_tab(page.get("id") or "")
            self._prepared = True
            return
        if refused.startswith("account 40"):
            raise ClaudeSignedOutError(f"{self.serial}: claude.ai is signed out ({refused})")
        logger.warning(f"[{self.serial}] claude.ai never finished loading; asking anyway")
        self._prepared = True

    # -- signing in -------------------------------------------------------------

    def is_logged_in(self) -> bool:
        try:
            self.prepare(force=True)
        except ClaudeSignedOutError:
            return False
        return True

    def ensure_logged_in(self, **kw) -> bool:
        """Sign in (creating the account if need be) unless already signed in."""
        if self.is_logged_in():
            return True
        logger.info(f"[{self.serial}] claude is signed out; signing in with the "
                    "phone's Google account")
        self.log_in(**kw)
        return True

    def log_in(self, *, email: str | None = None, name: str | None = None,
               birthday: str = DEFAULT_BIRTHDAY, allow_training: bool = False,
               timeout_s: int = LOGIN_TIMEOUT_S) -> str:
        """Sign claude.ai in with the Google account on this phone; return the address.

        A first sign-in creates the Claude account: it accepts the Consumer Terms and
        Privacy Policy, declines promotional email, picks the free plan, and sets
        "Help improve our AI models" to `allow_training`. `name` defaults to the
        Google profile's name.
        """
        self.session.require_open()
        if email is None:
            accounts = self.session.google_accounts()
            if len(accounts) != 1:
                raise ClaudeLoginError(
                    f"{self.serial} has {len(accounts)} Google accounts ({accounts}); "
                    "pass email= to say which one to sign in with")
            email = accounts[0]

        self.session.unlock()
        self.session.foreground_chrome()
        for stale in ("claude.ai/onboarding", "claude.ai/login", "accounts.google.com"):
            self.session.close_tabs_matching(stale, keep=0)
        self.session.launch(CLAUDE_LOGIN_URL)
        time.sleep(6)

        deadline = time.monotonic() + timeout_s
        google_name, last, stalled = "", "", 0
        while time.monotonic() < deadline:
            page, state, step = self._furthest_flow_page()
            self.session.ensure_visible()
            if page is not None:
                self.session.activate(page)
            if step == "done" and "memory" not in (self.session.evaluate(page, _MEMORY_JS) or {}):
                step = "waiting"   # the shell /new paints before a signed-out redirect
            if step == "done":
                logger.info(f"[{self.serial}] claude signed in as {email}")
                for host in ("claude.ai/onboarding", "claude.ai/login", "accounts.google.com"):
                    self.session.close_tabs_matching(host, keep=0)
                self._prepared = False
                return email
            stalled = stalled + 1 if step == last else 0
            last = step
            if stalled == 2:
                self.session.unlock()
                self.session.foreground_chrome()
            if stalled >= 6:
                raise ClaudeLoginError(
                    f"{self.serial}: claude login stuck at '{step}' "
                    f"({state.get('url', '')[:80]}): {state.get('body', '')[:200]}")
            logger.debug(f"[{self.serial}] claude login: {step}")

            if step == "error" or (step == "waiting" and page is None):
                self.session.launch(CLAUDE_LOGIN_URL)
            elif step in ("start", "waiting"):
                google_name = self._tap_google_sign_in() or google_name
            elif step == "choose_account":
                if email not in state["accounts"]:
                    raise ClaudeLoginError(f"{self.serial}: {email} is not offered by the "
                                           f"Google chooser ({state['accounts']})")
                self.session.evaluate(
                    page, _CLICK_SELECTOR_JS % json.dumps(f'[data-identifier="{email}"]'))
            elif step == "consent":
                self.session.evaluate(page, _CLICK_BY_TEXT_JS % json.dumps("continue"))
            elif step == "terms":
                self.session.evaluate(page, _TERMS_JS)
            elif step == "birthday":
                self._fill_birthday(page, birthday)
                self.session.evaluate(page, _CLICK_BY_TEXT_JS % json.dumps("continue"))
            elif step == "plan":
                self.session.evaluate(page, _CLICK_BY_TEXT_JS % json.dumps("use claude for free"))
            elif step == "app":
                self.session.evaluate(page, _CLICK_BY_TEXT_JS % json.dumps("skip"))
            elif step == "training":
                self.session.evaluate(page, _TRAINING_JS % json.dumps(allow_training))
                self.session.evaluate(page, _CLICK_BY_TEXT_JS % json.dumps("continue"))
            elif step == "name":
                if (self.session.evaluate(page, _FOCUS_JS % json.dumps("input[type=text]"))
                        or {}).get("ok"):
                    self.session.insert_text(page, name or google_name or _local_part(email))
                time.sleep(1)
                self.session.evaluate(page, _CLICK_BY_TEXT_JS % json.dumps("continue"))
            elif step == "role":
                self.session.evaluate(page, _CLICK_BY_TEXT_JS % json.dumps("set up later"))
            time.sleep(4)

        raise ClaudeLoginError(f"{self.serial}: claude login did not finish within "
                               f"{timeout_s}s (last step '{last}')")

    def _tap_google_sign_in(self) -> str:
        """Tap through Chrome's native Google sheet; return the profile name it showed.

        The sheet usually opens on its own over the login page. If not, the page's
        "Continue with Google" button is tapped for real, since a JS click is refused.
        """
        self._clear_native_overlays()
        sheet = self.session.ui_bounds(_SHEET_CONTINUE)
        if sheet is None:
            button = self.session.ui_bounds(_GOOGLE_BUTTON)
            if button is None:
                return ""
            self.session.tap(*button)
            time.sleep(4)
            sheet = self.session.ui_bounds(_SHEET_CONTINUE)
            if sheet is None:
                return ""   # went to accounts.google.com instead; the loop handles it
        name = self._sheet_name()
        self.session.tap(*sheet)
        return name

    def _clear_native_overlays(self) -> None:
        """Dismiss Android UI drawn over Chrome, which hides the sheet from uiautomator.

        Seen on these phones: Samsung's "Circle to Search" tip (a separate window
        with focus) and Chrome's own "notifications make things easier" prompt.
        """
        if "com.android.chrome" not in self.session.focused_app():
            self.session.adb.shell(self.serial, "input keyevent KEYCODE_BACK")
            time.sleep(1)
        no_thanks = self.session.ui_bounds(_NO_THANKS)
        if no_thanks:
            self.session.tap(*no_thanks)
            time.sleep(2)

    def _sheet_name(self) -> str:
        """The profile name on the sheet, from the dump `ui_bounds` just left behind."""
        xml = self.session.adb.shell(self.serial, "cat /sdcard/ui.xml")
        node = next((n for n in re.findall(r"<node [^>]*>", xml) if re.search(_SHEET_NAME, n)), "")
        text = re.search(r'text="([^"]*)"', node)
        return text.group(1).strip() if text else ""

    def _fill_birthday(self, page: dict, birthday: str) -> None:
        """Type each segment by its label, since their order follows the locale."""
        year, month, day = birthday.split("-")
        for label, value in (("day", day), ("month", month), ("year", year)):
            sel = f'[role="spinbutton"][aria-label^="{label}" i]'
            if (self.session.evaluate(page, _FOCUS_JS % json.dumps(sel)) or {}).get("ok"):
                self.session.type_keys(page, value)
        time.sleep(1)

    def _furthest_flow_page(self) -> tuple[dict | None, dict, str]:
        """The open sign-in tab furthest along the flow, with its state and step."""
        best: tuple[dict | None, dict, str] = (None, {}, "waiting")
        best_rank = -1
        for page in self.session.pages():
            if not any(h in page.get("url", "") for h in (CLAUDE_HOST, "accounts.google.com")):
                continue
            state = self.session.evaluate(page, _LOGIN_STATE_JS)
            if state is None:
                continue
            step = _login_step(state)
            if _LOGIN_STEP_ORDER[step] > best_rank:
                best, best_rank = (page, state, step), _LOGIN_STEP_ORDER[step]
        return best

    def clear_tab_backlog(self, keep: int = 1) -> int:
        """Close the claude.ai tabs already open on this phone, returning how many."""
        self.session.require_open()
        closed = self.session.close_tabs_matching(CLAUDE_HOST, keep=keep)
        if closed:
            logger.info(f"[{self.serial}] closed {closed} leftover Claude tab(s)")
        return closed

    # -- internals --------------------------------------------------------------

    def _result(self, prompt: str, scraped_at: str, **kw) -> ChatResult:
        return ChatResult(provider="claude", prompt=prompt, scraped_at=scraped_at,
                          surface="phone_farm", serial=self.serial, **kw)

    def _snapshot(self, prompt: str, data: dict, gaps: list[str]) -> None:
        """Log an answer the extractor could not fully read, and keep its page."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = DEBUG_DIR / f"claude-{self.serial}-{stamp}.json"
        try:
            DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"serial": self.serial, "prompt": prompt,
                                        "gaps": gaps, **data}, ensure_ascii=False),
                            encoding="utf-8")
        except OSError as e:
            path = f"nowhere: {e}"
        logger.warning(f"[{self.serial}] claude: {'; '.join(gaps)} (page kept at {path})")

    def _close_last_tab(self) -> None:
        target, self._last_target_id = self._last_target_id, ""
        self.session.close_tab(target)

    def _await_answer(self, prompt: str) -> dict | None:
        """Poll until the turn has stopped streaming and its text stopped growing."""
        time.sleep(self.settle_ms / 1000.0)
        deadline = time.monotonic() + self.answer_timeout_s
        last_len, latest = -1, None
        while time.monotonic() < deadline:
            data = self._extract_once(prompt)
            if data:
                latest = data
                if not data.get("signedIn") and not data.get("nUser"):
                    return data
                if not data.get("nUser") and data.get("composerText"):
                    self._submit()
                body = data.get("bodyText", "")
                if not data.get("nAssistant") and (_LIMIT_RE.search(body)
                                                   or _PAUSED_RE.search(body)):
                    return data
                current = len(data.get("text", "") or "")
                if data.get("nAssistant") and current and not data.get("streaming") \
                        and current == last_len:
                    return data
                last_len = current
            time.sleep(1.5)
        if latest is None and self.debug:
            logger.warning(f"[{self.serial}] no Claude conversation for {prompt!r}")
        return latest

    def _extract_once(self, prompt: str) -> dict | None:
        page, data = _chat_page(self.session.pages(), prompt, self.session)
        if page is None:
            return None
        self._last_target_id = page.get("id") or ""
        self._last_page = page
        return data

    def _submit(self) -> None:
        """Press send on this ask's tab, foregrounding it first (background tabs freeze)."""
        if not self._last_page:
            return
        self.session.activate(self._last_page)
        if (self.session.evaluate(self._last_page, _SEND_JS) or {}).get("sent"):
            logger.debug(f"[{self.serial}] sent the prompt from the composer")


# ── page choice ──────────────────────────────────────────────────────────────

def _any_claude_page(pages: list[dict]) -> dict | None:
    return next((p for p in pages if CLAUDE_HOST in p.get("url", "")), None)


def _chat_page(pages: list[dict], prompt: str,
               session: PhoneChromeSession) -> tuple[dict | None, dict | None]:
    """The tab holding this ask, matched on its first user turn or its unsent composer.

    Returns the page with the read that matched it, so the poll pays one eval per tab.
    """
    wanted = _normalise(prompt)
    candidates = [p for p in pages if CLAUDE_HOST in p.get("url", "")]
    candidates.sort(key=lambda p: "/chat/" not in p.get("url", ""))
    for page in candidates:
        data = session.evaluate(page, _CHAT_EXTRACT_JS)
        if not data:
            continue
        if _normalise(data.get("prompt", "")) == wanted \
                or _normalise(data.get("composerText", "")) == wanted:
            return page, data
    return None, None


# ── parsing ──────────────────────────────────────────────────────────────────

def _login_step(state: dict) -> str:
    """Which step of the sign-in flow a page is at."""
    for step in ("role", "name", "training", "app", "plan", "birthday", "terms"):
        if state.get(step):
            return step
    if state.get("composer") and "/login" not in state.get("url", "") \
            and "/onboarding" not in state.get("url", ""):
        return "done"
    if state.get("accounts"):
        return "choose_account"
    if state.get("consent"):
        return "consent"
    if state.get("error"):
        return "error"
    if state.get("googleButton"):
        return "start"
    return "waiting"


def _strip_turn_prefix(text: str) -> str:
    return _TURN_PREFIX_RE.sub("", (text or "").strip(), count=1).strip()


def _parse_citations(citations: list[dict], anchors: list[dict]) -> list[Reference]:
    """Cited sources → `Reference`s, deduped by URL, in the order they were cited.

    `citations` come from the conversation JSON ({url, title}); `anchors` are the
    page's links, used only when the JSON could not be read.
    """
    by_url: dict[str, Reference] = {}
    order: list[str] = []

    def add(url: str, title: str) -> None:
        url = _real_url(url)
        dom = _domain(url)
        if not url.startswith("http") or not dom:
            return
        # Anthropic's own pages (help, policies) are product furniture, not sources.
        if dom in ("claude.ai", "anthropic.com") or dom.endswith(".anthropic.com"):
            return
        if url in by_url:
            if len(title) > len(by_url[url].title):
                by_url[url].title = title
            return
        by_url[url] = Reference(title=title or dom, url=url, domain=dom)
        order.append(url)

    for c in citations:
        add(c.get("url", ""), (c.get("title") or "").strip())
    for a in anchors:
        # The page appends "+N" to a chip that folds N more sources behind it.
        add(a.get("href", ""), re.sub(r"\s*\+\d+$", "", (a.get("text") or "").strip()))
    return [by_url[u] for u in order]


# ── CLI: python -m aiscrape.phone_claude ──────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Ask Claude on an Android phone over adb + CDP.")
    ap.add_argument("prompts", nargs="*", help="prompt(s) to ask")
    ap.add_argument("--serial", required=True, help="adb serial of the phone")
    ap.add_argument("--ssh-host", default=None,
                    help="SSH host the phone hangs off (env AISCRAPE_PHONE_SSH)")
    ap.add_argument("--adb", default=None, help="adb binary path on that host (env AISCRAPE_ADB)")
    ap.add_argument("--cdp-port", type=int, default=DEFAULT_CDP_PORT)
    ap.add_argument("--settle-ms", type=int, default=DEFAULT_SETTLE_MS)
    ap.add_argument("--sleep-on-exit", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--login", action="store_true",
                    help="sign claude.ai in with the phone's Google account, creating the "
                         "Claude account if needed (a no-op if already signed in)")
    ap.add_argument("--email", default=None,
                    help="which Google account to sign in with, if the phone has several")
    ap.add_argument("--name", default=None, help="name for a new account (default: Google's)")
    ap.add_argument("--birthday", default=DEFAULT_BIRTHDAY,
                    help=f"birthday for a new account (default {DEFAULT_BIRTHDAY})")
    ap.add_argument("--status", action="store_true",
                    help="report whether this phone is signed in, and exit")
    args = ap.parse_args()

    with PhoneClaudeScraper(args.serial, ssh_host=args.ssh_host, adb=args.adb,
                            cdp_port=args.cdp_port, settle_ms=args.settle_ms,
                            sleep_on_exit=args.sleep_on_exit, debug=args.debug) as s:
        if args.status:
            logged_in = s.is_logged_in()
            print(json.dumps({"serial": args.serial, "logged_in": logged_in,
                              "memory_enabled": s.memory_enabled if logged_in else None,
                              "google_accounts": s.session.google_accounts()}, indent=2))
            return
        if args.login:
            if s.is_logged_in():
                print(json.dumps({"serial": args.serial, "signed_in": True}, indent=2))
            else:
                print(json.dumps({"serial": args.serial,
                                  "signed_in_as": s.log_in(email=args.email, name=args.name,
                                                           birthday=args.birthday)}, indent=2))
        for p in args.prompts:
            print(json.dumps(s.ask(p).to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
