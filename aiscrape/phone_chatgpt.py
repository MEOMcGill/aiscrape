"""ChatGPT scraper that drives chatgpt.com in Chrome on a real Android phone.

The chatbot counterpart of `phone_farm.PhoneFarmAIOverviewScraper`, on the same
handsets and the same transport (`PhoneChromeSession`: adb + the Chrome DevTools
Protocol):

    launch chatgpt.com/?q=<prompt>  →  the mobile page auto-submits it  →  poll the
    transcript over CDP until the answer stops growing  →  return a `ChatResult`.

Why a phone rather than the desktop `chatbots.send_to_chatgpt` path: the same
reason as AI Mode. It is a real Chrome on a residential mobile IP, so it does not
have to look like a browser to a bot-detector -- it is one -- and the query-bank
runner can then ask Google and ChatGPT from one place instead of needing a
logged-in camoufox profile alongside the farm.

**Signed in or anonymous, and the difference matters.** A phone can ask either way,
and `log_in` signs it in with the Google account the handset already carries (give
each phone its own), so one handset gets one ChatGPT account and no password is
stored anywhere. Anonymous still works and needs nothing
set up. What changes:

  * **the wall.** OpenAI verifies an *anonymous* visitor in the background just after
    the page loads, and an ask that beats that check comes back "Chat verification
    could not be completed" with no answer at all. That check can also go permanently
    hostile on an individual handset, and once it has, re-warming, retrying and
    clearing chatgpt.com's site data over CDP do not lift it. Signing in does -- the
    check is for visitors without an account, and that is the reason `log_in` exists.
  * **the app.** Signed out serves a mobile-only shell; signed in serves the ordinary
    ChatGPT web app, the same markup `chatbots.py` drives on the desktop. Different
    selectors for everything, handled in `_CHAT_EXTRACT_JS` and nowhere else.
  * **sending.** Signed out, `?q=` fills the composer *and* submits, so an ask is one
    URL launch. Signed in it only fills the composer, so the ask has to press send
    (`_submit`) -- which is also why a tab is matched on its composer text as well as
    on its transcript.
  * **the model.** Signed in, each answered turn carries `data-message-model-slug`
    and `ChatResult.served_model` records it (e.g. "gpt-5-6"). Anonymous names no
    model anywhere in the DOM, so it stays None rather than a guess.
  * **state.** Anonymous has no memory and no custom instructions, so every ask
    starts blank. A signed-in account remembers, unless its memory is turned off:
    pass `memory=False` and each session sets it before asking (`apply_memory`),
    and every `ChatResult.memory_enabled` says which way it was.

A wall that survives all this is reported as `blocked`, exactly like Google's CAPTCHA,
so `phone_pool` rotates to the next handset and a run degrades to the phones that
answer instead of failing.

`prepare` opens chatgpt.com and waits before the first ask, which is what keeps an
anonymous ask from losing the verification race; it also clears the cookie banner and
the login modal, both of which otherwise sit over the transcript.

DOM notes, signed out (a mobile-only shell -- none of `chatbots.py`'s selectors exist):
  * turns are `li[data-message-role="user"|"assistant"]` inside
    `ol[data-conversation-transcript]`.
  * the answer body is `[data-assistant-markdown]` inside the assistant turn.
  * cited sources are not links. Each inline chip (and the "Sources" strip that
    closes an answer) is a button whose `data-assistant-sources-payload` holds its
    sources as JSON -- `{attribution, title, url, snippet}` each -- so the citations
    come out already structured, and an answer that searched the web has no `<a>` in
    its prose at all.
  * every visible class is a build-hashed `_wdUoQG_*`, so nothing here selects on
    class -- only on those data attributes, which have survived the hashing.
  * `innerText` of a turn is prefixed with a screen-reader line ("ChatGPT said:"),
    stripped in `_clean_answer`.

DOM notes, signed in: turns are `[data-message-author-role]`, and citations *are*
ordinary anchors, each stamped `utm_source=chatgpt.com` (stripped by
`_strip_chatgpt_utm`, so a cited URL compares equal to the same page ranked by
Google). No source-payload chips in this shell.

Tab choice cannot work the way AI Mode's does: on submit the URL becomes
`/c/<uuid>` and the prompt disappears from it, so `_chat_page` matches on the
*first user turn's text* instead (or, before sending, on the composer's). Same
principle -- never read a tab without confirming it belongs to this ask -- since one
Chrome serves every ask on a phone.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import date, datetime, timezone
from urllib.parse import parse_qsl, quote_plus, urlencode, urlsplit, urlunsplit

from .chatbots import RATE_LIMIT_TEXT_RE
from .google_aimode import Reference, _clean_title, _domain, _real_url
from .logger import logger
from .models import ChatResult, now_iso
from .phone_farm import DEFAULT_CDP_PORT, PhoneChromeSession

# Asking is a URL: the mobile page fills its composer from `q` and submits.
CHATGPT_ASK_URL = "https://chatgpt.com/?q={q}"
CHATGPT_HOME_URL = "https://chatgpt.com/"
# Every tab this scraper is allowed to look at. Also what gets tidied up.
CHATGPT_HOST = "chatgpt.com"

# How long to let the page load and start answering before the first read, and how
# long to keep polling after that. ChatGPT streams a multi-paragraph answer in much
# the same way AI Mode does, so these are in the same range as the AI Mode settle.
DEFAULT_SETTLE_MS = 6_000
DEFAULT_ANSWER_TIMEOUT_S = 90

# How long chatgpt.com must have been open before an ask is submitted. See
# `PhoneChatGPTScraper.prepare`: the site verifies a visitor in the background just
# after the page loads, and an ask that beats it gets no answer at all. Measured
# rather than guessed -- a 2s dwell walled three handsets in a row, 8s walled none.
DEFAULT_WARMUP_S = 8.0

# Where the Google profile name is read from when the account chooser was skipped.
GOOGLE_HOME_URL = "https://www.google.com/"

# The signed-in avatar names the account: "Google Account: Lab Phone 9
# (labphone009@gmail.com)".
_GOOGLE_NAME_JS = r"""
(() => {
  const a = document.querySelector('a[aria-label*="Google Account"]');
  const label = a ? (a.getAttribute('aria-label') || '') : '';
  const m = label.match(/Google Account:\s*([^\n(]+)/i);
  return JSON.stringify({name: m ? m[1].trim() : ''});
})()
"""


def _epoch(value) -> float:
    """A conversation timestamp as a unix float, from either shape the API uses.

    The list endpoint gives ISO ("2026-09-10T22:00:27.084180Z"), the conversation
    itself a float. Unparseable reads as 0 rather than raising -- an odd timestamp on
    one conversation must not stop a recovery.
    """
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _local_part(email: str) -> str:
    """"labphone009@gmail.com" -> "Labphone009" — the last-resort account name."""
    return (email.split("@")[0] or email).capitalize()


# The screen-reader prefixes the mobile transcript puts on each turn.
_TURN_PREFIX_RE = re.compile(r"^(ChatGPT said|You said)\s*:\s*", re.IGNORECASE)

# OpenAI's anti-bot wall. The ChatGPT equivalent of Google's "unusual traffic": the
# ask never reached a model, and the fix is a different handset, so it is reported
# as `blocked` and the pool rotates rather than storing an empty answer.
_VERIFICATION_TEXT = ("chat verification could not be completed",
                      "chat verification required",
                      "verify you are human")

# ChatGPT's own failed-generation turn, which arrives in place of an answer:
# "Something went wrong. If this issue persists please contact us through our help
# center at help.openai.com." Anchored at the start of the turn and required to name
# the help centre, because "something went wrong" is a phrase a real answer could
# open with, and throwing away a real answer is the worse error. Not a block: the
# phone and the account are fine, this ask simply produced nothing.
_TURN_ERROR_RE = re.compile(r"^\s*something went wrong\b.{0,200}help\.openai\.com",
                            re.IGNORECASE | re.DOTALL)

# The anonymous-usage cap. Distinct from verification: the phone is fine, it has
# just asked enough for now, so it is worth saying which of the two happened.
_ANON_LIMIT_TEXT = ("you've reached our limit of messages",
                    "log in to continue", "sign up to continue")

# One CDP round-trip: the whole state of one ask. Deliberately one call rather than
# several — every read crosses adb, ssh and a websocket, so the poll loop wants to
# pay that once per tick.
_CHAT_EXTRACT_JS = r"""
(() => {
  // Two app shells, and which one is on screen depends on whether the phone is
  // signed in: logged out gets the "wm-" mobile site (data-message-role), logged in
  // gets the ordinary ChatGPT web app (data-message-author-role, the same markup the
  // desktop scrapers in chatbots.py drive). Everything below returns the same shape
  // either way, so only this block knows the difference.
  const app = !!document.querySelector('[data-message-author-role], #prompt-textarea');
  const roleAttr = app ? 'data-message-author-role' : 'data-message-role';
  const turns = [...document.querySelectorAll('[' + roleAttr + ']')];
  const users = turns.filter(t => t.getAttribute(roleAttr) === 'user');
  const bots = turns.filter(t => t.getAttribute(roleAttr) === 'assistant');
  const last = bots.length ? bots[bots.length - 1] : null;
  // The markdown node is the answer without the turn's action row (Copy, Share).
  const body = last ? (last.querySelector('[data-assistant-markdown]') || last) : null;
  const composerEl = document.querySelector('#prompt-textarea, #mobile-composer-prompt');
  const composerText = composerEl
    ? (composerEl.value !== undefined && composerEl.value !== null
        ? composerEl.value : (composerEl.innerText || '')).trim()
    : '';
  const anchors = body ? [...body.querySelectorAll('a[href]')]
    .filter(a => /^https?:/.test(a.href))
    .map(a => ({
      href: a.href,
      text: (a.innerText || '').trim(),
      aria: (a.getAttribute('aria-label') || '').trim(),
    })) : [];
  // The sources ChatGPT cites are NOT links in the prose: each inline chip and the
  // trailing "Sources" strip is a button carrying its sources as a JSON payload
  // ({attribution, title, url, snippet}). Parsed here, in the page, so one round
  // trip returns citations already structured. Malformed payloads are skipped
  // rather than failing the read -- a citation is not worth losing an answer over.
  const citations = [];
  if (body) {
    for (const el of body.querySelectorAll('[data-assistant-sources-payload]')) {
      try {
        for (const s of JSON.parse(el.getAttribute('data-assistant-sources-payload'))) {
          if (s && s.url) citations.push({url: s.url, title: s.title || '',
                                          attribution: s.attribution || ''});
        }
      } catch (e) { /* not JSON this build; the anchors fallback still applies */ }
    }
  }
  // While a turn streams the composer's send control becomes a stop control. Its
  // presence is a positive "still generating", so a slow first token is not read
  // as a finished empty answer.
  const streaming = !!document.querySelector('button[data-testid="stop-button"]')
    || [...document.querySelectorAll('button')].some(b =>
         /^(stop|stop streaming|stop generating)$/i.test(
           (b.getAttribute('aria-label') || '').trim()));
  return JSON.stringify({
    found: true,
    url: location.href,
    app,
    prompt: users.length ? (users[0].innerText || '').trim() : '',
    // What is sitting unsent in the composer. The logged-in app fills this from
    // `?q=` but does NOT send it, so this is how an ask finds its own pending tab.
    composerText,
    // Which model served the last turn. Only the logged-in app says.
    servedModel: last ? (last.getAttribute('data-message-model-slug') || '') : '',
    nUser: users.length,
    nAssistant: bots.length,
    streaming,
    text: body ? (body.innerText || '').trim() : '',
    html: body ? (body.innerHTML || '').slice(0, 200000) : '',
    anchors,
    citations,
    // For the block checks: a verification wall replaces the answer rather than
    // rendering inside it, so it is only visible on the page as a whole.
    bodyText: (document.body.innerText || '').slice(0, 3000),
  });
})()
"""

# ── signing in with the phone's own Google account ───────────────────────────
#
# A handset carries a Google account, and chatgpt.com offers "Continue with Google",
# so a phone can sign itself in with the account it already has -- no password
# anywhere, and each handset gets its own ChatGPT account rather than the farm
# sharing one.
#
# The flow crosses three hosts and a variable number of steps (the signup page only
# appears the first time a given Google account meets ChatGPT), so `log_in` is a
# state machine over what is on screen rather than a fixed script. Two things were
# learned the hard way and are load-bearing:
#
#   * every page must be foregrounded before it is clicked. Android Chrome freezes
#     background tabs, and SSO opens a new one -- clicks into the frozen chooser did
#     nothing at all until `session.activate` was added.
#   * the signup form has to be *typed* into. Its name field rejected a value set
#     from JS ("Hmm, that doesn't look right"), and its birthday is a react-aria
#     field of contenteditable segments that only responds to keystrokes. Both are
#     filled with `session.type_text` (adb), which is real keyboard input.

# Total budget for the whole sign-in, not for one step. Generous because a first-time
# account walks five or six pages, each with its own load: the phone that timed out at
# 150s was progressing normally, just not finished.
LOGIN_TIMEOUT_S = 300

# Conversations per list request. The endpoint's own maximum.
HISTORY_PAGE = 100

# How long one batch of conversation fetches may take. Well past an ordinary CDP read,
# because the work is the phone's -- a batch is that many HTTP requests made and
# parsed on the handset. At the 25s default every batch came back "no response".
HISTORY_TIMEOUT_S = 180.0

# Between conversation fetches, inside the page. The account rate limits them, and a
# batch fired off at once is enough concurrency to trip it on its own.
HISTORY_FETCH_PAUSE_MS = 400

# How long to wait before asking again for conversations the account refused with a
# 429, and how many times. Lengthening, because the limit is a rate rather than a cap.
HISTORY_RETRY_WAITS_S = (20.0, 45.0, 90.0)

# How far along the flow each step is. Used to choose between open tabs: the flow
# leaves spent ones behind (the signup form stays on screen after the account it
# created has landed), so the tab to act on is the one furthest along, not the one
# whose host sorts first. "error" ranks below everything real -- a stale error page
# must not restart a flow that is progressing in another tab.
_LOGIN_STEP_ORDER = {
    "error": 1, "waiting": 2, "start": 3, "choose_provider": 4, "choose_account": 5,
    "consent": 6, "signup": 7, "welcome": 8, "done": 9,
}

# The birthday the signup form is given. Required, and 18+ or ChatGPT hands back a
# minor's account; deliberately a round institutional date rather than anything
# resembling a real person's, since these accounts belong to handsets.
DEFAULT_BIRTHDAY = "2000-01-01"

# OpenAI's signup rejects a name containing a digit, and a farm's Google profiles are
# usually numbered per handset ("Lab Phone 6"). Spelling the number keeps each account
# identifiable as its phone, where stripping the digit would collapse them all to one
# name.
_DIGIT_WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
                "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def _name_for_signup(name: str) -> str:
    """A display name OpenAI's signup will accept: digits spelled out."""
    out = []
    for ch in name:
        if ch.isdigit():
            # Space the word off from whatever it was jammed against ("Phone6" -> "Phone Six").
            if out and out[-1] not in " -":
                out.append(" ")
            out.append(_DIGIT_WORDS[ch])
        else:
            out.append(ch)
    return "".join(out).strip()


def _age_on(birthday: str, today: date | None = None) -> int:
    """Whole years from `birthday` to today — for the form variant that asks an age.

    Derived rather than configured so the two shapes of the signup cannot disagree
    about the same account.
    """
    born = datetime.strptime(birthday, "%Y-%m-%d").date()
    today = today or datetime.now(timezone.utc).date()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


# One probe answering "where in the login flow is this page, and what can I click".
# Every branch of `log_in` reads this, so a step that never fires shows up as a state
# the loop reports rather than as a click into nothing.
_LOGIN_STATE_JS = r"""
(() => {
  const txt = e => ((e.innerText || '') + ' ' + (e.getAttribute('aria-label') || '')).trim();
  const clickables = [...document.querySelectorAll(
    'button, a[href], [role="button"], [role="link"]')];
  const byText = re => clickables.find(e => re.test(txt(e)));
  const body = document.body.innerText || '';
  return JSON.stringify({
    url: location.href,
    // Logged in: the composer is there and the logged-out calls to action are not.
    composer: !!document.querySelector('#mobile-composer-prompt, #prompt-textarea'),
    // The signed-in app, by furniture only it has. Needed as well as the composer
    // because the app paints its shell first and its composer a moment later, and
    // in that gap a finished sign-in looked like a page still loading.
    appShell: !!document.querySelector(
      '[data-testid="model-switcher-dropdown-button"], [data-testid="composer-plus-btn"], '
      + '#prompt-textarea'),
    loginButton: !!byText(/^log in$/i),
    signupButton: !!byText(/^sign up for free$/i),
    googleButton: !!byText(/^continue with google$/i),
    // The Google account chooser marks each row with the address it signs in as.
    accounts: [...document.querySelectorAll('[data-identifier]')].map(e => ({
      email: e.getAttribute('data-identifier'),
      name: (e.innerText || '').trim().split('\n')[0].trim(),
    })),
    consent: /will allow OpenAI to access|to continue to OpenAI/i.test(body)
             && !!byText(/^continue$/i),
    // ChatGPT's own "You're all set" page, shown once after a new account is
    // created. Straight prose rather than a form, but the flow stops here until
    // its Continue is clicked. (The apostrophe is typographic on the live page.)
    welcome: /you.{0,3}re all set/i.test(body) && !!byText(/^continue$/i),
    signup: !!document.querySelector('input[name="birthday"], input[name="age"]'),
    signupName: (document.querySelector('input[name="name"]') || {}).value || '',
    authError: /we ran into an issue while signing you in/i.test(body),
    body: body.slice(0, 400),
  });
})()
"""

# Click one control by its visible text (or aria-label). The login flow's controls are
# framework-rendered divs as often as they are buttons, so this matches on role rather
# than on tag, and takes the innermost match -- the account row wraps both the name and
# the address, and only the row itself carries the handler.
_CLICK_BY_TEXT_JS = """
(() => {
  const want = %s.toLowerCase();
  const els = [...document.querySelectorAll(
    'button, a[href], input[type=submit], [role="button"], [role="link"]')];
  const hits = els.filter(e =>
    ((e.innerText || '') + ' ' + (e.getAttribute('aria-label') || ''))
      .trim().toLowerCase().includes(want));
  if (!hits.length) return JSON.stringify({clicked: false});
  const hit = hits[hits.length - 1];
  hit.click();
  return JSON.stringify({clicked: true, on: (hit.innerText || '').trim().slice(0, 60)});
})()
"""

_CLICK_SELECTOR_JS = """
(() => {
  const el = document.querySelector(%s);
  if (!el) return JSON.stringify({clicked: false});
  el.click();
  return JSON.stringify({clicked: true, on: (el.innerText || '').trim().slice(0, 60)});
})()
"""

# Empty the signup's name field and put the caret in it, so `type_text` fills it as a
# person would. Clearing through React's own setter (rather than assigning `.value`)
# is what makes the form's state agree with what is on screen.
_FOCUS_SIGNUP_NAME_JS = r"""
(() => {
  const el = document.querySelector('input[name="name"]');
  if (!el) return JSON.stringify({ok: false});
  const set = Object.getOwnPropertyDescriptor(
    window.HTMLInputElement.prototype, 'value').set;
  set.call(el, '');
  el.dispatchEvent(new Event('input', {bubbles: true}));
  el.focus();
  return JSON.stringify({ok: true});
})()
"""

# The other shape of the same form: a plain "Age" box instead of a date. Emptied
# through React's setter first, like the name, so the value that gets typed is the
# only one the form's state has ever seen.
_FOCUS_AGE_JS = r"""
(() => {
  const el = document.querySelector('input[name="age"]');
  if (!el) return JSON.stringify({ok: false});
  const set = Object.getOwnPropertyDescriptor(
    window.HTMLInputElement.prototype, 'value').set;
  set.call(el, '');
  el.dispatchEvent(new Event('input', {bubbles: true}));
  el.focus();
  return JSON.stringify({ok: true});
})()
"""

# The birthday's first segment. Typing eight digits into it fills year, month and day:
# react-aria advances to the next segment as each one completes.
_FOCUS_BIRTHDAY_JS = r"""
(() => {
  const y = document.querySelector('[role="spinbutton"][data-type="year"]');
  if (!y) return JSON.stringify({ok: false});
  y.focus();
  return JSON.stringify({ok: true});
})()
"""


# Press send. `send-button` is what the signed-in app calls it; the logged-out
# mobile site uses `composer-send-button` and sends on its own anyway, so this is
# only ever needed for the former -- but both are matched, cheaply.
_SEND_JS = r"""
(() => {
  const b = document.querySelector('button[data-testid="send-button"]')
        || document.querySelector('button[data-testid="composer-send-button"]');
  if (!b || b.disabled) return JSON.stringify({sent: false});
  b.click();
  return JSON.stringify({sent: true});
})()
"""


# ── reading an account's own history ─────────────────────────────────────────
#
# Every signed-in ask leaves a conversation behind, and the account keeps it. That
# makes the handset's history a second copy of everything it ever answered -- worth
# having when the first copy is lost, and the only route back to an ask that has
# already happened, since re-asking gets a fresh answer rather than the old one.
#
# Read through the same endpoints the web app uses, from a page on chatgpt.com so the
# session cookie is already there. `/api/auth/session` hands out the bearer token the
# app itself carries; nothing here logs in or stores a credential.
#
# Anonymous asks have no account and therefore no history. Neither do temporary chats
# (`is_temporary_chat`), which are skipped rather than returned empty.

# Is this tab far enough along to make the calls below? Distinguishes a page that is
# still loading from one whose account is signed out, which otherwise both look like
# an account with no history.
_HISTORY_READY_JS = """
(async () => {
  try {
    const r = await fetch('/api/auth/session', {credentials: 'include'});
    if (!r.ok) return JSON.stringify({ready: false});
    const s = await r.json();
    return JSON.stringify({ready: !!(s && s.accessToken), signedOut: !(s && s.accessToken)});
  } catch (e) { return JSON.stringify({ready: false}); }
})()
"""

# One page of the conversation list, newest first. `total` in the response is
# page-relative rather than the account's count, so paging stops on a short page.
_HISTORY_PAGE_JS = """
(async () => {
  const s = await fetch('/api/auth/session', {credentials: 'include'}).then(r => r.json());
  if (!s || !s.accessToken) return JSON.stringify({error: 'not signed in'});
  const r = await fetch('/backend-api/conversations?offset=%d&limit=%d&order=updated',
                        {headers: {Authorization: 'Bearer ' + s.accessToken}});
  if (!r.ok) return JSON.stringify({error: 'list ' + r.status});
  const j = await r.json();
  return JSON.stringify({
    account: (s.user && s.user.email) || '',
    items: (j.items || []).map(c => ({id: c.id, created: c.create_time})),
  });
})()
"""

# A batch of conversations, each reduced to the one ask it holds. Batched because
# every round trip crosses adb, ssh and a websocket, and a day of asks is hundreds of
# conversations.
_HISTORY_FETCH_JS = """
(async () => {
  const ids = %s;
  const s = await fetch('/api/auth/session', {credentials: 'include'}).then(r => r.json());
  if (!s || !s.accessToken) return JSON.stringify({error: 'not signed in'});
  const H = {Authorization: 'Bearer ' + s.accessToken};
  const one = async (id) => {
    try {
      const r = await fetch('/backend-api/conversation/' + id, {headers: H});
      // 429 is the account saying "slower", not "no". Marked so the caller can wait
      // and come back for it rather than dropping the conversation.
      if (r.status === 429) return {id, error: 'get 429', retry: true};
      if (!r.ok) return {id, error: 'get ' + r.status};
      const c = await r.json();
      const nodes = Object.values(c.mapping || {})
        .filter(n => n && n.message && n.message.author && n.message.content);
      const text = (n) => (n.message.content.parts || [])
        .filter(x => typeof x === 'string').join('\\n').trim();
      // The ask is the first thing the user said; the answer is the last prose the
      // assistant produced. Everything between is tool calls and reasoning, which
      // carry the same role but not the answer.
      const asks = nodes.filter(n => n.message.author.role === 'user' && text(n))
        .sort((a, b) => (a.message.create_time || 0) - (b.message.create_time || 0));
      const said = nodes.filter(n => n.message.author.role === 'assistant'
                                 && n.message.content.content_type === 'text' && text(n))
        .sort((a, b) => (a.message.create_time || 0) - (b.message.create_time || 0));
      const answer = said.length ? said[said.length - 1] : null;
      const refs = [];
      for (const ref of ((answer && answer.message.metadata) || {}).content_references || []) {
        for (const item of ref.items || []) {
          if (item && item.url) refs.push({url: item.url, title: item.title || '',
                                           attribution: item.attribution || ''});
        }
      }
      return {
        id,
        temporary: !!c.is_temporary_chat,
        title: c.title || '',
        // When the ask was made, not when the answer finished: it is the question's
        // timestamp that a series of repeats is ordered by.
        asked: asks.length ? asks[0].message.create_time : c.create_time,
        prompt: asks.length ? text(asks[0]) : '',
        answer: answer ? text(answer) : '',
        model: (answer && answer.message.metadata
                && answer.message.metadata.model_slug) || '',
        references: refs,
      };
    } catch (e) { return {id, error: String(e)}; }
  };
  // One at a time, with a beat between: fetched all at once, a batch is itself
  // enough concurrency to get the account rate limited.
  const out = [];
  for (const id of ids) {
    out.push(await one(id));
    await new Promise(r => setTimeout(r, %d));
  }
  return JSON.stringify({conversations: out});
})()
"""


# ── the account's memory ─────────────────────────────────────────────────────
#
# Settings → Personalization → "Enable memory" is the `m3m` flag, and flipping it
# takes "reference saved memories" (`sunshine`) and "reference chat history"
# (`moonshine`) with it. Those two can no longer be written on their own (the API
# answers IMMUTABLE_SETTING), so they are only written on an account that has no
# `m3m` yet. Turning memory off deletes nothing: history and saved memories stay.
_MEMORY_KEYS = ("m3m", "sunshine", "moonshine")

# Read the flags, write them if `want` (true/false, or null to only read) differs,
# and read them back. `want` is substituted in as JSON.
_MEMORY_JS = """
(async () => {
  const want = %s;
  const keys = %s;
  let s;
  try {
    const r = await fetch('/api/auth/session', {credentials: 'include'});
    if (!r.ok) return JSON.stringify({error: 'session ' + r.status});
    s = await r.json();
  } catch (e) { return JSON.stringify({error: 'session ' + e}); }
  if (!s || !s.accessToken) return JSON.stringify({signedOut: true});
  const H = {Authorization: 'Bearer ' + s.accessToken};
  const read = async () => {
    const r = await fetch('/backend-api/settings/user', {headers: H});
    if (!r.ok) throw new Error('settings ' + r.status);
    const all = (await r.json()).settings || {};
    const out = {};
    for (const k of keys) if (typeof all[k] === 'boolean') out[k] = all[k];
    return out;
  };
  try {
    const before = await read();
    const on = Object.values(before).some(v => v);
    const writes = [];
    if (want !== null && on !== want) {
      const targets = 'm3m' in before ? ['m3m'] : keys.filter(k => k in before);
      for (const k of targets) {
        const r = await fetch('/backend-api/settings/account_user_setting?feature='
                              + k + '&value=' + want, {method: 'PATCH', headers: H});
        writes.push({key: k, status: r.status,
                     body: r.ok ? '' : (await r.text()).slice(0, 200)});
      }
    }
    return JSON.stringify({before, writes, after: writes.length ? await read() : before});
  } catch (e) { return JSON.stringify({error: String(e)}); }
})()
"""


class ChatGPTLoginError(RuntimeError):
    """The Google sign-in flow could not be completed on this phone."""


class ChatGPTMemoryError(RuntimeError):
    """The account's memory setting could not be read or set as asked."""


# Clicked once per session. Both are "get the page out of the way": without the
# cookie click every ask fails verification, and the login modal (which appears on
# its own schedule) covers the transcript. Written to be safe to run at any time --
# it clicks nothing that is not there.
_PREPARE_JS = r"""
(() => {
  const done = [];
  const cookies = [...document.querySelectorAll('button')]
    .find(b => /^accept all$/i.test((b.innerText || '').trim()));
  if (cookies) { cookies.click(); done.push('cookies'); }
  const close = [...document.querySelectorAll('button')].find(b =>
    /^(close|dismiss)$/i.test((b.getAttribute('aria-label') || '').trim()));
  if (close) { close.click(); done.push('modal'); }
  return JSON.stringify({
    done,
    // Has the app itself painted? Reported separately from the clicks because a tab
    // that answers CDP is not yet a tab that has rendered its cookie banner, and
    // "no banner" means "already accepted" only once this is true.
    ready: !!document.querySelector('#mobile-composer-prompt, #prompt-textarea'),
    banner: !!cookies,
  });
})()
"""


class PhoneChatGPTScraper:
    """Ask ChatGPT on one Android phone, over adb + CDP.

    Args:
        serial:       the phone's adb serial (from `adb devices`). Optional when
                      `session` is given.
        session:      an open `PhoneChromeSession` to ask on, instead of opening one
                      for `serial` — how one handset serves ChatGPT and the Google
                      surfaces off a single tunnel. Whoever created it closes it.
        ssh_host/adb/cdp_port: as `PhoneFarmAIOverviewScraper`; ignored when a
                      session is passed, since it carries its own.
        settle_ms:    wait after launching the ask before the first read.
        answer_timeout_s: give up on an answer that never settles after this.
        warmup_s:     how long chatgpt.com must be open before the first ask, so the
                      site's anti-bot check has cleared (see `prepare`). Paid once a
                      session, not per ask.
        memory:       set the signed-in account's memory on (True) or off (False)
                      once a session, before the first ask; None leaves it as it is
                      and only reads it. See `apply_memory`.

    Use as a context manager; reuse one instance for many prompts on one phone::

        with PhoneChatGPTScraper("R58MEXAMPLE", ssh_host="phone-farm",
                                 adb=r"C:\\platform-tools\\adb.exe") as s:
            r = s.ask("what are the economic consequences of ...")
            print(r.response)
    """

    def __init__(self, serial: str | None = None, *,
                 session: PhoneChromeSession | None = None,
                 ssh_host: str | None = None, adb: str | None = None,
                 cdp_port: int = DEFAULT_CDP_PORT,
                 settle_ms: int = DEFAULT_SETTLE_MS,
                 answer_timeout_s: int = DEFAULT_ANSWER_TIMEOUT_S,
                 warmup_s: float = DEFAULT_WARMUP_S,
                 memory: bool | None = None,
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
        self.warmup_s = warmup_s
        self.memory = memory
        self.debug = debug
        self._prepared = False
        # The account's memory as last read this session; None until it has been.
        self.memory_enabled: bool | None = None
        self._memory_checked = False
        self._last_target_id = ""
        # The tab the current ask is reading. Kept whole (not just its id) because
        # pressing send needs to foreground and evaluate on it.
        self._last_page: dict | None = None

    @property
    def serial(self) -> str:
        return self.session.serial

    # -- lifecycle --------------------------------------------------------------

    def __enter__(self) -> "PhoneChatGPTScraper":
        if self._owns_session:
            self.session.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._owns_session:
            self.session.__exit__(*exc)

    # -- asking -----------------------------------------------------------------

    def ask(self, prompt: str) -> ChatResult:
        """Put one prompt to ChatGPT on the phone, in a fresh conversation.

        A verification wall is retried once on this handset, re-warming the site
        first, before it is reported as `blocked`. The wall is usually about *when*
        the ask was submitted rather than about the phone -- the verification simply
        having lapsed since the session warmed up -- and rotating the farm away from
        a handset for that would burn phones for a condition a second attempt clears.
        """
        self.session.require_open()
        result = self._ask_once(prompt)
        if result.blocked and result.note.startswith("chatgpt verification"):
            logger.info(f"[{self.serial}] chatgpt verification wall; re-warming and "
                        "asking once more")
            self.prepare(force=True)
            result = self._ask_once(prompt)
        return result

    def _ask_once(self, prompt: str) -> ChatResult:
        self.prepare()
        # Before every ask, not once per session: the display times out mid-run and
        # the keyguard comes back over Chrome. A tab behind it is frozen, so the ask
        # opens its conversation and then never paints the reply -- which arrives
        # here as an answer turn that rendered empty, on a handset that looks fine.
        self.session.ensure_visible()
        scraped_at = now_iso()
        self.session.launch(CHATGPT_ASK_URL.format(q=quote_plus(prompt)))

        self._last_target_id = ""
        self._last_page = None
        try:
            data = self._await_answer(prompt)
            if data is None:
                return self._result(prompt, scraped_at,
                                    note="no ChatGPT conversation found for this ask")

            page_text = data.get("bodyText", "") or ""
            wall = _blocked_reason(page_text)
            if wall:
                # Not an answer and not a refusal: the ask never reached a model. Same
                # treatment as a Google CAPTCHA, so the pool rotates handsets.
                return self._result(prompt, scraped_at, blocked=True, note=wall,
                                    rate_limited="limit" in wall)
            if RATE_LIMIT_TEXT_RE.search(page_text):
                return self._result(prompt, scraped_at, blocked=True, rate_limited=True,
                                    note="chatgpt rate limit banner")

            answer = _clean_answer(data.get("text", ""))
            if _TURN_ERROR_RE.search(answer):
                # ChatGPT failed to generate. Reported as an ask that produced
                # nothing rather than stored as an answer -- it is the product's
                # error message, not the model's output.
                return self._result(prompt, scraped_at,
                                    note="chatgpt failed to generate a response")
            return self._result(
                prompt, scraped_at, response=answer,
                # Only the signed-in app names its model; anonymous stays None.
                served_model=data.get("servedModel") or None,
                memory_enabled=self.memory_enabled if data.get("app") else False,
                references=_parse_citations(data.get("citations", []),
                                            data.get("anchors", [])),
                note="" if answer else "the answer turn rendered empty",
            )
        finally:
            # One tab per ask, closed on the way out however we leave -- Chrome keeps
            # tabs across sessions, so without this a phone accumulates one tab per ask
            # ever made, and a long tab list is exactly what makes the right tab slow to
            # find (the lesson `phone_farm._ai_mode_page` was written for).
            self._close_last_tab()

    def ask_many(self, prompts: list[str]) -> list[ChatResult]:
        return [self.ask(p) for p in prompts]

    def prepare(self, force: bool = False) -> None:
        """Open chatgpt.com, clear the banners, and let it clear its anti-bot check.

        Not optional housekeeping — it is what makes an anonymous ask work at all.
        chatgpt.com verifies a visitor in the background shortly after the page
        loads, and an ask submitted before that finishes comes back "Chat
        verification could not be completed" with no answer at all. `?q=` submits the
        instant the page loads, so an ask on a cold Chrome is *always* too early:
        the page has to have been open once, for a moment, first. Hence a plain
        chatgpt.com tab, a `warmup_s` dwell on it, and only then asking.

        The same visit accepts the cookie banner and closes the login modal, both of
        which otherwise sit over the transcript.

        Once per session, since the check holds for a good while afterwards. `force`
        re-does it, which is what an ask that hit the wall anyway tries next.
        """
        if self._prepared and not force:
            return
        # Chrome has to be the app on screen, not just running: Android freezes the
        # tabs of a backgrounded browser, so on a locked phone (or one left with the
        # notification shade pulled down) the ask's tab never renders its answer and
        # the send button can never be pressed. Once per session is enough, since
        # launching a URL raises Chrome by itself afterwards.
        self.session.unlock()
        self.session.foreground_chrome()
        self.session.launch(CHATGPT_HOME_URL)
        deadline = time.monotonic() + 45
        clicked, ready_since = False, 0.0
        while time.monotonic() < deadline:
            time.sleep(2.0)
            page = _any_chatgpt_page(self.session.pages())
            if page is None:
                continue
            state = self.session.evaluate(page, _PREPARE_JS)
            if state is None:
                continue
            if state.get("done"):
                logger.debug(f"[{self.serial}] chatgpt: dismissed {state['done']}")
            clicked = clicked or state.get("banner", False)
            if not state.get("ready"):
                continue
            # Painted -- now wait, and keep clicking whatever appears while we do.
            # Returning as soon as a tab answered is what made this fail: every ask
            # then came straight back "Chat verification could not be completed",
            # because the page had not been open long enough to be verified.
            ready_since = ready_since or time.monotonic()
            if time.monotonic() - ready_since >= self.warmup_s:
                self._prepared = True
                self._last_target_id = page.get("id") or ""
                try:
                    if not self._memory_checked:
                        self._apply_memory_on(page)
                finally:
                    self._close_last_tab()
                return
        logger.warning(f"[{self.serial}] chatgpt.com never finished loading to accept "
                       "cookies on; asking anyway (the ask may fail verification)")
        self._prepared = True

    # -- the account's memory ---------------------------------------------------

    def apply_memory(self) -> bool | None:
        """Set the account's memory to `self.memory` and return the state it ends in.

        Reads before writing, so an account already set as asked costs one read.
        With `memory=None` it only reads. A signed-out phone has no setting to read,
        so it counts as unknown.

        Raises ChatGPTMemoryError when the setting cannot be read, or does not end
        up as asked: an ask must not go out with memory in an unknown state.
        """
        self.session.require_open()
        self.session.ensure_visible()
        self.session.launch(CHATGPT_HOME_URL)
        deadline = time.monotonic() + 60
        page = None
        while time.monotonic() < deadline and page is None:
            time.sleep(2.0)
            page = _any_chatgpt_page(self.session.pages())
        if page is None:
            raise ChatGPTMemoryError(f"{self.serial}: chatgpt.com would not open "
                                     "to read the memory setting")
        self._last_target_id = page.get("id") or ""
        try:
            return self._apply_memory_on(page)
        finally:
            self._close_last_tab()

    def _apply_memory_on(self, page: dict) -> bool | None:
        """`apply_memory` on a chatgpt.com tab that is already open."""
        want = json.dumps(self.memory)
        data, state = None, None
        # Foregrounded every try: a background tab is frozen and never answers. A
        # tab that has only just painted can also drop the first evaluate.
        for _ in range(3):
            self.session.activate(page)
            data = self.session.evaluate(page, _MEMORY_JS % (want, json.dumps(_MEMORY_KEYS)),
                                         timeout=30)
            state = _memory_state(data)
            if state is not None or (data or {}).get("signedOut"):
                break
            time.sleep(3.0)
        if state is None or (self.memory is not None and state != self.memory):
            msg = (f"{self.serial}: chatgpt memory is {_memory_word(state)}, "
                   f"wanted {_memory_word(self.memory)} ({_memory_detail(data)})")
            if self.memory is not None:
                raise ChatGPTMemoryError(msg)
            if not (data or {}).get("signedOut"):
                logger.warning(f"[{self.serial}] could not read chatgpt memory setting "
                               f"({_memory_detail(data)})")
        elif data.get("writes"):
            logger.info(f"[{self.serial}] chatgpt memory turned {_memory_word(state)} "
                        f"(was {data.get('before')})")
        else:
            logger.debug(f"[{self.serial}] chatgpt memory is {_memory_word(state)}")
        self.memory_enabled = state
        self._memory_checked = True
        return state

    # -- signing in -------------------------------------------------------------

    def is_logged_in(self) -> bool:
        """Is chatgpt.com signed in on this phone?

        Read off the logged-out calls to action rather than off an account widget:
        "Log in" and "Sign up for free" are gone the moment a session exists, and
        they are the same two controls the login flow starts from.
        """
        self.session.launch(CHATGPT_HOME_URL)
        time.sleep(6)
        return self._furthest_flow_page()[2] == "done"

    def ensure_logged_in(self, **kw) -> bool:
        """Sign in if this phone is not signed in already. True if it ends up signed in.

        The form a caller wants before a run: one page read when the session is
        already there (the usual case, since Chrome keeps it), the full flow when it
        is not.
        """
        if self.is_logged_in():
            return True
        logger.info(f"[{self.serial}] chatgpt is signed out; signing in with the "
                    "phone's Google account")
        self.log_in(**kw)
        return True

    def log_in(self, *, email: str | None = None, name: str | None = None,
               birthday: str = DEFAULT_BIRTHDAY,
               timeout_s: int = LOGIN_TIMEOUT_S) -> str:
        """Sign chatgpt.com in with the Google account already on this phone.

        Returns the address signed in as. A no-op when the phone is signed in
        already, so it is safe to call before a run.

        Args:
            email:    which Google account, when the phone carries more than one.
                      Defaults to its only one.
            name:     the name ChatGPT's signup gets, if this Google account has no
                      ChatGPT account yet. Defaults to the Google profile's own
                      display name (read off the account chooser), with any digits
                      spelled out because the signup rejects them.
            birthday: YYYY-MM-DD for that same form. See DEFAULT_BIRTHDAY.

        Raises ChatGPTLoginError if the flow stalls or ends somewhere unexpected;
        the phone is left wherever it stopped, which is what to look at.
        """
        self.session.require_open()
        if email is None:
            accounts = self.session.google_accounts()
            if len(accounts) != 1:
                raise ChatGPTLoginError(
                    f"{self.serial} has {len(accounts)} Google accounts ({accounts}); "
                    "pass email= to say which one to sign in with")
            email = accounts[0]

        # Unlike an ask, which only ever *reads* pages, signing in clicks its way
        # through them — so Chrome has to be the app on screen, not merely running.
        self.session.unlock()
        self.session.foreground_chrome()
        self.prepare()
        deadline = time.monotonic() + timeout_s
        # The display name is only legible on the account chooser, which is also the
        # only step that knows it -- caught there, used later if signup asks.
        google_name, last, stalled = "", "", 0
        looked_up_name = False
        while time.monotonic() < deadline:
            page, state, step = self._furthest_flow_page()
            if page is None:
                self.session.launch(CHATGPT_HOME_URL)
                time.sleep(3)
                continue
            # Foreground it before touching it: a frozen tab ignores clicks. Both
            # halves matter -- the right tab within Chrome, and Chrome itself out
            # from under a lock screen that has come back.
            self.session.ensure_visible()
            self.session.activate(page)
            if step == "done":
                # "You're all set" is the last step of *creating* an account, and it
                # appears while another tab already shows the signed-in app -- which
                # outranks it, so the flow reaches `done` with the gate still open
                # behind it. Whatever is left open is what the next run's
                # `is_logged_in` reads, and a stranded gate makes a signed-in phone
                # report itself signed out on every handset pick from then on.
                self._dismiss_welcome()
                logger.info(f"[{self.serial}] chatgpt signed in as {email}")
                # Spent sign-in tabs, closed so the next run does not have to rank
                # its way past them.
                for host in ("auth.openai.com", "accounts.google.com"):
                    self.session.close_tabs_matching(host, keep=0)
                self._prepared = False   # the session is a different one now
                self._memory_checked = False
                return email
            # A flow that stops moving is a flow that needs a person to look at it,
            # so the same state twice running is not treated as progress.
            stalled = stalled + 1 if step == last else 0
            last = step
            if stalled == 2:
                # A step that has not moved twice is usually a step whose page is not
                # actually on screen -- something has come up over Chrome and frozen
                # it. Cheaper to raise Chrome again than to fail the login over it.
                logger.debug(f"[{self.serial}] '{step}' is not moving; raising Chrome")
                self.session.unlock()
                self.session.foreground_chrome()
            if stalled >= 6:
                raise ChatGPTLoginError(
                    f"{self.serial}: chatgpt login stuck at '{step}' "
                    f"({state.get('url', '')[:80]}): {state.get('body', '')[:200]}")
            logger.debug(f"[{self.serial}] chatgpt login: {step}")

            if step == "error":
                # The OAuth round trip expired or was refused. Start it over rather
                # than clicking on inside a dead flow -- the state and nonce in that
                # URL are spent.
                logger.warning(f"[{self.serial}] chatgpt sign-in errored; restarting the flow")
                self.session.launch(CHATGPT_HOME_URL)
                time.sleep(4)
            elif step == "start":
                self.session.evaluate(page, _CLICK_BY_TEXT_JS % json.dumps("log in"))
            elif step == "choose_provider":
                self.session.evaluate(page,
                                      _CLICK_BY_TEXT_JS % json.dumps("continue with google"))
            elif step == "choose_account":
                row = next((a for a in state["accounts"] if a.get("email") == email), None)
                if row is None:
                    raise ChatGPTLoginError(
                        f"{self.serial}: {email} is not offered by the Google account "
                        f"chooser (it lists {[a.get('email') for a in state['accounts']]})")
                google_name = google_name or row.get("name", "")
                self.session.evaluate(
                    page, _CLICK_SELECTOR_JS % json.dumps(f'[data-identifier="{email}"]'))
            elif step in ("consent", "welcome"):
                self.session.evaluate(page, _CLICK_BY_TEXT_JS % json.dumps("continue"))
            elif step == "signup":
                if not (name or google_name) and not looked_up_name:
                    # The form, without having passed the account chooser: a sign-in
                    # Google let straight through because it remembered the choice.
                    # Ask google.com for the profile name rather than naming the new
                    # account after an email address.
                    looked_up_name = True
                    google_name = self.google_display_name(email)
                    self.session.launch(CHATGPT_HOME_URL)
                    time.sleep(4)
                    continue
                self._fill_signup(page, name or _name_for_signup(google_name or _local_part(email)),
                                  birthday)
            time.sleep(3)

        raise ChatGPTLoginError(
            f"{self.serial}: chatgpt login did not finish within {timeout_s}s "
            f"(last step '{last}')")

    def google_display_name(self, email: str) -> str:
        """The Google profile's name for `email`, read off a signed-in google.com.

        The account chooser shows it, but the chooser is skipped once Google has
        remembered the choice for this OAuth client -- so a sign-in resumed halfway
        never sees it. This is the second way to the same string, and it keeps a new
        ChatGPT account named after its handset's profile ("Lab Phone 9")
        rather than after its email address.

        Best-effort: returns "" if google.com is walled or the widget has moved.
        """
        self.session.launch(GOOGLE_HOME_URL)
        time.sleep(5)
        page = next((p for p in self.session.pages()
                     if "google.com" in p.get("url", "") and "/sorry/" not in p.get("url", "")),
                    None)
        data = self.session.evaluate(page, _GOOGLE_NAME_JS) if page else None
        if page:
            # Closed here rather than left for a backlog sweep: the Google one only
            # matches search tabs, so these accumulated one per login attempt.
            self.session.close_tab(page.get("id") or "")
        name = (data or {}).get("name", "")
        if name and email.split("@")[0] not in name:
            return name
        return ""

    def _dismiss_welcome(self) -> None:
        """Click through any "You're all set" page still open on this phone.

        Every tab, not just the best-ranked one: the gate is reached once per
        account and is easy to leave behind, and clicking a Continue that is not
        there costs nothing.
        """
        for page in self.session.pages():
            if CHATGPT_HOST not in page.get("url", ""):
                continue
            state = self.session.evaluate(page, _LOGIN_STATE_JS)
            if not state or self._login_step(state) != "welcome":
                continue
            logger.debug(f"[{self.serial}] closing the 'you're all set' gate")
            self.session.activate(page)
            self.session.evaluate(page, _CLICK_BY_TEXT_JS % json.dumps("continue"))

    def _login_step(self, state: dict) -> str:
        """Which step of the sign-in flow the page on screen is at."""
        if state.get("authError"):
            return "error"
        if state.get("signup"):
            return "signup"
        if state.get("accounts"):
            return "choose_account"
        if state.get("consent"):
            return "consent"
        if state.get("welcome"):
            return "welcome"
        if state.get("googleButton"):
            return "choose_provider"
        if (state.get("composer") or state.get("appShell")) \
           and not state.get("loginButton") and not state.get("signupButton"):
            return "done"
        if state.get("loginButton"):
            return "start"
        return "waiting"

    def _fill_signup(self, page: dict, name: str, birthday: str) -> None:
        """Fill ChatGPT's "confirm your age" form and submit it.

        Everything is typed rather than assigned: see the notes above the login
        constants for what each field rejects otherwise. Typed *through CDP*, which
        is what makes it independent of the lock screen -- `adb input text` goes to
        whatever is on the display, so on a locked handset it types onto the lock
        screen and leaves the form empty.

        The form comes in two shapes, and which one a phone gets is OpenAI's choice:
        a birthday (three react-aria segments) or a plain "Age" number. Both are
        filled from the same `birthday`, so the accounts agree however they were
        asked.
        """
        logger.info(f"[{self.serial}] creating a ChatGPT account for this phone "
                    f"(name={name!r}, birthday={birthday})")
        if (self.session.evaluate(page, _FOCUS_SIGNUP_NAME_JS) or {}).get("ok"):
            self.session.insert_text(page, name)
        if (self.session.evaluate(page, _FOCUS_AGE_JS) or {}).get("ok"):
            self.session.insert_text(page, str(_age_on(birthday)))
        elif (self.session.evaluate(page, _FOCUS_BIRTHDAY_JS) or {}).get("ok"):
            # Keystrokes, not an inserted string: react-aria's segments read keys.
            # YYYYMMDD in one go, since each segment hands focus to the next as it
            # fills.
            self.session.type_keys(page, birthday.replace("-", ""))
        time.sleep(1)
        if not self._submit_signup(page):
            logger.warning(f"[{self.serial}] the signup form is filled in but nothing "
                           "on it could be clicked to submit; the flow will stall here")

    def _submit_signup(self, page: dict) -> bool:
        """Press the signup form's submit control. True if something was clicked.

        By role before by words, because OpenAI relabels the button (it reads
        "Continue" now, and read "Finish creating account" before). Matching only
        the words turned a form that was filled in correctly into a flow that
        stalled with the finished form on screen -- the click helper reports that
        it found nothing, so the wording is worth failing loudly about.
        """
        for js in (_CLICK_SELECTOR_JS % json.dumps('button[type="submit"], '
                                                   'input[type="submit"]'),
                   _CLICK_BY_TEXT_JS % json.dumps("finish creating account"),
                   _CLICK_BY_TEXT_JS % json.dumps("continue")):
            if (self.session.evaluate(page, js) or {}).get("clicked"):
                return True
        return False

    def _furthest_flow_page(self) -> tuple[dict | None, dict, str]:
        """The open tab that is furthest along the sign-in flow, and its state.

        Every tab is read and then *ranked*, rather than one being picked by URL,
        because the flow leaves working tabs behind it: SSO opens its own tab over
        the chatgpt.com one it started from, and a signup page stays open on screen
        after the account it created has already landed. Picking by URL kept
        choosing that spent signup form while the finished session sat in the tab
        beside it, and the flow read as stuck at a step it had in fact passed.
        """
        best: tuple[dict | None, dict, str] = (None, {}, "waiting")
        best_rank = -1
        for page in self.session.pages():
            if not any(h in page.get("url", "")
                       for h in ("accounts.google.com", "auth.openai.com", CHATGPT_HOST)):
                continue
            state = self.session.evaluate(page, _LOGIN_STATE_JS)
            if state is None:
                continue
            step = self._login_step(state)
            rank = _LOGIN_STEP_ORDER.get(step, 0)
            if rank > best_rank:
                best, best_rank = (page, state, step), rank
        return best

    # -- reading the account's own history --------------------------------------

    def history(self, *, since: datetime | None = None, limit: int | None = None,
                batch: int = 4) -> list[ChatResult]:
        """Every ask still in this handset's ChatGPT account, newest first.

        A second copy of what the phone has answered, read back out of the account
        rather than re-asked -- which is the point: re-asking a query produces a new
        answer, where this returns the one that was actually given, with its own
        timestamp, model and citations.

        Returns the same `ChatResult` an ask returns, so a caller can put these
        through whatever it puts live results through.

        Args:
            since: stop once conversations are older than this (UTC). The list comes
                   back newest first, so this bounds the work rather than filtering
                   after the fact.
            limit: stop after this many conversations.
            batch: conversations per round trip. Every one is an HTTP call inside the
                   page, so this trades a longer evaluate against fewer of them.

        Only what the account kept: an anonymous ask was never in an account, and a
        temporary chat is not stored, so neither can be recovered.
        """
        self.session.require_open()
        self.session.ensure_visible()
        page = self._history_page()
        cutoff = since.timestamp() if since else None
        wanted: list[str] = []
        offset, account = 0, ""
        while limit is None or len(wanted) < limit:
            data = self.session.evaluate(page, _HISTORY_PAGE_JS % (offset, HISTORY_PAGE))
            if not data or data.get("error"):
                logger.warning(f"[{self.serial}] could not list chatgpt history "
                               f"({(data or {}).get('error', 'no response')})")
                break
            account = account or data.get("account", "")
            items = data.get("items") or []
            if not items:
                break
            for item in items:
                if cutoff is not None and _epoch(item.get("created")) < cutoff:
                    items = []          # newest first, so everything after is older too
                    break
                wanted.append(item["id"])
                if limit is not None and len(wanted) >= limit:
                    break
            if not items or len(items) < HISTORY_PAGE:
                break
            offset += HISTORY_PAGE

        logger.info(f"[{self.serial}] {len(wanted)} conversation(s) in {account or 'this account'}"
                    + (f" since {since.isoformat()}" if since else ""))
        results: list[ChatResult] = []
        retryable: list[str] = []
        unreadable: list[str] = []
        for start in range(0, len(wanted), batch):
            chunk = wanted[start:start + batch]
            results.extend(self._fetch_batch(page, chunk, retry_into=retryable,
                                             lost_into=unreadable))

        # The account rate limits these, and a conversation refused with a 429 is one
        # that exists and was simply asked for too fast. Dropping it would under-recover
        # in silence -- the failure mode this whole module exists to undo -- so it is
        # waited out and asked for again.
        for delay in HISTORY_RETRY_WAITS_S:
            if not retryable:
                break
            logger.info(f"[{self.serial}] {len(retryable)} conversation(s) were rate "
                        f"limited; waiting {delay:.0f}s and asking again")
            time.sleep(delay)
            again, retryable = retryable, []
            for start in range(0, len(again), batch):
                results.extend(self._fetch_batch(page, again[start:start + batch],
                                                 retry_into=retryable,
                                                 lost_into=unreadable))
        if retryable or unreadable:
            logger.warning(
                f"[{self.serial}] {len(retryable) + len(unreadable)} of {len(wanted)} "
                f"conversation(s) could not be read, so they are NOT in what this "
                f"returned ({len(retryable)} still rate limited)")
        return results

    def _fetch_batch(self, page: dict, ids: list[str], *, retry_into: list[str],
                     lost_into: list[str]) -> list[ChatResult]:
        """One round trip's worth of conversations.

        A conversation that was refused goes into `retry_into` if asking again could
        help and `lost_into` if it could not, so that neither is quietly a zero.
        """
        data = self.session.evaluate(
            page, _HISTORY_FETCH_JS % (json.dumps(ids), HISTORY_FETCH_PAUSE_MS),
            timeout=HISTORY_TIMEOUT_S)
        if not data or data.get("error"):
            # The whole batch, not one conversation: worth retrying as a batch.
            logger.debug(f"[{self.serial}] batch of {len(ids)} did not come back "
                         f"({(data or {}).get('error', 'no response')})")
            retry_into.extend(ids)
            return []
        out = []
        for conv in data.get("conversations") or []:
            if conv.get("retry"):
                retry_into.append(conv["id"])
                continue
            if conv.get("error"):
                lost_into.append(conv["id"])
                continue
            result = self._history_result(conv)
            if result is not None:
                out.append(result)
        return out

    def _history_page(self) -> dict:
        """A chatgpt.com tab that is actually answering, to run the history calls from.

        Opened fresh and then proved, rather than taken from whatever is already on the
        phone: the calls below run *in* the page for its session cookie, so a tab that
        is merely present -- still loading, or left over and frozen -- fails them as a
        transport error that reads like an empty account.
        """
        self.session.launch(CHATGPT_HOME_URL)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            time.sleep(2.0)
            page = _any_chatgpt_page(self.session.pages())
            if page is None:
                continue
            ready = self.session.evaluate(page, _HISTORY_READY_JS)
            if ready and ready.get("ready"):
                return page
            if ready and ready.get("signedOut"):
                raise ChatGPTLoginError(
                    f"{self.serial}: chatgpt is signed out, so it has no history to read")
        raise ChatGPTLoginError(f"{self.serial}: chatgpt.com would not open to read history")

    def _history_result(self, conv: dict) -> ChatResult | None:
        """One recovered conversation as a `ChatResult`, or None if it holds no ask."""
        # A temporary chat was never stored and an empty one holds no ask; neither is
        # a conversation that failed to read, which `_fetch_batch` has already counted.
        if conv.get("temporary") or not conv.get("prompt"):
            return None
        asked = datetime.fromtimestamp(_epoch(conv.get("asked")), tz=timezone.utc)
        return ChatResult(
            provider="chatgpt", prompt=conv["prompt"], response=conv.get("answer", ""),
            served_model=conv.get("model") or None,
            references=_parse_citations(conv.get("references") or [], []),
            scraped_at=asked.isoformat(), surface="phone_farm", serial=self.serial,
            # Said plainly, because a recovered row is not evidence the phone answered
            # today -- it is evidence the account holds an answer from whenever this is.
            note="recovered from chatgpt history",
        )

    def clear_tab_backlog(self, keep: int = 1) -> int:
        """Close the ChatGPT tabs already open on this phone, returning how many."""
        self.session.require_open()
        closed = self.session.close_tabs_matching(CHATGPT_HOST, keep=keep)
        if closed:
            logger.info(f"[{self.serial}] closed {closed} leftover ChatGPT tab(s)")
        return closed

    # -- internals --------------------------------------------------------------

    def _result(self, prompt: str, scraped_at: str, **kw) -> ChatResult:
        return ChatResult(provider="chatgpt", prompt=prompt, scraped_at=scraped_at,
                          surface="phone_farm", serial=self.serial, **kw)

    def _close_last_tab(self) -> None:
        target, self._last_target_id = self._last_target_id, ""
        self.session.close_tab(target)

    def _await_answer(self, prompt: str) -> dict | None:
        """Poll until the answer stops growing (or timeout).

        Two things have to be true to stop: nothing is streaming, and the answer's
        length is the same as it was a tick ago. The streaming flag alone would stop
        on the gap before the first token; the length alone would stop on a pause
        mid-answer.
        """
        time.sleep(self.settle_ms / 1000.0)
        deadline = time.monotonic() + self.answer_timeout_s
        last_len, latest = -1, None
        while time.monotonic() < deadline:
            data = self._extract_once(prompt)
            if data:
                latest = data
                # A wall is a final state, not something to keep waiting on.
                if _blocked_reason(data.get("bodyText", "") or ""):
                    return data
                # Signed in, `?q=` only fills the composer — somebody has to press
                # send. Done here rather than before the poll because it is exactly
                # the "no turns yet" case the loop is already waiting through, and
                # a send that does not take is simply retried on the next tick.
                if not data.get("nUser") and data.get("composerText"):
                    self._submit()
                current = len(data.get("text", "") or "")
                if current and not data.get("streaming") and current == last_len:
                    return data
                last_len = current
            time.sleep(1.5)
        if latest is None and self.debug:
            logger.warning(f"[{self.serial}] no ChatGPT conversation for {prompt!r}")
        return latest

    def _extract_once(self, prompt: str) -> dict | None:
        page = _chat_page(self.session.pages(), prompt, self.session)
        if page is None:
            return None
        self._last_target_id = page.get("id") or ""
        self._last_page = page
        return self.session.evaluate(page, _CHAT_EXTRACT_JS)

    def _submit(self) -> None:
        """Press send on the tab holding this ask, for the signed-in app.

        The tab has to be foregrounded first: Android Chrome freezes background
        tabs, and a click into a frozen one is simply lost.
        """
        if not self._last_page:
            return
        self.session.activate(self._last_page)
        done = self.session.evaluate(self._last_page, _SEND_JS) or {}
        if done.get("sent"):
            logger.debug(f"[{self.serial}] sent the prompt from the composer")


# ── page choice ──────────────────────────────────────────────────────────────

def _any_chatgpt_page(pages: list[dict]) -> dict | None:
    for page in pages:
        if CHATGPT_HOST in page.get("url", ""):
            return page
    return None


def _chat_page(pages: list[dict], prompt: str, session: PhoneChromeSession) -> dict | None:
    """The open tab holding *this* ask's conversation, or None.

    AI Mode can pick its tab from the URL (the query is in it); here it cannot,
    because submitting replaces `?q=...` with `/c/<uuid>`. So each ChatGPT tab is
    asked what its first user turn says and the one echoing this prompt wins -- the
    same guarantee by a different route, and it matters for the same reason: one
    Chrome serves every ask on a phone, so reading "the ChatGPT tab" would hand back
    a previous ask's answer for this query.

    Conversation tabs (`/c/<uuid>`) are checked before any other chatgpt.com tab,
    because a match is not enough on its own: the mobile site is one SPA, so a
    chatgpt.com tab left open on the home screen keeps the previous conversation
    mounted and answers with *this* prompt's echo while rendering none of its
    answer. Reading that tab would report an empty answer for a question that was
    answered fine one tab over.

    Costs one CDP eval per open ChatGPT tab per poll, which is why asks close their
    tab behind them and a session clears the backlog on open.
    """
    wanted = _normalise(prompt)
    candidates = [p for p in pages if CHATGPT_HOST in p.get("url", "")]
    candidates.sort(key=lambda p: "/c/" not in p.get("url", ""))
    for page in candidates:
        data = session.evaluate(page, _CHAT_EXTRACT_JS)
        if not data:
            continue
        if _normalise(_strip_turn_prefix(data.get("prompt", ""))) == wanted:
            return page
        # Or the tab where this ask is still sitting in the composer, unsent: the
        # logged-in app takes `?q=` as a prefill rather than as a submission, so its
        # tab has no turns at all until `_submit` clicks send on it.
        if _normalise(data.get("composerText", "")) == wanted:
            return page
    return None


# ── parsing ──────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Whitespace-insensitive form, for comparing a prompt to the page's echo of it.

    The transcript re-wraps what it was sent, so an echo can differ from the prompt
    in newlines and runs of spaces without being a different prompt.
    """
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _strip_turn_prefix(text: str) -> str:
    """Drop the transcript's screen-reader lead-in ("ChatGPT said:", "You said:")."""
    return _TURN_PREFIX_RE.sub("", (text or "").strip(), count=1).strip()


def _clean_answer(text: str) -> str:
    """The answer prose: the turn text without its screen-reader prefix.

    Much less work than the AI Mode cleaner needs, because `[data-assistant-markdown]`
    is already just the answer -- the turn's action row (Copy, Share) lives outside
    it. Guarded anyway: when the markdown node is missing the extractor falls back to
    the whole turn, and then the prefix is there.
    """
    return _strip_turn_prefix(text)


def _blocked_reason(page_text: str) -> str:
    """Why this ask never reached a model, or "" if it did.

    Read off the page rather than the answer turn: both walls *replace* the answer,
    so there is nothing in the turn to match on.
    """
    low = (page_text or "").lower().replace("’", "'")
    if any(m in low for m in _VERIFICATION_TEXT):
        return "chatgpt verification wall (anti-bot)"
    if any(m in low for m in _ANON_LIMIT_TEXT):
        return "chatgpt anonymous usage limit"
    return ""


def _memory_state(data: dict | None) -> bool | None:
    """Whether `_MEMORY_JS` found memory on: any of its flags set. None if unread."""
    if not data or data.get("error") or data.get("signedOut"):
        return None
    after = data.get("after") or {}
    if not after:
        return None
    return any(after.values())


def _memory_word(state: bool | None) -> str:
    return {True: "on", False: "off", None: "unknown"}[state]


def _memory_detail(data: dict | None) -> str:
    if not data:
        return "no response"
    if data.get("error"):
        return data["error"]
    if data.get("signedOut"):
        return "signed out"
    return f"read {data.get('after')}, writes {data.get('writes')}"


def _strip_chatgpt_utm(url: str) -> str:
    """Drop the `utm_source=chatgpt.com` ChatGPT stamps on every cited URL.

    It is ChatGPT's attribution tag rather than part of the source's identity, and
    leaving it on would mean the same page cited by ChatGPT and ranked by Google
    never compares equal -- which is the one comparison these rows exist for. Only
    that parameter goes; anything else in the query string is the URL as published.
    """
    parsed = urlsplit(url)
    if not parsed.query:
        return url
    kept = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if not (k == "utm_source" and v == "chatgpt.com")]
    return urlunsplit(parsed._replace(query=urlencode(kept)))


def _parse_citations(citations: list[dict], anchors: list[dict]) -> list[Reference]:
    """The answer's sources → `Reference`s, deduped, in the order it cited them.

    Two inputs because ChatGPT has two ways of pointing at a page: the source chips
    it renders after searching the web (`citations`, carrying a title and a URL
    apiece) and the occasional ordinary markdown link in the prose (`anchors`).
    Chips first, since they are the citations proper; a link that repeats a chip's
    URL folds into it.

    An empty list is the ordinary case, not a parsing failure: ChatGPT cites when it
    has searched and stays silent when it answered from the model. Deliberately the
    same `Reference` shape the Google surfaces produce, so a caller writes one row
    schema and "what ChatGPT cites" against "what Google ranks" is a query over one
    index rather than a join.
    """
    by_url: dict[str, Reference] = {}
    order: list[str] = []

    def add(url: str, title: str) -> None:
        url = _strip_chatgpt_utm(_real_url(url))
        dom = _domain(url)
        if not url.startswith("http") or not dom:
            return
        # openai.com links are the product's own furniture (policy pages, the "learn
        # more" in a refusal), not a source it went and read.
        if dom == "openai.com" or dom.endswith(".openai.com") or dom == CHATGPT_HOST:
            return
        if url in by_url:
            if len(title) > len(by_url[url].title):
                by_url[url].title = title
            return
        by_url[url] = Reference(title=title or dom, url=url, domain=dom)
        order.append(url)

    for c in citations:
        add(c.get("url", ""), (c.get("title") or c.get("attribution") or "").strip())
    for a in anchors:
        url = a.get("href", "")
        add(url, _clean_title(a.get("text", ""), a.get("aria", ""),
                              _domain(_real_url(url))))
    return [by_url[u] for u in order]


# ── CLI: python -m aiscrape.phone_chatgpt ─────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ask ChatGPT on an Android phone over adb + CDP.")
    ap.add_argument("prompts", nargs="*", help="prompt(s) to ask")
    ap.add_argument("--serial", required=True, help="adb serial of the phone")
    ap.add_argument("--ssh-host", default=None,
                    help="SSH host the phone hangs off (env AISCRAPE_PHONE_SSH)")
    ap.add_argument("--adb", default=None,
                    help="adb binary path on that host (env AISCRAPE_ADB)")
    ap.add_argument("--cdp-port", type=int, default=DEFAULT_CDP_PORT)
    ap.add_argument("--settle-ms", type=int, default=DEFAULT_SETTLE_MS)
    ap.add_argument("--sleep-on-exit", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--login", action="store_true",
                    help="sign chatgpt.com in with the phone's own Google account "
                         "(a no-op if it already is), then ask any prompts given")
    ap.add_argument("--email", default=None,
                    help="which Google account to sign in with, if the phone has several")
    ap.add_argument("--name", default=None,
                    help="name for ChatGPT's signup form; defaults to the Google "
                         "profile's display name")
    ap.add_argument("--birthday", default=DEFAULT_BIRTHDAY,
                    help=f"birthday for that form (default {DEFAULT_BIRTHDAY})")
    ap.add_argument("--status", action="store_true",
                    help="report whether this phone is signed in, and exit")
    ap.add_argument("--memory", choices=("on", "off"), default=None,
                    help="set the account's memory before asking (default: leave it)")
    args = ap.parse_args()
    memory = None if args.memory is None else args.memory == "on"

    with PhoneChatGPTScraper(
            args.serial, ssh_host=args.ssh_host, adb=args.adb,
            cdp_port=args.cdp_port, settle_ms=args.settle_ms,
            memory=memory, sleep_on_exit=args.sleep_on_exit, debug=args.debug) as s:
        if args.status:
            accounts = s.session.google_accounts()
            logged_in = s.is_logged_in()
            print(json.dumps({"serial": args.serial,
                              "logged_in": logged_in,
                              "memory_enabled": s.apply_memory() if logged_in else False,
                              "google_accounts": accounts}, indent=2))
            return
        if args.login:
            print(json.dumps({"serial": args.serial,
                              "signed_in_as": s.log_in(email=args.email, name=args.name,
                                                       birthday=args.birthday)}, indent=2))
        if memory is not None and not args.prompts:
            print(json.dumps({"serial": args.serial,
                              "memory_enabled": s.apply_memory()}, indent=2))
            return
        prompts = args.prompts or ([] if args.login else
                                   ["what are the economic consequences of "
                                    "how does photosynthesis work"])
        for p in prompts:
            print(json.dumps(s.ask(p).to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
