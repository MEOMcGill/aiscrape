"""PhoneClaudeScraper's parsing and the pool's Claude handling — no handset."""

import asyncio

import pytest

from aiscrape import phone_claude
from aiscrape.phone_claude import (ClaudeSignedOutError, PhoneClaudeScraper, _login_step,
                                   _parse_citations, _strip_turn_prefix)
from aiscrape.phone_pool import PhoneBackend, SerialPool


def test_citations_are_deduped_in_order_and_skip_anthropic():
    refs = _parse_citations([
        {"url": "https://www.ctvnews.ca/a", "title": "A"},
        {"url": "https://support.anthropic.com/help", "title": "Help"},
        {"url": "https://www.ctvnews.ca/a", "title": "A, longer title"},
        {"url": "https://globalnews.ca/b", "title": ""},
    ], [])
    assert [r.url for r in refs] == ["https://www.ctvnews.ca/a", "https://globalnews.ca/b"]
    assert refs[0].title == "A, longer title"
    assert refs[1].title == "globalnews.ca"


def test_page_anchors_lose_the_folded_source_count():
    refs = _parse_citations([], [{"href": "https://example.com/x", "text": "Some story +3"}])
    assert refs[0].title == "Some story"


def test_turn_prefix_is_stripped():
    assert _strip_turn_prefix("Claude responded: Hi there.") == "Hi there."


@pytest.mark.parametrize("state,step", [
    ({"googleButton": True, "url": "https://claude.ai/login"}, "start"),
    ({"error": True, "googleButton": True}, "error"),
    ({"terms": True}, "terms"),
    ({"birthday": True, "terms": False}, "birthday"),
    ({"role": True, "composer": False}, "role"),
    ({"composer": True, "url": "https://claude.ai/new"}, "done"),
    ({"composer": True, "url": "https://claude.ai/onboarding"}, "waiting"),
    ({}, "waiting"),
])
def test_login_step(state, step):
    assert _login_step(state) == step


class _Session:
    serial = "R58MEXAMPLE"

    def __init__(self, conv):
        self.conv = conv

    def require_open(self): pass
    def ensure_visible(self): pass
    def launch(self, url): pass
    def close_tab(self, target): pass
    def evaluate(self, page, expression, timeout=25.0): return self.conv


def _scraper(page_data, conv):
    s = object.__new__(PhoneClaudeScraper)
    s.session = _Session(conv)
    s.memory_enabled = False
    s._last_target_id, s._last_page = "", {"id": "1"}
    s.prepare = lambda force=False: None
    s._await_answer = lambda prompt: page_data
    return s


_PAGE = {"signedIn": True, "nUser": 1, "nAssistant": 1, "text": "Claude responded: Iqaluit.",
         "anchors": [], "bodyText": "Claude responded: Iqaluit."}


def test_ask_reads_the_answer_model_and_sources_from_the_conversation():
    conv = {"model": "claude-sonnet-5-5", "answer": "Iqaluit.", "searches": ["nunavut"],
            "citations": [{"url": "https://gov.nu.ca/news", "title": "Newsroom"}]}
    r = _scraper(_PAGE, conv).ask("capital of nunavut")
    assert (r.provider, r.response, r.served_model) == ("claude", "Iqaluit.", "claude-sonnet-5-5")
    assert [ref.domain for ref in r.references] == ["gov.nu.ca"]
    assert r.memory_enabled is False
    assert r.note == "searched the web 1x"


def test_ask_falls_back_to_the_page_when_the_conversation_is_unreadable(tmp_path, monkeypatch):
    monkeypatch.setattr(phone_claude, "DEBUG_DIR", tmp_path)
    page = dict(_PAGE, anchors=[{"href": "https://gov.nu.ca/news", "text": "Newsroom +2"}])
    r = _scraper(page, {"error": "conversation 500"}).ask("capital of nunavut")
    assert r.response == "Iqaluit."
    assert r.served_model is None
    assert "conversation JSON unreadable" in r.note
    assert len(list(tmp_path.glob("claude-*.json"))) == 1


def test_a_usage_limit_is_blocked_not_an_answer():
    page = {"signedIn": True, "nUser": 1, "nAssistant": 0, "text": "", "anchors": [],
            "bodyText": "You've reached your usage limit. Resets at 5pm."}
    r = _scraper(page, {}).ask("anything")
    assert r.blocked and r.rate_limited and not r.response


def test_a_signed_out_page_raises():
    page = {"signedIn": False, "nUser": 0, "nAssistant": 0, "text": "", "bodyText": ""}
    with pytest.raises(ClaudeSignedOutError):
        _scraper(page, {}).ask("anything")


class _Claude:
    def __init__(self, ok):
        self.ok = ok

    def ensure_logged_in(self):
        if not self.ok:
            raise RuntimeError("stuck at 'terms'")
        return True


@pytest.mark.parametrize("ok", [True, False])
def test_claude_sign_in_result_is_remembered_without_walling(ok):
    async def run():
        pool = SerialPool(["S1"])
        backend = PhoneBackend(pool)
        backend._claude = _Claude(ok)
        await backend._check_claude("S1")
        return await pool.claude_state("S1"), pool.snapshot()
    state, snapshot = asyncio.run(run())
    assert state is ok
    assert "0 walled" in snapshot


class _LimitedClaude:
    def ask(self, prompt):
        from aiscrape.models import ChatResult
        return ChatResult(provider="claude", prompt=prompt, blocked=True,
                          rate_limited=True, note="claude usage limit")


def test_a_usage_limit_takes_the_handset_out_of_claude_without_walling():
    async def run():
        pool = SerialPool(["S1"])
        backend = PhoneBackend(pool, ask_delay_min_s=0, ask_delay_max_s=0)
        backend._serial, backend._session = "S1", object()
        backend._claude = _LimitedClaude()
        await pool.set_claude_ok("S1", True)
        rested = []

        async def rest():
            rested.append(backend._serial)
            raise RuntimeError("stop after the first rest")
        backend._rest = rest
        with pytest.raises(RuntimeError, match="first rest"):
            await backend.claude("anything")
        return await pool.claude_state("S1"), pool.snapshot(), rested
    state, snapshot, rested = asyncio.run(run())
    assert state is False
    assert "0 walled" in snapshot
    assert rested == ["S1"]
