"""PhoneBackend's handling of the ChatGPT memory setting — no handset."""

import asyncio

from aiscrape.phone_chatgpt import ChatGPTMemoryError
from aiscrape.phone_pool import PhoneBackend, SerialPool


class _Chat:
    def __init__(self, error=None):
        self.error = error

    def apply_memory(self):
        if self.error:
            raise self.error
        return False


def _memory_verdict(chat) -> bool | None:
    async def run():
        pool = SerialPool(["S1"])
        backend = PhoneBackend(pool, chatgpt_memory=False)
        backend._chatgpt = chat
        await backend._apply_chatgpt_memory("S1")
        return await pool.chatgpt_state("S1")
    return asyncio.run(run())


def test_memory_that_cannot_be_set_takes_the_handset_out_of_chat():
    assert _memory_verdict(_Chat(ChatGPTMemoryError("S1: memory is on, wanted off"))) is False


def test_memory_set_leaves_the_handset_unproven():
    assert _memory_verdict(_Chat()) is None


# ── per-provider walls ──────────────────────────────────────────────────────

import pytest
from types import SimpleNamespace

from aiscrape import phone_pool
from aiscrape.phone_pool import PhoneFarmExhausted


def test_a_provider_wall_only_keeps_the_phone_from_that_provider():
    async def run():
        pool = SerialPool(["S1", "S2"])
        await pool.wall_provider("S1", "google")
        got_google = await pool.acquire("google")
        got_chat = await pool.acquire("chatgpt")
        return got_google, got_chat, pool.snapshot()
    google, chat, snapshot = asyncio.run(run())
    assert (google, chat) == ("S2", "S1")
    assert "0 walled; walled for google: 1" in snapshot


def test_a_provider_wall_lapses_after_the_cooldown(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(phone_pool.time, "monotonic", lambda: clock[0])

    async def run():
        pool = SerialPool(["S1"], wall_cooldown_s=1800)
        await pool.wall_provider("S1", "claude")
        with pytest.raises(PhoneFarmExhausted, match="for claude.*clears in 30 min"):
            await pool.acquire("claude")
        clock[0] += 1800
        return await pool.acquire("claude")
    assert asyncio.run(run()) == "S1"


def test_no_cooldown_keeps_a_provider_wall_for_the_run(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(phone_pool.time, "monotonic", lambda: clock[0])

    async def run():
        pool = SerialPool(["S1"], wall_cooldown_s=0)
        await pool.wall_provider("S1", "google")
        clock[0] += 10 ** 6
        await pool.acquire("google")
    with pytest.raises(PhoneFarmExhausted, match="walls last the run"):
        asyncio.run(run())


class _Session:
    def __exit__(self, *exc):
        pass


def _blocked_backend(surface_attr, method, result):
    pool = SerialPool(["S1"])
    backend = PhoneBackend(pool, ask_delay_min_s=0, ask_delay_max_s=0)
    pool._free.remove("S1")
    pool._in_use.add("S1")
    backend._serial, backend._session = "S1", _Session()
    setattr(backend, surface_attr, SimpleNamespace(**{method: lambda *a: result}))
    return pool, backend


@pytest.mark.parametrize("surface_attr,method,ask,provider", [
    ("_scraper", "search", lambda b: b.search("q"), "google"),
    ("_chatgpt", "ask", lambda b: b.chat("q"), "chatgpt"),
    ("_claude", "ask", lambda b: b.claude("q"), "claude"),
])
def test_a_blocked_ask_walls_the_phone_for_its_provider_only(surface_attr, method, ask,
                                                             provider):
    result = SimpleNamespace(blocked=True, note="walled", has_overview=False, response="")

    async def run():
        pool, backend = _blocked_backend(surface_attr, method, result)
        await pool.set_claude_ok("S1", True)
        # One phone, now walled for this provider: nothing is left to re-ask on.
        with pytest.raises(PhoneFarmExhausted, match=f"for {provider}"):
            await ask(backend)
        others = [p for p in ("google", "chatgpt", "claude") if p != provider]
        return (await pool.is_walled_for("S1", provider),
                [await pool.is_walled_for("S1", p) for p in others],
                await pool.claude_state("S1"), pool.snapshot())
    walled, others, claude_ok, snapshot = asyncio.run(run())
    assert walled
    assert others == [False, False]
    assert claude_ok is True           # a usage limit is not "out for the run"
    assert "1 free / 0 in use / 0 walled" in snapshot


def test_a_signed_out_google_phone_still_serves_the_chatbots(monkeypatch):
    async def run():
        pool = SerialPool(["S1"])
        backend = PhoneBackend(pool, chatgpt_login=False)
        monkeypatch.setattr(phone_pool, "PhoneChromeSession",
                            lambda serial, **kw: SimpleNamespace(
                                __enter__=lambda: None, __exit__=lambda *e: None))

        async def signed_out(serial):
            return False
        backend._google_signed_in = signed_out
        monkeypatch.setattr(phone_pool.PhoneFarmAIOverviewScraper, "clear_tab_backlog",
                            lambda self, keep=1: 0)
        monkeypatch.setattr(phone_pool.PhoneChatGPTScraper, "clear_tab_backlog",
                            lambda self, keep=1: 0)
        monkeypatch.setattr(phone_pool.PhoneClaudeScraper, "clear_tab_backlog",
                            lambda self, keep=1: 0)
        await backend._open("chatgpt")
        held = backend._serial
        with pytest.raises(PhoneFarmExhausted, match="for google"):
            await backend._rest("google")
        return held, await pool.is_walled_for("S1", "google")
    held, walled = asyncio.run(run())
    assert held == "S1"
    assert walled
