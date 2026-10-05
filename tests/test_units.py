"""Unit tests for the pure helpers — no network, no browser, no handset."""

from datetime import date

import pytest

from aiscrape.google_aimode import _domain, _real_url
from aiscrape.phone_farm import PhoneFarmAIOverviewScraper, _clean_overview_text
from aiscrape import phone_chatgpt
from aiscrape.phone_chatgpt import (PhoneChatGPTScraper, _age_on, _extraction_gaps,
                                    _local_part, _memory_state, _name_for_signup)
from aiscrape.utils import PROVIDERS, get_env_bool, normalize_storage_state


@pytest.mark.parametrize("href,expected", [
    ("/url?q=https%3A%2F%2Fexample.com%2Fa", "https://example.com/a"),
    ("https://www.google.com/url?url=https%3A%2F%2Fexample.com", "https://example.com"),
    ("https://example.com/direct", "https://example.com/direct"),
    ("", ""),
])
def test_real_url_unwraps_google_redirects(href, expected):
    assert _real_url(href) == expected


@pytest.mark.parametrize("url,expected", [
    ("https://www.example.com/a/b", "example.com"),
    ("https://sub.example.com", "sub.example.com"),
    ("not a url", ""),
])
def test_domain(url, expected):
    assert _domain(url) == expected


@pytest.mark.parametrize("name,expected", [
    ("Lab Phone 6", "Lab Phone Six"),
    ("Phone6", "Phone Six"),          # a digit jammed against a word gets spaced off
    ("Phone-6", "Phone-Six"),         # ...but not one already next to a separator
    ("No Digits Here", "No Digits Here"),
])
def test_name_for_signup_spells_out_digits(name, expected):
    """OpenAI's signup form rejects a display name containing a digit."""
    assert _name_for_signup(name) == expected


def test_local_part_is_the_last_resort_account_name():
    assert _local_part("labphone009@gmail.com") == "Labphone009"


def test_age_on_counts_whole_years():
    assert _age_on("2000-01-01", today=date(2026, 1, 1)) == 26
    assert _age_on("2000-01-02", today=date(2026, 1, 1)) == 25


def test_get_env_bool(monkeypatch):
    monkeypatch.delenv("AISCRAPE_TEST_FLAG", raising=False)
    assert get_env_bool("AISCRAPE_TEST_FLAG", False) is False
    monkeypatch.setenv("AISCRAPE_TEST_FLAG", "1")
    assert get_env_bool("AISCRAPE_TEST_FLAG") is True


def test_normalize_storage_state_accepts_a_dict():
    state = {"cookies": [], "origins": []}
    assert normalize_storage_state(state) == state


def test_normalize_storage_state_defaults_to_empty():
    assert normalize_storage_state(None) == {"cookies": [], "origins": []}


def test_every_provider_has_its_own_db_file():
    from aiscrape.accounts_pool import default_db_file

    files = {default_db_file(p) for p in PROVIDERS}
    assert len(files) == len(PROVIDERS)


@pytest.mark.parametrize("data,expected", [
    ({"before": {"m3m": True}, "writes": [], "after": {"m3m": True, "sunshine": True}}, True),
    ({"after": {"m3m": False, "sunshine": False, "moonshine": False}}, False),
    ({"after": {"m3m": False, "sunshine": True}}, True),   # any flag on counts as on
    ({"signedOut": True}, None),
    ({"error": "settings 500"}, None),
    ({"after": {}}, None),
    (None, None),
])
def test_memory_state(data, expected):
    assert _memory_state(data) is expected


# What AI Mode shows when it loads `udm=50` but never runs the query.
START_SCREEN = "Maps\nShopping\nBooks\nFlights\nFinance\nHi MEO 11, what's on your mind?"


def test_clean_overview_text_without_the_query_echo_is_empty():
    assert _clean_overview_text(START_SCREEN, "is the economy doing well") == ""


def test_clean_overview_text_cuts_at_the_query_echo():
    text = "AI Mode\nAll\nis the economy doing well\nIt depends on the measure."
    assert _clean_overview_text(text, "is the economy doing well") == "It depends on the measure."


class _StubSession:
    serial = "R5CY30YVNKN"

    def require_open(self): pass
    def ensure_visible(self): pass
    def launch(self, url): pass
    def close_tab(self, target): pass


def test_search_reports_the_start_screen_as_a_failed_ask():
    scraper = object.__new__(PhoneFarmAIOverviewScraper)
    scraper.session = _StubSession()
    scraper._await_answer = lambda prompt: {"found": True, "text": START_SCREEN,
                                            "anchors": [], "url": ""}
    result = scraper.search("is the economy doing well")
    assert not result.has_overview
    assert not result.blocked
    assert result.note == "ai mode never answered the query"


def test_search_keeps_an_answer_with_no_sources():
    prompt = "is the economy doing well"
    answer = "It depends on the measure."
    scraper = object.__new__(PhoneFarmAIOverviewScraper)
    scraper.session = _StubSession()
    scraper._await_answer = lambda p: {"found": True, "text": f"AI Mode\n{prompt}\n{answer}",
                                       "anchors": [], "url": ""}
    result = scraper.search(prompt)
    assert result.has_overview
    assert result.overview_text == answer


@pytest.mark.parametrize("data,refs,expected", [
    ({"app": True, "servedModel": "gpt-5-6", "chips": []}, [], []),
    ({"app": False, "servedModel": "", "chips": []}, [], []),
    ({"app": True, "servedModel": "", "chips": []}, [],
     ["signed-in turn carried no model slug"]),
    ({"app": False, "servedModel": "", "chips": ["Elections Alberta"]}, [],
     ["answer shows source chips the extractor could not read"]),
    ({"app": False, "servedModel": "", "chips": ["Elections Alberta"]}, ["a ref"], []),
])
def test_extraction_gaps(data, refs, expected):
    assert _extraction_gaps(data, refs) == expected


def _chatgpt_scraper(data):
    scraper = object.__new__(PhoneChatGPTScraper)
    scraper.session = _StubSession()
    scraper._prepared = True
    scraper.memory_enabled = None
    scraper._last_target_id = ""
    scraper.prepare = lambda force=False: None
    scraper._await_answer = lambda prompt: data
    return scraper


def test_ask_keeps_an_unlabelled_answer_and_flags_it(tmp_path, monkeypatch):
    monkeypatch.setattr(phone_chatgpt, "DEBUG_DIR", tmp_path)
    data = {"found": True, "app": True, "servedModel": "", "text": "Polls close at 8.",
            "citations": [], "anchors": [], "chips": ["Elections Alberta"],
            "bodyText": "Polls close at 8."}
    result = _chatgpt_scraper(data).ask("when do polls close")
    assert result.response == "Polls close at 8."
    assert result.served_model is None
    assert "no model slug" in result.note
    assert "source chips" in result.note
    snapshots = list(tmp_path.glob("chatgpt-*.json"))
    assert len(snapshots) == 1


def test_ask_leaves_a_clean_answer_unflagged(tmp_path, monkeypatch):
    monkeypatch.setattr(phone_chatgpt, "DEBUG_DIR", tmp_path)
    data = {"found": True, "app": True, "servedModel": "gpt-5-6", "text": "Hello.",
            "citations": [], "anchors": [], "chips": [], "bodyText": "Hello."}
    result = _chatgpt_scraper(data).ask("say hello")
    assert result.served_model == "gpt-5-6"
    assert result.note == ""
    assert not list(tmp_path.iterdir())
