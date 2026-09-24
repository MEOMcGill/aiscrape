"""Unit tests for the pure helpers — no network, no browser, no handset."""

from datetime import date

import pytest

from aiscrape.google_aimode import _domain, _real_url
from aiscrape.phone_chatgpt import _age_on, _local_part, _memory_state, _name_for_signup
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
