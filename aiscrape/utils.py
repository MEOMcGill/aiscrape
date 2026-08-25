"""Small utilities for the aiscrape account pool."""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

# Providers that have a saved session + a scraper behind them. Each gets its own
# account-pool DB file (db/<provider>.db) — "separate account pools per
# provider" — because a Google account, a ChatGPT account and a Claude account
# are unrelated credentials with independent rate-limits and block state.
# gemini rides the Google account (same cookie host) but is still its own pool:
# its rate-limits and safety blocks are tracked separately from Search.
PROVIDERS: tuple[str, ...] = ("google", "chatgpt", "claude", "gemini", "meta")

# An empty logged-in session, in Playwright storage_state shape.
EMPTY_STORAGE_STATE: dict = {"cookies": [], "origins": []}


class utc:
    @staticmethod
    def now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def from_iso(iso: str) -> datetime:
        return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)

    @staticmethod
    def ts() -> int:
        return int(utc.now().timestamp())


def get_env_bool(key: str, default_val: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default_val
    return val.lower() in ("1", "true", "yes")


def get_device_os() -> str:
    """Detect current OS for the camoufox fingerprint."""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    return "linux"


def normalize_storage_state(value: str | dict | list | Path | None) -> dict:
    """Coerce various inputs into a Playwright `storage_state` dict.

    Accepts:
      - a `storage_state` dict (``{"cookies": [...], "origins": [...]}``),
      - a bare list of Playwright cookie dicts (wrapped as cookies, no origins),
      - a path to an ``auth/<platform>.json`` storage_state file,
      - a JSON string of either of the above,
      - ``None`` / empty -> an empty session.

    Always returns a dict with both ``cookies`` and ``origins`` keys.
    """
    if value is None or value == "":
        return {"cookies": [], "origins": []}

    if isinstance(value, Path):
        value = value.read_text(encoding="utf-8")

    if isinstance(value, str):
        # A filesystem path to a storage_state file, or a JSON blob.
        p = Path(value)
        if p.exists() and p.is_file():
            value = p.read_text(encoding="utf-8")
        value = json.loads(value)

    if isinstance(value, list):
        return {"cookies": value, "origins": []}

    if isinstance(value, dict):
        if "cookies" in value or "origins" in value:
            return {
                "cookies": value.get("cookies", []),
                "origins": value.get("origins", []),
            }
        # A {name: value} cookie mapping — no domain context, so unusable as a
        # session; reject rather than silently produce a broken storage_state.
        raise ValueError(
            "storage_state dict must contain 'cookies'/'origins'; "
            "got a bare mapping with neither"
        )

    raise ValueError(f"Cannot interpret storage_state value of type {type(value)}")
