"""
Read cookies for a domain out of the local Firefox profile's cookie jar,
without disturbing a running Firefox.

Firefox holds a lock on the live cookies.sqlite, so it's copied (+ WAL/SHM) to
a temp dir and read from there -- read-only, the live profile is never touched.
Used to keep a scraper session "warm" with cookies from real day-to-day
browsing: google_aimode.py imports the Google jar, and callers scraping other
sites (e.g. Facebook) read the whole jar with `host_like=None`.
"""

from __future__ import annotations

import configparser
import shutil
import sqlite3
import tempfile
from pathlib import Path

FF_PROFILES_ROOT = Path.home() / "snap/firefox/common/.mozilla/firefox"
# Firefox sameSite int -> Playwright string.
_SAMESITE = {0: "None", 1: "Lax", 2: "Strict"}


def default_firefox_profile() -> Path | None:
    """Path to the snap Firefox default profile (via profiles.ini Default=1)."""
    ini = FF_PROFILES_ROOT / "profiles.ini"
    if ini.exists():
        cfg = configparser.ConfigParser()
        cfg.read(ini)
        for sec in cfg.sections():
            if sec.startswith("Profile") and cfg[sec].get("Default") == "1":
                return FF_PROFILES_ROOT / cfg[sec]["Path"]
    cands = sorted(FF_PROFILES_ROOT.glob("*.default"))
    return cands[0] if cands else None


def read_cookies(profile: Path, host_like: str | None = "%google.%") -> list[dict]:
    """Cookies from the profile's default (non-container) jar, converted to
    Playwright `add_cookies` dicts.

    `host_like` is a SQL LIKE pattern matched against the cookie host (e.g.
    ``"%google.%"``); pass ``None`` to read the whole jar (every host).
    """
    src = profile / "cookies.sqlite"
    if not src.exists():
        return []
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "cookies.sqlite"
        for suffix in ("", "-wal", "-shm"):
            s = Path(str(src) + suffix)
            if s.exists():
                shutil.copy2(s, str(dst) + suffix)
        con = sqlite3.connect(str(dst))
        # originAttributes = '' -> the default (non-container) jar.
        cols = ("host, name, value, path, expiry, isSecure, isHttpOnly, sameSite")
        if host_like is None:
            rows = con.execute(
                f"SELECT {cols} FROM moz_cookies WHERE originAttributes = ''"
            ).fetchall()
        else:
            rows = con.execute(
                f"SELECT {cols} FROM moz_cookies "
                "WHERE originAttributes = '' AND host LIKE ?",
                (host_like,),
            ).fetchall()
        con.close()

    out = []
    for host, name, value, path, expiry, secure, http_only, samesite in rows:
        secure, http_only = bool(secure), bool(http_only)
        ss = _SAMESITE.get(samesite, "Lax")
        # Playwright rejects SameSite=None on a non-Secure cookie.
        if ss == "None" and not secure:
            ss = "Lax"
        # Playwright wants `expires` in Unix SECONDS; Firefox stores `expiry`
        # in milliseconds. Any real seconds value is < 1e11 (year 5138), so
        # anything larger is ms -> divide. 0 => session cookie.
        if not expiry:
            expires = -1.0
        else:
            expires = float(expiry) / 1000.0 if expiry > 1e11 else float(expiry)
        out.append({
            "name": name,
            "value": value,
            "domain": host,          # Firefox already stores leading-dot domains
            "path": path or "/",
            "expires": expires,
            "httpOnly": http_only,
            "secure": secure,
            "sameSite": ss,
        })
    return out
