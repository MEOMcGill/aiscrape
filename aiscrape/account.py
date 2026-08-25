import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime

from .utils import utc


@dataclass
class Account:
    """A saved logged-in session for one AI provider.

    Keyed on `label` (a human-chosen name, usually the account's email). The
    session itself lives in `storage_state` — the Playwright
    ``{"cookies": [...], "origins": [...]}`` blob loaded into a browser context
    via ``new_context(storage_state=...)``. There is no password-based login
    here (the products gate fresh automated browsers), so `password` is optional
    reference data only.
    """

    label: str
    email: str | None = None
    password: str | None = None
    storage_state: dict = field(default_factory=lambda: {"cookies": [], "origins": []})
    active: bool = False
    locks: dict[str, datetime] = field(default_factory=dict)
    query_count_per_endpoint: dict[str, int] = field(default_factory=dict)
    proxy_server: str | None = None
    proxy_username: str | None = None
    proxy_password: str | None = None
    fingerprint: str | None = None
    os: str = "linux"
    locale: str | None = None
    error_msg: str | None = None
    last_used: datetime | None = None
    in_use: bool = False
    queries_since_rest: int = 0
    query_count_24h: int = 0

    @property
    def identifier(self) -> str:
        return self.label

    @property
    def display_name(self) -> str:
        return self.label

    @property
    def cookies(self) -> list[dict]:
        """The cookie list from the session (convenience accessor)."""
        return self.storage_state.get("cookies", [])

    @property
    def proxy_dict(self) -> dict | None:
        """Playwright proxy settings for this account, or None if no proxy."""
        if not self.proxy_server:
            return None
        proxy: dict = {"server": self.proxy_server}
        if self.proxy_username and self.proxy_password:
            proxy["username"] = self.proxy_username
            proxy["password"] = self.proxy_password
        return proxy

    @staticmethod
    def from_rs(rs: sqlite3.Row) -> "Account":
        doc = dict(rs)
        doc.pop("_tx", None)
        doc["locks"] = {
            k: utc.from_iso(v) for k, v in json.loads(doc["locks"]).items()
        }
        doc["query_count_per_endpoint"] = {
            k: v
            for k, v in json.loads(doc["query_count_per_endpoint"]).items()
            if isinstance(v, int)
        }
        doc["storage_state"] = json.loads(doc["storage_state"])
        doc["active"] = bool(doc["active"])
        doc["in_use"] = bool(doc["in_use"])
        doc["last_used"] = (
            utc.from_iso(doc["last_used"]) if doc["last_used"] else None
        )
        return Account(**doc)

    def to_rs(self) -> dict:
        rs = asdict(self)
        rs["locks"] = json.dumps(rs["locks"], default=lambda x: x.isoformat())
        rs["query_count_per_endpoint"] = json.dumps(rs["query_count_per_endpoint"])
        rs["storage_state"] = json.dumps(rs["storage_state"])
        rs["last_used"] = rs["last_used"].isoformat() if rs["last_used"] else None
        return rs
