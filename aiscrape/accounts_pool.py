"""Per-provider pool of saved sessions ("accounts").

One `AccountsPool` instance manages one provider's accounts, stored in that
provider's own database file (``db/<provider>.db`` by default). Hand out an
available account with `get_available`, lock it on a rate-limit with
`lock_until`, mark it inactive when its session expires, and refresh its cookies
back with `update_storage_state`. Ported from igscrape's `AccountsPool`, keyed
on `label` and storing `storage_state` instead of a username/cookies pair.
"""

import asyncio
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from .account import Account
from .db import execute, fetchall, fetchone
from .exceptions import NoAccountError
from .logger import logger
from .utils import get_env_bool, normalize_storage_state, utc


def default_db_file(provider: str) -> str:
    """The default database path for a provider — CWD-relative, one per provider."""
    return str(Path("db") / f"{provider}.db")


class AccountsPool:
    # Least-recently-used-first: hand out the account with the fewest queries in
    # the current window so load spreads evenly and no single session gets hot.
    _order_by: str = "query_count_24h ASC"

    def __init__(
        self,
        provider: str,
        db_file: str | None = None,
        _raise_when_no_account: bool = get_env_bool(
            "AISCRAPE_RAISE_WHEN_NO_ACCOUNT"
        ),
    ):
        if not provider:
            raise ValueError("Must provide a provider (e.g. 'google', 'claude')")
        self.provider = provider
        self._db_file = db_file or default_db_file(provider)
        self._raise_when_no_account = _raise_when_no_account

    @staticmethod
    def _id_cond(label: str) -> str:
        return f"label = '{label}'"

    @staticmethod
    def _ids_cond(labels: list[str]) -> str:
        quoted = ",".join([f"'{x}'" for x in labels])
        return f"label IN ({quoted})"

    async def add_account(
        self,
        label: str,
        storage_state: str | dict | list | None = None,
        email: str | None = None,
        password: str | None = None,
        proxy_server: str | None = None,
        proxy_username: str | None = None,
        proxy_password: str | None = None,
        fingerprint: str | None = None,
        os: str = "linux",
        locale: str | None = None,
    ):
        """Add an account (a saved session), keyed on `label`."""
        if not label:
            raise ValueError("Must provide a label")

        qs = f"SELECT * FROM accounts WHERE {self._id_cond(label)}"
        if await fetchone(self._db_file, qs):
            logger.warning(f"[{self.provider}] account {label} already exists")
            return

        state = normalize_storage_state(storage_state)

        account = Account(
            label=label,
            email=email,
            password=password,
            storage_state=state,
            # A session with cookies is presumed usable until a scrape proves
            # otherwise; an empty session starts inactive (nothing to load).
            active=bool(state.get("cookies")),
            proxy_server=proxy_server,
            proxy_username=proxy_username,
            proxy_password=proxy_password,
            fingerprint=fingerprint,
            os=os,
            locale=locale,
        )

        await self.save(account)
        logger.info(
            f"[{self.provider}] account {label} added "
            f"(active={account.active}, {len(account.cookies)} cookies)"
        )

    async def delete_account(self, label: str | list[str]):
        labels = label if isinstance(label, list) else [label]
        labels = list(set(labels))
        if not labels:
            return
        qs = f"DELETE FROM accounts WHERE {self._ids_cond(labels)}"
        await execute(self._db_file, qs)
        logger.info(f"[{self.provider}] deleted {len(labels)} account(s)")

    async def get_inactive_accounts(self) -> list[Account]:
        rs = await fetchall(self._db_file, "SELECT * FROM accounts WHERE active = false")
        return [Account.from_rs(x) for x in rs]

    async def get_active_accounts(self) -> list[Account]:
        rs = await fetchall(self._db_file, "SELECT * FROM accounts WHERE active = true")
        return [Account.from_rs(x) for x in rs]

    async def get(self, label: str | list[str] | None) -> Account | list[Account]:
        if label is None:
            rs = await fetchall(self._db_file, "SELECT * FROM accounts")
            return [Account.from_rs(x) for x in rs]
        elif isinstance(label, list):
            labels = list(set(label))
            qs = f"SELECT * FROM accounts WHERE {self._ids_cond(labels)}"
            rs = await fetchall(self._db_file, qs)
            return [Account.from_rs(x) for x in rs]
        else:
            qs = f"SELECT * FROM accounts WHERE {self._id_cond(label)}"
            rs = await fetchone(self._db_file, qs)
            if not rs:
                raise ValueError(f"Account {label} not found")
            return Account.from_rs(rs)

    async def save(self, account: Account):
        data = account.to_rs()
        cols = list(data.keys())
        label = account.label

        existing = await fetchone(
            self._db_file, f"SELECT * FROM accounts WHERE {self._id_cond(label)}"
        )
        if existing:
            set_clause = ",".join([f"{x}=:{x}" for x in cols if x != "label"])
            qs = f"UPDATE accounts SET {set_clause} WHERE {self._id_cond(label)}"
        else:
            qs = (
                f"INSERT INTO accounts ({','.join(cols)}) "
                f"VALUES ({','.join([f':{x}' for x in cols])})"
            )
        await execute(self._db_file, qs, data)

    async def reset_locks(self, label: str | list[str] | None = None):
        if label is None:
            qs = "UPDATE accounts SET locks = json_object()"
        else:
            labels = label if isinstance(label, list) else [label]
            qs = (
                "UPDATE accounts SET locks = json_object() "
                f"WHERE {self._ids_cond(list(set(labels)))}"
            )
        await execute(self._db_file, qs)
        logger.info(f"[{self.provider}] reset locks for {label or 'all accounts'}")

    async def set_active(
        self,
        label: str | list[str] | None,
        active: bool,
        error_message: str | None = None,
    ):
        params = {"active": active, "error_msg": error_message}
        if label is None:
            qs = "UPDATE accounts SET active = :active, error_msg = :error_msg"
        else:
            labels = label if isinstance(label, list) else [label]
            qs = (
                "UPDATE accounts SET active = :active, error_msg = :error_msg "
                f"WHERE {self._ids_cond(list(set(labels)))}"
            )
        await execute(self._db_file, qs, params)
        logger.info(
            f"[{self.provider}] set active={active} for {label or 'all accounts'}"
        )

    async def lock_until(self, label: str | list[str] | None, until: str):
        """Lock account(s) until a SQLite datetime expression, e.g.
        ``"datetime('now', '+15 minutes')"`` — a rate-limit cooldown."""
        labels = label if isinstance(label, list) else [label] if label else []
        where = self._ids_cond(list(set(labels))) if labels else "TRUE"
        qs = f"""
        UPDATE accounts SET
            locks = json_set(locks, '$.locked_until', {until}),
            last_used = datetime({utc.ts()}, 'unixepoch')
        WHERE {where}
        """
        await execute(self._db_file, qs)

    async def unlock(self, label: str | list[str] | None = None):
        labels = label if isinstance(label, list) else [label] if label else []
        where = self._ids_cond(list(set(labels))) if labels else "TRUE"
        qs = f"""
        UPDATE accounts SET
            locks = json_remove(locks, '$.locked_until'),
            last_used = datetime({utc.ts()}, 'unixepoch')
        WHERE {where}
        """
        await execute(self._db_file, qs)

    async def _get_and_mark_in_use(self, subquery: str) -> Account | None:
        if int(sqlite3.sqlite_version_info[1]) >= 35:
            qs = f"""
            UPDATE accounts SET
                last_used = datetime({utc.ts()}, 'unixepoch'),
                in_use = true
            WHERE label = ({subquery})
            RETURNING *
            """
            rs = await fetchone(self._db_file, qs)
        else:
            tx = uuid.uuid4().hex
            qs = f"""
            UPDATE accounts SET
                last_used = datetime({utc.ts()}, 'unixepoch'),
                in_use = true,
                _tx = '{tx}'
            WHERE label = ({subquery})
            """
            await execute(self._db_file, qs)
            rs = await fetchone(
                self._db_file, f"SELECT * FROM accounts WHERE _tx = '{tx}'"
            )
        return Account.from_rs(rs) if rs else None

    async def get_available(self) -> Account | None:
        q = f"""
        SELECT label FROM accounts
        WHERE active = true
          AND in_use = false
          AND (
                locks IS NULL
                OR json_extract(locks, '$.locked_until') IS NULL
                OR json_extract(locks, '$.locked_until') < datetime('now')
          )
        ORDER BY {self._order_by}
        LIMIT 1
        """
        return await self._get_and_mark_in_use(q)

    async def get_available_or_wait(self) -> Account | None:
        msg_shown = False
        while True:
            account = await self.get_available()
            if account:
                if msg_shown:
                    logger.info(f"[{self.provider}] continuing with {account.label}")
                return account

            if self._raise_when_no_account or get_env_bool(
                "AISCRAPE_RAISE_WHEN_NO_ACCOUNT"
            ):
                raise NoAccountError(f"No {self.provider} account available")

            if not msg_shown:
                nat = await self.next_available_at()
                if not nat:
                    logger.warning(
                        f"[{self.provider}] no active accounts. Stopping..."
                    )
                    return None
                logger.info(
                    f"[{self.provider}] no account available. Next available at {nat}"
                )
                msg_shown = True

            await asyncio.sleep(5)

    async def next_available_at(self) -> str | None:
        qs = """
        SELECT json_extract(locks, '$.locked_until') AS locked_until
        FROM accounts
        WHERE active = true
          AND json_extract(locks, '$.locked_until') IS NOT NULL
          AND json_extract(locks, '$.locked_until') > datetime('now')
        ORDER BY locked_until ASC
        LIMIT 1
        """
        rs = await fetchone(self._db_file, qs)
        if rs and rs["locked_until"]:
            now, trg = utc.now(), utc.from_iso(rs["locked_until"])
            if trg < now:
                return "now"
            at_local = datetime.now() + (trg - now)
            return at_local.strftime("%H:%M:%S")
        return None

    async def release_account(self, label: str | list[str] | None):
        labels = label if isinstance(label, list) else [label] if label else []
        where = self._ids_cond(list(set(labels))) if labels else "TRUE"
        qs = f"""
        UPDATE accounts SET
            in_use = false,
            last_used = datetime({utc.ts()}, 'unixepoch')
        WHERE {where}
        """
        await execute(self._db_file, qs)

    async def mark_inactive(self, label: str, error_msg: str | None):
        qs = (
            "UPDATE accounts SET active = false, error_msg = :error_msg, in_use = false "
            f"WHERE {self._id_cond(label)}"
        )
        await execute(self._db_file, qs, {"error_msg": error_msg})
        logger.warning(f"[{self.provider}] marked {label} inactive: {error_msg}")

    async def update_storage_state(self, label: str, storage_state: str | dict | list):
        """Persist a refreshed session (called after each scrape so cookies the
        product rotated stay current in the pool)."""
        state = normalize_storage_state(storage_state)
        qs = (
            "UPDATE accounts SET storage_state = :state "
            f"WHERE {self._id_cond(label)}"
        )
        await execute(self._db_file, qs, {"state": json.dumps(state)})
        logger.info(
            f"[{self.provider}] updated session for {label} "
            f"({len(state.get('cookies', []))} cookies)"
        )

    async def update_last_used(self, label: str):
        qs = (
            f"UPDATE accounts SET last_used = datetime({utc.ts()}, 'unixepoch') "
            f"WHERE {self._id_cond(label)}"
        )
        await execute(self._db_file, qs)

    async def update_query_count(self, label: str, endpoint: str, increment: int = 1):
        qs = f"""
        UPDATE accounts SET
            query_count_per_endpoint = json_set(
                query_count_per_endpoint,
                '$.{endpoint}',
                COALESCE(json_extract(query_count_per_endpoint, '$.{endpoint}'), 0) + :increment
            ),
            query_count_24h = query_count_24h + :increment,
            queries_since_rest = queries_since_rest + :increment,
            last_used = datetime({utc.ts()}, 'unixepoch')
        WHERE {self._id_cond(label)}
        """
        await execute(self._db_file, qs, {"increment": increment})

    async def get_query_count(self, label: str, endpoint: str | None = None) -> int:
        if endpoint:
            qs = (
                f"SELECT json_extract(query_count_per_endpoint, '$.{endpoint}') AS count "
                f"FROM accounts WHERE {self._id_cond(label)}"
            )
        else:
            qs = f"SELECT query_count_24h AS count FROM accounts WHERE {self._id_cond(label)}"
        rs = await fetchone(self._db_file, qs)
        return (rs["count"] or 0) if rs else 0

    async def reset_query_counts(
        self, label: str | None = None, endpoint: str | None = None
    ):
        if endpoint:
            base = (
                "UPDATE accounts SET "
                f"query_count_per_endpoint = json_remove(query_count_per_endpoint, '$.{endpoint}')"
            )
        else:
            base = (
                "UPDATE accounts SET "
                "query_count_per_endpoint = '{}', query_count_24h = 0"
            )
        qs = base if label is None else f"{base} WHERE {self._id_cond(label)}"
        await execute(self._db_file, qs)
        logger.info(
            f"[{self.provider}] reset query counts for {label or 'all'}"
            + (f" endpoint={endpoint}" if endpoint else "")
        )

    async def reset_queries_since_rest(self, label: str):
        qs = (
            "UPDATE accounts SET queries_since_rest = 0 "
            f"WHERE {self._id_cond(label)}"
        )
        await execute(self._db_file, qs)

    async def get_queries_since_rest(self, label: str) -> int:
        qs = (
            "SELECT queries_since_rest AS c FROM accounts "
            f"WHERE {self._id_cond(label)}"
        )
        rs = await fetchone(self._db_file, qs)
        return rs["c"] if rs else 0

    _updatable_fields = {
        "email",
        "password",
        "active",
        "proxy_server",
        "proxy_username",
        "proxy_password",
        "fingerprint",
        "os",
        "locale",
        "error_msg",
    }

    async def update_field(self, label: str, field: str, value):
        if field not in self._updatable_fields:
            raise ValueError(
                f"Field '{field}' is not updatable. "
                f"Allowed: {', '.join(sorted(self._updatable_fields))}"
            )
        existing = await fetchone(
            self._db_file, f"SELECT * FROM accounts WHERE {self._id_cond(label)}"
        )
        if not existing:
            raise ValueError(f"Account {label} not found")
        if field == "active" and isinstance(value, str):
            value = value.lower() in ("true", "1", "yes", "y")
        qs = f"UPDATE accounts SET {field} = :value WHERE {self._id_cond(label)}"
        await execute(self._db_file, qs, {"value": value})
        logger.info(f"[{self.provider}] updated {field}={value} for {label}")

    async def stats(self) -> dict:
        config = [
            ("total", "SELECT COUNT(*) FROM accounts"),
            ("active", "SELECT COUNT(*) FROM accounts WHERE active = true"),
            ("inactive", "SELECT COUNT(*) FROM accounts WHERE active = false"),
            ("in_use", "SELECT COUNT(*) FROM accounts WHERE in_use = true"),
            (
                "locked",
                "SELECT COUNT(*) FROM accounts "
                "WHERE json_extract(locks, '$.locked_until') IS NOT NULL "
                "AND json_extract(locks, '$.locked_until') > datetime('now')",
            ),
        ]
        qs = f"SELECT {','.join([f'({q}) as {k}' for k, q in config])}"
        rs = await fetchone(self._db_file, qs)
        return dict(rs) if rs else {}
