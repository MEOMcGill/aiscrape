"""Click CLI for managing the per-provider account pools.

Every command takes ``--provider`` (google / chatgpt / claude / gemini / meta),
which selects that provider's own database file (``db/<provider>.db`` by
default). Accounts are saved logged-in sessions, so the usual way to populate a
pool is `seed-from-auth`, which imports the `auth/<provider>.json` files that
`aiscrape.auth` writes.

Examples:
  aiscrape --provider google seed-from-auth
  aiscrape --provider claude add --label work@x.com --session auth/claude.json
  aiscrape --provider google list
  aiscrape --provider google stats
  aiscrape --provider claude unlock --label work@x.com
"""

import asyncio
import json
from pathlib import Path

import click
from tabulate import tabulate

from .accounts_pool import AccountsPool, default_db_file
from .logger import set_log_level
from .utils import PROVIDERS


def run_async(coro):
    return asyncio.run(coro)


@click.group()
@click.option(
    "--provider",
    required=True,
    type=click.Choice(PROVIDERS),
    help="Which provider's account pool to operate on",
)
@click.option("--db", "db_file", default=None, help="Override the DB path")
@click.option("--log-level", default=None, help="TRACE/DEBUG/INFO/WARNING/ERROR")
@click.pass_context
def cli(ctx, provider, db_file, log_level):
    """aiscrape — per-provider account pool management."""
    if log_level:
        set_log_level(log_level.upper())
    ctx.ensure_object(dict)
    ctx.obj["provider"] = provider
    ctx.obj["pool"] = AccountsPool(provider, db_file=db_file)


def _pool(ctx) -> AccountsPool:
    return ctx.obj["pool"]


# ── Populate ──────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--label", required=True, help="Account label (e.g. its email)")
@click.option("--session", default=None,
              help="Path to a storage_state JSON file (e.g. auth/<provider>.json)")
@click.option("--email", default=None)
@click.option("--password", default=None, help="Reference only; no login is performed")
@click.option("--proxy", default=None, help="Proxy server URL")
@click.option("--proxy-user", default=None)
@click.option("--proxy-pass", default=None)
@click.option("--os", "os_type", default="linux", help="OS fingerprint")
@click.option("--locale", default=None, help="e.g. en-CA")
@click.pass_context
def add(ctx, label, session, email, password, proxy, proxy_user, proxy_pass,
        os_type, locale):
    """Add an account from a storage_state file (or an empty session)."""

    async def _add():
        await _pool(ctx).add_account(
            label=label,
            storage_state=session,
            email=email,
            password=password,
            proxy_server=proxy,
            proxy_username=proxy_user,
            proxy_password=proxy_pass,
            os=os_type,
            locale=locale,
        )
        click.echo(f"Added {ctx.obj['provider']} account: {label}")

    run_async(_add())


@cli.command("seed-from-auth")
@click.option("--auth-dir", default="auth", help="Directory holding <provider>.json")
@click.option("--label", default=None,
              help="Label for the account (default: the provider name)")
@click.pass_context
def seed_from_auth(ctx, auth_dir, label):
    """Import this provider's `auth/<provider>.json` session into the pool."""
    provider = ctx.obj["provider"]
    auth_file = Path(auth_dir) / f"{provider}.json"
    if not auth_file.exists():
        raise click.UsageError(
            f"No session file at {auth_file}. "
            f"Save one first: python -m aiscrape.auth --platform {provider}"
        )

    async def _seed():
        await _pool(ctx).add_account(
            label=label or provider,
            storage_state=str(auth_file),
        )
        click.echo(f"Seeded {provider} account '{label or provider}' from {auth_file}")

    run_async(_seed())


# ── Inspect ─────────────────────────────────────────────────────────────────


@cli.command(name="list")
@click.pass_context
def list_accounts(ctx):
    """List all accounts for this provider."""

    async def _list():
        accounts = await _pool(ctx).get(None)
        if not accounts:
            click.echo("No accounts.")
            return
        rows = []
        for a in accounts:
            locked = a.locks.get("locked_until")
            rows.append([
                a.label,
                "yes" if a.active else "no",
                "yes" if a.in_use else "no",
                len(a.cookies),
                a.query_count_24h,
                locked.strftime("%H:%M:%S") if locked else "-",
                (a.error_msg or "")[:40],
            ])
        click.echo(tabulate(
            rows,
            headers=["label", "active", "in_use", "cookies", "queries", "locked_until", "error"],
        ))

    run_async(_list())


@cli.command()
@click.option("--label", required=True)
@click.pass_context
def info(ctx, label):
    """Show one account's full record (session cookies elided)."""

    async def _info():
        a = await _pool(ctx).get(label)
        d = a.to_rs()
        state = json.loads(d["storage_state"])
        d["storage_state"] = (
            f"<{len(state.get('cookies', []))} cookies, "
            f"{len(state.get('origins', []))} origins>"
        )
        click.echo(json.dumps(d, indent=2, default=str))

    run_async(_info())


@cli.command()
@click.pass_context
def stats(ctx):
    """Show pool counts for this provider."""

    async def _stats():
        s = await _pool(ctx).stats()
        click.echo(f"provider: {ctx.obj['provider']}  db: {default_db_file(ctx.obj['provider'])}")
        click.echo(json.dumps(s, indent=2))

    run_async(_stats())


# ── Mutate ──────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--label", required=True)
@click.pass_context
def delete(ctx, label):
    """Delete an account."""
    run_async(_pool(ctx).delete_account(label))
    click.echo(f"Deleted {label}")


@cli.command()
@click.option("--label", required=True)
@click.pass_context
def activate(ctx, label):
    """Mark an account active."""
    run_async(_pool(ctx).set_active(label, True))


@cli.command()
@click.option("--label", required=True)
@click.pass_context
def deactivate(ctx, label):
    """Mark an account inactive."""
    run_async(_pool(ctx).set_active(label, False))


@cli.command()
@click.option("--label", default=None, help="Account label (all if omitted)")
@click.pass_context
def unlock(ctx, label):
    """Clear the rate-limit lock on an account (or all accounts)."""
    run_async(_pool(ctx).unlock(label))
    click.echo(f"Unlocked {label or 'all accounts'}")


@cli.command()
@click.option("--label", default=None, help="Account label (all if omitted)")
@click.pass_context
def release(ctx, label):
    """Clear the in_use flag on an account (or all) — recover from a crash."""
    run_async(_pool(ctx).release_account(label))
    click.echo(f"Released {label or 'all accounts'}")


@cli.command(name="set")
@click.option("--label", required=True)
@click.argument("field")
@click.argument("value")
@click.pass_context
def set_field(ctx, label, field, value):
    """Update a single field on an account (e.g. `set --label x proxy_server ...`)."""
    run_async(_pool(ctx).update_field(label, field, value))


if __name__ == "__main__":
    cli()
