"""Task + result models shared by the Worker / WorkerPool.

A `Task` is a provider-agnostic unit of work put on the WorkerPool's queue; the
Worker dispatches it to the right scraper by `endpoint`. Google's endpoints
return the existing `AIOverviewResult` / `SearchResult` (from `google_aimode`);
the chatbot endpoint returns a `ChatResult`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import ClassVar


@dataclass
class Task:
    """A unit of scraping work for one provider.

    `endpoint` selects what the Worker does; `payload` carries its inputs:

        google : "ai_mode"  -> {"prompt": str}
                 "search"   -> {"prompt": str, "top_n": int?}
        chatbot: "chat"     -> {"prompt": str, "model": str?}   (chatgpt/claude/gemini/meta)
    """

    ENDPOINT_REQUIRED_FIELDS: ClassVar[dict[str, list[str]]] = {
        "ai_mode": ["prompt"],
        "search": ["prompt"],
        "chat": ["prompt"],
    }

    endpoint: str
    payload: dict
    # Non-serialisable per-call options (e.g. callbacks); excluded from equality
    # and serialisation so a Task stays JSON-safe.
    runtime_options: dict | None = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        if self.endpoint not in self.ENDPOINT_REQUIRED_FIELDS:
            raise ValueError(
                f"Unsupported endpoint: '{self.endpoint}'. "
                f"Supported: {list(self.ENDPOINT_REQUIRED_FIELDS)}"
            )
        missing = [
            f
            for f in self.ENDPOINT_REQUIRED_FIELDS[self.endpoint]
            if f not in self.payload
        ]
        if missing:
            raise ValueError(
                f"Task '{self.endpoint}' missing required payload fields: {missing}"
            )

    def to_dict(self) -> dict:
        return {"endpoint": self.endpoint, "payload": self.payload}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class ChatResult:
    """Result of one chatbot turn (ChatGPT / Claude / Gemini / Meta).

    One shape for every way a chatbot can be driven — the desktop camoufox
    scrapers in `chatbots` and the phone-farm ones in `phone_chatgpt` and
    `phone_claude` — so a caller
    that has a prompt and wants an answer does not care which produced it. The
    fields after `note` are the ones only a scrape *from a handset* fills in.
    """

    provider: str
    prompt: str
    response: str = ""
    # The model the UI actually served (chatgpt/gemini, and claude on the phone
    # farm), or the requested model (desktop claude, which has a working picker);
    # None if not applicable — which includes
    # anonymous mobile-web ChatGPT, where the DOM names no model at all.
    served_model: str | None = None
    requested_model: str | None = None
    # Whether the account's memory could shape this answer. False for an anonymous
    # ask (no account, so no memory); None where nobody read the setting.
    memory_enabled: bool | None = None
    # Invalid-response signals — not real model output; the run should stop.
    rate_limited: bool = False
    l2_block: bool = False
    note: str = ""
    scraped_at: str = ""
    # Sources the answer cited, when it cited any (ChatGPT links them once it has
    # searched the web). Same `Reference` shape the Google scrapers produce —
    # deliberately, so the two can be compared without a translation step. Typed
    # loosely to keep this module free of a dependency on `google_aimode`.
    references: list = field(default_factory=list)
    # The ask never reached a model — an anti-bot wall or a usage cap, not a
    # refusal. Named as in `AIOverviewResult` so a pool can rotate handsets on
    # either without knowing which surface it is holding.
    blocked: bool = False
    # Which machinery answered ("phone_farm"), and which handset, when one did.
    surface: str = ""
    serial: str = ""
    # Multi-turn asks (`ask_conversation`). A single ask is turn 1 of a one-turn
    # conversation, so these defaults describe it without anything being set.
    turn: int = 1
    # The page the conversation lives at once sent (chatgpt.com/c/<uuid>), which is
    # what ties the turns of one conversation together in the account's history.
    conversation_url: str = ""
    # How a follow-up turn's prompt was written: generator model, prompt-template
    # hash, style, and the transcript the generator saw. Empty on turn 1, whose
    # prompt came from the bank. See `aiscrape.followups`.
    followup: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


@dataclass
class FollowUp:
    """The next user message of a conversation, and how it was written."""

    text: str
    # Provenance kept on the turn's row: re-running a generator does not reproduce
    # its text, so the stored text and this record are the only account of it.
    meta: dict = field(default_factory=dict)


@dataclass
class Conversation:
    """The turns of one multi-turn ask, in order.

    `turns[0]` is the bank's prompt; each later turn's prompt came from a follow-up
    generator. A conversation that stopped early (a turn walled or came back empty,
    or the generator gave up) keeps the turns it has, with `complete` False and
    `stop_reason` saying why -- the turns that were answered are still evidence.
    """

    turns: list[ChatResult] = field(default_factory=list)
    planned: int = 1
    stop_reason: str = ""

    @property
    def complete(self) -> bool:
        return (len(self.turns) == self.planned and not self.stop_reason
                and all(t.response for t in self.turns))

    # The pool rotates handsets on these, exactly as on a `ChatResult`. Only the
    # first turn decides: a wall on turn 1 means no conversation happened and the
    # prompt belongs on another phone; a wall later is a conversation that stopped
    # partway, and re-asking it elsewhere would give a second, different turn 1.
    @property
    def blocked(self) -> bool:
        return bool(self.turns) and self.turns[0].blocked

    @property
    def note(self) -> str:
        return self.turns[0].note if self.turns else self.stop_reason

    @property
    def response(self) -> str:
        return self.turns[0].response if self.turns else ""


def now_iso() -> str:
    """UTC ISO-8601 timestamp — stamped onto results when a task is issued."""
    return datetime.now(timezone.utc).isoformat()
