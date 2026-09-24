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
    scrapers in `chatbots` and the phone-farm one in `phone_chatgpt` — so a caller
    that has a prompt and wants an answer does not care which produced it. The
    fields after `note` are the ones only a scrape *from a handset* fills in.
    """

    provider: str
    prompt: str
    response: str = ""
    # The model the UI actually served (chatgpt/gemini), or the requested model
    # (claude, which has a working picker); None if not applicable — which includes
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

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


def now_iso() -> str:
    """UTC ISO-8601 timestamp — stamped onto results when a task is issued."""
    return datetime.now(timezone.utc).isoformat()
