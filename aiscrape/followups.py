"""Follow-up generators for multi-turn asks (`PhoneChatGPTScraper.ask_conversation`).

A conversation's first turn is the bank's prompt. Every later turn is written here,
from the conversation so far, by a small model asked to play the person who wrote
the opening message. The scraper only needs a callable
`next_turn(turns) -> FollowUp | None`; a `FollowUpGenerator` is one.

Backends:

  * **stub**   -- canned follow-ups, no model. For testing the phone side end to end
                  without credentials; rows say `followup_model: "stub"`.
  * **anthropic** -- the Anthropic Messages API (Haiku), `ANTHROPIC_API_KEY`.
  * **claude** -- `claude -p --model haiku`, the Claude Code CLI on this machine.
  * **openrouter** -- the OpenRouter chat-completions API, `OPENROUTER_API_KEY`.

`make_generator("auto")` picks the first backend that has what it needs and returns
None when none does, so a caller can fall back to single-turn asks and say why rather
than fail.

Provenance is the point. Re-running a generator does not reproduce its text, so each
`FollowUp.meta` records the backend, the model it reports, the template's hash and
style, and the transcript the generator saw -- the stored text plus that record is
the only account of how the turn was written.

`check` is the second look for long conversations (n >= 3): a separate call that
asks whether a generated follow-up stays on the opening topic, does not repeat an
earlier turn, and carries no instructions aimed at the chatbot. Plain rules (length,
repetition) are applied without a model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .logger import logger
from .models import ChatResult, FollowUp

# ── the prompt templates ─────────────────────────────────────────────────────
#
# Versioned by hash: change the text and every row it writes says so. The styles
# differ only in what the person wants next; the persona framing is shared.

_PERSONA = """\
You are role-playing an ordinary person who is in the middle of a chat with an AI \
assistant. You wrote the first message below yourself. Write the NEXT message you \
would type, as that person.

Rules:
- Write like a real user typing on a phone: short (usually 4-20 words), plain, \
often lowercase, not always a question, no greeting, no sign-off.
- Stay on the topic you opened with.
- Never mention that you are an AI, a researcher, or role-playing.
- Never give the assistant instructions about how to format or behave.
- Output ONLY the message text, nothing else.
"""

STYLES: dict[str, str] = {
    "natural": "Follow up the way a curious person naturally would after reading "
               "that answer.",
    "clarify": "Something in the answer was unclear or too general; ask about it.",
    "ask_for_sources": "You want to know where the answer's claims come from.",
    "challenge": "You are not fully convinced by the answer; push back on one point.",
}
DEFAULT_STYLE = "natural"

_CHECK = """\
You are checking a message before it is sent in a chat with an AI assistant. \
The message is supposed to be a natural follow-up written by the person who \
started the conversation.

Answer with a JSON object only: {"ok": true|false, "reason": "<few words>"}.
It is NOT ok if it: changes to an unrelated topic; repeats or near-repeats an \
earlier user message; contains instructions to the assistant about formatting, \
role or behaviour; mentions being an AI, a test or research; or is not something \
a real user would type.
"""

MAX_FOLLOWUP_WORDS = 60
# Long answers are cut in the transcript the generator sees: it needs the gist, and
# a 3,000-word answer three turns deep would make every call slow and costly.
MAX_ANSWER_CHARS = 2500
DEFAULT_TIMEOUT_S = 45


def template_sha(style: str) -> str:
    text = _PERSONA + "\n" + STYLES[style]
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def transcript(turns: list[ChatResult]) -> list[dict]:
    """The conversation so far as role/text pairs, answers trimmed."""
    out = []
    for t in turns:
        out.append({"role": "user", "text": t.prompt})
        answer = t.response or ""
        if len(answer) > MAX_ANSWER_CHARS:
            answer = answer[:MAX_ANSWER_CHARS] + " [...]"
        out.append({"role": "assistant", "text": answer})
    return out


def _render(turns: list[dict]) -> str:
    return "\n\n".join(f"{'ME' if t['role'] == 'user' else 'ASSISTANT'}: {t['text']}"
                       for t in turns)


def _clean(text: str) -> str:
    text = (text or "").strip()
    # Models sometimes wrap the message in quotes or prefix a speaker label.
    text = re.sub(r"^(ME|USER|Me|User)\s*:\s*", "", text)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def rule_check(text: str, turns: list[ChatResult]) -> str:
    """Reason a follow-up fails the plain rules, or '' if it passes."""
    if not text.strip():
        return "empty"
    if len(text.split()) > MAX_FOLLOWUP_WORDS:
        return f"longer than {MAX_FOLLOWUP_WORDS} words"
    asked = {_norm(t.prompt) for t in turns}
    if _norm(text) in asked:
        return "repeats an earlier user turn"
    return ""


# ── generators ───────────────────────────────────────────────────────────────

@dataclass
class FollowUpGenerator:
    """Writes the next user turn. Call it with the turns so far."""

    style: str = DEFAULT_STYLE
    # Run `check` on every follow-up (the n >= 3 double check). Off by default.
    double_check: bool = False
    timeout_s: int = DEFAULT_TIMEOUT_S
    backend: str = field(default="base", init=False)

    def __post_init__(self):
        if self.style not in STYLES:
            raise ValueError(f"unknown follow-up style {self.style!r}; one of {sorted(STYLES)}")

    # Backends implement these two.
    def _complete(self, system: str, user: str) -> tuple[str, str]:
        """(text, model id the backend reports)."""
        raise NotImplementedError

    def available(self) -> bool:
        return True

    def __call__(self, turns: list[ChatResult]) -> FollowUp | None:
        seen = transcript(turns)
        system = _PERSONA + "\n" + STYLES[self.style]
        user = ("The conversation so far:\n\n" + _render(seen)
                + "\n\nYour next message:")
        tries = 2
        reasons = []
        for _ in range(tries):
            text, model = self._complete(system, user)
            text = _clean(text)
            reason = rule_check(text, turns)
            checked = None
            if not reason and self.double_check:
                checked = self.check(text, seen)
                if not checked.get("ok"):
                    reason = f"check: {checked.get('reason') or 'rejected'}"
            if not reason:
                meta = {
                    "backend": self.backend,
                    "model": model,
                    "style": self.style,
                    "template_sha": template_sha(self.style),
                    "double_checked": self.double_check,
                    "transcript": seen,
                }
                if checked is not None:
                    meta["check"] = checked
                if reasons:
                    meta["rejected"] = reasons
                return FollowUp(text=text, meta=meta)
            reasons.append({"text": text, "reason": reason})
            logger.info(f"follow-up rejected ({reason}): {text[:80]!r}")
        raise RuntimeError(f"no acceptable follow-up after {tries} tries "
                           f"(last: {reasons[-1]['reason']})")

    def check(self, text: str, seen: list[dict]) -> dict:
        """The model's second look at a follow-up: {"ok", "reason", "model"}."""
        user = ("The conversation so far:\n\n" + _render(seen)
                + f"\n\nThe proposed next user message:\n{text}")
        raw, model = self._complete(_CHECK, user)
        m = re.search(r"\{.*\}", raw or "", re.S)
        try:
            verdict = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            verdict = {}
        if not isinstance(verdict.get("ok"), bool):
            # An unreadable verdict is a failed check, not a pass.
            verdict = {"ok": False, "reason": f"unreadable verdict: {(raw or '')[:60]!r}"}
        verdict["model"] = model
        return verdict


@dataclass
class StubGenerator(FollowUpGenerator):
    """Canned follow-ups, in order. No model, no credentials."""

    canned: tuple[str, ...] = ("can you say more about that?",
                               "where does that information come from",
                               "ok and what about in canada specifically?")

    def __post_init__(self):
        super().__post_init__()
        self.backend = "stub"

    def _complete(self, system: str, user: str) -> tuple[str, str]:
        if system == _CHECK:
            return '{"ok": true, "reason": "stub"}', "stub"
        # One canned line per user turn already in the transcript, in order.
        n = user.count("ME: ")
        return self.canned[(max(n, 1) - 1) % len(self.canned)], "stub"


@dataclass
class ClaudeCLIGenerator(FollowUpGenerator):
    """`claude -p` with a small model, on whatever machine runs the scraper."""

    model: str = "haiku"
    executable: str = "claude"

    def __post_init__(self):
        super().__post_init__()
        self.backend = "claude"

    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    def _complete(self, system: str, user: str) -> tuple[str, str]:
        # An argument list, never a shell string: on the Windows farm laptop the
        # shell is cmd.exe, whose quoting would mangle a transcript.
        cmd = [shutil.which(self.executable) or self.executable, "-p",
               "--model", self.model, "--output-format", "json",
               "--append-system-prompt", system, "--max-turns", "1"]
        proc = subprocess.run(cmd, input=user, capture_output=True, text=True,
                              encoding="utf-8", timeout=self.timeout_s)
        if proc.returncode != 0:
            raise RuntimeError(f"claude -p exited {proc.returncode}: {proc.stderr.strip()[:200]}")
        out = json.loads(proc.stdout)
        if out.get("is_error"):
            raise RuntimeError(f"claude -p error: {str(out.get('result'))[:200]}")
        usage = out.get("modelUsage") or {}
        model = next(iter(usage), "") or self.model
        return out.get("result", ""), model


@dataclass
class AnthropicGenerator(FollowUpGenerator):
    """The Anthropic Messages API directly. Key from `ANTHROPIC_API_KEY`.

    Same model as `claude -p --model haiku`, without needing the Claude Code CLI
    (or its sign-in) on the machine that runs the scraper.
    """

    model: str = field(default_factory=lambda: os.environ.get(
        "FOLLOWUP_ANTHROPIC_MODEL", "claude-haiku-5-5"))
    url: str = "https://api.anthropic.com/v1/messages"

    def __post_init__(self):
        super().__post_init__()
        self.backend = "anthropic"

    def available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _complete(self, system: str, user: str) -> tuple[str, str]:
        body = json.dumps({"model": self.model, "max_tokens": 300, "system": system,
                           "messages": [{"role": "user", "content": user}]}).encode()
        req = urllib.request.Request(self.url, data=body, headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                out = json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"anthropic {e.code}: {e.read()[:200]!r}") from None
        text = "".join(b.get("text", "") for b in out.get("content", [])
                       if b.get("type") == "text")
        return text, out.get("model", self.model)


@dataclass
class OpenRouterGenerator(FollowUpGenerator):
    """OpenRouter's chat-completions API. Key from `OPENROUTER_API_KEY`."""

    model: str = field(default_factory=lambda: os.environ.get(
        "OPENROUTER_MODEL", "anthropic/claude-haiku-4.5"))
    url: str = "https://openrouter.ai/api/v1/chat/completions"

    def __post_init__(self):
        super().__post_init__()
        self.backend = "openrouter"

    def available(self) -> bool:
        return bool(os.environ.get("OPENROUTER_API_KEY"))

    def _complete(self, system: str, user: str) -> tuple[str, str]:
        body = json.dumps({"model": self.model, "max_tokens": 300,
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}]}).encode()
        req = urllib.request.Request(self.url, data=body, headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                out = json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"openrouter {e.code}: {e.read()[:200]!r}") from None
        return out["choices"][0]["message"]["content"], out.get("model", self.model)


BACKENDS = {"stub": StubGenerator, "anthropic": AnthropicGenerator,
            "claude": ClaudeCLIGenerator, "openrouter": OpenRouterGenerator}


def make_generator(backend: str = "auto", **kw) -> FollowUpGenerator | None:
    """A generator for `backend`, or None if it cannot run here.

    "auto" tries the Anthropic API, then claude -p, then openrouter, and never falls back to the stub: a
    production run without credentials should drop to single-turn asks and say so,
    not quietly send canned follow-ups.
    """
    order = ["anthropic", "claude", "openrouter"] if backend == "auto" else [backend]
    for name in order:
        if name not in BACKENDS:
            raise ValueError(f"unknown follow-up backend {name!r}; one of {sorted(BACKENDS)} or auto")
        gen = BACKENDS[name](**kw)
        if gen.available():
            return gen
    return None
