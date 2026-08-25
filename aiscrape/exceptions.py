"""Custom exceptions for the aiscrape account pool.

These map loosely onto the ways a scrape against an AI product can come back
invalid — the same signals the scrapers already detect (`detect_rate_limit`,
`detect_l2_classifier`, Google's CAPTCHA `blocked=True`) — so the Worker can
decide whether to lock-and-rotate, mark inactive, or surface the failure.
"""


class AIScraperError(Exception):
    """Base exception for aiscrape."""


class NoAccountError(AIScraperError):
    """No account available in the pool for this provider."""


class RateLimitError(AIScraperError):
    """The provider served a rate-limit banner / throttle.

    Not real model output; the account should be locked out for a cooldown and
    the task retried on a rotated account.
    """


class AccountBlockedError(AIScraperError):
    """The provider applied an application-layer (L2) safety block.

    Distinct from an in-context RLHF refusal (which is valid model output) — an
    L2 block terminates the conversation, so the run is aborted for this account.
    """


class SessionExpiredError(AIScraperError):
    """The saved session is no longer logged in.

    Covers Google's 'unusual traffic' / CAPTCHA interstitial (which it serves to
    fresh/expired sessions) and any chatbot that bounces to a login wall — the
    account needs its session re-saved via `aiscrape.auth`.
    """
