"""retry.py — exponential-backoff retry for transient provider errors.

Carried from the proven Simpler Arch implementation. Wraps each provider's
network call so a rate-limit (429 / RESOURCE_EXHAUSTED / TPM cap) or transient
overload is retried with exponential backoff + jitter instead of dropping the
sample. Hard quota/billing failures fail fast (retrying just backs off against a
wall). Applied at the leaf (each provider's raw send) so there is exactly one
retry layer — no nesting.
"""
from __future__ import annotations

import tenacity

# Substrings marking a transient, retry-worthy error across providers.
_RETRYABLE_MARKERS = (
    "rate limit", "rate_limit", "ratelimit",
    "429", "resource_exhausted", "too many requests",
    "overloaded", "503", "529", "service unavailable",
    "timeout", "timed out",
)

# Hard quota/billing walls — NOT transient; retrying just backs off against a wall.
# High-precision phrases only: a bare "monthly"/"spend" would false-match transient
# 429 bodies like "you've hit your monthly request limit, retry shortly".
_HARD_FAIL_MARKERS = (
    "insufficient_quota", "quota exceeded", "exceeded your current quota",
    "spending cap", "spending limit", "billing hard limit", "credit balance", "hard limit",
)


def _is_hard_fail(exc: BaseException) -> bool:
    """A genuine quota/billing wall: a hard-fail phrase AND no transient signal.

    The 'and not retryable' gate ensures a rate-limit message that merely mentions
    a quota in passing still wins as retryable. Shared by the collect engine's
    budget gate so retry and budget detection can never disagree.
    """
    msg = str(exc).lower()
    return any(m in msg for m in _HARD_FAIL_MARKERS) and not any(m in msg for m in _RETRYABLE_MARKERS)


def _is_retryable(exc: BaseException) -> bool:
    if _is_hard_fail(exc):
        return False
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    if "ratelimit" in name or "overloaded" in name:
        return True
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if code in (429, 503, 529):
        return True
    return any(m in msg for m in _RETRYABLE_MARKERS)


# Up to 6 attempts; backoff ~2,4,8,16,32s (capped 60) + jitter. reraise so the
# real error surfaces to the caller's per-call error handling if all retries fail.
retry_on_rate_limit = tenacity.retry(
    retry=tenacity.retry_if_exception(_is_retryable),
    wait=tenacity.wait_random_exponential(multiplier=2, max=60),
    stop=tenacity.stop_after_attempt(6),
    reraise=True,
)
