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

# Hard quota/billing failures are NOT transient — fail fast even on a 429.
_HARD_FAIL_MARKERS = ("spend", "spending cap", "billing", "monthly", "exceeded its")


def _is_retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if any(m in msg for m in _HARD_FAIL_MARKERS):
        return False
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
