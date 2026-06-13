"""Exponential-backoff retry for transient provider errors.

Wraps each provider's network call so a rate-limit (429 / RESOURCE_EXHAUSTED /
TPM cap) or transient overload is retried with exponential backoff + jitter,
instead of dropping the sample. After the attempt budget is exhausted the
original exception is re-raised, so callers' existing per-sample error handling
still applies.

Applied at the leaf level (each get_*_response / get_*_chat and every
Conversation.send) so each network call has exactly one retry layer — no nesting.
"""
import tenacity

# Substrings marking a transient, retry-worthy error across providers
# (OpenAI/Groq RateLimitError, Anthropic 429/529 overloaded, Gemini RESOURCE_EXHAUSTED).
_RETRYABLE_MARKERS = (
    "rate limit", "rate_limit", "ratelimit",
    "429", "resource_exhausted", "too many requests",
    "overloaded", "503", "529", "service unavailable",
    "timeout", "timed out",
)


# Hard quota/billing failures are NOT transient — retrying just wastes time
# backing off against a wall. Fail fast on these even if the HTTP code is 429.
_HARD_FAIL_MARKERS = ("spend", "spending cap", "billing", "monthly", "exceeded its")


def _is_retryable(exc):
    msg = str(exc).lower()
    if any(m in msg for m in _HARD_FAIL_MARKERS):
        return False  # e.g. "exceeded its monthly spending cap" — won't clear on retry
    name = type(exc).__name__.lower()
    if "ratelimit" in name or "overloaded" in name:
        return True
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if code in (429, 503, 529):
        return True
    return any(m in msg for m in _RETRYABLE_MARKERS)


# Up to 6 attempts; backoff ~2,4,8,16,32s (capped at 60) with random jitter.
# reraise=True so the real error surfaces to the caller if all retries fail.
retry_on_rate_limit = tenacity.retry(
    retry=tenacity.retry_if_exception(_is_retryable),
    wait=tenacity.wait_random_exponential(multiplier=2, max=60),
    stop=tenacity.stop_after_attempt(6),
    reraise=True,
)
