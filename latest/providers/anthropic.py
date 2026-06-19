"""Anthropic provider.

Structured output via client.messages.parse(output_format=Schema) -> .parsed_output
(verified present in anthropic 0.109). Client-side transcript (Anthropic has no
server-side conversation state by design). Anthropic REQUIRES max_tokens, so a
sane default is supplied when the caller passes None.

Lazy SDK import so the package imports without the anthropic package installed.
"""
from __future__ import annotations

import os
import time

from latest.config.loader import get_price
from latest.providers import router
from latest.providers.base import Conversation
from latest.providers.retry import retry_on_rate_limit
from latest.records import ProviderResponse

_FALLBACK = {"input": 3.00, "output": 15.00}
# Anthropic's API REQUIRES max_tokens — can't be removed like the other providers.
# Per Harsha 2026-06-19 ("if not possible remove for all"): use a high floor so it's
# effectively uncapped for our short answers, and ignore any smaller caller cap.
_DEFAULT_MAX_TOKENS = 8192
_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=30.0)
    return _client


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = get_price("anthropic", model, _FALLBACK)
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


class AnthropicConversation(Conversation):
    provider = "anthropic"

    @retry_on_rate_limit
    def _raw_send(self, transcript, schema, max_tokens, temperature) -> ProviderResponse:
        client = _get_client()
        start = time.monotonic()
        # temperature omitted (deprecated/rejected on newer snapshots e.g. opus-4-8);
        # max_tokens forced to the high floor (ignore caller cap).
        resp = client.messages.parse(
            model=self.model,
            max_tokens=_DEFAULT_MAX_TOKENS,
            output_format=schema,
            messages=transcript,
        )
        latency_ms = (time.monotonic() - start) * 1000

        parsed = resp.parsed_output
        u = resp.usage
        cost = estimate_cost(self.model, u.input_tokens, u.output_tokens)
        return ProviderResponse(
            provider="anthropic",
            model=self.model,
            model_version=getattr(resp, "model", None),
            temperature=temperature,
            answer=parsed,
            raw=parsed.model_dump(),
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cost_usd=round(cost, 6),
            latency_ms=round(latency_ms, 2),
        )


@retry_on_rate_limit
def chat(messages, model, max_tokens=None, temperature=0.0) -> ProviderResponse:
    """Single-turn free-form chat (judges, moral scenarios)."""
    client = _get_client()
    start = time.monotonic()
    # temperature omitted; max_tokens forced to the high floor — see _raw_send note.
    resp = client.messages.create(
        model=model,
        max_tokens=_DEFAULT_MAX_TOKENS,
        messages=messages,
    )
    latency_ms = (time.monotonic() - start) * 1000

    # Concatenate all text blocks, skipping any leading thinking/redacted-thinking
    # blocks (which have no .text) — robust to extended-thinking judge snapshots.
    text = "".join(b.text for b in (resp.content or []) if getattr(b, "type", None) == "text").strip()
    u = resp.usage
    cost = estimate_cost(model, u.input_tokens, u.output_tokens)
    return ProviderResponse(
        provider="anthropic",
        model=model,
        model_version=getattr(resp, "model", None),
        temperature=temperature,
        text=text,
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cost_usd=round(cost, 6),
        latency_ms=round(latency_ms, 2),
    )


router.register("anthropic", conversation_cls=AnthropicConversation, chat_fn=chat, estimate_cost_fn=estimate_cost)
