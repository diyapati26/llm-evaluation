"""OpenAI provider.

Multi-turn structured output via chat.completions.parse (native messages list +
Pydantic response_format -> .parsed). This replaces the old server-side
Responses/Conversations path so OpenAI uses the same client-side transcript as
every other provider, which is what makes the content-addressed cache correct.

Free-form chat (judges/moral) via chat.completions.create.
Lazy SDK import so the package imports without the openai package installed.
"""
from __future__ import annotations

import os
import time

from latest.config.loader import get_price
from latest.providers import router
from latest.providers.base import Conversation
from latest.providers.retry import retry_on_rate_limit
from latest.records import ProviderResponse

_FALLBACK = {"input": 0.75, "output": 4.50}
_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=30.0)
    return _client


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = get_price("openai", model, _FALLBACK)
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


class OpenAIConversation(Conversation):
    provider = "openai"

    @retry_on_rate_limit
    def _raw_send(self, transcript, schema, max_tokens, temperature) -> ProviderResponse:
        client = _get_client()
        kwargs = {
            "model": self.model,
            "messages": transcript,
            "response_format": schema,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens

        start = time.monotonic()
        resp = client.chat.completions.parse(**kwargs)
        latency_ms = (time.monotonic() - start) * 1000

        parsed = resp.choices[0].message.parsed
        u = resp.usage
        cost = estimate_cost(self.model, u.prompt_tokens, u.completion_tokens)
        return ProviderResponse(
            provider="openai",
            model=self.model,
            model_version=getattr(resp, "model", None),
            temperature=temperature,
            answer=parsed,
            raw=parsed.model_dump(),
            input_tokens=u.prompt_tokens,
            output_tokens=u.completion_tokens,
            cost_usd=round(cost, 6),
            latency_ms=round(latency_ms, 2),
        )


@retry_on_rate_limit
def chat(messages, model, max_tokens=None, temperature=0.0) -> ProviderResponse:
    """Single-turn free-form chat (judges, moral scenarios)."""
    client = _get_client()
    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens

    start = time.monotonic()
    resp = client.chat.completions.create(**kwargs)
    latency_ms = (time.monotonic() - start) * 1000

    text = (resp.choices[0].message.content or "").strip()
    u = resp.usage
    cost = estimate_cost(model, u.prompt_tokens, u.completion_tokens)
    return ProviderResponse(
        provider="openai",
        model=model,
        model_version=getattr(resp, "model", None),
        temperature=temperature,
        text=text,
        input_tokens=u.prompt_tokens,
        output_tokens=u.completion_tokens,
        cost_usd=round(cost, 6),
        latency_ms=round(latency_ms, 2),
    )


router.register("openai", conversation_cls=OpenAIConversation, chat_fn=chat, estimate_cost_fn=estimate_cost)
