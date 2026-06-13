"""OpenAI-compatible chat-completions providers: Groq + OpenRouter.

Both speak the OpenAI chat.completions API with strict response_format=json_schema.
A shared ChatCompletionsConversation implements the turn; thin GroqConversation /
OpenRouterConversation subclasses supply the client + cost function. A fallback
path (inline-schema instruction) covers smaller OpenRouter models whose strict
json_schema support is flaky.

OpenRouter pricing is fetched live from /api/v1/models (cached in-process) so
logged cost matches what was actually billed; Groq uses pricing.yaml.

Lazy SDK import so the package imports without the openai package installed.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

from latest.config.loader import get_price
from latest.providers import router
from latest.providers.base import Conversation
from latest.providers.retry import retry_on_rate_limit
from latest.providers.schema_util import make_strict
from latest.records import ProviderResponse

# ───────────────────────────── clients ─────────────────────────────────────

_groq_client = None
_openrouter_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from openai import OpenAI

        _groq_client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
            timeout=30.0,
        )
    return _groq_client


def _get_openrouter_client():
    global _openrouter_client
    if _openrouter_client is None:
        from openai import OpenAI

        _openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            timeout=30.0,
        )
    return _openrouter_client


# ───────────────────────────── pricing ─────────────────────────────────────

_GROQ_FALLBACK = {"input": 0.0, "output": 0.0}


def groq_estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = get_price("groq", model, _GROQ_FALLBACK)
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


_openrouter_pricing: dict | None = None


def _openrouter_pricing_table() -> dict:
    """Fetch /models once -> {model_id: {input,output}} in $/1M tokens."""
    global _openrouter_pricing
    if _openrouter_pricing is not None:
        return _openrouter_pricing
    table: dict = {}
    try:
        with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for m in data.get("data", []):
            mid = m.get("id")
            if not mid:
                continue
            p = m.get("pricing", {}) or {}
            try:
                inp = float(p.get("prompt", 0) or 0) * 1_000_000
                out = float(p.get("completion", 0) or 0) * 1_000_000
            except (TypeError, ValueError):
                inp, out = 0.0, 0.0
            table[mid] = {"input": inp, "output": out}
    except Exception:  # noqa: BLE001 - pricing fetch is best-effort
        table = {}
    _openrouter_pricing = table
    return table


def openrouter_estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = _openrouter_pricing_table().get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


# ───────────────────────── shared conversation ─────────────────────────────


class ChatCompletionsConversation(Conversation):
    """Base for OpenAI-compatible providers. Subclasses set provider + supply
    _client() and _estimate(). Strict json_schema with an inline-schema fallback.
    """

    provider = "chat"

    def _client(self):
        raise NotImplementedError

    def _estimate(self, model, in_tok, out_tok) -> float:
        raise NotImplementedError

    @retry_on_rate_limit
    def _raw_send(self, transcript, schema, max_tokens, temperature) -> ProviderResponse:
        client = self._client()
        schema_dict = make_strict(schema.model_json_schema())

        def _kwargs(messages, response_format=None):
            k = {"model": self.model, "messages": messages, "temperature": temperature}
            if max_tokens is not None:
                k["max_tokens"] = max_tokens
            if response_format is not None:
                k["response_format"] = response_format
            return k

        start = time.monotonic()
        try:
            resp = client.chat.completions.create(**_kwargs(
                transcript,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema.__name__, "schema": schema_dict, "strict": True},
                },
            ))
            raw_text = (resp.choices[0].message.content or "").strip()
            parsed = schema(**json.loads(raw_text))
        except Exception:  # noqa: BLE001 - retry strict-mode failures via inline-schema fallback
            resp = client.chat.completions.create(**_kwargs(
                transcript + [{
                    "role": "system",
                    "content": "Reply with ONLY a JSON object matching this schema, no prose or code fences:\n"
                    + json.dumps(schema_dict),
                }],
            ))
            raw_text = (resp.choices[0].message.content or "").strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`")
                if raw_text.lower().startswith("json"):
                    raw_text = raw_text[4:].lstrip()
            parsed = schema(**json.loads(raw_text))

        latency_ms = (time.monotonic() - start) * 1000
        u = resp.usage
        cost = self._estimate(self.model, u.prompt_tokens, u.completion_tokens)
        return ProviderResponse(
            provider=self.provider,
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


class GroqConversation(ChatCompletionsConversation):
    provider = "groq"

    def _client(self):
        return _get_groq_client()

    def _estimate(self, model, in_tok, out_tok):
        return groq_estimate_cost(model, in_tok, out_tok)


class OpenRouterConversation(ChatCompletionsConversation):
    provider = "openrouter"

    def _client(self):
        return _get_openrouter_client()

    def _estimate(self, model, in_tok, out_tok):
        return openrouter_estimate_cost(model, in_tok, out_tok)


# ───────────────────────────── free-form chat ──────────────────────────────


def _chat(client, provider, estimate, messages, model, max_tokens, temperature) -> ProviderResponse:
    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    start = time.monotonic()
    resp = client.chat.completions.create(**kwargs)
    latency_ms = (time.monotonic() - start) * 1000
    text = (resp.choices[0].message.content or "").strip()
    u = resp.usage
    cost = estimate(model, u.prompt_tokens, u.completion_tokens)
    return ProviderResponse(
        provider=provider, model=model, model_version=getattr(resp, "model", None),
        temperature=temperature, text=text, input_tokens=u.prompt_tokens,
        output_tokens=u.completion_tokens, cost_usd=round(cost, 6), latency_ms=round(latency_ms, 2),
    )


@retry_on_rate_limit
def groq_chat(messages, model, max_tokens=None, temperature=0.0) -> ProviderResponse:
    return _chat(_get_groq_client(), "groq", groq_estimate_cost, messages, model, max_tokens, temperature)


@retry_on_rate_limit
def openrouter_chat(messages, model, max_tokens=2048, temperature=0.0) -> ProviderResponse:
    return _chat(_get_openrouter_client(), "openrouter", openrouter_estimate_cost, messages, model, max_tokens, temperature)


router.register("groq", conversation_cls=GroqConversation, chat_fn=groq_chat, estimate_cost_fn=groq_estimate_cost)
router.register("openrouter", conversation_cls=OpenRouterConversation, chat_fn=openrouter_chat, estimate_cost_fn=openrouter_estimate_cost)
