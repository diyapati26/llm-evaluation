"""Gemini provider (google-genai SDK).

Structured output via generate_content(response_mime_type="application/json",
response_schema=Schema) -> response.parsed. Client-side transcript converted to
Gemini's contents format (role user/system -> "user", assistant -> "model").
Client carries an explicit 60s timeout so a stalled connection errors (and is
retried) instead of hanging a worker thread forever.

Lazy SDK import so the package imports without google-genai installed.
"""
from __future__ import annotations

import os
import time

from latest.config.loader import get_price
from latest.providers import router
from latest.providers.base import Conversation
from latest.providers.retry import retry_on_rate_limit
from latest.records import ProviderResponse

_FALLBACK = {"input": 0.0, "output": 0.0}
_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        from google.genai import types

        _client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(timeout=60_000),
        )
    return _client


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = get_price("gemini", model, _FALLBACK)
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def _to_gemini_contents(transcript):
    """[{role, content}] -> Gemini contents [{role: user|model, parts:[{text}]}]."""
    contents = []
    for m in transcript:
        role = "user" if m["role"] in ("user", "system") else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    return contents


class GeminiConversation(Conversation):
    provider = "gemini"

    @retry_on_rate_limit
    def _raw_send(self, transcript, schema, max_tokens, temperature) -> ProviderResponse:
        from google.genai import types

        client = _get_client()
        cfg = {
            "response_mime_type": "application/json",
            "response_schema": schema,
            "temperature": temperature,
        }
        if max_tokens is not None:
            cfg["max_output_tokens"] = max_tokens

        start = time.monotonic()
        resp = client.models.generate_content(
            model=self.model,
            contents=_to_gemini_contents(transcript),
            config=types.GenerateContentConfig(**cfg),
        )
        latency_ms = (time.monotonic() - start) * 1000

        parsed = resp.parsed
        u = resp.usage_metadata
        in_tok = u.prompt_token_count or 0
        out_tok = u.candidates_token_count or 0
        cost = estimate_cost(self.model, in_tok, out_tok)
        return ProviderResponse(
            provider="gemini",
            model=self.model,
            model_version=getattr(resp, "model_version", None),
            temperature=temperature,
            answer=parsed,
            raw=parsed.model_dump(),
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=round(cost, 6),
            latency_ms=round(latency_ms, 2),
        )


@retry_on_rate_limit
def chat(messages, model, max_tokens=None, temperature=0.0) -> ProviderResponse:
    """Single-turn free-form chat (judges, moral scenarios)."""
    from google.genai import types

    client = _get_client()
    cfg = {"temperature": temperature}
    if max_tokens is not None:
        cfg["max_output_tokens"] = max_tokens

    start = time.monotonic()
    resp = client.models.generate_content(
        model=model,
        contents=_to_gemini_contents(messages),
        config=types.GenerateContentConfig(**cfg),
    )
    latency_ms = (time.monotonic() - start) * 1000

    text = (resp.text or "").strip()
    u = resp.usage_metadata
    in_tok = u.prompt_token_count or 0
    out_tok = u.candidates_token_count or 0
    cost = estimate_cost(model, in_tok, out_tok)
    return ProviderResponse(
        provider="gemini",
        model=model,
        model_version=getattr(resp, "model_version", None),
        temperature=temperature,
        text=text,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=round(cost, 6),
        latency_ms=round(latency_ms, 2),
    )


router.register("gemini", conversation_cls=GeminiConversation, chat_fn=chat, estimate_cost_fn=estimate_cost)
