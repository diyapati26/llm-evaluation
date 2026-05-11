"""Gemini provider — function-based, no classes.

Uses the new google-genai SDK (`google.genai`), not the legacy google-generativeai.

Structured output: client.models.generate_content with
    config=GenerateContentConfig(response_mime_type="application/json",
                                 response_schema=PydanticModel)
The parsed Pydantic instance comes back via `response.parsed`.

Multi-turn: handled by GeminiConversation (in providers/conversation.py) — keeps
a client-side `contents` list with role="user"/"model" entries.

Pricing values below are approximate $/1M tokens. Verify on
https://ai.google.dev/pricing before reporting paper-final costs.
"""
import os
import time

from google import genai
from google.genai import types

from Output_Formats.output_format import ProviderResponse

PRICING = {
    # Approximate $/1M tokens; verify before paper publication
    "gemini-2.5-pro":         {"input": 1.25,   "output": 10.00},
    "gemini-2.5-flash":       {"input": 0.075,  "output":  0.30},
    "gemini-2.5-flash-lite":  {"input": 0.0375, "output":  0.15},
    "gemini-2.0-flash":       {"input": 0.075,  "output":  0.30},
    "gemini-2.0-flash-lite":  {"input": 0.0375, "output":  0.15},
}

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def estimate_cost(model, input_tokens, output_tokens):
    p = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def _to_gemini_contents(messages):
    """Convert [{role, content}] OpenAI-style messages → Gemini contents list."""
    contents = []
    for m in messages:
        role = "user" if m["role"] in ("user", "system") else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    return contents


def get_gemini_response(prompt, model, output_format, max_tokens=None, temperature=0.0):
    """Call Gemini with a Pydantic output_format. Returns ProviderResponse.

    Uses response_mime_type='application/json' + response_schema for native
    structured output; response.parsed is the Pydantic instance.
    """
    client = _get_client()

    config_kwargs = {
        "response_mime_type": "application/json",
        "response_schema": output_format,
        "temperature": temperature,
    }
    if max_tokens is not None:
        config_kwargs["max_output_tokens"] = max_tokens

    start = time.monotonic()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )

    latency_ms = (time.monotonic() - start) * 1000
    parsed = response.parsed
    usage = response.usage_metadata
    in_tok = usage.prompt_token_count or 0
    out_tok = usage.candidates_token_count or 0
    cost = estimate_cost(model, in_tok, out_tok)

    return ProviderResponse(
        provider="gemini",
        model=model,
        answer=parsed.answer if hasattr(parsed, "answer") else parsed,
        raw=parsed.model_dump() if hasattr(parsed, "model_dump") else None,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=round(cost, 6),
        latency_ms=round(latency_ms, 2),
    )


def get_gemini_chat(messages, model, max_tokens=None, temperature=0.0):
    """Free-form chat (no Pydantic). Returns ProviderResponse with text."""
    client = _get_client()

    config_kwargs = {"temperature": temperature}
    if max_tokens is not None:
        config_kwargs["max_output_tokens"] = max_tokens

    start = time.monotonic()
    response = client.models.generate_content(
        model=model,
        contents=_to_gemini_contents(messages),
        config=types.GenerateContentConfig(**config_kwargs),
    )

    latency_ms = (time.monotonic() - start) * 1000
    text = (response.text or "").strip()
    usage = response.usage_metadata
    in_tok = usage.prompt_token_count or 0
    out_tok = usage.candidates_token_count or 0
    cost = estimate_cost(model, in_tok, out_tok)

    return ProviderResponse(
        provider="gemini",
        model=model,
        text=text,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=round(cost, 6),
        latency_ms=round(latency_ms, 2),
    )
