"""
Uses Anthropic's structured-output API: client.messages.parse(output_format=PydanticModel).
"""

import os
import time

import anthropic

import pricing
from schemas import ProviderResponse

# Authoritative prices live in config/pricing.yaml; this dict is the offline
# fallback. Kept in sync with the YAML (verified 2026-05-30).
PRICING = {
    "claude-opus-4-7":           {"input":  5.00, "output": 25.00},
    "claude-opus-4-6":           {"input":  5.00, "output": 25.00},
    "claude-sonnet-4-6":         {"input":  3.00, "output": 15.00},
    "claude-sonnet-4-5":         {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5":          {"input":  1.00, "output":  5.00},
    "claude-haiku-4-5-20251001": {"input":  1.00, "output":  5.00},
}

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=30.0)
    return _client


def estimate_cost(model, input_tokens, output_tokens):
    p = pricing.get_price("anthropic", model, PRICING.get(model, {"input": 3.00, "output": 15.00}))
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def get_anthropic_response(prompt, model, output_format, max_tokens=4096, temperature=0.0):
    """Call Anthropic with a Pydantic output schema. Returns ProviderResponse.

    Anthropic's API REQUIRES max_tokens — unlike OpenAI/Gemini there's no
    "let the model decide". Default is 4096 (plenty for typed answers).
    """
    client = _get_client()
    start = time.monotonic()
    response = client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        output_format=output_format,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = (time.monotonic() - start) * 1000
    parsed = response.parsed_output
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = estimate_cost(model, input_tokens, output_tokens)

    return ProviderResponse(
        provider="anthropic",
        model=model,
        model_version=getattr(response, "model", None),
        temperature=temperature,
        answer=parsed.answer if hasattr(parsed, "answer") else parsed,
        raw=parsed.model_dump(),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost, 6),
        latency_ms=round(latency_ms, 2),
    )


def get_anthropic_chat(messages, model, max_tokens=4096, temperature=0.0):
    """Free-form chat (no Pydantic). Anthropic API requires max_tokens (default 4096)."""
    client = _get_client()
    start = time.monotonic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
    )

    latency_ms = (time.monotonic() - start) * 1000
    text = response.content[0].text.strip()
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = estimate_cost(model, input_tokens, output_tokens)

    return ProviderResponse(
        provider="anthropic",
        model=model,
        model_version=getattr(response, "model", None),
        temperature=temperature,
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost, 6),
        latency_ms=round(latency_ms, 2),
    )
