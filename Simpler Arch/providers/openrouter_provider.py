"""OpenRouter provider — function-based, no classes.

Uses OpenAI SDK pointed at OpenRouter's OpenAI-compatible endpoint. Single key,
hundreds of models from Anthropic, OpenAI, Meta, Mistral, DeepSeek, Qwen, xAI,
etc., namespaced like "anthropic/claude-3.5-sonnet" or "qwen/qwen-2.5-72b-instruct".

Pricing is fetched once from https://openrouter.ai/api/v1/models on first cost
calc and cached in-process — live prices rather than a hardcoded dict, so the
costs in your results match what OpenRouter actually billed on the day of run.

STRICT MODE ONLY: structured-output requests use response_format=json_schema
with strict=True. If the chosen model doesn't support strict json_schema, the
call fails — we don't best-effort parse, since silent fallback can produce
garbage results that look successful. Pick a model that supports strict mode
(Claude 3.5+, GPT-4o, Llama 3.3 70B, Qwen 72B, DeepSeek-V3, etc.).
"""
import os
import json
import time
import urllib.request

from openai import OpenAI

from Output_Formats.output_format import ProviderResponse


_client = None
_pricing_cache = None  # {model_id: {"input": $/Mtok, "output": $/Mtok}}


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    return _client


def _load_pricing():
    """Fetch /models once, build model_id -> {input,output} dict in $/1M tokens.

    OpenRouter returns prompt/completion prices in $/token. We multiply by 1M
    so the unit matches PRICING dicts in the other providers.
    """
    global _pricing_cache
    if _pricing_cache is not None:
        return _pricing_cache

    with urllib.request.urlopen(
        "https://openrouter.ai/api/v1/models", timeout=10
    ) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    pricing = {}
    for m in data.get("data", []):
        model_id = m.get("id")
        if not model_id:
            continue
        p = m.get("pricing", {}) or {}
        try:
            inp = float(p.get("prompt", 0) or 0) * 1_000_000
            out = float(p.get("completion", 0) or 0) * 1_000_000
        except (TypeError, ValueError):
            inp, out = 0.0, 0.0
        pricing[model_id] = {"input": inp, "output": out}

    _pricing_cache = pricing
    return pricing


def estimate_cost(model, input_tokens, output_tokens):
    p = _load_pricing().get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def _make_strict(node):
    """Recursively pin additionalProperties=False on every object node. Required
    when the underlying provider (Claude, Groq, etc.) enforces it for strict mode.
    """
    if isinstance(node, dict):
        if node.get("type") == "object" and "additionalProperties" not in node:
            node["additionalProperties"] = False
        for v in node.values():
            _make_strict(v)
    elif isinstance(node, list):
        for v in node:
            _make_strict(v)
    return node


def get_openrouter_response(prompt, model, output_format, max_tokens=None, temperature=0.0):
    """Call OpenRouter with strict json_schema. Returns ProviderResponse.

    Strict mode only — raises if the underlying model doesn't support
    response_format=json_schema. max_tokens=None lets the model use its full
    output budget.
    """
    client = _get_client()
    schema = _make_strict(output_format.model_json_schema())

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": output_format.__name__,
                "schema": schema,
                "strict": True,
            },
        },
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    start = time.monotonic()
    response = client.chat.completions.create(**kwargs)
    parsed = output_format(**json.loads(response.choices[0].message.content))

    latency_ms = (time.monotonic() - start) * 1000
    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    cost = estimate_cost(model, input_tokens, output_tokens)

    return ProviderResponse(
        provider="openrouter",
        model=model,
        answer=parsed.answer if hasattr(parsed, "answer") else parsed,
        raw=parsed.model_dump() if hasattr(parsed, "model_dump") else None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost, 6),
        latency_ms=round(latency_ms, 2),
    )


def get_openrouter_chat(messages, model, max_tokens=None, temperature=0.0):
    """Free-form chat (no Pydantic). Returns ProviderResponse with text."""
    client = _get_client()

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    start = time.monotonic()
    response = client.chat.completions.create(**kwargs)

    latency_ms = (time.monotonic() - start) * 1000
    text = (response.choices[0].message.content or "").strip()
    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    cost = estimate_cost(model, input_tokens, output_tokens)

    return ProviderResponse(
        provider="openrouter",
        model=model,
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost, 6),
        latency_ms=round(latency_ms, 2),
    )
