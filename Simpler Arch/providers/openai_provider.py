"""OpenAI provider — function-based, no classes.

Uses the new Responses API (matches the notebook):
    client.responses.parse(model, input, text_format=PydanticModel)
"""
import os
import time
from openai import OpenAI

# Per-1M-token pricing (USD)
PRICING = {
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.4":      {"input": 2.50, "output": 10.00},
    "gpt-5":        {"input": 5.00, "output": 20.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1":      {"input": 2.00, "output": 8.00},
}

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def estimate_cost(model, input_tokens, output_tokens):
    p = PRICING.get(model, {"input": 0.75, "output": 4.50})
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def get_openai_response(prompt, model, output_format, max_tokens=256, temperature=0.0):
    """Call OpenAI Responses API with a Pydantic text_format. Returns dict with parsed answer + metadata."""
    client = _get_client()
    start = time.monotonic()

    response = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text_format=output_format,
        max_output_tokens=max_tokens,
        temperature=temperature,
    )

    latency_ms = (time.monotonic() - start) * 1000
    parsed = response.output_parsed
    usage = response.usage
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    cost = estimate_cost(model, input_tokens, output_tokens)

    return {
        "provider": "openai",
        "model": model,
        "answer": parsed.answer if hasattr(parsed, "answer") else parsed,
        "raw": parsed.model_dump() if hasattr(parsed, "model_dump") else parsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
        "latency_ms": round(latency_ms, 2),
    }


def get_openai_chat(messages, model, max_tokens=512, temperature=0.0):
    """Multi-turn free-form chat — no Pydantic, returns text + metadata.
    Used for manipulation testing where the model gives reasoned letter answers.
    Responses API accepts the same role/content message list via `input`.
    """
    client = _get_client()
    start = time.monotonic()

    response = client.responses.create(
        model=model,
        input=messages,
        max_output_tokens=max_tokens,
        temperature=temperature,
    )

    latency_ms = (time.monotonic() - start) * 1000
    text = (response.output_text or "").strip()
    usage = response.usage
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    cost = estimate_cost(model, input_tokens, output_tokens)

    return {
        "provider": "openai",
        "model": model,
        "text": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
        "latency_ms": round(latency_ms, 2),
    }
