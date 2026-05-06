"""OpenAI provider — function-based, no classes.

Uses OpenAI's structured-output: client.beta.chat.completions.parse(response_format=PydanticModel).
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
    """Call OpenAI with a Pydantic response_format. Returns dict with parsed answer + metadata."""
    client = _get_client()
    start = time.monotonic()

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=output_format,
        max_completion_tokens=max_tokens,
        temperature=temperature,
    )

    latency_ms = (time.monotonic() - start) * 1000
    parsed = response.choices[0].message.parsed
    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
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
