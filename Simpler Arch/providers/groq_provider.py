"""Groq provider — function-based, no classes.

Uses OpenAI SDK pointed at Groq's OpenAI-compatible endpoint.
Default model: openai/gpt-oss-120b — natively honors response_format=json_schema,
so no regex post-processing or system-prompt hacks needed (unlike Llama 4 Scout).
"""
import os
import json
import time
from openai import OpenAI

# Per-1M-token pricing (USD). Groq paid tier; free tier = $0.
PRICING = {
    "openai/gpt-oss-120b":                            {"input": 0.15, "output": 0.75},
    "openai/gpt-oss-20b":                             {"input": 0.05, "output": 0.25},
    "meta-llama/llama-4-scout-17b-16e-instruct":      {"input": 0.00, "output": 0.00},
    "meta-llama/llama-3.3-70b-versatile":             {"input": 0.00, "output": 0.00},
}

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
        )
    return _client


def estimate_cost(model, input_tokens, output_tokens):
    p = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def _make_strict(node):
    """Groq's strict json_schema mode rejects any object that doesn't pin
    additionalProperties=False. Pydantic v2's model_json_schema doesn't add it.
    Walk the schema and add it on every object node.
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


def get_groq_response(prompt, model, output_format, max_tokens=256, temperature=0.0):
    """Call Groq with a Pydantic response_format. Returns dict with parsed answer + metadata.

    output_format must be a Pydantic BaseModel subclass — its model_json_schema() is sent.
    """
    client = _get_client()
    schema = _make_strict(output_format.model_json_schema())

    start = time.monotonic()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": output_format.__name__,
                "schema": schema,
                "strict": True,
            },
        },
        max_tokens=max_tokens,
        temperature=temperature,
    )

    latency_ms = (time.monotonic() - start) * 1000
    raw_text = response.choices[0].message.content
    payload = json.loads(raw_text)
    parsed = output_format(**payload)

    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    cost = estimate_cost(model, input_tokens, output_tokens)

    return {
        "provider": "groq",
        "model": model,
        "answer": parsed.answer if hasattr(parsed, "answer") else parsed,
        "raw": parsed.model_dump() if hasattr(parsed, "model_dump") else parsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
        "latency_ms": round(latency_ms, 2),
    }


def get_groq_chat(messages, model, max_tokens=512, temperature=0.0):
    """Multi-turn free-form chat — no Pydantic, returns text + metadata.
    Used for manipulation testing where the model gives reasoned letter answers.
    """
    client = _get_client()
    start = time.monotonic()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    latency_ms = (time.monotonic() - start) * 1000
    text = response.choices[0].message.content.strip()
    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    cost = estimate_cost(model, input_tokens, output_tokens)

    return {
        "provider": "groq",
        "model": model,
        "text": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
        "latency_ms": round(latency_ms, 2),
    }
