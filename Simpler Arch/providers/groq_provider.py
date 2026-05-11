"""Groq provider — function-based, no classes.

Uses OpenAI SDK pointed at Groq's OpenAI-compatible endpoint.
Default model: openai/gpt-oss-120b — natively honors response_format=json_schema,
"""

import json
import os
import time

from openai import OpenAI

from schemas import ProviderResponse

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


def get_groq_response(prompt, model, output_format, max_tokens=None, temperature=0.0):
    """Call Groq with a Pydantic response_format. Returns dict with parsed answer + metadata.

    output_format must be a Pydantic BaseModel subclass — its model_json_schema() is sent.
    max_tokens=None lets the model use its full output budget (no truncation).
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

    latency_ms = (time.monotonic() - start) * 1000
    raw_text = response.choices[0].message.content
    payload = json.loads(raw_text)
    parsed = output_format(**payload)

    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    cost = estimate_cost(model, input_tokens, output_tokens)

    return ProviderResponse(
        provider="groq",
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


def get_groq_chat(messages, model, max_tokens=None, temperature=0.0):
    """Free-form chat (no Pydantic). Returns text + metadata.

    Used by moral.py's LLM-judge calls. NOT used by manipulation any more — that
    path now goes through providers.conversation.start_conversation.
    """
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
    text = response.choices[0].message.content.strip()
    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    cost = estimate_cost(model, input_tokens, output_tokens)

    return ProviderResponse(
        provider="groq",
        model=model,
        model_version=getattr(response, "model", None),
        temperature=temperature,
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost, 6),
        latency_ms=round(latency_ms, 2),
    )
