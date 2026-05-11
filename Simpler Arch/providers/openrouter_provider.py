"""OpenRouter provider — function-based, no classes.

Uses OpenAI SDK pointed at OpenRouter's OpenAI-compatible endpoint. Single key,
hundreds of models from Anthropic, OpenAI, Meta, Mistral, DeepSeek, Qwen, xAI,
etc., namespaced like "anthropic/claude-3.5-sonnet" or "qwen/qwen-2.5-72b-instruct".

Pricing is fetched once from https://openrouter.ai/api/v1/models on first cost
calc and cached in-process — accuracy comes from OpenRouter's live prices rather
than a hardcoded dict, which is what we want for paper reproducibility (cite
the day-of-run prices from the same source the code reads).

OpenRouter routes structured-output requests through to the underlying provider.
Not every model honors `response_format=json_schema` — get_openrouter_response
falls back to a plain chat completion + best-effort JSON parse when the strict
form errors out.
"""
import os
import json
import time
import urllib.request

from openai import OpenAI


_client = None
_pricing_cache = None  # {model_id: {"input": $/Mtok, "output": $/Mtok}}


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            default_headers={
                # Optional but recommended — OpenRouter uses these for attribution
                "HTTP-Referer": "https://github.com/your-repo",
                "X-Title": "llm-robustness-eval",
            },
        )
    return _client


def _load_pricing():
    """Fetch /models once, build model_id -> {input,output} dict in $/1M tokens.

    OpenRouter returns prompt/completion prices in $/token. We multiply by 1M
    so the unit matches PRICING dicts in the other providers (where the cost
    formula divides by 1_000_000 at the end).
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
    """OpenRouter passes strict json_schema requests through to the underlying
    provider, which may require additionalProperties=False on every object node
    (same constraint Groq enforces). Walk the schema and pin it.
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


def _parse_payload(raw_text, output_format):
    """Best-effort: strip code fences if present, then parse JSON into the schema."""
    text = (raw_text or "").strip()
    if text.startswith("```"):
        # ```json ... ```  or  ``` ... ```
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    payload = json.loads(text)
    return output_format(**payload)


def get_openrouter_response(prompt, model, output_format, max_tokens=256, temperature=0.0):
    """Call OpenRouter with a Pydantic response_format. Returns dict with parsed answer + metadata.

    Tries response_format=json_schema first (strict). If the underlying model
    doesn't support it, falls back to a plain completion and best-effort parses
    JSON from the raw text. Raises RuntimeError only if both paths fail.
    """
    client = _get_client()
    schema = _make_strict(output_format.model_json_schema())
    start = time.monotonic()

    response = None
    parsed = None
    try:
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
        parsed = _parse_payload(response.choices[0].message.content, output_format)
    except Exception:
        # Some OpenRouter-hosted models don't support strict json_schema.
        # Fall back to plain chat and best-effort JSON parse.
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": (
                    prompt
                    + "\n\nReturn ONLY a JSON object matching this schema, "
                      "no prose, no code fences:\n"
                    + json.dumps(schema)
                )}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            parsed = _parse_payload(response.choices[0].message.content, output_format)
        except Exception as e:
            raise RuntimeError(
                f"OpenRouter model {model} did not return parseable structured "
                f"output (both strict and fallback paths failed): {e}"
            )

    latency_ms = (time.monotonic() - start) * 1000
    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    cost = estimate_cost(model, input_tokens, output_tokens)

    return {
        "provider": "openrouter",
        "model": model,
        "answer": parsed.answer if hasattr(parsed, "answer") else parsed,
        "raw": parsed.model_dump() if hasattr(parsed, "model_dump") else parsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
        "latency_ms": round(latency_ms, 2),
    }


def get_openrouter_chat(messages, model, max_tokens=512, temperature=0.0):
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
    text = (response.choices[0].message.content or "").strip()
    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    cost = estimate_cost(model, input_tokens, output_tokens)

    return {
        "provider": "openrouter",
        "model": model,
        "text": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
        "latency_ms": round(latency_ms, 2),
    }
