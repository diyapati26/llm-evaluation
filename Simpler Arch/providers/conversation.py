"""Conversation abstraction — three subclasses hide the provider asymmetry.

Each Conversation maintains its own state and exposes a uniform .send(message, schema)
that returns the same dict shape as the existing get_*_response functions. The
manipulation module drives all three the same way.

Subclasses:
  OpenAIConversation             — server-side state via conversations.create()
                                    + responses.parse(conversation=id, ..., text_format=schema)
  AnthropicConversation          — client-side messages list, messages.parse(output_format=schema).
                                    Anthropic has no server-side conversation state by design.
  ChatCompletionsConversation    — client-side messages, chat.completions.create
                                    with response_format=json_schema strict mode.
                                    Used by Groq and OpenRouter (Groq probe confirmed
                                    /v1/conversations is unavailable; OpenRouter has never
                                    exposed that endpoint either).

Factory:
  start_conversation(model)      — routes by 'provider:model' prefix or name inference.
"""
import json
import time

from google.genai import types as gemini_types

from Output_Formats.output_format import ProviderResponse
from providers import (
    anthropic_provider,
    gemini_provider,
    groq_provider,
    openai_provider,
    openrouter_provider,
)


def _make_strict(node):
    """Recursively pin additionalProperties=False on every object node.

    Required by Groq and (often) OpenRouter when using response_format=json_schema
    with strict=True. Pydantic v2's model_json_schema doesn't emit it.
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


# ── Base ──────────────────────────────────────────────────────────


class Conversation:
    """Base class. Subclasses implement .send().

    Tracks cumulative cost / tokens / latency across all turns so callers can
    report per-conversation totals without bookkeeping at the call site.
    """
    provider = "base"

    def __init__(self, model):
        self.model = model
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency_ms = 0.0
        self.turn_count = 0

    def _accumulate(self, cost, in_tok, out_tok, latency_ms):
        self.total_cost += cost
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok
        self.total_latency_ms += latency_ms
        self.turn_count += 1

    def send(self, message, schema, max_tokens=None, temperature=0.0):
        raise NotImplementedError


# ── OpenAI: server-side conversation state ────────────────────────


class OpenAIConversation(Conversation):
    provider = "openai"

    def __init__(self, model):
        super().__init__(model)
        self.client = openai_provider._get_client()
        self.conv_id = None  # created lazily on first send

    def send(self, message, schema, max_tokens=None, temperature=0.0):
        start = time.monotonic()
        if self.conv_id is None:
            self.conv_id = self.client.conversations.create().id

        kwargs = {
            "model": self.model,
            "conversation": self.conv_id,
            "input": [{"role": "user", "content": message}],
            "text_format": schema,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens

        response = self.client.responses.parse(**kwargs)

        latency_ms = (time.monotonic() - start) * 1000
        parsed = response.output_parsed
        usage = response.usage
        cost = openai_provider.estimate_cost(
            self.model, usage.input_tokens, usage.output_tokens
        )
        self._accumulate(cost, usage.input_tokens, usage.output_tokens, latency_ms)

        return ProviderResponse(
            provider="openai",
            model=self.model,
            answer=parsed,
            raw=parsed.model_dump() if hasattr(parsed, "model_dump") else None,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=round(cost, 6),
            latency_ms=round(latency_ms, 2),
        )


# ── Anthropic: client-side messages, no server-side state ────────


class AnthropicConversation(Conversation):
    provider = "anthropic"

    def __init__(self, model):
        super().__init__(model)
        self.client = anthropic_provider._get_client()
        self.messages = []

    def send(self, message, schema, max_tokens=None, temperature=0.0):
        start = time.monotonic()
        self.messages.append({"role": "user", "content": message})

        kwargs = {
            "model": self.model,
            "temperature": temperature,
            "output_format": schema,
            "messages": self.messages,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = self.client.messages.parse(**kwargs)

        latency_ms = (time.monotonic() - start) * 1000
        parsed = response.parsed_output

        # Feed the parsed JSON back as the assistant turn so the next call sees
        # the full history. Using model_dump_json keeps the content stable across
        # turns (no whitespace or markup drift).
        self.messages.append({
            "role": "assistant",
            "content": parsed.model_dump_json(),
        })

        in_tok = response.usage.input_tokens
        out_tok = response.usage.output_tokens
        cost = anthropic_provider.estimate_cost(self.model, in_tok, out_tok)
        self._accumulate(cost, in_tok, out_tok, latency_ms)

        return ProviderResponse(
            provider="anthropic",
            model=self.model,
            answer=parsed,
            raw=parsed.model_dump() if hasattr(parsed, "model_dump") else None,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=round(cost, 6),
            latency_ms=round(latency_ms, 2),
        )


# ── Gemini: client-side contents list, native response_schema ─────


class GeminiConversation(Conversation):
    provider = "gemini"

    def __init__(self, model):
        super().__init__(model)
        self.client = gemini_provider._get_client()
        # Gemini contents list: [{"role": "user"|"model", "parts": [{"text": ...}]}]
        self.contents = []

    def send(self, message, schema, max_tokens=None, temperature=0.0):
        start = time.monotonic()
        self.contents.append({"role": "user", "parts": [{"text": message}]})

        config_kwargs = {
            "response_mime_type": "application/json",
            "response_schema": schema,
            "temperature": temperature,
        }
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens

        response = self.client.models.generate_content(
            model=self.model,
            contents=self.contents,
            config=gemini_types.GenerateContentConfig(**config_kwargs),
        )

        latency_ms = (time.monotonic() - start) * 1000
        parsed = response.parsed

        # Append model turn (JSON text) so next call sees full history
        self.contents.append({
            "role": "model",
            "parts": [{"text": parsed.model_dump_json() if hasattr(parsed, "model_dump_json") else (response.text or "")}],
        })

        usage = response.usage_metadata
        in_tok = usage.prompt_token_count or 0
        out_tok = usage.candidates_token_count or 0
        cost = gemini_provider.estimate_cost(self.model, in_tok, out_tok)
        self._accumulate(cost, in_tok, out_tok, latency_ms)

        return ProviderResponse(
            provider="gemini",
            model=self.model,
            answer=parsed,
            raw=parsed.model_dump() if hasattr(parsed, "model_dump") else None,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=round(cost, 6),
            latency_ms=round(latency_ms, 2),
        )


# ── Generic OpenAI-compatible chat completions (Groq + OpenRouter) ─


class ChatCompletionsConversation(Conversation):
    """OpenAI-compatible chat completions with strict json_schema.

    Used for any provider whose endpoint is chat.completions-only (no Responses
    API or no Conversations API). Falls back to a plain completion if strict
    mode errors out — relevant for some smaller OpenRouter-hosted models.
    """

    def __init__(self, model, provider_name, client, pricing_fn):
        super().__init__(model)
        self.provider = provider_name
        self.client = client
        self.pricing_fn = pricing_fn
        self.messages = []

    def send(self, message, schema, max_tokens=None, temperature=0.0):
        start = time.monotonic()
        self.messages.append({"role": "user", "content": message})
        schema_dict = _make_strict(schema.model_json_schema())

        def _build_kwargs(messages, response_format=None):
            k = {"model": self.model, "messages": messages, "temperature": temperature}
            if max_tokens is not None:
                k["max_tokens"] = max_tokens
            if response_format is not None:
                k["response_format"] = response_format
            return k

        try:
            response = self.client.chat.completions.create(**_build_kwargs(
                messages=self.messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": schema_dict,
                        "strict": True,
                    },
                },
            ))
            raw_text = (response.choices[0].message.content or "").strip()
            parsed = schema(**json.loads(raw_text))
        except Exception:
            # Fallback: tell the model the schema inline and parse best-effort
            response = self.client.chat.completions.create(**_build_kwargs(
                messages=self.messages + [{
                    "role": "system",
                    "content": (
                        "Reply with ONLY a JSON object matching this schema, "
                        "no prose or code fences:\n" + json.dumps(schema_dict)
                    ),
                }],
            ))
            raw_text = (response.choices[0].message.content or "").strip()
            # Strip code fences some models add
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`")
                if raw_text.lower().startswith("json"):
                    raw_text = raw_text[4:].lstrip()
            parsed = schema(**json.loads(raw_text))

        # Append assistant turn (the JSON string) so history persists for next call
        self.messages.append({"role": "assistant", "content": raw_text})

        latency_ms = (time.monotonic() - start) * 1000
        usage = response.usage
        in_tok = usage.prompt_tokens
        out_tok = usage.completion_tokens
        cost = self.pricing_fn(self.model, in_tok, out_tok)
        self._accumulate(cost, in_tok, out_tok, latency_ms)

        return ProviderResponse(
            provider=self.provider,
            model=self.model,
            answer=parsed,
            raw=parsed.model_dump() if hasattr(parsed, "model_dump") else None,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=round(cost, 6),
            latency_ms=round(latency_ms, 2),
        )


# ── Factory ───────────────────────────────────────────────────────


def _resolve(model):
    """Return (provider_name, real_model) using the same rules as runner.route()."""
    if ":" in model:
        provider_name, real_model = model.split(":", 1)
        return provider_name, real_model
    if model.startswith(("gpt-", "o1", "o3")):
        return "openai", model
    if model.startswith("claude-"):
        return "anthropic", model
    if model.startswith("gemini-"):
        return "gemini", model
    if model.startswith(("openai/gpt-oss", "meta-llama/", "llama")):
        return "groq", model
    raise ValueError(
        f"Don't know how to route '{model}'. Use 'provider:model' syntax."
    )


def chat(model, messages, max_tokens=None, temperature=0.0):
    """Single-turn free-form chat across any provider. Returns ProviderResponse.

    Routes by 'provider:model' prefix or name inference. Use this when you want
    raw text output (LLM judges, moral scenarios) — Conversation is for multi-turn
    structured-output flows.
    """
    provider_name, real_model = _resolve(model)
    if provider_name == "openai":
        return openai_provider.get_openai_chat(messages, real_model, max_tokens=max_tokens, temperature=temperature)
    if provider_name == "anthropic":
        return anthropic_provider.get_anthropic_chat(messages, real_model, max_tokens=max_tokens, temperature=temperature)
    if provider_name == "groq":
        return groq_provider.get_groq_chat(messages, real_model, max_tokens=max_tokens, temperature=temperature)
    if provider_name == "openrouter":
        return openrouter_provider.get_openrouter_chat(messages, real_model, max_tokens=max_tokens, temperature=temperature)
    if provider_name == "gemini":
        return gemini_provider.get_gemini_chat(messages, real_model, max_tokens=max_tokens, temperature=temperature)
    raise ValueError(f"Unknown provider '{provider_name}'")


def start_conversation(model):
    """Create the right Conversation subclass for this model."""
    provider_name, real_model = _resolve(model)

    if provider_name == "openai":
        return OpenAIConversation(real_model)
    if provider_name == "anthropic":
        return AnthropicConversation(real_model)
    if provider_name == "groq":
        return ChatCompletionsConversation(
            model=real_model,
            provider_name="groq",
            client=groq_provider._get_client(),
            pricing_fn=groq_provider.estimate_cost,
        )
    if provider_name == "openrouter":
        return ChatCompletionsConversation(
            model=real_model,
            provider_name="openrouter",
            client=openrouter_provider._get_client(),
            pricing_fn=openrouter_provider.estimate_cost,
        )
    if provider_name == "gemini":
        return GeminiConversation(real_model)
    raise ValueError(f"Unknown provider '{provider_name}'")
