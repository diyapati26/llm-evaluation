"""router.py — the ONE place a model string maps to a provider.

The old code had two drifting resolvers (runner.route and conversation._resolve).
Here there is a single resolve(), and a registry that the concrete provider
modules populate at import time via register(). Every collector goes through
start_conversation() / chat() / estimate_cost() here.

Routing rules (resolve):
  - explicit 'provider:model'         -> (provider, model)
  - 'gpt-*', 'o1*', 'o3*'             -> openai
  - 'claude-*'                        -> anthropic
  - 'gemini-*'                        -> gemini
  - 'openai/gpt-oss*', 'meta-llama/*', 'llama*' -> groq
  - otherwise                         -> ValueError (use 'provider:model')
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from latest.providers.base import CallContext, Conversation


@dataclass
class ProviderEntry:
    conversation_cls: type[Conversation] | None = None
    chat_fn: Callable | None = None          # (messages, model, max_tokens, temperature) -> ProviderResponse
    estimate_cost_fn: Callable | None = None  # (model, in_tok, out_tok) -> float


_REGISTRY: dict[str, ProviderEntry] = {}


def register(
    name: str,
    *,
    conversation_cls: type[Conversation] | None = None,
    chat_fn: Callable | None = None,
    estimate_cost_fn: Callable | None = None,
) -> None:
    """Register (or extend) a provider. Concrete provider modules call this at import."""
    entry = _REGISTRY.setdefault(name, ProviderEntry())
    if conversation_cls is not None:
        entry.conversation_cls = conversation_cls
    if chat_fn is not None:
        entry.chat_fn = chat_fn
    if estimate_cost_fn is not None:
        entry.estimate_cost_fn = estimate_cost_fn


def registered_providers() -> list[str]:
    return sorted(_REGISTRY)


def resolve(model: str) -> tuple[str, str]:
    """Return (provider_name, real_model). Pure string logic — no SDKs needed."""
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
    raise ValueError(f"Cannot route '{model}'. Use explicit 'provider:model' syntax.")


def _entry(provider_name: str) -> ProviderEntry:
    entry = _REGISTRY.get(provider_name)
    if entry is None:
        raise ValueError(
            f"Provider '{provider_name}' is not registered. "
            f"Registered: {registered_providers() or '(none)'}. "
            f"Did you `import latest.providers.<name>`?"
        )
    return entry


def start_conversation(
    model: str,
    ctx: CallContext | None = None,
    cache: Any = None,
    ledger: Any = None,
) -> Conversation:
    """Create the right Conversation subclass for a model string."""
    provider_name, real_model = resolve(model)
    entry = _entry(provider_name)
    if entry.conversation_cls is None:
        raise ValueError(f"Provider '{provider_name}' registered no conversation class.")
    return entry.conversation_cls(real_model, ctx=ctx, cache=cache, ledger=ledger)


def chat(model: str, messages: list, max_tokens: int | None = None, temperature: float = 0.0):
    """Single-turn free-form chat (LLM judges, moral scenarios). Returns ProviderResponse."""
    provider_name, real_model = resolve(model)
    entry = _entry(provider_name)
    if entry.chat_fn is None:
        raise ValueError(f"Provider '{provider_name}' registered no chat function.")
    return entry.chat_fn(messages, real_model, max_tokens=max_tokens, temperature=temperature)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    provider_name, real_model = resolve(model)
    entry = _entry(provider_name)
    if entry.estimate_cost_fn is None:
        raise ValueError(f"Provider '{provider_name}' registered no cost estimator.")
    return entry.estimate_cost_fn(real_model, input_tokens, output_tokens)
