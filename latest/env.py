"""env.py — .env loading + API-key presence helpers (used by CLI/main AND tests).

Lives in the package (not the test folder) so production code never imports from
tests. Loads latest/.env (and a repo-root .env if present).
"""
from __future__ import annotations

import os
from pathlib import Path

_PKG = Path(__file__).resolve().parent  # latest/

# provider -> (env var, a cheap model to smoke-test with)
PROVIDER_KEYS = {
    "openai": ("OPENAI_API_KEY", "openai:gpt-5.4-nano"),
    "anthropic": ("ANTHROPIC_API_KEY", "anthropic:claude-haiku-4-5"),
    "gemini": ("GEMINI_API_KEY", "gemini:gemini-2.5-flash"),
    "groq": ("GROQ_API_KEY", "groq:openai/gpt-oss-20b"),
}

ENV_BY_PROVIDER = {
    "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY",
}


def load_env() -> None:
    """Load latest/.env (and repo-root .env if present)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for p in (_PKG / ".env", _PKG.parent / ".env"):
        if p.exists():
            load_dotenv(p)


def is_real_key(value: str) -> bool:
    return bool(value) and "your_" not in value and "_here" not in value


def available_providers() -> dict[str, tuple[str, str]]:
    """{provider: (env_var, smoke_model)} for providers with a real key set."""
    load_env()
    return {name: (env, model) for name, (env, model) in PROVIDER_KEYS.items()
            if is_real_key(os.environ.get(env, ""))}
