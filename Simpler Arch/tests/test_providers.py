"""Smoke test — send 'Hi' to each provider and verify a non-empty response.

Run from repo root with venv active:
    python "Simpler Arch/tests/test_providers.py"

Skips providers whose API key isn't set in the env.
"""
import os
import sys

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from providers import (
    openai_provider,
    anthropic_provider,
    groq_provider,
    openrouter_provider,
)

load_dotenv()

PROVIDERS = [
    ("openai",     openai_provider.get_openai_chat,         "gpt-5.4-mini"),
    ("anthropic",  anthropic_provider.get_anthropic_chat,   "claude-sonnet-4-6"),
    ("groq",       groq_provider.get_groq_chat,             "openai/gpt-oss-120b"),
    ("openrouter", openrouter_provider.get_openrouter_chat, "meta-llama/llama-3.3-70b-instruct"),
]

REQUIRED_KEYS = {
    "openai":     "OPENAI_API_KEY",
    "anthropic":  "ANTHROPIC_API_KEY",
    "groq":       "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def check(name, fn, model):
    """Send 'Hi' through `fn`, return True if a non-empty text comes back."""
    try:
        # Anthropic still requires max_tokens. Pass a small positive value so
        # the unified test works for all four providers.
        resp = fn(
            [{"role": "user", "content": "Hi"}],
            model,
            max_tokens=50,
        )
        ok = bool(resp.text and resp.text.strip())
        marker = "PASS" if ok else "FAIL"
        preview = (resp.text or "")[:50].replace("\n", " ")
        print(
            f"  {marker}  {name:10s}  {model:42s}  "
            f"text={preview!r:54s}  "
            f"cost=${resp.cost_usd:.4f}  "
            f"latency={resp.latency_ms:.0f}ms"
        )
        return ok
    except Exception as e:
        print(f"  FAIL  {name:10s}  {model:42s}  error: {type(e).__name__}: {e}")
        return False


def main():
    available = {p: bool(os.environ.get(k)) for p, k in REQUIRED_KEYS.items()}
    missing = [k for p, k in REQUIRED_KEYS.items() if not os.environ.get(k)]
    if missing:
        print(f"[warn] Missing env vars: {', '.join(missing)}  (skipping those providers)\n")

    print("Smoke-testing providers: send 'Hi', expect a non-empty response.\n")
    results = []
    for name, fn, model in PROVIDERS:
        if not available.get(name, True):
            print(f"  SKIP  {name:10s}  {model:42s}  no API key in env")
            continue
        results.append(check(name, fn, model))

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} providers passed (skipped {len(PROVIDERS) - total}).")
    sys.exit(0 if passed == total and total > 0 else 1)


if __name__ == "__main__":
    main()
