"""Smoke test — verify all 3 providers respond to a trivial typed-answer prompt.

Run from repo root with venv active:
    python "Simpler Arch/tests/test_providers.py"

Expected output:
    ✓ PASS  openai       gpt-5.4-mini             answer=2  cost=$...  latency=...ms
    ✓ PASS  anthropic    claude-sonnet-4-6        answer=2  cost=$...  latency=...ms
    ✓ PASS  groq         openai/gpt-oss-120b      answer=2  cost=$...  latency=...ms

Exit code 0 if all pass, 1 if any fail.
"""
import os
import sys
import traceback

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from Output_Formats.output_format import MMLU_Answer
from providers import openai_provider, anthropic_provider, groq_provider

load_dotenv()

# Trivial 4-choice prompt — answer is "2" (Paris).
PROMPT = (
    "Answer the following multiple-choice question.\n\n"
    "Question: What is the capital of France?\n"
    "1. London\n"
    "2. Paris\n"
    "3. Berlin\n"
    "4. Madrid\n\n"
    "Reply with the single number 1, 2, 3, or 4."
)
EXPECTED = "2"

PROVIDERS_TO_TEST = [
    ("openai",    openai_provider.get_openai_response,    "gpt-5.4-mini"),
    ("anthropic", anthropic_provider.get_anthropic_response, "claude-sonnet-4-6"),
    ("groq",      groq_provider.get_groq_response,        "openai/gpt-oss-120b"),
]


def check(provider_name, fn, model):
    try:
        resp = fn(PROMPT, model, MMLU_Answer)
        ok = str(resp.get("answer")) == EXPECTED
        marker = "✓ PASS" if ok else "✗ FAIL"
        print(
            f"{marker}  {provider_name:10s}  {model:32s}  "
            f"answer={resp.get('answer')}  "
            f"cost=${resp.get('cost_usd', 0):.4f}  "
            f"latency={resp.get('latency_ms', 0):.0f}ms"
        )
        return ok
    except Exception as e:
        print(f"✗ FAIL  {provider_name:10s}  {model:32s}  error: {type(e).__name__}: {e}")
        return False


def main():
    # Quick env-key sanity check
    missing = []
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"):
        if not os.environ.get(k):
            missing.append(k)
    if missing:
        print(f"⚠️  Missing env vars: {', '.join(missing)}")
        print("   Copy .env.example to .env and fill in your keys, then re-run.")
        sys.exit(1)

    print("Smoke-testing 3 providers with a trivial typed-answer prompt...\n")
    results = [check(name, fn, model) for name, fn, model in PROVIDERS_TO_TEST]
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} providers passed.")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
