"""
Provider integration tests.
Run with: python3 tests/test_providers.py
"""
import asyncio
import os
from dotenv import load_dotenv
from evals.providers.openai_provider    import OpenAIProvider
from evals.providers.anthropic_provider import AnthropicProvider
from evals.providers.groq_provider      import GroqProvider
from evals.cache                        import ResponseCache

load_dotenv()

async def test_all_providers():
    prompt    = "What is the capital of Japan? One word only."
    cache     = ResponseCache()
    providers = [
        OpenAIProvider(
            model="gpt-5.4-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            cache=cache,
        ),
        AnthropicProvider(
            model="claude-sonnet-4-6",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            cache=cache,
        ),
        GroqProvider(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=os.getenv("GROQ_API_KEY"),
            cache=cache,
        ),
    ]

    print("Testing all providers...")
    print("=" * 55)
    all_passed = True

    for provider in providers:
        try:
            response = await provider.generate(prompt, "test_001")
            status   = "✓ PASS"
            print(f"{status}  {response.provider:<12} "
                  f"{response.model:<35} "
                  f"answer={response.output:<10} "
                  f"${response.cost_usd:.6f}  "
                  f"{response.latency_ms:.0f}ms")
        except Exception as e:
            print(f"✗ FAIL  {provider.model:<35} {e}")
            all_passed = False

    print("=" * 55)
    print(f"Cache stats: {cache.stats()}")
    print(f"\nAll providers: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed

asyncio.run(test_all_providers())