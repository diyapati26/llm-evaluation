import asyncio
import os
from dotenv import load_dotenv

from evals.datasets.loaders              import load_all, load_mmlu_hard
from evals.datasets.manipulation_dataset import build_manipulation_samples
from evals.providers.openai_provider     import OpenAIProvider
from evals.providers.anthropic_provider  import AnthropicProvider
from evals.providers.groq_provider       import GroqProvider
from evals.modules.manipulation_runner   import ManipulationRunner
from evals.cache                         import ResponseCache

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_TOKEN", "")


async def main():

    # ── 1. Load datasets ──────────────────────────────────────
    print("Loading datasets...")
    standard_samples = load_all(max_per_dataset=10)
    hard_samples     = load_mmlu_hard(max_per_subject=5)
    print(f"Hard samples: {len(hard_samples)}\n")

    # ── 2. Build tests ────────────────────────────────────────
    standard_tests = build_manipulation_samples(
        base_samples=hard_samples,
        attack_types=[
            "confident_assertion",
            "authority_pressure",
            "emotional_pressure",
            "false_consensus",
            "false_evidence",
            "identity_challenge",
            "concession_override",
        ],
    )

    drift_tests = build_manipulation_samples(
        base_samples=hard_samples,
        attack_types=["incremental_drift"],
    )

    # ── 3. Cache ──────────────────────────────────────────────
    cache = ResponseCache()

    # ── 4. Providers — split by use case ─────────────────────
    # Standard attacks: 1 call per sample → use 70B for quality
    providers_standard = [
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

    # Drift tests: 6 calls per sample → use 8B to save daily budget
    # llama-3.1-8b-instant has 500k TPD vs 70B's 100k TPD
    providers_drift = [
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
            model="meta-llama/llama-4-scout-17b-16e-instruct",   # 5x more daily budget
            api_key=os.getenv("GROQ_API_KEY"),
            cache=cache,
        ),
    ]

    # ── 5. Runners ────────────────────────────────────────────
    runner_standard = ManipulationRunner(
        providers=providers_standard,
        cache=cache,
        concurrency=1,
    )
    runner_drift = ManipulationRunner(
        providers=providers_drift,
        cache=cache,
        concurrency=1,
    )

    # ── 6. Run standard attacks ───────────────────────────────
    print("\n" + "=" * 60)
    print("PART 1 — STANDARD ATTACKS (7 types, hard MMLU questions)")
    print("=" * 60)
    standard_results = await runner_standard.run(standard_tests)

    # ── 7. Run drift tests ────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 2 — INCREMENTAL DRIFT (6 turns, hard MMLU questions)")
    print("=" * 60)
    drift_results = await runner_drift.run_drift_test(
        drift_tests,
        num_turns=6,
    )

    # ── 8. Combined leaderboard ───────────────────────────────
    print("\n" + "=" * 65)
    print("FULL MANIPULATION RESISTANCE LEADERBOARD")
    print("=" * 65)
    print(
        f"{'Model':<35} "
        f"{'Standard':>9} "
        f"{'Drift(6)':>9} "
        f"{'Combined':>10}"
    )
    print("-" * 65)

    # Collect all unique model names across both runners
    all_models = list(dict.fromkeys(
        [p.model for p in providers_standard] +
        [p.model for p in providers_drift]
    ))

    for model in all_models:
        std   = standard_results.get(model, [])
        drift = drift_results.get(model, [])

        sv = [r for r in std   if r["outcome"] != "invalid"]
        dv = [r for r in drift if r["outcome"] != "invalid"]

        sr = (
            len([r for r in sv if r["outcome"] == "resistant"]) / len(sv)
            if sv else 0
        )
        dr = (
            len([r for r in dv if r["outcome"] == "resistant"]) / len(dv)
            if dv else 0
        )

        # Drift weighted higher — harder test
        combined = sr * 0.4 + dr * 0.6
        bar      = ("█" * int(combined * 10) +
                    "░" * (10 - int(combined * 10)))

        print(
            f"{model:<35} "
            f"{sr:>8.0%} "
            f"{dr:>8.0%} "
            f"  {bar} {combined:.0%}"
        )

    print("-" * 65)
    print(f"\nCache: {cache.stats()}")
    print(
        f"\nNote: Standard attacks use llama-3.3-70b-versatile. "
        f"Drift tests use llama-3.1-8b-instant to preserve "
        f"Groq's 100k daily token budget for 70B."
    )


asyncio.run(main())