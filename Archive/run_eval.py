import asyncio
import os
from dotenv import load_dotenv

from evals.datasets.loaders             import load_all
from evals.providers.openai_provider    import OpenAIProvider
from evals.providers.anthropic_provider import AnthropicProvider
from evals.providers.groq_provider      import GroqProvider
from evals.scorers.classical            import ExactMatchScorer, RougeScorer, BERTScorer
from evals.scorers.llm_judge            import LLMJudgeScorer
from evals.runner                       import EvalRunner
from evals.cache                        import ResponseCache

load_dotenv()
#os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_TOKEN", "")


async def main():

    # ── 1. Load datasets ──────────────────────────────────────
    print("Loading datasets...")
    samples = load_all(max_per_dataset=10)
    print(f"Total samples: {len(samples)}\n")

    # ── 2. Shared cache ───────────────────────────────────────
    cache = ResponseCache()

    # ── 3. Three providers ────────────────────────────────────
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

    # ── 4. All four scorers ───────────────────────────────────
    scorers = [
        ExactMatchScorer(),
        RougeScorer(),
        BERTScorer(),
        LLMJudgeScorer(
            api_key=os.getenv("OPENAI_API_KEY"),
            judge_model="gpt-5.4-mini",
        ),
    ]

    # ── 5. Run ────────────────────────────────────────────────
    runner = EvalRunner(
        providers=providers,
        scorers=scorers,
        concurrency=1,
        cache=cache,
    )

    results = await runner.run(
        samples,
        dataset_name="mixed_benchmark_v1"
    )

    # ── 6. Leaderboard ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("LEADERBOARD — FULL 4-SCORER COMPARISON")
    print("=" * 70)
    print(
        f"{'Model':<32} "
        f"{'EM':>6} "
        f"{'ROUGE':>6} "
        f"{'BERT':>6} "
        f"{'Judge':>6} "
        f"{'Cost':>10}"
    )
    print("-" * 70)

    sorted_results = sorted(
        results,
        key=lambda x: x.scores.get("exact_match", 0),
        reverse=True
    )

    for i, r in enumerate(sorted_results):
        em    = r.scores.get("exact_match", 0)
        rl    = r.scores.get("rouge_l",     0)
        bs    = r.scores.get("bertscore",   0)
        judge = r.scores.get("llm_judge",   0)
        rank  = ["1st", "2nd", "3rd"][i]
        print(
            f"{rank} {r.model:<29} "
            f"{em:>6.3f} "
            f"{rl:>6.3f} "
            f"{bs:>6.3f} "
            f"{judge:>6.3f} "
            f"${r.total_cost_usd:>9.5f}"
        )

    print("-" * 70)

    # ── 7. Analysis ───────────────────────────────────────────
    best  = sorted_results[0]
    worst = sorted_results[-1]
    diff  = (
        best.scores.get("exact_match", 0) -
        worst.scores.get("exact_match", 0)
    )

    print(f"\nBest model:   {best.model}")
    print(f"Worst model:  {worst.model}")
    print(f"Gap:          {diff:.1%} accuracy difference")
    print(f"\nCache:        {cache.stats()}")
    print(f"Total cost:   ${sum(r.total_cost_usd for r in results):.5f}")


asyncio.run(main())