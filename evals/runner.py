import asyncio
import uuid
from evals.schemas import EvalSample, LLMResponse, MetricScore, EvalResult
from evals.providers.base import BaseLLMProvider
from evals.cache import ResponseCache

class EvalRunner:
    """
    Orchestrates the entire eval pipeline.
    Takes samples + providers + scorers,
    runs everything concurrently,
    returns aggregated EvalResults.
    """

    def __init__(
        self,
        providers:   list,
        scorers:     list,
        concurrency: int = 5,
        cache:       ResponseCache = None,
    ):
        self.providers   = providers
        self.scorers     = scorers
        # Semaphore limits how many API calls fire simultaneously
        # 5 = safe for free tiers, won't trigger rate limits
        self.concurrency = concurrency
        self.cache       = cache

    async def run(
        self,
        samples:      list[EvalSample],
        dataset_name: str = "unknown",
    ) -> list[EvalResult]:
        """
        Run all providers against all samples.
        Returns one EvalResult per provider.
        """
        run_id = str(uuid.uuid4())[:8]

        print(f"\n{'='*55}")
        print(f"Run ID:   {run_id}")
        print(f"Samples:  {len(samples)}")
        print(f"Models:   {len(self.providers)}")
        print(f"Scorers:  {len(self.scorers)}")
        print(f"{'='*55}\n")

        all_results = []

        for provider in self.providers:
            print(f"Running {provider.model}...")

            responses, metric_scores = await self._run_provider(
                provider, samples
            )

            # Aggregate — mean score per scorer across all samples
            aggregated = {}
            for scorer in self.scorers:
                scorer_scores = [
                    s.score for s in metric_scores
                    if s.scorer_name == scorer.name
                ]
                if scorer_scores:
                    aggregated[scorer.name] = round(
                        sum(scorer_scores) / len(scorer_scores), 4
                    )

            total_cost = round(sum(r.cost_usd for r in responses), 6)

            result = EvalResult(
                run_id=run_id,
                model=provider.model,
                dataset=dataset_name,
                scores=aggregated,
                total_cost_usd=total_cost,
                total_samples=len(samples),
            )
            all_results.append(result)

            # Print per-model summary
            scores_str = "  ".join(
                f"{k}={v:.3f}" for k, v in aggregated.items()
            )
            print(f"  {provider.model}")
            print(f"  Scores:  {scores_str}")
            print(f"  Cost:    ${total_cost:.5f}")
            print()

        return all_results

    async def _run_provider(
        self,
        provider: BaseLLMProvider,
        samples:  list[EvalSample],
    ) -> tuple[list[LLMResponse], list[MetricScore]]:
        """
        Run one provider against all samples concurrently.
        Semaphore ensures max `concurrency` calls in flight at once.
        """
        sem       = asyncio.Semaphore(self.concurrency)
        responses = []
        all_scores = []

        async def run_one(sample: EvalSample):
            async with sem:
                try:
                    # Get model response
                    response = await provider.generate(
                        sample.prompt,
                        sample.id
                    )
                    responses.append(response)

                    # Score the response with every scorer
                    for scorer in self.scorers:
                        try:
                            score_fn = scorer.score

                            if asyncio.iscoroutinefunction(score_fn):
                                # Async scorer (LLM judge)
                                result = await score_fn(
                                    prediction=response.output,
                                    reference=sample.reference or "",
                                    question=sample.prompt,
                                )
                            else:
                                # Sync scorer (ROUGE, BERTScore, ExactMatch)
                                result = score_fn(
                                    prediction=response.output,
                                    reference=sample.reference or "",
                                )

                            all_scores.append(MetricScore(
                                sample_id=sample.id,
                                model=provider.model,
                                scorer_name=scorer.name,
                                score=result["score"],
                                explanation=result.get("reasoning"),
                                metadata=result,
                            ))

                        except Exception as e:
                            print(f"  Scorer {scorer.name} failed on "
                                  f"{sample.id}: {e}")

                except Exception as e:
                    print(f"  Provider failed on {sample.id}: {e}")

        # Fire all samples concurrently (limited by semaphore)
        await asyncio.gather(*[run_one(s) for s in samples])

        return responses, all_scores