import time
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from evals.schemas import LLMResponse
from evals.providers.base import BaseLLMProvider
from evals.cache import ResponseCache

PRICING = {
    "gpt-5.4-mini": {"input": 0.75,  "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20,  "output": 1.25},
    "gpt-4.1-mini": {"input": 0.40,  "output": 1.60},
    "gpt-4.1":      {"input": 2.00,  "output": 8.00},
}


class OpenAIProvider(BaseLLMProvider):

    def __init__(
        self,
        model:       str = "gpt-5.4-mini",
        api_key:     str = None,
        max_tokens:  int = 256,
        temperature: float = 0.0,
        cache:       ResponseCache = None,
    ):
        self.model       = model
        self.client      = AsyncOpenAI(api_key=api_key)
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.cache       = cache

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10)
    )
    async def generate(
        self,
        prompt:    str,
        sample_id: str,
    ) -> LLMResponse:

        # 1. Check cache
        if self.cache:
            cached = self.cache.get(self.model, prompt)
            if cached:
                cached.sample_id = sample_id
                return cached

        # 2. API call
        start = time.monotonic()

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        latency_ms = (time.monotonic() - start) * 1000
        usage      = response.usage
        cost       = self.estimate_cost(
            usage.prompt_tokens,
            usage.completion_tokens,
        )

        result = LLMResponse(
            sample_id=sample_id,
            model=self.model,
            provider="openai",
            output=response.choices[0].message.content.strip(),
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost_usd=cost,
            latency_ms=round(latency_ms, 2),
        )

        # 3. Save to cache
        if self.cache:
            self.cache.save(result, prompt)

        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10)
    )
    async def generate_conversation(
        self,
        messages:  list[dict],
        sample_id: str,
    ) -> LLMResponse:
        """Multi-turn — no cache, each conversation is unique."""

        start = time.monotonic()

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        latency_ms = (time.monotonic() - start) * 1000
        usage      = response.usage
        cost       = self.estimate_cost(
            usage.prompt_tokens,
            usage.completion_tokens,
        )

        return LLMResponse(
            sample_id=sample_id,
            model=self.model,
            provider="openai",
            output=response.choices[0].message.content.strip(),
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost_usd=cost,
            latency_ms=round(latency_ms, 2),
        )

    def estimate_cost(
        self,
        input_tokens:  int,
        output_tokens: int,
    ) -> float:
        p = PRICING.get(self.model, {"input": 0.75, "output": 4.50})
        return (
            input_tokens  * p["input"]  / 1_000_000 +
            output_tokens * p["output"] / 1_000_000
        )