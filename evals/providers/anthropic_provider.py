import time
import anthropic
from evals.schemas import LLMResponse
from evals.providers.base import BaseLLMProvider
from evals.cache import ResponseCache

PRICING = {
    "claude-sonnet-4-6":         {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5":          {"input": 1.00,  "output": 5.00},
    "claude-haiku-4-5-20251001": {"input": 1.00,  "output": 5.00},
}


class AnthropicProvider(BaseLLMProvider):

    def __init__(
        self,
        model:       str = "claude-sonnet-4-6",
        api_key:     str = None,
        max_tokens:  int = 256,
        temperature: float = 0.0,
        cache:       ResponseCache = None,
    ):
        self.model       = model
        self.client      = anthropic.AsyncAnthropic(api_key=api_key)
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.cache       = cache

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

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        latency_ms    = (time.monotonic() - start) * 1000
        input_tokens  = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost          = self.estimate_cost(input_tokens, output_tokens)

        result = LLMResponse(
            sample_id=sample_id,
            model=self.model,
            provider="anthropic",
            output=response.content[0].text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=round(latency_ms, 2),
        )

        # 3. Save to cache
        if self.cache:
            self.cache.save(result, prompt)

        return result

    async def generate_conversation(
        self,
        messages:  list[dict],
        sample_id: str,
    ) -> LLMResponse:
        """Multi-turn — no cache, each conversation is unique."""

        start = time.monotonic()

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        )

        latency_ms    = (time.monotonic() - start) * 1000
        input_tokens  = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost          = self.estimate_cost(input_tokens, output_tokens)

        return LLMResponse(
            sample_id=sample_id,
            model=self.model,
            provider="anthropic",
            output=response.content[0].text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=round(latency_ms, 2),
        )

    def estimate_cost(
        self,
        input_tokens:  int,
        output_tokens: int,
    ) -> float:
        p = PRICING.get(self.model, {"input": 3.00, "output": 15.00})
        return (
            input_tokens  * p["input"]  / 1_000_000 +
            output_tokens * p["output"] / 1_000_000
        )