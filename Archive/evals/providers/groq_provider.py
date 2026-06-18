import time
import asyncio
import re
from openai import AsyncOpenAI
from evals.schemas import LLMResponse
from evals.providers.base import BaseLLMProvider
from evals.cache import ResponseCache


class DailyLimitExceeded(Exception):
    """Raised when Groq's daily token limit is hit."""
    pass


class GroqProvider(BaseLLMProvider):

    def __init__(
        self,
        model:       str = "meta-llama/llama-4-scout-17b-16e-instruct",
        api_key:     str = None,
        max_tokens:  int = 256,
        temperature: float = 0.0,
        cache:       ResponseCache = None,
    ):
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.cache       = cache
        self.client      = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )

    def _is_multiple_choice(self, prompt: str) -> bool:
        return (
            "A)" in prompt or
            "A) " in prompt or
            "Answer with A, B, C, or D" in prompt or
            "\nA " in prompt
        )

    def _build_messages(self, prompt: str) -> list[dict]:
        """Build messages list with appropriate system prompt for Llama 4."""
        if "llama-4" in self.model:
            is_mc = self._is_multiple_choice(prompt)
            if is_mc:
                return [
                    {
                        "role": "system",
                        "content": (
                            "You are a precise answering assistant. "
                            "When asked a multiple choice question, "
                            "respond with ONLY the single letter: "
                            "A, B, C, or D. "
                            "No explanation. No reasoning. "
                            "Just the single letter."
                        )
                    },
                    {"role": "user", "content": prompt}
                ]
            else:
                return [
                    {
                        "role": "system",
                        "content": (
                            "You are a precise, concise answering "
                            "assistant. Answer truthfully and briefly. "
                            "One or two sentences maximum."
                        )
                    },
                    {"role": "user", "content": prompt}
                ]
        return [{"role": "user", "content": prompt}]

    def _post_process(self, output: str, prompt: str) -> str:
        """Extract single letter from Llama 4 MC responses."""
        if "llama-4" in self.model:
            if self._is_multiple_choice(prompt):
                match = re.search(r'\b([A-D])\b', output.upper())
                return match.group(1) if match else output
        return output

    async def _call_api(
        self,
        messages:  list[dict],
        sample_id: str,
        prompt_for_postprocess: str = "",
    ) -> LLMResponse:
        """Core API call with retry logic."""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                start = time.monotonic()

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

                latency_ms    = (time.monotonic() - start) * 1000
                raw_output    = (
                    response.choices[0].message.content.strip()
                )
                input_tokens  = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

                output = self._post_process(
                    raw_output,
                    prompt_for_postprocess,
                )

                return LLMResponse(
                    sample_id=sample_id,
                    model=self.model,
                    model_version=response.model,
                    temperature=self.temperature,
                    provider="groq",
                    output=output,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=0.0,
                    latency_ms=round(latency_ms, 2),
                )

            except Exception as e:
                error_str = str(e)

                if "TPD" in error_str or "per day" in error_str:
                    wait_match = re.search(
                        r'try again in (\d+)m([\d.]+)s', error_str
                    )
                    if wait_match:
                        wait_mins = int(wait_match.group(1))
                        print(
                            f"\n  Groq DAILY limit reached for "
                            f"{self.model}.\n"
                            f"  Resets in ~{wait_mins} minutes.\n"
                        )
                    raise DailyLimitExceeded(
                        f"Groq daily limit reached for {self.model}"
                    )

                elif (
                    "429" in error_str
                    or "rate_limit" in error_str
                    or "TPM" in error_str
                ):
                    if attempt < max_retries - 1:
                        wait = 2 * (attempt + 1)
                        print(
                            f"  Groq rate limit — waiting {wait}s "
                            f"(attempt {attempt+1}/{max_retries})"
                        )
                        await asyncio.sleep(wait)
                        continue
                    raise e
                else:
                    raise e

        raise Exception(f"Groq failed after {max_retries} attempts")

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

        # 2. Build messages and call API
        messages = self._build_messages(prompt)
        result   = await self._call_api(
            messages=messages,
            sample_id=sample_id,
            prompt_for_postprocess=prompt,
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
        """Multi-turn — inject system prompt for Llama 4 if needed."""

        if "llama-4" in self.model:
            has_system = any(
                m["role"] == "system" for m in messages
            )
            if not has_system:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a precise answering assistant. "
                            "When asked a multiple choice question, "
                            "respond with ONLY the single letter: "
                            "A, B, C, or D. "
                            "No explanation. No reasoning. "
                            "Just the single letter."
                        )
                    }
                ] + messages

        # Use last user message for post-processing detection
        last_user = next(
            (m["content"] for m in reversed(messages)
             if m["role"] == "user"),
            ""
        )

        return await self._call_api(
            messages=messages,
            sample_id=sample_id,
            prompt_for_postprocess=last_user,
        )

    def estimate_cost(
        self,
        input_tokens:  int,
        output_tokens: int,
    ) -> float:
        return 0.0