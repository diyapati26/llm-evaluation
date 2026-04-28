from abc import ABC, abstractmethod
from evals.schemas import LLMResponse


class BaseLLMProvider(ABC):

    @abstractmethod
    async def generate(
        self,
        prompt:    str,
        sample_id: str,
    ) -> LLMResponse:
        """Single-turn generation — one prompt, one response."""
        pass

    @abstractmethod
    async def generate_conversation(
        self,
        messages:  list[dict],
        sample_id: str,
    ) -> LLMResponse:
        """
        Multi-turn generation — full conversation history.

        messages format:
        [
            {"role": "user",      "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user",      "content": "Are you sure?"},
        ]
        """
        pass