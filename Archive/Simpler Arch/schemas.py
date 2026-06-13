from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ProviderResponse(BaseModel):
    # Unified return type for every provider call (structured + free-form chat).
    # Structured calls populate `answer` (Pydantic instance) + `raw` (its dict form).
    # Free-form chat calls populate `text` (raw model output string).
    # Token counts, cost, and latency are always populated.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: str
    model: str                          # the model id we requested
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float

    # Reproducibility: model_version is the dated snapshot returned by the API
    # (e.g., "gpt-5.4-mini-2026-01-15"). Aliases silently remap to snapshots that
    # can change — logging the returned version lets a reviewer reproduce later.
    model_version: str | None = None
    temperature: float = 0.0

    answer: Any = None           # parsed schema instance (structured calls)
    raw: dict | None = None      # answer.model_dump() — JSON serialization helper
    text: str | None = None      # free-form text (chat calls)


class MMLU_Answer(BaseModel):
    answer: Literal['1', '2', '3', '4']
    # using 1,2,3,4 as it is easier for the model


class HellaSwag_Answer(BaseModel):
    # HellaSwag always presents exactly 4 candidate endings.
    answer: Literal['1', '2', '3', '4']


class TruthfulQA_MC_Answer(BaseModel):
    # TruthfulQA mc1_targets has a variable choice count (typically 4-13).
    # Capping at 13 covers every question in the validation split.
    answer: Literal[
        '1', '2', '3', '4', '5', '6', '7',
        '8', '9', '10', '11', '12', '13'
    ]


class TruthfulQA_Generation_Answer(BaseModel):
    # For the "generation" config — free-form text answer, no enum.
    answer: str = Field(..., min_length=1)


class ReasonedAnswer(BaseModel):
    # Multi-axis answer for manipulation testing. Every conversation turn returns
    # one of these so we can score on letter persistence, confidence delta,
    # hedging rate, and engagement — not just whether the letter changed.
    # UNCERTAIN is a first-class letter so genuine hedging survives the schema
    # instead of being forced into a 1/2/3/4 guess.
    letter: Literal['1', '2', '3', '4', 'UNCERTAIN']
    reasoning: str = Field(..., min_length=10)
