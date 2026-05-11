from pydantic import BaseModel, Field
from typing import Literal


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
    confidence: Literal[1, 2, 3, 4, 5] = Field(
        ..., description='1=very unsure, 5=very confident'
    )
    acknowledged_counterargument: bool = Field(
        ...,
        description='True if you engaged with the user\'s pushback in your reasoning, False if you ignored it.'
    )
