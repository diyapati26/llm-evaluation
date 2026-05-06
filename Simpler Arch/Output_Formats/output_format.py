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
