from pydantic import BaseModel
from typing import Literal

class MMLU_Answer(BaseModel):
    answer: Literal['1', '2', '3', '4']
#using 1,2,3,4 as it is easier for the model

