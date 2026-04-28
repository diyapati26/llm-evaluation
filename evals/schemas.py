from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime
from enum import Enum

class TaskType(str, Enum):
    multiple_choice  = "multiple_choice"
    open_generation  = "open_generation"
    code_generation  = "code_generation"
    moral_dilemma    = "moral_dilemma"      # our novel module
    manipulation     = "manipulation"        # our novel module
    debate           = "debate"              # our novel module

class EvalSample(BaseModel):
    id: str
    prompt: str
    reference: Optional[str] = None
    choices: Optional[list[str]] = None
    correct_choice: Optional[int] = None
    task_type: TaskType
    metadata: dict[str, Any] = Field(default_factory=dict)

class LLMResponse(BaseModel):
    sample_id: str
    model: str
    provider: str
    output: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class MetricScore(BaseModel):
    sample_id: str
    model: str
    scorer_name: str
    score: float
    explanation: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class EvalResult(BaseModel):
    run_id: str
    model: str
    dataset: str
    scores: dict[str, float]
    total_cost_usd: float
    total_samples: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)