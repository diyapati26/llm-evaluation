import asyncio
import os
from openai import AsyncOpenAI
from pydantic import BaseModel

class JudgeOutput(BaseModel):
    """Structured output the judge must return."""
    score:     int         # 1 to 5
    reasoning: str         # Why this score
    issues:    list[str]   # Specific problems found, empty if none

JUDGE_PROMPT = """You are an expert evaluator assessing the quality of an AI model's response.

QUESTION:
{question}

MODEL RESPONSE:
{response}

REFERENCE ANSWER:
{reference}

Evaluate the response on these criteria:
1. Factual accuracy — is the information correct?
2. Completeness — does it fully address the question?
3. Clarity — is it clearly expressed?

Think step by step. Then provide:
- score: integer from 1 (very poor) to 5 (excellent)
- reasoning: your evaluation in 1-2 sentences
- issues: list of specific problems, empty list [] if none

Respond in JSON only. No other text."""


class LLMJudgeScorer:
    """
    Uses GPT-5.4-mini to evaluate response quality.
    Implements G-Eval style chain-of-thought scoring.
    Returns structured scores with reasoning.
    """
    name = "llm_judge"

    def __init__(self, api_key: str, judge_model: str = "gpt-5.4-mini"):  # ← updated
        self.client      = AsyncOpenAI(api_key=api_key)
        self.judge_model = judge_model

    async def score(
        self,
        prediction: str,
        reference:  str,
        question:   str = "",
        **kwargs
    ) -> dict:

        if not prediction:
            return {
                "score":     0.0,
                "raw_score": 0,
                "reasoning": "Empty prediction",
                "issues":    ["No response generated"],
            }

        prompt = JUDGE_PROMPT.format(
            question=question or "Not provided",
            response=prediction,
            reference=reference or "Not provided",
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_completion_tokens=300,              # ← fixed
                response_format={"type": "json_object"},
            )

            import json
            raw = json.loads(response.choices[0].message.content)

            # Validate score is in range
            raw_score = max(1, min(5, int(raw.get("score", 3))))

            # Normalize 1-5 to 0-1
            normalized = (raw_score - 1) / 4

            return {
                "score":     round(normalized, 4),
                "raw_score": raw_score,
                "reasoning": raw.get("reasoning", ""),
                "issues":    raw.get("issues", []),
            }

        except Exception as e:
            return {
                "score":     0.0,
                "raw_score": 0,
                "reasoning": f"Judge error: {str(e)}",
                "issues":    ["Scoring failed"],
            }