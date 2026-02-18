from __future__ import annotations

from pathlib import Path
from typing import Any

import openai
import pandas as pd
from pydantic import BaseModel, Field

from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool

# Grader template for 3-way classification (Correct/Incorrect/Not Attempted)
GRADER_TEMPLATE = """You are an expert evaluator for Chinese-language question answering tasks.

Your task: Classify the student's answer into one of three categories based on the reference answer.

**Question (Chinese):**
{question}

**Reference Answer (Chinese):**
{reference_answer}

**Student Answer (Chinese):**
{student_answer}

**Classification Guidelines:**
1. CORRECT: The answer is semantically equivalent to the reference answer. Accept synonyms, paraphrasing, and different wordings that convey the same meaning.
2. INCORRECT: The answer contradicts the reference answer or provides wrong information.
3. NOT_ATTEMPTED: The answer indicates the student doesn't know (e.g., "我不知道", "无法回答", "不清楚") or is empty/evasive.

**Important**: Focus on semantic meaning in Chinese, not exact character matching.

**Output Format:**
First, provide brief reasoning (1-2 sentences) explaining your decision.
Then, on a new line, write EXACTLY one of:
- CORRECT
- INCORRECT
- NOT_ATTEMPTED
"""

# Data loading with path checking (production /orwd_data first, then local fallback)
if Path("/orwd_data/").exists():
    DATA_PATH = Path("/orwd_data/")
else:
    DATA_PATH = Path(__file__).parent

# Load dataset at module import time
df = pd.read_parquet(DATA_PATH / "chinese_simpleqa.parquet")
TASKS = df[["id", "question", "answer"]].to_dict(orient="records")


class ChineseSimpleQATaskSpec(BaseModel):
    """Task specification for Chinese-SimpleQA."""

    id: str
    question: str
    answer: str  # Hidden from agent, used for grading


class SubmitAnswerInput(BaseModel):
    """Input schema for submit_answer tool."""

    answer: str = Field(..., description="Your answer in Chinese (你的中文答案)")


class ChineseSimpleQA(Environment):
    """
    Chinese-SimpleQA environment for evaluating factuality in Chinese language.

    This environment implements a 3,000-task Chinese Q&A benchmark with
    LLM-based grading using gpt-5-mini to classify answers as:
    - CORRECT: Semantically equivalent to reference
    - INCORRECT: Wrong or contradictory
    - NOT_ATTEMPTED: Model indicates uncertainty or doesn't answer

    Dataset: https://huggingface.co/datasets/OpenStellarTeam/Chinese-SimpleQA
    Paper: https://arxiv.org/abs/2411.07140
    """

    @classmethod
    def list_splits(cls) -> list[str]:
        """Return available data splits."""
        return ["test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        """Return task specifications for given split."""
        if split != "test":
            raise ValueError(f"Unknown split: {split}. Available splits: test")
        return TASKS

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        """
        Initialize environment with task specification and secrets.

        Args:
            task_spec: Task specification containing id, question, answer
            secrets: Must contain 'openai_api_key' for grading

        Raises:
            ValueError: If openai_api_key not provided in secrets
        """
        super().__init__(task_spec)

        # CRITICAL: Validate API key from secrets (no environment variable fallback)
        api_key = secrets.get("openai_api_key")
        if not api_key:
            raise ValueError(
                "OpenAI API key required for grading. "
                "Pass via secrets={'openai_api_key': 'sk-...'}"
            )

        self.client = openai.AsyncClient(api_key=api_key)
        self.config = ChineseSimpleQATaskSpec.model_validate(task_spec)

    async def get_prompt(self) -> str:
        """Return the Chinese question prompt for the agent."""
        return [TextBlock(text=self.config.question)]

    async def _grade_answer(self, student_answer: str) -> dict[str, Any]:
        """
        Grade answer using gpt-5-mini (3-way classification).

        Args:
            student_answer: The answer submitted by the agent

        Returns:
            dict with keys:
                - classification: "correct" | "incorrect" | "not_attempted"
                - grading_response: str (full LLM reasoning)
                - reward: float (1.0 for correct, 0.0 otherwise)
        """
        grader_prompt = GRADER_TEMPLATE.format(
            question=self.config.question,
            reference_answer=self.config.answer,
            student_answer=student_answer,
        )

        try:
            # Use gpt-5-mini (NO temperature parameter per documentation)
            response = await self.client.chat.completions.create(
                model="gpt-5-mini", messages=[{"role": "user", "content": grader_prompt}]
            )

            grading_response = response.choices[0].message.content or ""
            upper_response = grading_response.upper()

            # Parse classification (priority order matters)
            if "NOT_ATTEMPTED" in upper_response or "NOT ATTEMPTED" in upper_response:
                classification = "not_attempted"
                reward = 0.0
            elif "CORRECT" in upper_response and "INCORRECT" not in upper_response:
                classification = "correct"
                reward = 1.0
            else:
                classification = "incorrect"
                reward = 0.0

            return {
                "classification": classification,
                "grading_response": grading_response,
                "reward": reward,
            }

        except Exception as e:
            # Conservative fallback on error: assume incorrect
            return {
                "classification": "incorrect",
                "grading_response": f"Grading error: {str(e)}",
                "reward": 0.0,
            }

    @tool
    async def submit_answer(self, params: SubmitAnswerInput) -> ToolOutput:
        """
        Submit your answer for evaluation.

        The answer will be graded using an LLM-based evaluator that classifies
        it as CORRECT, INCORRECT, or NOT_ATTEMPTED based on semantic equivalence
        to the reference answer.
        """
        # Validate non-empty (early check before calling grader)
        if not params.answer.strip():
            return ToolOutput(
                blocks=[TextBlock(text="❌ Empty answer submitted.")],
                metadata={"error": "empty_answer"},
                reward=0.0,
                finished=True,
            )

        # Grade answer using LLM
        grader_result = await self._grade_answer(params.answer)

        # Format display with emoji
        emoji_map = {
            "correct": "✅",
            "incorrect": "❌",
            "not_attempted": "⚠️",
        }
        emoji = emoji_map.get(grader_result["classification"], "❓")
        label = grader_result["classification"].upper().replace("_", " ")

        display_text = f"""{emoji} {label}

{grader_result['grading_response']}

Your Answer: {params.answer}
Reference Answer: {self.config.answer}
"""

        return ToolOutput(
            blocks=[TextBlock(text=display_text)],
            metadata={
                "task_id": self.config.id,
                "question": self.config.question,
                "student_answer": params.answer,
                "reference_answer": self.config.answer,
                "classification": grader_result["classification"],
                "grader_reasoning": grader_result["grading_response"],
            },
            reward=grader_result["reward"],
            finished=True,
        )
