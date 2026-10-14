"""City-scale visual grounding environment: stateless tools, OpenAI tool protocol."""

from .env import CityGroundingEnv
from .tool_env import (
    CityToolEnv,
    OPENAI_TOOL_SCHEMAS,
    SYSTEM_PROMPT,
    TOOL_METHODS,
    parse_answer,
)
from .reward import (
    compute_terminal_reward,
    reason_to_id,
    IOU_WEIGHT,
    STEP_COST,
    IOU_THRESHOLD,
)

__all__ = [
    "CityGroundingEnv",
    "CityToolEnv",
    "OPENAI_TOOL_SCHEMAS",
    "SYSTEM_PROMPT",
    "TOOL_METHODS",
    "parse_answer",
    "compute_terminal_reward",
    "reason_to_id",
    "IOU_WEIGHT",
    "STEP_COST",
    "IOU_THRESHOLD",
]
