"""Paper rewards for city-grounding RL.

    R = iou_weight * IoU
        + dist_weight * exp(-center_dist / dist_decay)
        - step_cost * (turns / max_turns)

Tool shaping is added on the tool step, separately from this terminal
reward. Successful oblique rendering receives a capped candidate-IoU
reward; there is no additional render bonus when submitting an answer.
"""

from __future__ import annotations

import math

IOU_WEIGHT    = 1.0
DIST_WEIGHT   = 0.3
DIST_DECAY    = 80.0          # metres; e-folding distance
STEP_COST     = 0.05
RENDER_IOU_WEIGHT = 0.05
RENDER_REWARD_CAP = 0.05
IOU_THRESHOLD = 0.1          # used by env.py for the metric only,
                              # not by the reward formula


def compute_terminal_reward(
    iou_2d: float,
    center_dist: float,
    assistant_turns: int,
    max_assistant_turns: int,
) -> dict:
    """Return a dict of scalar terms (sum is the reward).

    The dict keeps each term separate so callers can surface them as
    metrics for tensorboard.
    """
    iou_term  = IOU_WEIGHT * float(iou_2d)
    dist_reward = DIST_WEIGHT * math.exp(-float(center_dist) / DIST_DECAY)

    norm_turns = min(int(assistant_turns), max_assistant_turns) / max(1, max_assistant_turns)
    step_term = STEP_COST * norm_turns

    return {
        "reward":      iou_term + dist_reward - step_term,
        "iou_term":    iou_term,
        "dist_reward": dist_reward,
        "step_term":   step_term,
    }


def compute_render_reward(iou_2d: float) -> float:
    """Paper tool shaping: candidate overlap, capped per successful render."""
    return min(RENDER_REWARD_CAP, RENDER_IOU_WEIGHT * max(0.0, float(iou_2d)))


REASON_IDS = {
    "answer":       0.0,
    "no_answer":    1.0,
    "truncated":    2.0,
    "invalid":      3.0,
    "tool_error":   4.0,
}


def reason_to_id(reason: str) -> float:
    return REASON_IDS.get(reason, 5.0)
