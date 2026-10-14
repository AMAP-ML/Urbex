"""Mock-data smoke test for CityGroundingEnv (no BEV files needed).

Stubs the inner CityToolEnv methods so we can exercise the OpenAI native
protocol without needing real PNGs:
  • reset returns (messages, info) with info["tools"] = OPENAI_TOOL_SCHEMAS
  • <tool_call> dispatch appends role="tool" message with image content
  • <answer> commit emits the terminal reward
  • no tool_call + no answer -> "no_answer" terminal
  • unknown tool -> graceful error message in tool reply

Run with `pytest scripts/tests/agentic/env/test_city_grounding.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from roll.pipeline.agentic.env.city_grounding import CityGroundingEnv
from roll.pipeline.agentic.env.city_grounding.env import TOOL_ERROR_PENALTY
from roll.pipeline.agentic.env.city_grounding.reward import IOU_WEIGHT, STEP_COST


def _make_dataset(tmp: Path) -> str:
    p = tmp / "ds.jsonl"
    p.write_text(json.dumps({
        "query": "the red car next to the bench",
        "scene_id": "scene_x",
        "dataset": "cityrefer",
        "gt_bbox": [10.0, 20.0, 0.0, 4.0, 4.0, 1.0],
        "bev_png": "<unused>",
    }) + "\n")
    return str(p)


class _StubInner:
    """Replaces CityToolEnv so reset/tool methods don't need real PNGs."""

    def __init__(self, *_, **__):
        self.MAX_TOOL_CALLS = 6

    def reset(self, _sample):
        return {"image": Image.new("RGB", (8, 8), (1, 2, 3))}

    def tool_zoom_in(self, args):
        return f"zoomed to {args}", [Image.new("RGB", (4, 4), (10, 0, 0))], {"view_id": "Z1"}

    def tool_get_landmark_xy(self, args):
        return f"landmarks for {args}", [], {"n_resolved": 1}

    def tool_render_oblique(self, args):
        return f"oblique {args}", [Image.new("RGB", (4, 4), (0, 10, 0))], {}

    def score_answer(self, _bbox, _view_id):
        return {"world_bbox": [1, 2, 3, 4], "iou_2d": 0.42, "success": True}


def _make_env(tmp: Path) -> CityGroundingEnv:
    data_path = _make_dataset(tmp)
    with patch(
        "roll.pipeline.agentic.env.city_grounding.env.CityToolEnv",
        _StubInner,
    ):
        return CityGroundingEnv(
            data_path=data_path,
            mode="val",
            seed=0,
            max_tool_calls=6,
            max_assistant_turns=7,
        )


def test_reset_returns_messages_and_tools(tmp_path):
    env = _make_env(tmp_path)
    messages, info = env.reset(seed=0)

    # OpenAI shape: [system, user-with-image]
    assert isinstance(messages, list) and len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"]  # bare SYSTEM_PROMPT, not empty
    assert messages[1]["role"] == "user"
    user_content = messages[1]["content"]
    assert isinstance(user_content, list)
    assert any(c.get("type") == "text" and "Query: the red car" in c["text"] for c in user_content)
    assert any(c.get("type") == "image" and c.get("image") is not None for c in user_content)

    # Tools list is populated -- env_manager will pass to apply_chat_template
    assert isinstance(info["tools"], list) and len(info["tools"]) == 3
    names = [t["function"]["name"] for t in info["tools"]]
    assert names == ["get_landmark_xy", "zoom_in", "render_oblique"]


def test_tool_call_appends_assistant_then_tool(tmp_path):
    env = _make_env(tmp_path)
    env.reset(seed=0)
    action = (
        '<think>look closer</think>'
        '<tool_call>{"name":"zoom_in","arguments":{"view_id":"global","bbox_2d":[100,200,300,400]}}</tool_call>'
    )
    messages, reward, terminated, truncated, info = env.step(action)
    assert not terminated and not truncated
    assert reward == 0.0

    # Last two messages: assistant (model output verbatim) + tool reply
    assert messages[-2]["role"] == "assistant"
    assert "<tool_call>" in messages[-2]["content"]
    assert messages[-1]["role"] == "tool"
    assert messages[-1].get("name") == "zoom_in"
    tool_content = messages[-1]["content"]
    assert any(c.get("type") == "text" and "zoomed" in c["text"] for c in tool_content)
    assert any(c.get("type") == "image" and c.get("image") is not None for c in tool_content)
    assert info["metrics"]["tool_call_valid"] == 1.0


def test_unknown_tool_is_graceful(tmp_path):
    env = _make_env(tmp_path)
    env.reset(seed=0)
    messages, reward, terminated, truncated, info = env.step(
        '<tool_call>{"name":"frobnicate","arguments":{}}</tool_call>'
    )
    assert not terminated
    assert reward == -TOOL_ERROR_PENALTY
    assert messages[-1]["role"] == "tool"
    assert "unknown tool" in messages[-1]["content"]
    assert info["metrics"]["tool_call_valid"] == 0.0


def test_terminal_commit_success(tmp_path):
    env = _make_env(tmp_path)
    env.reset(seed=0)
    action = '<answer>{"bbox_2d":[100,100,200,200],"view_id":"global"}</answer>'
    messages, reward, terminated, truncated, info = env.step(action)
    assert terminated and not truncated
    expected =  IOU_WEIGHT * 0.42 - STEP_COST * (1 / 7)
    assert abs(reward - expected) < 1e-6
    # Final assistant turn recorded; no tool reply was appended.
    assert messages[-1]["role"] == "assistant"
    m = info["metrics"]
    assert m["city_success"] == 1.0
    assert abs(m["city_iou_2d"] - 0.42) < 1e-6
    assert m["city_reason_id"] == 0.0


def test_terminal_no_answer(tmp_path):
    env = _make_env(tmp_path)
    env.reset(seed=0)
    messages, reward, terminated, truncated, info = env.step("free-form ramble, no tags")
    assert terminated and truncated
    # IoU 0 and infinite centre distance leave only the step cost.
    expected = -STEP_COST * (1 / 7)
    assert abs(reward - expected) < 1e-6
    assert info["metrics"]["city_reason_id"] == 1.0


def test_last_turn_tool_call_emits_no_answer(tmp_path):
    """When the manager will force-terminate on the next step
    (assistant_turns == max), a tool_call action must trigger the no_answer
    terminal so the step cost is recorded. Otherwise "ran out of turns"
    rollouts score 0 and contribute no variance to their GRPO group.
    """
    env = _make_env(tmp_path)
    env.reset(seed=0)
    # Burn 6 of 7 turns with valid tool_calls.
    for _ in range(6):
        messages, _r, terminated, _, _ = env.step(
            '<tool_call>{"name":"zoom_in","arguments":{"view_id":"global","bbox_2d":[1,2,3,4]}}</tool_call>'
        )
        assert not terminated
    # Turn 7 with another tool_call: env should NOT dispatch -- must terminate
    # as no_answer with the step_cost penalty.
    _msgs, reward, terminated, truncated, info = env.step(
        '<tool_call>{"name":"zoom_in","arguments":{"view_id":"global","bbox_2d":[1,2,3,4]}}</tool_call>'
    )
    assert terminated and truncated
    # Failure at turn 7 of 7 → full step cost, no IoU or distance credit.
    expected = -STEP_COST
    assert abs(reward - expected) < 1e-6
    assert info["metrics"]["city_reason_id"] == 1.0  # no_answer


def test_last_turn_commit_still_works(tmp_path):
    """The force-terminate path must not block a legitimate last-turn commit."""
    env = _make_env(tmp_path)
    env.reset(seed=0)
    for _ in range(6):
        env.step('<tool_call>{"name":"get_landmark_xy","arguments":{"names":["X"]}}</tool_call>')
    _msgs, reward, terminated, truncated, info = env.step(
        '<answer>{"bbox_2d":[1,2,3,4],"view_id":"global"}</answer>'
    )
    assert terminated and not truncated
    assert info["metrics"]["city_reason_id"] == 0.0  # answer
    expected = IOU_WEIGHT * 0.42 - STEP_COST  # 7/7
    assert abs(reward - expected) < 1e-6


def test_multi_turn_tool_then_commit(tmp_path):
    env = _make_env(tmp_path)
    env.reset(seed=0)
    env.step('<tool_call>{"name":"get_landmark_xy","arguments":{"names":["X"]}}</tool_call>')
    env.step('<tool_call>{"name":"zoom_in","arguments":{"view_id":"global","bbox_2d":[1,2,3,4]}}</tool_call>')
    messages, reward, terminated, _, info = env.step(
        '<answer>{"bbox_2d":[100,100,200,200],"view_id":"Z1"}</answer>'
    )
    assert terminated
    assert info["metrics"]["city_assistant_turns"] == 3.0
    expected = IOU_WEIGHT * 0.42 - STEP_COST * (3 / 7)
    assert abs(reward - expected) < 1e-6
    # Conversation: system, user, asst, tool, asst, tool, asst -> 7 messages
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
