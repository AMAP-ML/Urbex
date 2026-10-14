"""Reward regressions using real BEV projection and tool dispatch.

Only the graphics boundary is mocked; no OpenGL or model is needed.
"""

import json
import math

import pytest
from PIL import Image

from roll.pipeline.agentic.env.city_grounding import CityGroundingEnv


TARGET_BOX = [400, 400, 600, 600]
ANSWER_CASES = [
    (TARGET_BOX, 1.0, 0.0),
    ([400, 400, 500, 500], 0.25, math.sqrt(50)),
    ([50, 50, 100, 100], 0.0, math.hypot(42.5, 42.5)),
]


@pytest.fixture
def env(tmp_path, monkeypatch):
    bev_path = tmp_path / "fixture.png"
    Image.new("RGB", (100, 100), (20, 30, 40)).save(bev_path)
    bev_path.with_suffix(".json").write_text(json.dumps({
        "x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100,
        "res_x": 1, "res_y": 1,
    }))
    data_path = tmp_path / "fixture.jsonl"
    data_path.write_text(json.dumps({
        "query": "the building in the scene",
        "scene_id": "fixture",
        "dataset": "cityrefer",
        "gt_bbox": [50, 50, 0, 20, 20, 2],
        "bev_png": str(bev_path),
    }) + "\n")
    instance = CityGroundingEnv(data_path=str(data_path), mode="val")
    monkeypatch.setattr(
        instance._env, "_render_oblique",
        lambda world_bbox, only_direction=None: [Image.new("RGB", (4, 4)) for _ in range(4)],
    )
    instance.reset(seed=0)
    return instance


def _render(bbox=TARGET_BOX, **overrides):
    args = {"view_id": "global", "bbox_2d": bbox, "direction": "east", **overrides}
    return '<tool_call>' + json.dumps({"name": "render_oblique", "arguments": args}) + '</tool_call>'


def _answer(bbox=TARGET_BOX):
    return '<answer>' + json.dumps({"view_id": "global", "bbox_2d": bbox}) + '</answer>'


@pytest.mark.parametrize("bbox,iou,_distance", ANSWER_CASES)
def test_render_rewards_actual_candidate_overlap(env, bbox, iou, _distance):
    messages, reward, terminated, truncated, info = env.step(_render(bbox))
    assert not terminated and not truncated
    assert info["render_iou"] == pytest.approx(iou)
    assert reward == pytest.approx(0.05 * iou)
    assert info["metrics"]["city_step_reward_render_iou"] == pytest.approx(reward)
    assert sum(item["type"] == "image" for item in messages[-1]["content"]) == 1


@pytest.mark.parametrize("render_count", [0, 1, 2])
@pytest.mark.parametrize("bbox,iou,distance", ANSWER_CASES)
def test_terminal_reward_has_no_render_bonus(env, render_count, bbox, iou, distance):
    tool_reward = 0.0
    for _ in range(render_count):
        tool_reward += env.step(_render())[1]
    _messages, reward, terminated, truncated, info = env.step(_answer(bbox))
    assert terminated and not truncated
    expected = iou + 0.3 * math.exp(-distance / 80) - 0.05 * (render_count + 1) / 8
    assert reward == pytest.approx(expected)
    assert tool_reward + reward == pytest.approx(0.05 * render_count + expected)
    assert info["metrics"]["city_assistant_turns"] == render_count + 1


def test_no_answer_after_render_receives_only_terminal_step_cost(env):
    assert env.step(_render())[1] == pytest.approx(0.05)
    _messages, reward, terminated, truncated, _info = env.step("no answer")
    assert terminated and truncated
    assert reward == pytest.approx(-0.05 * 2 / 8)


@pytest.mark.parametrize("views,error", [
    (None, "no_oblique_views"),
    ([None, None, None, None], "missing_oblique_direction"),
])
def test_unavailable_render_is_penalized(env, monkeypatch, views, error):
    monkeypatch.setattr(env._env, "_render_oblique", lambda *args, **kwargs: views)
    messages, reward, terminated, truncated, info = env.step(_render())
    assert not terminated and not truncated
    assert reward == pytest.approx(-0.05)
    assert info["error"] == error
    assert info["metrics"]["tool_error"] == 1.0
    assert len(messages[-1]["content"]) == 1
    assert messages[-1]["content"][0]["text"].startswith("error:")


def test_ground_truth_changes_reward_without_changing_observation(env):
    messages, reward, *_ = env.step(_render())
    first_text = messages[-1]["content"][0]["text"]
    assert reward == pytest.approx(0.05)
    env.reset(seed=0)
    env._env.sample["gt_bbox"] = [10, 10, 0, 20, 20, 2]
    messages, reward, *_ = env.step(_render())
    assert reward == 0.0
    assert messages[-1]["content"][0]["text"] == first_text


@pytest.mark.parametrize("commit", [True, False])
def test_default_budget_allows_seven_tools_and_final_answer(env, commit):
    assert env._env.MAX_TOOL_CALLS == 7
    for call in range(1, 8):
        messages, reward, terminated, truncated, _info = env.step(_render())
        assert not terminated and not truncated
        assert reward == pytest.approx(0.05)
        if call == 6:
            assert "ONE TOOL CALL LEFT" in messages[-1]["content"][0]["text"]
        if call == 7:
            assert "BUDGET EXHAUSTED" in messages[-1]["content"][0]["text"]
    action = _answer() if commit else _render()
    _messages, reward, terminated, truncated, info = env.step(action)
    assert terminated
    assert truncated == (not commit)
    assert reward == pytest.approx(1.3 - 0.05 if commit else -0.05)
    assert info["metrics"]["city_assistant_turns"] == 8
    assert env._env.tool_calls_used == 7
