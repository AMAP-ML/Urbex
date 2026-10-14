"""Tests for controlled landmark robustness conditions."""

from __future__ import annotations

from unittest.mock import patch

from roll.pipeline.agentic.env.city_grounding.tool_env import CityToolEnv


def make_env(mode: str, **kwargs) -> CityToolEnv:
    """Construct a tool env without loading scene assets."""
    env = CityToolEnv(landmark_mode=mode, **kwargs)
    env.sample = {
        "scene_id": "scene-a",
        "query": "the car near Alpha Hall",
        "dataset": "cityrefer",
    }
    return env


def test_missing_landmark_always_returns_none():
    env = make_env("missing")
    hit = {
        "object_name": "Alpha Hall",
        "bbox": [100.0, 200.0],
        "score": 1.0,
    }
    assert env._perturb_landmark_hit("Alpha Hall", hit) is None


def test_noisy_landmark_is_deterministic_and_does_not_mutate_input():
    env = make_env("noisy", landmark_noise_std_m=10.0, landmark_seed=7)
    hit = {
        "object_name": "Alpha Hall",
        "bbox": [100.0, 200.0],
        "score": 1.0,
    }
    first = env._perturb_landmark_hit("Alpha Hall", hit)
    second = env._perturb_landmark_hit("Alpha Hall", hit)

    assert first == second
    assert first is not None
    assert first["bbox"] != hit["bbox"]
    assert hit["bbox"] == [100.0, 200.0]


def test_wrong_landmark_uses_different_scene_landmark():
    env = make_env("wrong", landmark_wrong_min_distance_m=50.0, landmark_seed=3)
    hit = {
        "object_name": "Alpha Hall",
        "bbox": [100.0, 200.0],
        "score": 1.0,
    }
    alternatives = [
        {
            "object_name": "Alpha Hall",
            "bbox": [100.0, 200.0],
            "score": 1.0,
        },
        {
            "object_name": "Beta Library",
            "bbox": [160.0, 200.0],
            "score": 1.0,
        },
        {
            "object_name": "Gamma House",
            "bbox": [300.0, 200.0],
            "score": 1.0,
        },
    ]
    with patch(
        "roll.pipeline.agentic.env.city_grounding.tools.retrieve.list_landmarks_for_scene",
        return_value=alternatives,
    ):
        wrong = env._perturb_landmark_hit("Alpha Hall", hit)

    assert wrong is not None
    assert wrong["object_name"] != "Alpha Hall"
    assert wrong["bbox"] != hit["bbox"]


def test_invalid_landmark_mode_is_rejected():
    try:
        CityToolEnv(landmark_mode="random")
    except ValueError as error:
        assert "unsupported landmark_mode" in str(error)
    else:
        raise AssertionError("invalid landmark mode should raise ValueError")
