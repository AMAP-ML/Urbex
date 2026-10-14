from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


RELEASE_ROOT = Path(__file__).resolve().parents[2]


def _load_api_baseline():
    path = RELEASE_ROOT / "scripts" / "eval" / "api_baseline.py"
    spec = importlib.util.spec_from_file_location("release_api_baseline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def api_baseline():
    return _load_api_baseline()


def _record(scene_id: str, query: str) -> dict:
    return {
        "scene_id": scene_id,
        "query": query,
        "dataset": "cityrefer",
        "iou_2d": 0.0,
        "center_dist": 1.0,
        "success": False,
        "assistant_turns": 1,
        "reason": "answer",
        "messages": [],
    }


def _worker_args() -> SimpleNamespace:
    return SimpleNamespace(
        provider="local",
        base_url="http://localhost:8000/v1",
        model="test-model",
        max_tokens=16,
        temperature=0.0,
        top_p=1.0,
        no_thinking=True,
        reasoning_effort="none",
        display_max_side=64,
        display_patch=16,
        display_max_pixels=4096,
        multiview_cache_dir="",
        max_tool_calls=1,
        bbox_norm_max=1000.0,
        landmark_mode="clean",
        landmark_noise_std_m=0.0,
        landmark_wrong_min_distance_m=50.0,
        landmark_seed=42,
        blacklist_file=None,
        vis_dir="",
        max_turns=1,
        bbox_order="xyxy",
        forced_tool_sequence=[],
        iou_threshold=0.0,
    )


def test_worker_resume_writes_unique_append_only_shards(
    api_baseline, monkeypatch, tmp_path
):
    monkeypatch.setattr(api_baseline, "DashScopeClient", lambda **_: object())
    monkeypatch.setattr(api_baseline, "CityToolEnv", lambda **_: object())
    monkeypatch.setattr(api_baseline, "make_recorder", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        api_baseline,
        "run_episode",
        lambda sample, *_args, **_kwargs: sample["record"],
    )

    first = {"scene_id": "scene-a", "query": "query-a", "record": _record("scene-a", "query-a")}
    second = {"scene_id": "scene-b", "query": "query-b", "record": _record("scene-b", "query-b")}
    args = _worker_args()

    assert api_baseline._worker_loop([first], 0, "run-one", str(tmp_path), args) == 1
    first_path = tmp_path / "api_rollout_worker0_run-one.jsonl"
    first_contents = first_path.read_text()

    assert api_baseline._worker_loop([second], 0, "run-two", str(tmp_path), args) == 1

    assert first_path.read_text() == first_contents
    assert json.loads(first_contents)["scene_id"] == "scene-a"
    assert json.loads(
        (tmp_path / "api_rollout_worker0_run-two.jsonl").read_text()
    )["scene_id"] == "scene-b"


def test_resume_and_merge_skip_truncated_lines(api_baseline, tmp_path):
    first = _record("scene-a", "query-a")
    second = _record("scene-b", "query-b")
    third = _record("scene-c", "query-c")
    legacy = tmp_path / "api_rollout_worker0.jsonl"
    current = tmp_path / "api_rollout_worker0_run-two.jsonl"
    legacy.write_text(
        json.dumps(first) + "\n" + '{"scene_id": "truncated"\n' + json.dumps(second) + "\n"
    )
    current.write_text(json.dumps(first) + "\n" + json.dumps(third))

    assert api_baseline._get_done_keys(str(tmp_path)) == {
        ("scene-a", "query-a"),
        ("scene-b", "query-b"),
        ("scene-c", "query-c"),
    }

    api_baseline._dedup_dump(str(tmp_path))
    merged = [json.loads(line) for line in (tmp_path / "merged.jsonl").open()]
    dedup = [json.loads(line) for line in (tmp_path / "dedup.jsonl").open()]
    assert len(merged) == 4
    assert {(row["scene_id"], row["query"]) for row in dedup} == {
        ("scene-a", "query-a"),
        ("scene-b", "query-b"),
        ("scene-c", "query-c"),
    }


def test_main_rebuilds_aggregate_when_every_sample_is_done(
    api_baseline, monkeypatch, tmp_path
):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps(_record("scene-a", "query-a")) + "\n")
    shard = tmp_path / "api_rollout_worker0_previous.jsonl"
    shard.write_text(json.dumps(_record("scene-a", "query-a")) + "\n")
    calls = []
    monkeypatch.setattr(api_baseline, "_dedup_dump", calls.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "api_baseline.py",
            "--provider",
            "local",
            "--data_path",
            str(dataset),
            "--output_dir",
            str(tmp_path),
        ],
    )

    api_baseline.main()

    assert calls == [str(tmp_path)]


def test_provider_credentials_are_isolated(api_baseline, monkeypatch):
    for env_var in ("DASHSCOPE_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-secret")

    assert api_baseline._resolve_api_key("dashscope") == "dashscope-secret"
    assert api_baseline._resolve_api_key("openai") == "openai-secret"
    assert api_baseline._resolve_api_key("gemini") == "gemini-secret"
    assert api_baseline._resolve_api_key("local") == "EMPTY"

    monkeypatch.delenv("GEMINI_API_KEY")
    assert api_baseline._resolve_api_key("gemini") == "google-secret"


def test_explicit_programmatic_key_takes_precedence(api_baseline, monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(api_baseline, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")

    client = api_baseline.DashScopeClient(
        api_key="explicit-secret",
        provider="openai",
    )

    assert client.api_key == "explicit-secret"
    assert captured["api_key"] == "explicit-secret"


def test_wrapper_keeps_api_key_out_of_argv_and_logs(tmp_path):
    capture_path = tmp_path / "argv.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$CAPTURE_PATH\"\n"
    )
    fake_python.chmod(0o755)
    sentinel = "SECRET_SENTINEL_DO_NOT_LOG"
    env = os.environ.copy()
    env.update(
        {
            "PYTHON": str(fake_python),
            "CAPTURE_PATH": str(capture_path),
            "OPENAI_API_KEY": sentinel,
            "CITY_AGENT_DATA": str(tmp_path / "data"),
            "MODEL": "test-model",
            "OUTPUT_ROOT": str(tmp_path / "output"),
            "NUM_SAMPLES": "1",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/eval/run_api_baseline.sh", "openai"],
        cwd=RELEASE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    argv = capture_path.read_text()

    assert sentinel not in result.stdout
    assert sentinel not in result.stderr
    assert sentinel not in argv
    assert "--api_key" not in argv
    assert "--api-key" not in argv
    assert "--provider\nopenai\n" in argv
