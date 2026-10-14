from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from roll.pipeline.agentic.env.city_grounding import _asset_build_parallel as parallel_builder
from roll.pipeline.agentic.env.city_grounding.asset_paths import (
    load_scene_ids,
    resolve_ply_path,
)
from roll.pipeline.agentic.env.city_grounding.build_bev import (
    build_scene_bev,
    render_bev,
)
from roll.pipeline.agentic.env.city_grounding import build_bev as bev_builder
from roll.pipeline.agentic.env.city_grounding import build_verify_cache as cache_builder
from roll.pipeline.agentic.env.city_grounding.render import (
    _load_bev_cached,
    load_bev,
    pixel_to_world,
    world_to_pixel,
)
from roll.pipeline.agentic.env.city_grounding.verify_cache import BboxCache


def _parallel_probe_job(job: dict[str, object]) -> dict:
    time.sleep(float(job.get("delay", 0)))
    if job.get("fail"):
        raise RuntimeError("probe failure")
    return {"value": job["scene_id"], "pid": os.getpid()}


def _write_jsonl(path: Path, dataset: str, scenes: list[str]) -> None:
    path.write_text(
        "".join(json.dumps({"dataset": dataset, "scene_id": scene}) + "\n" for scene in scenes),
        encoding="utf-8",
    )


def test_scene_loading_and_dataset_paths(tmp_path):
    city_jsonl = tmp_path / "city.jsonl"
    _write_jsonl(city_jsonl, "cityrefer", ["scene_b", "scene_a", "scene_a"])
    assert load_scene_ids(city_jsonl, "cityrefer") == ["scene_a", "scene_b"]
    assert load_scene_ids(city_jsonl, "cityrefer", ["scene_b"]) == ["scene_b"]
    with pytest.raises(ValueError, match="absent"):
        load_scene_ids(city_jsonl, "cityrefer", ["missing"])

    city_root = tmp_path / "raw_city"
    city_ply = city_root / "sensaturban" / "ply" / "test" / "scene_a.ply"
    city_ply.parent.mkdir(parents=True)
    city_ply.touch()
    assert resolve_ply_path("scene_a", "cityrefer", city_root) == city_ply.resolve()

    stpls_root = tmp_path / "raw_stpls"
    stpls_ply = stpls_root / "STPLS3D" / "RealWorldData" / "USC_points.ply"
    stpls_ply.parent.mkdir(parents=True)
    stpls_ply.touch()
    assert resolve_ply_path("USC", "stpls3d", stpls_root) == stpls_ply.resolve()


def test_scene_loading_merges_splits_before_selecting(tmp_path):
    nd = tmp_path / "nd.jsonl"
    no = tmp_path / "no.jsonl"
    empty = tmp_path / "empty.jsonl"
    _write_jsonl(nd, "cityrefer", ["seen", "shared"])
    _write_jsonl(no, "cityrefer", ["held_out", "shared"])
    empty.touch()
    paths = [nd, no, empty]
    assert load_scene_ids(paths, "cityrefer") == ["held_out", "seen", "shared"]
    assert load_scene_ids(paths, "cityrefer", ["held_out", "seen"]) == ["held_out", "seen"]
    with pytest.raises(ValueError, match="absent"):
        load_scene_ids(paths, "cityrefer", ["missing"])
    _write_jsonl(no, "stpls3d", ["USC"])
    with pytest.raises(ValueError, match="dataset mismatch"):
        load_scene_ids(paths, "cityrefer")


@pytest.mark.parametrize("builder", [bev_builder, cache_builder])
def test_builders_schedule_each_scene_once_across_splits(tmp_path, monkeypatch, builder):
    nd = tmp_path / "nd.jsonl"
    no = tmp_path / "no.jsonl"
    _write_jsonl(nd, "cityrefer", ["seen", "shared"])
    _write_jsonl(no, "cityrefer", ["held_out", "shared"])
    monkeypatch.setattr(sys, "argv", [
        "builder", "--dataset", "cityrefer", "--jsonl", str(nd), str(no),
        "--ply_root", str(tmp_path), "--out", str(tmp_path / "out"),
    ])
    scheduled = []

    def capture_jobs(jobs, _worker, _workers):
        scheduled.extend(job["scene_id"] for job in jobs)
        return []

    monkeypatch.setattr(builder, "run_scene_jobs", capture_jobs)
    assert builder.main() == 0
    assert scheduled == ["held_out", "seen", "shared"]


@pytest.mark.parametrize("selection", ["all", "subset", "custom"])
def test_asset_launcher_builds_bev_from_unified_cityrefer(tmp_path, selection):
    data_root = tmp_path / "city data"
    meta = data_root / "meta"
    meta.mkdir(parents=True)
    if selection != "custom":
        for split, scenes in {
            "train": ["train_scene", "nd_scene"],
            "val": ["train_scene"],
            "test_ND": ["nd_scene"],
            "test_NO": ["held_out"],
        }.items():
            _write_jsonl(meta / f"cityrefer_{split}.jsonl", "cityrefer", scenes)

    raw = tmp_path / "raw"
    raw.mkdir()
    for scene in ("train_scene", "nd_scene", "held_out", "custom_scene"):
        (raw / f"{scene}.ply").write_text(
            "ply\nformat ascii 1.0\nelement vertex 4\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n0 0 0 255 0 0\n0 1 1 0 255 0\n"
            "1 0 1 0 0 255\n1 1 2 255 255 0\n",
            encoding="utf-8",
        )

    env = os.environ.copy()
    for key in ("DATA_PATH", "OUT", "WORKERS"):
        env.pop(key, None)
    env.update(CITY_AGENT_DATA=str(data_root), URBEX_RAW_DATA=str(raw), PYTHON=sys.executable)
    args = ["bash", "scripts/data/build_assets.sh", "cityrefer", "bev", "--resolution", "0.5"]
    expected = {"train_scene", "nd_scene", "held_out"}
    if selection == "subset":
        args.extend(["--scenes", "held_out"])
        expected = {"held_out"}
    elif selection == "custom":
        custom = tmp_path / "custom.jsonl"
        _write_jsonl(custom, "cityrefer", ["custom_scene"])
        env.update(DATA_PATH=str(custom), OUT=str(tmp_path / "custom bev"))
        expected = {"custom_scene"}

    completed = subprocess.run(
        args, cwd=Path(__file__).resolve().parents[4], env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    output = Path(env.get("OUT", str(data_root / "bev_cityrefer")))
    assert {path.stem for path in output.glob("*.png")} == expected
    for scene in expected:
        image, mapping = load_bev(output / f"{scene}.png")
        assert image.size == (mapping["width"], mapping["height"])


@pytest.mark.parametrize("filename", ["WMSC.ply", "WMSC_points.ply"])
def test_stpls_original_and_processed_ply_names(tmp_path, monkeypatch, filename):
    from roll.pipeline.agentic.env.city_grounding.tools import retrieve

    ply = tmp_path / "STPLS3D" / "RealWorldData" / filename
    ply.parent.mkdir(parents=True)
    ply.touch()
    # The raw root, dataset directory and direct PLY directory are all
    # documented builder inputs. Runtime fallback uses the raw root.
    for root in (tmp_path, ply.parent.parent, ply.parent):
        assert resolve_ply_path("WMSC", "stpls3d", root) == ply.resolve()
    monkeypatch.setattr(retrieve, "_RAW_DATA", str(tmp_path))
    assert retrieve.ply_path_for_scene("WMSC", "stpls3d") == str(ply)


def test_scene_loading_rejects_malformed_or_mismatched_jsonl(tmp_path):
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_scene_ids(malformed, "cityrefer")

    mismatched = tmp_path / "mismatched.jsonl"
    _write_jsonl(mismatched, "stpls3d", ["USC"])
    with pytest.raises(ValueError, match="dataset mismatch"):
        load_scene_ids(mismatched, "cityrefer")


def test_render_bev_uses_highest_point_and_pixel_edge_mapping():
    points = np.asarray(
        [[0, 0, 0], [0, 0, 5], [1, 0, 1], [1, 1, 2]],
        dtype=np.float32,
    )
    colors = np.asarray(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
        dtype=np.uint8,
    )
    image, mapping = render_bev(points, colors, resolution=0.5, fill=False)
    array = np.asarray(image)

    assert image.size == (2, 2)
    assert array[1, 0].tolist() == [0, 255, 0]
    assert world_to_pixel(mapping, mapping["x_max"], mapping["y_min"]) == (2.0, 2.0)
    for world_x, world_y in ((0.0, 0.0), (0.25, 0.75), (1.0, 1.0)):
        pixel = world_to_pixel(mapping, world_x, world_y)
        restored = pixel_to_world(mapping, *pixel)
        assert restored == pytest.approx((world_x, world_y))


def test_render_bev_subsampling_is_deterministic_and_keeps_bounds():
    rng = np.random.default_rng(4)
    points = rng.uniform([0, 0, 0], [10, 20, 5], size=(100, 3)).astype(np.float32)
    colors = rng.integers(0, 256, size=(100, 3), dtype=np.uint8)
    first, first_mapping = render_bev(
        points, colors, resolution=1.0, subsample=20, seed=7, fill=False
    )
    second, second_mapping = render_bev(
        points, colors, resolution=1.0, subsample=20, seed=7, fill=False
    )
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert first_mapping == second_mapping
    assert first_mapping["x_min"] == pytest.approx(float(points[:, 0].min()))
    assert first_mapping["y_max"] == pytest.approx(float(points[:, 1].max()))


def test_build_scene_bev_writes_loadable_pair_and_skips_existing(tmp_path, monkeypatch):
    ply_path = tmp_path / "scene.ply"
    points = np.asarray(
        [[0, 0, 0], [0, 0, 5], [1, 0, 1], [1, 1, 2]],
        dtype=np.float32,
    )
    colors = np.asarray(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
        dtype=np.uint8,
    )
    monkeypatch.setattr(bev_builder, "_load_ply_cached", lambda _path: (points, colors))
    output_dir = tmp_path / "bev"

    built = build_scene_bev("scene", ply_path, output_dir, resolution=0.5, fill=False)
    assert built["status"] == "built"
    _load_bev_cached.cache_clear()
    image, mapping = load_bev(output_dir / "scene.png")
    assert image.size == (mapping["width"], mapping["height"])

    before = (output_dir / "scene.png").read_bytes()
    skipped = build_scene_bev("scene", ply_path, output_dir, resolution=0.25, fill=False)
    assert skipped["status"] == "skipped"
    assert (output_dir / "scene.png").read_bytes() == before


def _fake_points() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 60, 20)
    y = np.linspace(0, 10, 20)
    points = np.column_stack((x, y, np.linspace(0, 3, 20))).astype(np.float32)
    colors = np.full((len(points), 3), 128, dtype=np.uint8)
    return points, colors


def _fake_cameras(_region, target_z, **kwargs):
    return [
        {
            "extrinsic": np.eye(4),
            "intrinsic": np.asarray([[4.0, 0.0, 4.0], [0.0, 4.0, 4.0], [0.0, 0.0, 1.0]]),
            "target": [0.0, 0.0, target_z],
            "box_size": kwargs["min_size"],
        }
        for _ in range(4)
    ]


def test_cache_builder_compacts_successes_and_is_readable(tmp_path, monkeypatch):
    points, colors = _fake_points()
    monkeypatch.setattr(cache_builder, "_load_ply_cached", lambda _path: (points, colors))
    render_kwargs = []
    camera_kwargs = []

    def fake_render(_ply, region, **kwargs):
        render_kwargs.append(kwargs)
        if region[0] < 10:
            return []
        return {
            "views": [Image.new("RGB", (8, 8), (index * 20, 10, 10)) for index in range(4)],
            "depths": [np.full((8, 8), 10.0 + index, dtype=np.float32) for index in range(4)],
        }

    def fake_cameras(region, target_z, **kwargs):
        camera_kwargs.append(kwargs)
        return _fake_cameras(region, target_z, **kwargs)

    monkeypatch.setattr(cache_builder, "render_region", fake_render)
    monkeypatch.setattr(cache_builder, "compute_region_cameras", fake_cameras)
    output_root = tmp_path / "cache"
    result = cache_builder.build_scene_cache(
        "scene",
        "cityrefer",
        tmp_path / "scene.ply",
        output_root,
        region_size=40,
        step=30,
        output_size=8,
        depth_downsample=2,
    )

    assert result["status"] == "built"
    assert result["regions"] == 1
    scene_dir = output_root / "scene"
    regions = np.load(scene_dir / "regions.npy")
    assert regions.shape == (1, 4)
    assert (scene_dir / "imgs" / "000000_e.jpg").is_file()
    assert not (scene_dir / "imgs" / "000001_e.jpg").exists()
    with np.load(scene_dir / "depths" / "000000.npz") as depth:
        assert depth["depths"].shape == (4, 4, 4)
        assert depth["depths"].dtype == np.uint16

    common_keys = {
        "output_size", "scale", "min_size", "max_size", "elevation_deg", "camera_dist_factor"
    }
    successful_render = render_kwargs[-1]
    assert {key: successful_render[key] for key in common_keys} == {
        key: camera_kwargs[-1][key] for key in common_keys
    }

    BboxCache._global_registry.clear()
    cache = BboxCache.get(output_root, "scene", dist_threshold=1000)
    images, distance = cache.lookup([25.0, 1.0, 35.0, 9.0], only_direction="east")
    assert distance is not None
    assert images is not None and images[0] is not None
    assert images[1:] == [None, None, None]

    sentinel = scene_dir / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    skipped = cache_builder.build_scene_cache(
        "scene",
        "cityrefer",
        tmp_path / "scene.ply",
        output_root,
        region_size=40,
        step=30,
        output_size=8,
        depth_downsample=2,
    )
    assert skipped["status"] == "skipped"
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_cache_builder_does_not_publish_failed_or_overwrite_partial(tmp_path, monkeypatch):
    points, colors = _fake_points()
    monkeypatch.setattr(cache_builder, "_load_ply_cached", lambda _path: (points, colors))
    monkeypatch.setattr(cache_builder, "render_region", lambda *_args, **_kwargs: [])
    output_root = tmp_path / "cache"

    with pytest.raises(RuntimeError, match="no complete cache regions"):
        cache_builder.build_scene_cache(
            "failed",
            "cityrefer",
            tmp_path / "failed.ply",
            output_root,
            region_size=40,
            step=30,
            output_size=8,
        )
    assert not (output_root / "failed").exists()
    assert not list(output_root.glob(".failed.building-*"))

    partial = output_root / "partial"
    partial.mkdir()
    sentinel = partial / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete cache metadata"):
        cache_builder.build_scene_cache(
            "partial",
            "cityrefer",
            tmp_path / "partial.ply",
            output_root,
            region_size=40,
            step=30,
            output_size=8,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_scene_scheduler_serial_path_isolates_failures(monkeypatch):
    monkeypatch.setattr(
        parallel_builder,
        "ProcessPoolExecutor",
        lambda **_kwargs: pytest.fail("workers=1 must not create a process pool"),
    )
    outcomes = parallel_builder.run_scene_jobs(
        [
            {"scene_id": "ok"},
            {"scene_id": "bad", "fail": True},
            {"scene_id": "later"},
        ],
        _parallel_probe_job,
        workers=1,
    )
    assert [outcome["ok"] for outcome in outcomes] == [True, False, True]
    assert outcomes[1]["scene_id"] == "bad"
    assert "probe failure" in outcomes[1]["error"]


def test_scene_scheduler_uses_spawn_and_caps_workers(monkeypatch):
    captured = {}

    class FakeExecutor:
        def __init__(self, *, max_workers, mp_context):
            captured["max_workers"] = max_workers
            captured["start_method"] = mp_context.get_start_method()

        def submit(self, fn, job):
            future = Future()
            future.set_result(fn(job))
            return future

        def shutdown(self, *, wait, cancel_futures=False):
            captured["shutdown"] = (wait, cancel_futures)

    monkeypatch.setattr(parallel_builder, "ProcessPoolExecutor", FakeExecutor)
    outcomes = parallel_builder.run_scene_jobs(
        [{"scene_id": "a"}, {"scene_id": "b"}],
        _parallel_probe_job,
        workers=8,
    )
    assert captured == {
        "max_workers": 2,
        "start_method": "spawn",
        "shutdown": (True, False),
    }
    assert [outcome["scene_id"] for outcome in outcomes] == ["a", "b"]


def test_scene_scheduler_cancels_pending_jobs_on_interrupt(monkeypatch):
    captured = {"futures": []}

    class FakeExecutor:
        def __init__(self, **_kwargs):
            pass

        def submit(self, _fn, _job):
            future = Future()
            captured["futures"].append(future)
            return future

        def shutdown(self, *, wait, cancel_futures=False):
            captured["shutdown"] = (wait, cancel_futures)

    def interrupting_as_completed(_futures):
        raise KeyboardInterrupt

    monkeypatch.setattr(parallel_builder, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(parallel_builder, "as_completed", interrupting_as_completed)
    with pytest.raises(KeyboardInterrupt):
        parallel_builder.run_scene_jobs(
            [{"scene_id": "a"}, {"scene_id": "b"}],
            _parallel_probe_job,
            workers=2,
        )
    assert all(future.cancelled() for future in captured["futures"])
    assert captured["shutdown"] == (False, True)


def test_scene_scheduler_runs_spawn_workers_and_keeps_order():
    outcomes = parallel_builder.run_scene_jobs(
        [
            {"scene_id": "a", "delay": 0.05},
            {"scene_id": "bad", "fail": True},
            {"scene_id": "b", "delay": 0.05},
        ],
        _parallel_probe_job,
        workers=2,
    )
    assert [outcome["scene_id"] for outcome in outcomes] == ["a", "bad", "b"]
    assert [outcome["ok"] for outcome in outcomes] == [True, False, True]
    worker_pids = {
        outcome["result"]["pid"]
        for outcome in outcomes
        if outcome["ok"]
    }
    assert worker_pids and os.getpid() not in worker_pids


def test_bev_scene_job_always_clears_ply_cache(tmp_path, monkeypatch):
    cleared = []
    monkeypatch.setattr(bev_builder, "resolve_ply_path", lambda *_args: tmp_path / "scene.ply")
    monkeypatch.setattr(bev_builder, "_clear_ply_cache", lambda: cleared.append(True))
    monkeypatch.setattr(
        bev_builder,
        "build_scene_bev",
        lambda *_args, **_kwargs: {"scene_id": "scene", "status": "built", "output": "bev.png"},
    )
    job = {
        "scene_id": "scene",
        "dataset": "cityrefer",
        "ply_root": str(tmp_path),
        "out": str(tmp_path),
        "resolution": 1.0,
        "subsample": None,
        "seed": 0,
        "fill": False,
        "fill_radius": None,
    }
    result = bev_builder._build_bev_scene_job(job)
    assert result["status"] == "built"
    assert cleared == [True]

    monkeypatch.setattr(
        bev_builder,
        "build_scene_bev",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    with pytest.raises(RuntimeError, match="failed"):
        bev_builder._build_bev_scene_job(job)
    assert cleared == [True, True]


def test_verify_scene_job_always_clears_ply_cache(tmp_path, monkeypatch):
    cleared = []
    monkeypatch.setattr(cache_builder, "resolve_ply_path", lambda *_args: tmp_path / "scene.ply")
    monkeypatch.setattr(cache_builder, "_clear_ply_cache", lambda: cleared.append(True))
    monkeypatch.setattr(
        cache_builder,
        "build_scene_cache",
        lambda *_args, **_kwargs: {"scene_id": "scene", "status": "built", "output": "cache"},
    )
    job = {
        "scene_id": "scene",
        "dataset": "cityrefer",
        "ply_root": str(tmp_path),
        "out": str(tmp_path),
        "region_size": 40.0,
        "step": 15.0,
        "output_size": 8,
        "jpeg_quality": 82,
        "elevation_deg": 30.0,
        "camera_dist_factor": 1.3,
        "point_size": 5.0,
        "default_bbox_height": 3.0,
        "depth_scale": 0.05,
        "depth_downsample": 1,
    }
    result = cache_builder._build_verify_scene_job(job)
    assert result["status"] == "built"
    assert cleared == [True]

    monkeypatch.setattr(
        cache_builder,
        "build_scene_cache",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    with pytest.raises(RuntimeError, match="failed"):
        cache_builder._build_verify_scene_job(job)
    assert cleared == [True, True]
