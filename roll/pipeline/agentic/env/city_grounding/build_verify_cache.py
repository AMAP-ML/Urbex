from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from ._asset_build_parallel import SceneJob, run_scene_jobs
from .asset_paths import SUPPORTED_DATASETS, load_scene_ids, resolve_ply_path
from .render import _clear_ply_cache, _load_ply_cached
from .render_open3d import DIRECTIONS, compute_region_cameras, render_region
from .verify_cache import _region_feat

logger = logging.getLogger(__name__)
_DIRECTION_NAMES = tuple(name for name, _ in DIRECTIONS)
_CARDINALS = tuple(name[0] for name in _DIRECTION_NAMES)
_VIEW_COUNT = len(_CARDINALS)


def _axis_centers(lower: float, upper: float, region_size: float, step: float) -> np.ndarray:
    half = region_size / 2.0
    if upper - lower <= region_size:
        return np.asarray([(lower + upper) / 2.0], dtype=np.float64)
    first = lower + half
    last = upper - half
    centers = np.arange(first, last + 1e-9, step, dtype=np.float64)
    if centers.size == 0 or last - centers[-1] > 1e-6:
        centers = np.append(centers, last)
    return centers


def sample_regions(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    *,
    region_size: float,
    step: float,
) -> np.ndarray:
    if region_size <= 0 or step <= 0:
        raise ValueError("region_size and step must be positive")
    centers_x = _axis_centers(x_min, x_max, region_size, step)
    centers_y = _axis_centers(y_min, y_max, region_size, step)
    half = region_size / 2.0
    return np.asarray(
        [
            [cx - half, cy - half, cx + half, cy + half]
            for cx in centers_x
            for cy in centers_y
        ],
        dtype=np.float32,
    )


def _local_target_z(
    points: np.ndarray,
    region: np.ndarray,
    fallback: float,
    *,
    pad_factor: float = 0.3,
    percentile: float = 5.0,
    target_offset: float = 3.0,
) -> float:
    x_min, y_min, x_max, y_max = [float(value) for value in region]
    radius = max(x_max - x_min, y_max - y_min) / 2.0
    padding = radius * pad_factor
    mask = (
        (points[:, 0] >= x_min - padding)
        & (points[:, 0] <= x_max + padding)
        & (points[:, 1] >= y_min - padding)
        & (points[:, 1] <= y_max + padding)
    )
    local = points[mask]
    if len(local) < 5:
        return fallback
    return float(np.percentile(local[:, 2], percentile)) + target_offset


def _downsample_depth(depth: np.ndarray, factor: int) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2 or not depth.size:
        raise ValueError("each depth view must be a non-empty 2D array")
    if factor == 1:
        return depth
    height, width = depth.shape
    pooled_height, pooled_width = height // factor, width // factor
    if pooled_height == 0 or pooled_width == 0:
        raise ValueError("depth_downsample exceeds the rendered depth dimensions")
    blocks = depth[: pooled_height * factor, : pooled_width * factor].reshape(
        pooled_height, factor, pooled_width, factor
    )
    valid = blocks > 0
    pooled = np.where(valid, blocks, np.inf).min(axis=(1, 3))
    pooled[~np.isfinite(pooled)] = 0.0
    return pooled


def _cache_config(
    dataset: str,
    *,
    region_size: float,
    step: float,
    output_size: int,
    elevation_deg: float,
    camera_dist_factor: float,
    point_size: float,
    jpeg_quality: int,
    depth_scale: float,
    depth_downsample: int,
    default_bbox_height: float,
) -> dict:
    return {
        "dataset": dataset,
        "region_size": float(region_size),
        "step": float(step),
        "output_size": int(output_size),
        "elevation_deg": float(elevation_deg),
        "camera_dist_factor": float(camera_dist_factor),
        "point_size": float(point_size),
        "jpeg_quality": int(jpeg_quality),
        "save_depth": True,
        "depth_scale": float(depth_scale),
        "depth_downsample": int(depth_downsample),
        "depth_format": "npz",
        "default_bbox_height": float(default_bbox_height),
    }


def _values_match(actual: object, expected: object) -> bool:
    if isinstance(expected, float):
        try:
            return bool(np.isclose(float(actual), expected))
        except (TypeError, ValueError):
            return False
    return actual == expected


def validate_scene_cache(scene_dir: str | Path, expected_config: dict | None = None) -> dict:
    scene_dir = Path(scene_dir)
    meta_path = scene_dir / "scene_meta.json"
    regions_path = scene_dir / "regions.npy"
    index_path = scene_dir / "index.pkl"
    if not all(path.is_file() for path in (meta_path, regions_path, index_path)):
        raise ValueError(f"incomplete cache metadata in {scene_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if expected_config:
        mismatched = [
            key
            for key, expected in expected_config.items()
            if not _values_match(meta.get(key), expected)
        ]
        if mismatched:
            raise ValueError(f"cache parameters differ in {scene_dir}: {', '.join(mismatched)}")

    regions = np.asarray(np.load(regions_path), dtype=np.float32)
    if regions.ndim != 2 or regions.shape[1:] != (4,) or len(regions) == 0:
        raise ValueError(f"invalid regions.npy in {scene_dir}")
    if index_path.stat().st_size == 0:
        raise ValueError(f"empty index.pkl in {scene_dir}")
    if int(meta.get("region_count", -1)) != len(regions):
        raise ValueError(f"region_count does not match regions.npy in {scene_dir}")

    for entry_id in range(len(regions)):
        for cardinal in _CARDINALS:
            image_path = scene_dir / "imgs" / f"{entry_id:06d}_{cardinal}.jpg"
            if not image_path.is_file():
                raise ValueError(f"missing cache image {image_path}")
            with Image.open(image_path) as image:
                if image.size != (int(meta["output_size"]), int(meta["output_size"])):
                    raise ValueError(f"unexpected cache image size in {image_path}")
                image.verify()

        camera_path = scene_dir / "cams" / f"{entry_id:06d}.npz"
        depth_path = scene_dir / "depths" / f"{entry_id:06d}.npz"
        if not camera_path.is_file() or not depth_path.is_file():
            raise ValueError(f"missing camera or depth entry {entry_id} in {scene_dir}")
        with np.load(camera_path) as camera:
            if camera["extrinsics"].shape != (_VIEW_COUNT, 4, 4) or camera["intrinsics"].shape != (_VIEW_COUNT, 3, 3):
                raise ValueError(f"invalid camera matrices in {camera_path}")
        with np.load(depth_path) as depth:
            depths = depth["depths"]
            if depths.ndim != 3 or depths.shape[0] != _VIEW_COUNT or depths.dtype != np.uint16:
                raise ValueError(f"invalid depth stack in {depth_path}")
    return meta


def build_scene_cache(
    scene_id: str,
    dataset: str,
    ply_path: str | Path,
    output_root: str | Path,
    *,
    region_size: float = 40.0,
    step: float = 15.0,
    output_size: int = 512,
    jpeg_quality: int = 82,
    elevation_deg: float = 30.0,
    camera_dist_factor: float = 1.3,
    point_size: float = 5.0,
    default_bbox_height: float = 3.0,
    depth_scale: float = 0.05,
    depth_downsample: int = 4,
) -> dict:
    if output_size <= 0 or not 1 <= jpeg_quality <= 100:
        raise ValueError("output_size must be positive and jpeg_quality must be in [1, 100]")
    if depth_scale <= 0 or depth_downsample <= 0:
        raise ValueError("depth_scale and depth_downsample must be positive")
    if _DIRECTION_NAMES != ("east", "north", "west", "south"):
        raise RuntimeError("render direction order no longer matches the cache format")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / scene_id
    config = _cache_config(
        dataset,
        region_size=region_size,
        step=step,
        output_size=output_size,
        elevation_deg=elevation_deg,
        camera_dist_factor=camera_dist_factor,
        point_size=point_size,
        jpeg_quality=jpeg_quality,
        depth_scale=depth_scale,
        depth_downsample=depth_downsample,
        default_bbox_height=default_bbox_height,
    )
    if final_dir.exists():
        validate_scene_cache(final_dir, config)
        return {"scene_id": scene_id, "status": "skipped", "output": str(final_dir)}

    points, _ = _load_ply_cached(str(ply_path))
    points = np.asarray(points, dtype=np.float32)
    finite = np.isfinite(points[:, :3]).all(axis=1)
    points = points[finite, :3]
    if len(points) == 0:
        raise ValueError(f"point cloud has no finite points: {ply_path}")
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    scene_z_mid = float((xyz_min[2] + xyz_max[2]) / 2.0)
    sampled_regions = sample_regions(
        float(xyz_min[0]),
        float(xyz_min[1]),
        float(xyz_max[0]),
        float(xyz_max[1]),
        region_size=region_size,
        step=step,
    )

    staging_dir = Path(tempfile.mkdtemp(prefix=f".{scene_id}.building-", dir=output_root))
    (staging_dir / "imgs").mkdir()
    (staging_dir / "cams").mkdir()
    (staging_dir / "depths").mkdir()
    successful_regions: list[np.ndarray] = []
    camera_kwargs = {
        "output_size": output_size,
        "scale": 1.0,
        "min_size": region_size,
        "max_size": region_size,
        "elevation_deg": elevation_deg,
        "camera_dist_factor": camera_dist_factor,
    }

    try:
        for sampled_id, region in enumerate(sampled_regions):
            target_z = _local_target_z(points, region, scene_z_mid)
            try:
                rendered = render_region(
                    str(ply_path),
                    region.tolist(),
                    scene_key=scene_id,
                    point_size=point_size,
                    draw_bbox=False,
                    return_depth=True,
                    target_z=target_z,
                    **camera_kwargs,
                )
                if not isinstance(rendered, dict):
                    raise ValueError("renderer returned no depth payload")
                views = list(rendered.get("views") or [])
                depths = list(rendered.get("depths") or [])
                if len(views) != _VIEW_COUNT or len(depths) != _VIEW_COUNT:
                    raise ValueError("renderer did not return all directional views and depth maps")
                cameras = compute_region_cameras(region.tolist(), target_z, **camera_kwargs)
                if len(cameras) != _VIEW_COUNT:
                    raise ValueError("camera builder did not return all directional cameras")

                entry_id = len(successful_regions)
                for cardinal, view in zip(_CARDINALS, views):
                    image = view.convert("RGB")
                    if image.size != (output_size, output_size):
                        raise ValueError("rendered image size does not match output_size")
                    image.save(
                        staging_dir / "imgs" / f"{entry_id:06d}_{cardinal}.jpg",
                        "JPEG",
                        quality=jpeg_quality,
                        optimize=True,
                    )

                extrinsics = np.stack([np.asarray(camera["extrinsic"], dtype=np.float64) for camera in cameras])
                intrinsics = np.stack([np.asarray(camera["intrinsic"], dtype=np.float64) for camera in cameras])
                targets = np.stack([np.asarray(camera["target"], dtype=np.float64) for camera in cameras])
                box_sizes = np.asarray([camera["box_size"] for camera in cameras], dtype=np.float64)
                if extrinsics.shape != (_VIEW_COUNT, 4, 4) or intrinsics.shape != (_VIEW_COUNT, 3, 3):
                    raise ValueError("camera matrices have unexpected shapes")
                np.savez_compressed(
                    staging_dir / "cams" / f"{entry_id:06d}.npz",
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                    targets=targets,
                    box_sizes=box_sizes,
                )

                depth_stack = np.stack(
                    [_downsample_depth(depth, depth_downsample) for depth in depths]
                )
                encoded_depth = np.clip(depth_stack / depth_scale, 0, 65535).astype(np.uint16)
                np.savez_compressed(
                    staging_dir / "depths" / f"{entry_id:06d}.npz",
                    depths=encoded_depth,
                )
                successful_regions.append(region.copy())
            except Exception as exc:
                logger.warning("scene=%s region=%d failed: %s", scene_id, sampled_id, exc)

        if not successful_regions:
            raise RuntimeError(f"scene {scene_id} produced no complete cache regions")

        failed_regions = len(sampled_regions) - len(successful_regions)
        regions = np.asarray(successful_regions, dtype=np.float32)
        features = np.stack([_region_feat(region) for region in regions])
        np.save(staging_dir / "regions.npy", regions)
        with (staging_dir / "index.pkl").open("wb") as handle:
            pickle.dump({"tree": cKDTree(features), "feats": features}, handle)

        metadata = {
            "scene_id": scene_id,
            **config,
            "ground_z": scene_z_mid,
            "region_count": len(regions),
            "sampled_region_count": len(sampled_regions),
            "failed_region_count": failed_regions,
        }
        (staging_dir / "scene_meta.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        validate_scene_cache(staging_dir, config)
        try:
            staging_dir.rename(final_dir)
            status = "built"
        except FileExistsError:
            validate_scene_cache(final_dir, config)
            status = "skipped"
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    return {
        "scene_id": scene_id,
        "status": status,
        "output": str(final_dir),
        "regions": len(successful_regions),
        "failed_regions": failed_regions,
    }


def _build_verify_scene_job(job: SceneJob) -> dict:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(processName)s %(levelname)s %(message)s",
    )
    started = time.monotonic()
    try:
        scene_id = str(job["scene_id"])
        dataset = str(job["dataset"])
        ply_path = resolve_ply_path(scene_id, dataset, str(job["ply_root"]))
        result = build_scene_cache(
            scene_id,
            dataset,
            ply_path,
            str(job["out"]),
            region_size=float(job["region_size"]),
            step=float(job["step"]),
            output_size=int(job["output_size"]),
            jpeg_quality=int(job["jpeg_quality"]),
            elevation_deg=float(job["elevation_deg"]),
            camera_dist_factor=float(job["camera_dist_factor"]),
            point_size=float(job["point_size"]),
            default_bbox_height=float(job["default_bbox_height"]),
            depth_scale=float(job["depth_scale"]),
            depth_downsample=int(job["depth_downsample"]),
        )
        result["elapsed_s"] = time.monotonic() - started
        return result
    finally:
        _clear_ply_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the pre-rendered CityTool oblique-view cache from raw PLY scenes."
    )
    parser.add_argument(
        "--jsonl", nargs="+", required=True,
        help="Annotation JSONL files; scene IDs are merged and deduplicated.",
    )
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--ply_root", required=True, help="Raw dataset root or directory containing scene PLY files.")
    parser.add_argument("--out", required=True, help="Cache root; one atomic subdirectory is published per scene.")
    parser.add_argument("--scenes", nargs="*", help="Optional subset of scene IDs present in --jsonl.")
    parser.add_argument("--scene_limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Number of scenes to build concurrently.")
    parser.add_argument("--region_size", type=float, default=40.0)
    parser.add_argument("--step", type=float, default=15.0)
    parser.add_argument("--output_size", type=int, default=512)
    parser.add_argument("--jpeg_quality", type=int, default=82)
    parser.add_argument("--elevation_deg", type=float, default=30.0)
    parser.add_argument("--camera_dist_factor", type=float, default=1.3)
    parser.add_argument("--point_size", type=float, default=5.0)
    parser.add_argument("--default_bbox_height", type=float, default=3.0)
    parser.add_argument("--depth_scale", type=float, default=0.05)
    parser.add_argument("--depth_downsample", type=int, default=4)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(processName)s %(levelname)s %(message)s",
    )
    scene_ids = load_scene_ids(args.jsonl, args.dataset, args.scenes)
    if args.scene_limit is not None:
        if args.scene_limit <= 0:
            raise ValueError("scene_limit must be positive")
        scene_ids = scene_ids[: args.scene_limit]

    jobs: list[SceneJob] = [
        {
            "scene_id": scene_id,
            "dataset": args.dataset,
            "ply_root": args.ply_root,
            "out": args.out,
            "region_size": args.region_size,
            "step": args.step,
            "output_size": args.output_size,
            "jpeg_quality": args.jpeg_quality,
            "elevation_deg": args.elevation_deg,
            "camera_dist_factor": args.camera_dist_factor,
            "point_size": args.point_size,
            "default_bbox_height": args.default_bbox_height,
            "depth_scale": args.depth_scale,
            "depth_downsample": args.depth_downsample,
        }
        for scene_id in scene_ids
    ]
    try:
        outcomes = run_scene_jobs(jobs, _build_verify_scene_job, args.workers)
    except KeyboardInterrupt:
        logger.warning("interrupted; pending scene jobs cancelled")
        return 130

    failures = 0
    for outcome in outcomes:
        if outcome["ok"]:
            result = outcome["result"]
            assert isinstance(result, dict)
            logger.info(
                "scene=%s status=%s elapsed=%.1fs output=%s",
                outcome["scene_id"],
                result["status"],
                result["elapsed_s"],
                result["output"],
            )
        else:
            failures += 1
            logger.error("scene=%s failed: %s", outcome["scene_id"], outcome["error"])
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
