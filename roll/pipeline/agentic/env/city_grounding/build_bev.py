from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ._asset_build_parallel import SceneJob, run_scene_jobs
from .asset_paths import SUPPORTED_DATASETS, load_scene_ids, resolve_ply_path
from .render import _clear_ply_cache, _load_ply_cached, world_to_pixel

logger = logging.getLogger(__name__)


def _fill_holes(
    image: np.ndarray,
    occupied: np.ndarray,
    max_radius: float | None,
) -> np.ndarray:
    if occupied.all() or not occupied.any():
        return image
    from scipy.ndimage import distance_transform_edt

    fill_mask = ~occupied
    if max_radius is None:
        indices = distance_transform_edt(
            fill_mask,
            return_distances=False,
            return_indices=True,
        )
    else:
        distances, indices = distance_transform_edt(fill_mask, return_indices=True)
        fill_mask &= distances <= max_radius
    filled = image.copy()
    filled[fill_mask] = image[indices[0][fill_mask], indices[1][fill_mask]]
    return filled


def render_bev(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    resolution: float,
    subsample: int | None = None,
    seed: int = 0,
    fill: bool = True,
    fill_radius: float | None = None,
) -> tuple[Image.Image, dict]:
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    points = np.asarray(points)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
        raise ValueError("points must have shape (N, 3) with N > 0")
    if colors.shape != (len(points), 3):
        raise ValueError("colors must have shape (N, 3)")

    points = points[:, :3]
    finite = np.isfinite(points).all(axis=1)
    if not finite.all():
        points = points[finite]
        colors = colors[finite]
    if len(points) == 0:
        raise ValueError("point cloud contains no finite XYZ points")

    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    x_min, y_min = float(xyz_min[0]), float(xyz_min[1])
    x_span = max(float(xyz_max[0] - xyz_min[0]), resolution)
    y_span = max(float(xyz_max[1] - xyz_min[1]), resolution)
    x_max, y_max = x_min + x_span, y_min + y_span
    width = int(np.ceil(x_span / resolution))
    height = int(np.ceil(y_span / resolution))
    res_x = x_span / width
    res_y = y_span / height

    raster_points = points
    raster_colors = colors
    if subsample is not None:
        if subsample <= 0:
            raise ValueError("subsample must be positive")
        if len(points) > subsample:
            rng = np.random.default_rng(seed)
            selected = np.sort(rng.choice(len(points), subsample, replace=False))
            raster_points = points[selected]
            raster_colors = colors[selected]

    mapping = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "width": width,
        "height": height,
        "res_x": res_x,
        "res_y": res_y,
        "requested_resolution": float(resolution),
    }
    projected_x, projected_y = world_to_pixel(
        mapping,
        raster_points[:, 0],
        raster_points[:, 1],
    )
    pixel_x = np.clip(np.floor(projected_x).astype(np.int64), 0, width - 1)
    pixel_y = np.clip(np.floor(projected_y).astype(np.int64), 0, height - 1)
    linear = pixel_y * width + pixel_x

    point_order = np.arange(len(raster_points), dtype=np.int64)
    order = np.lexsort((point_order, -raster_points[:, 2], linear))
    sorted_linear = linear[order]
    _, first = np.unique(sorted_linear, return_index=True)
    winners = order[first]

    image = np.zeros((height, width, 3), dtype=np.uint8)
    occupied = np.zeros((height, width), dtype=bool)
    image[pixel_y[winners], pixel_x[winners]] = raster_colors[winners]
    occupied[pixel_y[winners], pixel_x[winners]] = True
    if fill:
        image = _fill_holes(image, occupied, fill_radius)

    return Image.fromarray(image, mode="RGB"), mapping


def _validate_bev_pair(png_path: Path, json_path: Path) -> None:
    if not png_path.is_file() or not json_path.is_file():
        raise ValueError(f"incomplete BEV pair: {png_path} and {json_path}")
    with Image.open(png_path) as image:
        size = image.size
        image.verify()
    mapping = json.loads(json_path.read_text(encoding="utf-8"))
    required = {"x_min", "x_max", "y_min", "y_max", "width", "height", "res_x", "res_y"}
    missing = sorted(required - mapping.keys())
    if missing:
        raise ValueError(f"mapping {json_path} is missing: {', '.join(missing)}")
    if size != (int(mapping["width"]), int(mapping["height"])):
        raise ValueError(f"image and mapping dimensions differ for {png_path}")
    for axis, size_key in (("x", "width"), ("y", "height")):
        span = float(mapping[f"{axis}_max"]) - float(mapping[f"{axis}_min"])
        covered = int(mapping[size_key]) * float(mapping[f"res_{axis}"])
        if not np.isclose(span, covered):
            raise ValueError(f"invalid {axis} mapping in {json_path}")


def build_scene_bev(
    scene_id: str,
    ply_path: str | Path,
    output_dir: str | Path,
    *,
    resolution: float,
    subsample: int | None = None,
    seed: int = 0,
    fill: bool = True,
    fill_radius: float | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{scene_id}.png"
    json_path = output_dir / f"{scene_id}.json"

    if png_path.exists() or json_path.exists():
        _validate_bev_pair(png_path, json_path)
        return {"scene_id": scene_id, "status": "skipped", "output": str(png_path)}

    points, colors = _load_ply_cached(str(ply_path))
    image, mapping = render_bev(
        points,
        colors,
        resolution=resolution,
        subsample=subsample,
        seed=seed,
        fill=fill,
        fill_radius=fill_radius,
    )

    with tempfile.TemporaryDirectory(
        prefix=f".{scene_id}.building-",
        dir=output_dir,
    ) as staging_name:
        staging_dir = Path(staging_name)
        staged_png = staging_dir / png_path.name
        staged_json = staging_dir / json_path.name
        image.save(staged_png, "PNG", optimize=True)
        staged_json.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
        _validate_bev_pair(staged_png, staged_json)
        os.replace(staged_json, json_path)
        os.replace(staged_png, png_path)

    return {
        "scene_id": scene_id,
        "status": "built",
        "output": str(png_path),
        "width": image.width,
        "height": image.height,
    }


def _build_bev_scene_job(job: SceneJob) -> dict:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(processName)s %(levelname)s %(message)s",
    )
    started = time.monotonic()
    try:
        scene_id = str(job["scene_id"])
        ply_path = resolve_ply_path(scene_id, str(job["dataset"]), str(job["ply_root"]))
        result = build_scene_bev(
            scene_id,
            ply_path,
            str(job["out"]),
            resolution=float(job["resolution"]),
            subsample=None if job["subsample"] is None else int(job["subsample"]),
            seed=int(job["seed"]),
            fill=bool(job["fill"]),
            fill_radius=None if job["fill_radius"] is None else float(job["fill_radius"]),
        )
        result["elapsed_s"] = time.monotonic() - started
        return result
    finally:
        _clear_ply_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build top-down BEV PNG and mapping JSON from raw PLY scenes.")
    parser.add_argument(
        "--jsonl", nargs="+", required=True,
        help="Annotation JSONL files; scene IDs are merged and deduplicated.",
    )
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--ply_root", required=True, help="Raw dataset root or directory containing scene PLY files.")
    parser.add_argument("--out", required=True, help="Output directory for <scene_id>.png/.json pairs.")
    parser.add_argument("--scenes", nargs="*", help="Optional subset of scene IDs present in --jsonl.")
    parser.add_argument("--resolution", type=float, default=0.05, help="Requested metres per pixel.")
    parser.add_argument("--subsample", type=int, default=None, help="Optional maximum number of rasterized points.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for deterministic subsampling.")
    parser.add_argument("--no_fill", action="store_true", help="Leave pixels without projected points black.")
    parser.add_argument("--workers", type=int, default=1, help="Number of scenes to build concurrently.")
    parser.add_argument(
        "--fill_radius",
        type=float,
        default=None,
        help="Maximum nearest-neighbour fill radius in pixels.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(processName)s %(levelname)s %(message)s",
    )
    scene_ids = load_scene_ids(args.jsonl, args.dataset, args.scenes)
    jobs: list[SceneJob] = [
        {
            "scene_id": scene_id,
            "dataset": args.dataset,
            "ply_root": args.ply_root,
            "out": args.out,
            "resolution": args.resolution,
            "subsample": args.subsample,
            "seed": args.seed,
            "fill": not args.no_fill,
            "fill_radius": args.fill_radius,
        }
        for scene_id in scene_ids
    ]
    try:
        outcomes = run_scene_jobs(jobs, _build_bev_scene_job, args.workers)
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
