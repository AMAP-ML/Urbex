"""Rebuild a local 3D scene from the released oblique cache and show a prediction in it.

Every cached region stores four oblique renders with their cameras and per-pixel depth, so
back-projecting them gives a coloured point cloud around a predicted box without the original
PLY. Only the box footprint comes from the prediction; the displayed Z extent is measured from
the rebuilt points and, like the camera, is a presentation choice.
"""
from __future__ import annotations

import base64
import html
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


CARDINALS = ("e", "n", "w", "s")


@dataclass
class LocalScene:
    """Points rebuilt around a predicted box, with the Z extent displayed for that box."""
    xyz: np.ndarray        # (N, 3) float32, world metres
    rgb: np.ndarray        # (N, 3) uint8
    world_bbox: list       # predicted [xmin, ymin, xmax, ymax]
    zs: tuple              # displayed (z_min, z_max)
    radius: float          # metres rebuilt around the box centre
    spacing: float         # typical distance between neighbouring points


def view_radius(world_bbox, margin=18.0, minimum=35.0, maximum=140.0):
    """Radius in metres of the disc rebuilt around a predicted XY box."""
    x0, y0, x1, y1 = [float(v) for v in world_bbox]
    return float(np.clip(math.hypot(x1 - x0, y1 - y0) / 2 * 1.25 + margin, minimum, maximum))


def _depths(root, entry, meta):
    """The four depth maps of a cache entry in metres, for the lossless cache formats."""
    scale = float(meta.get("depth_scale", 0.05))
    layout = meta.get("depth_format", "npz")
    if layout == "npz":
        with np.load(root / "depths" / f"{entry:06d}.npz") as data:
            return data["depths"].astype(np.float32) * scale
    if layout == "png":
        return np.stack([np.asarray(Image.open(root / "depths" / f"{entry:06d}_{cardinal}.png"), dtype=np.float32)
                         for cardinal in CARDINALS]) * scale
    raise ValueError(f"the 3D view needs lossless cached depth, not {layout!r}")


def _back_project(depth, extrinsic, intrinsic, image_size, step=1):
    """World XYZ of every ``step``-th depth pixel; cameras follow verify_cache (x right, y down, z forward).

    A depth map smaller than its colour image is read at the centres of the pooled pixels.
    """
    height, width = depth.shape
    u, v = np.meshgrid((np.arange(0, width, step, dtype=np.float32) + 0.5) * (image_size[0] / width) - 0.5,
                       (np.arange(0, height, step, dtype=np.float32) + 0.5) * (image_size[1] / height) - 0.5)
    depth = depth[::step, ::step]
    camera = np.stack([(u - intrinsic[0, 2]) * depth / intrinsic[0, 0],
                       (v - intrinsic[1, 2]) * depth / intrinsic[1, 1], depth], axis=-1)
    return (camera.reshape(-1, 3) - extrinsic[:3, 3]) @ extrinsic[:3, :3]


def _interior(valid):
    """Drop pixels that touch missing depth, where JPEG colour bleeds in from the background."""
    inner = valid.copy()
    inner[1:] &= valid[:-1]
    inner[:-1] &= valid[1:]
    inner[:, 1:] &= valid[:, :-1]
    inner[:, :-1] &= valid[:, 1:]
    return inner


def _voxel_mean(values, inverse, counts):
    return np.stack([np.bincount(inverse, values[:, k], len(counts)) for k in range(3)], axis=1) / counts[:, None]


def display_height(xyz, world_bbox, percentiles=(2, 95), default=3.0):
    """Z extent shown for a predicted XY box: percentiles of the rebuilt points inside it."""
    x0, y0, x1, y1 = [float(v) for v in world_bbox]
    inside = (xyz[:, 0] >= x0) & (xyz[:, 0] <= x1) & (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1)
    if inside.sum() < 5:
        ground = float(np.percentile(xyz[:, 2], 5))
        return ground, ground + default
    low, high = np.percentile(xyz[inside, 2], percentiles)
    return float(low), float(max(high, low + 0.3))


def rebuild_scene(cache_root, scene_id, world_bbox, radius=None, max_points=320_000, voxel=0.08):
    """Back-project the cached oblique views within ``radius`` metres of a predicted XY box.

    Each point comes from the region whose centre is nearest, where the cached views are
    sharpest, and points are averaged per voxel. The voxel grows until ``max_points`` fit.
    """
    root = Path(cache_root) / scene_id
    meta = json.loads((root / "scene_meta.json").read_text())
    if not meta.get("save_depth"):
        raise ValueError(f"{scene_id}: the 3D view needs a cache built with depth maps")
    x0, y0, x1, y1 = [float(v) for v in world_bbox]
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    radius = view_radius(world_bbox) if radius is None else float(radius)
    regions = np.load(root / "regions.npy")
    centres = (regions[:, :2] + regions[:, 2:]) / 2
    reach = radius + float(meta.get("region_size", 40.0))
    usable = [int(i) for i in np.flatnonzero(np.abs(centres - centre).max(axis=1) <= reach)
              if (root / "cams" / f"{i:06d}.npz").is_file()]
    if not usable:
        raise ValueError(f"{scene_id}: no cached region within {radius:.0f} m of the prediction")
    # A 1 m raster around the box records which usable region is nearest to each cell.
    side = int(math.ceil(2 * radius))
    cells = centre - radius + np.stack(np.meshgrid(np.arange(side) + 0.5, np.arange(side) + 0.5), axis=-1)
    owner = cKDTree(centres[usable]).query(cells.reshape(-1, 2))[1].reshape(side, side)
    owner[np.linalg.norm(cells - centre, axis=-1) > radius + 1] = -1
    step = 2 if radius > 70 else 1                # wide views end up with coarse voxels anyway
    points, colours = [], []
    for slot in np.unique(owner[owner >= 0]):
        entry = usable[slot]
        with np.load(root / "cams" / f"{entry:06d}.npz") as cameras:
            extrinsics, intrinsics = cameras["extrinsics"], cameras["intrinsics"]
        depths = _depths(root, entry, meta)
        for view, cardinal in enumerate(CARDINALS):
            image = Image.open(root / "imgs" / f"{entry:06d}_{cardinal}.jpg").convert("RGB")
            depth = depths[view]
            xyz = _back_project(depth, extrinsics[view], intrinsics[view], image.size, step)
            cell = np.floor(xyz[:, :2] - (centre - radius)).astype(np.int64)
            keep = _interior(depth > 0)[::step, ::step].ravel() & (cell >= 0).all(axis=1) & (cell < side).all(axis=1)
            keep[keep] = owner[cell[keep, 1], cell[keep, 0]] == slot
            keep[keep] = np.linalg.norm(xyz[keep, :2] - centre, axis=1) <= radius
            points.append(xyz[keep].astype(np.float32))
            colours.append(np.asarray(image.resize(depth.shape[::-1]))[::step, ::step].reshape(-1, 3)[keep])
    xyz, rgb = np.concatenate(points), np.concatenate(colours)
    if not len(xyz):
        raise ValueError(f"{scene_id}: the cached views hold no geometry around the prediction")
    origin = xyz.min(axis=0)
    while True:
        index = np.floor((xyz - origin) / voxel).astype(np.int64)
        _, inverse, counts = np.unique((index[:, 0] << 42) | (index[:, 1] << 21) | index[:, 2],
                                       return_inverse=True, return_counts=True)
        if len(counts) <= max_points:
            break
        voxel *= max(1.05, math.sqrt(len(counts) / max_points))
    xyz = _voxel_mean(xyz, inverse, counts).astype(np.float32)
    rgb = np.round(_voxel_mean(rgb, inverse, counts)).astype(np.uint8)
    spacing = float(np.median(cKDTree(xyz).query(xyz[:: len(xyz) // 2000 + 1], k=2)[0][:, 1]))
    return LocalScene(xyz, rgb, [x0, y0, x1, y1], display_height(xyz, world_bbox), radius, spacing)


def viewer_html(scene, title="", subtitle="", query="", bev=None, zooms=(), height=560):
    """Self-contained WebGL page, wrapped in an iframe, that orbits the scene and its predicted box.

    ``bev`` is the scene's ``(image, mapping)`` pair from ``load_bev``. With it the view opens on
    the global BEV and flies through ``zooms``, the world XY boxes of the zoom views, to the box.
    """
    x0, y0, x1, y1 = scene.world_bbox
    origin = np.array([(x0 + x1) / 2, (y0 + y1) / 2, float(np.percentile(scene.xyz[:, 2], 0.5)) - 0.2])
    local = scene.xyz - origin
    low, span = local.min(axis=0), np.maximum(np.ptp(local, axis=0), 1e-6)

    def shift(box):
        return [float(box[0] - origin[0]), float(box[1] - origin[1]),
                float(box[2] - origin[0]), float(box[3] - origin[1])]

    data = {
        "title": title, "subtitle": subtitle, "query": query, "count": len(local),
        "positions": base64.b64encode(np.round((local - low) / span * 65535).astype("<u2").tobytes()).decode(),
        "colours": base64.b64encode(np.ascontiguousarray(scene.rgb, dtype=np.uint8).tobytes()).decode(),
        "low": low.tolist(), "span": span.tolist(), "radius": scene.radius, "spacing": scene.spacing,
        "box": [*shift(scene.world_bbox), float(scene.zs[0] - origin[2]), float(scene.zs[1] - origin[2])],
        "zooms": [shift(box) for box in zooms], "bev": None, "extent": 4 * scene.radius,
    }
    if bev is not None:
        image, mapping = bev
        bounds = shift([mapping["x_min"], mapping["y_min"], mapping["x_max"], mapping["y_max"]])
        scale = min(1.0, 1536 / max(image.size))
        small = image.resize((round(image.width * scale), round(image.height * scale)),
                             Image.Resampling.LANCZOS, reducing_gap=3.0)
        encoded = io.BytesIO()
        small.convert("RGB").save(encoded, "JPEG", quality=80)
        data["bev"] = {"image": "data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode(),
                       "bounds": bounds}
        data["extent"] = max(data["extent"], math.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1]))
    page = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")
    page = page.replace("/*DATA*/null", json.dumps(data).replace("</", "<\\/"))
    return (f'<div style="max-width:1280px"><iframe srcdoc="{html.escape(page)}" '
            f'style="width:100%;height:{int(height)}px;border:0;border-radius:10px"></iframe></div>')
