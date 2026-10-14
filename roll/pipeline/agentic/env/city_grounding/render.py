"""BEV rendering utilities.

Crops a zoomed region from a pre-rendered city BEV PNG using the
coordinate-mapping JSON saved alongside it, and caches raw point clouds for the
live oblique renderer in render_open3d.
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

# A single STPLS3D BEV PNG can reach ~240M pixels, above PIL's ~179M
# decompression-bomb threshold. These are our own pre-rendered files.
Image.MAX_IMAGE_PIXELS = 1_000_000_000


# ─── BEV (pre-rendered PNG) helpers ──────────────────────────────────────────

def _resolve_bev_path(bev_png: str | Path) -> Path:
    """Resolve a jsonl ``bev_png`` reference to an existing file.

    The shipped jsonls use paths RELATIVE to ``$CITY_AGENT_DATA`` (e.g.
    ``bev_cityrefer/birmingham_block_3.png``). Set that env var to the artifact
    root described in README.md. Absolute paths are honoured unchanged; an
    absolute path written on a different host is re-rooted at
    ``$CITY_AGENT_DATA`` by matching the ``agent_data*`` path segment.
    """
    p = Path(bev_png)
    root = os.environ.get("CITY_AGENT_DATA")
    if p.is_absolute():
        if p.exists():
            return p
        if root:
            parts = p.parts
            for i, seg in enumerate(parts):
                if seg.startswith("agent_data"):
                    cand = Path(root, *parts[i + 1:])
                    if cand.exists():
                        return cand
                    break
        return p  # let Image.open raise with the original path
    if not root:
        raise RuntimeError(
            f"bev_png {bev_png!r} is relative but CITY_AGENT_DATA is unset. "
            "Export CITY_AGENT_DATA=/path/to/agent_data (see README.md)."
        )
    return Path(root) / p


@lru_cache(maxsize=64)
def _load_bev_cached(resolved_path: str):
    """Process-level cache: each scene's BEV is decoded once per worker.
    CityRefer has ~30 unique scenes; 64 slots covers all with headroom."""
    bev_png = Path(resolved_path)
    bev_json = bev_png.with_suffix(".json")
    img = Image.open(bev_png).convert("RGB")
    with open(bev_json) as f:
        mapping = json.load(f)
    return img, mapping


def load_bev(bev_png: str | Path):
    """Load BEV image and its coordinate mapping JSON."""
    resolved = _resolve_bev_path(bev_png)
    return _load_bev_cached(str(resolved))


def world_to_pixel(mapping: dict, wx: float, wy: float):
    """Convert world (x, y) → image pixel (col, row)."""
    px = (wx - mapping["x_min"]) / mapping["res_x"]
    py = (mapping["y_max"] - wy) / mapping["res_y"]
    return px, py


def pixel_to_world(mapping: dict, px: float, py: float):
    """Convert image pixel (col, row) → world (x, y)."""
    wx = mapping["x_min"] + px * mapping["res_x"]
    wy = mapping["y_max"] - py * mapping["res_y"]
    return wx, wy


def crop_bev_with_mapping(
    img: Image.Image,
    mapping: dict,
    world_x_min: float,
    world_y_min: float,
    world_x_max: float,
    world_y_max: float,
    output_size: int = 1024,
    padding_frac: float = 0.0,
) -> tuple[Image.Image, dict]:
    """Same as `crop_bev`, but also returns a `crop_mapping` whose
    (x_min / y_max / res_x / res_y) correctly describe the returned canvas,
    accounting for (a) `padding_frac` expansion of the world region and
    (b) the aspect-preserving black-canvas letterbox offset.

    With this mapping, `world_to_pixel(crop_mapping, wx, wy)` gives the
    pixel position of (wx, wy) *on the returned canvas image*.
    """
    w_range = world_x_max - world_x_min
    h_range = world_y_max - world_y_min
    pad_x = w_range * padding_frac
    pad_y = h_range * padding_frac

    # Padded world bounds actually shown in the crop
    padded_x_min = world_x_min - pad_x
    padded_x_max = world_x_max + pad_x
    padded_y_min = world_y_min - pad_y
    padded_y_max = world_y_max + pad_y

    left, top     = world_to_pixel(mapping, padded_x_min, padded_y_max)
    right, bottom = world_to_pixel(mapping, padded_x_max, padded_y_min)

    W, H = img.size
    left   = max(0, int(left))
    top    = max(0, int(top))
    right  = min(W, int(right) + 1)
    bottom = min(H, int(bottom) + 1)

    if right <= left or bottom <= top:
        canvas = img.resize((output_size, output_size), Image.LANCZOS)
        crop_mapping = {
            "x_min": padded_x_min,
            "x_max": padded_x_max,
            "y_min": padded_y_min,
            "y_max": padded_y_max,
            "res_x": (padded_x_max - padded_x_min) / output_size,
            "res_y": (padded_y_max - padded_y_min) / output_size,
        }
        return canvas, crop_mapping

    cropped = img.crop((left, top, right, bottom))
    cw, ch = cropped.size
    scale = output_size / max(cw, ch)
    new_w, new_h = int(cw * scale), int(ch * scale)
    resized = cropped.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (output_size, output_size), (0, 0, 0))
    offset_x = (output_size - new_w) // 2
    offset_y = (output_size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    # World bounds actually covered by the cropped source pixels
    shown_x_min, shown_y_max = pixel_to_world(mapping, left,  top)
    shown_x_max, shown_y_min = pixel_to_world(mapping, right, bottom)
    eff_res_x = (shown_x_max - shown_x_min) / max(1, new_w)
    eff_res_y = (shown_y_max - shown_y_min) / max(1, new_h)

    # Shift origin so canvas pixel (0,0) corresponds to the black-bar corner
    crop_mapping = {
        "x_min": shown_x_min - offset_x * eff_res_x,
        "x_max": shown_x_max + (output_size - offset_x - new_w) * eff_res_x,
        "y_min": shown_y_min - (output_size - offset_y - new_h) * eff_res_y,
        "y_max": shown_y_max + offset_y * eff_res_y,
        "res_x": eff_res_x,
        "res_y": eff_res_y,
    }
    return canvas, crop_mapping


def crop_bev_fit(
    img: Image.Image,
    mapping: dict,
    world_x_min: float,
    world_y_min: float,
    world_x_max: float,
    world_y_max: float,
    max_side: int = 768,
    patch: int = 32,
    max_pixels: int = 589_824,
    padding_frac: float = 0.0,
) -> tuple[Image.Image, dict]:
    """Like `crop_bev_with_mapping`, but returns an aspect-preserved rectangle
    (no square letterbox). Target size is computed by `compute_display_size`
    so the result is patch-aligned for the VLM image encoder.
    """
    w_range = world_x_max - world_x_min
    h_range = world_y_max - world_y_min
    pad_x = w_range * padding_frac
    pad_y = h_range * padding_frac
    padded_x_min = world_x_min - pad_x
    padded_x_max = world_x_max + pad_x
    padded_y_min = world_y_min - pad_y
    padded_y_max = world_y_max + pad_y

    left, top     = world_to_pixel(mapping, padded_x_min, padded_y_max)
    right, bottom = world_to_pixel(mapping, padded_x_max, padded_y_min)

    W, H = img.size
    left   = max(0, int(left))
    top    = max(0, int(top))
    right  = min(W, int(right) + 1)
    bottom = min(H, int(bottom) + 1)

    if right <= left or bottom <= top:
        # Degenerate crop — fall back to a square resize of the whole image.
        dw, dh = compute_display_size(W, H, max_side=max_side,
                                      patch=patch, max_pixels=max_pixels)
        canvas = img.resize((dw, dh), Image.Resampling.LANCZOS)
        crop_mapping = {
            "x_min": padded_x_min,
            "x_max": padded_x_max,
            "y_min": padded_y_min,
            "y_max": padded_y_max,
            "res_x": (padded_x_max - padded_x_min) / max(1, dw),
            "res_y": (padded_y_max - padded_y_min) / max(1, dh),
        }
        return canvas, crop_mapping

    cropped = img.crop((left, top, right, bottom))
    cw, ch = cropped.size
    dw, dh = compute_display_size(cw, ch, max_side=max_side,
                                  patch=patch, max_pixels=max_pixels)
    resized = cropped.resize((dw, dh), Image.Resampling.LANCZOS)

    shown_x_min, shown_y_max = pixel_to_world(mapping, left,  top)
    shown_x_max, shown_y_min = pixel_to_world(mapping, right, bottom)
    crop_mapping = {
        "x_min": shown_x_min,
        "x_max": shown_x_max,
        "y_min": shown_y_min,
        "y_max": shown_y_max,
        "res_x": (shown_x_max - shown_x_min) / max(1, dw),
        "res_y": (shown_y_max - shown_y_min) / max(1, dh),
    }
    return resized, crop_mapping


def compute_display_size(
    bev_w: int,
    bev_h: int,
    *,
    max_side: int = 1024,
    patch: int = 32,
    max_pixels: int = 1_200_000,
) -> tuple[int, int]:
    """Aspect-preserving target size for the BEV shown to the VLM.

    Rounds each side to a multiple of `patch` (Qwen-VL image-encoder patch),
    then shrinks proportionally if W*H exceeds `max_pixels`.
    """
    long_side = max(bev_w, bev_h)
    r = min(1.0, max_side / long_side)
    dw = max(patch, round(bev_w * r / patch) * patch)
    dh = max(patch, round(bev_h * r / patch) * patch)
    if dw * dh > max_pixels:
        shrink = (max_pixels / (dw * dh)) ** 0.5
        dw = max(patch, (int(dw * shrink) // patch) * patch)
        dh = max(patch, (int(dh * shrink) // patch) * patch)
    return dw, dh

# ─── Point-cloud loading (used by the live oblique renderer) ─────────────────

_PLY_CACHE: dict = {}   # ply_path -> (pts: np.ndarray, colors: np.ndarray)


def _clear_ply_cache() -> None:
    _PLY_CACHE.clear()


def _load_ply_cached(ply_path: str):
    """Load PLY once per process and cache as numpy arrays."""
    if ply_path in _PLY_CACHE:
        return _PLY_CACHE[ply_path]

    from plyfile import PlyData
    data = PlyData.read(ply_path)
    v = data["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    # Colors: try red/green/blue fields
    try:
        r = np.asarray(v["red"],   dtype=np.float32)
        g = np.asarray(v["green"], dtype=np.float32)
        b = np.asarray(v["blue"],  dtype=np.float32)
        # Values may be 0-255 or 0-1
        if r.max() <= 1.0:
            r, g, b = r * 255, g * 255, b * 255
        colors = np.stack([r, g, b], axis=1).astype(np.uint8)
    except (ValueError, KeyError):
        colors = np.full((len(pts), 3), 180, dtype=np.uint8)

    _PLY_CACHE[ply_path] = (pts, colors)
    return pts, colors
