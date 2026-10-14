"""Region-based multiview cache for CityToolEnv.

Cache layout per scene, produced by the dataset-construction pipeline
(see the Data section of README.md):

    <cache_root>/<scene_id>/
        scene_meta.json       {scene_id, dataset, ground_z, region_size,
                               output_size, elevation_deg, camera_dist_factor}
        regions.npy           float32 (N, 4)  XY world [x0, y0, x1, y1]
        index.pkl             {"tree": cKDTree, "feats": (N, 2) [cx, cy]}
        imgs/{id:06d}_{e,n,w,s}.jpg
        cams/{id:06d}.npz     extrinsics (4, 4, 4), intrinsics (4, 3, 3),
                              targets (4, 3), box_sizes (4,), output_size

At runtime (headless training host, no render stack), ``BboxCache`` looks up
the nearest region, projects the candidate bbox's 3D cuboid through the 4
stored cameras, and draws the cuboid edges onto copies of the cached images.
The VLM thus sees a per-query bbox marker WITHOUT any Open3D dependency.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


def _region_feat(world_bbox_xy: list | np.ndarray) -> np.ndarray:
    """(cx, cy) of a world bbox — the KDTree key."""
    x0, y0, x1, y1 = [float(v) for v in world_bbox_xy[:4]]
    return np.asarray([0.5 * (x0 + x1), 0.5 * (y0 + y1)], dtype=np.float32)


def _marker_points(world_bbox_xy: list, z: float) -> np.ndarray:
    """Centre + 4 footprint corners, all at ground height. Used for the
    point-style marker (not a cuboid) — since cached oblique imagery has no
    occlusion with the overlay, a flat marker is crisper than a wireframe."""
    x0, y0, x1, y1 = [float(v) for v in world_bbox_xy[:4]]
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    return np.asarray([
        [cx, cy, z],          # 0: centre
        [x0, y0, z],          # 1: footprint corners
        [x0, y1, z],
        [x1, y0, z],
        [x1, y1, z],
    ], dtype=np.float64)


# 8 cuboid corners indexed as (x, y, z) with bits 000..111 → x0y0z0, x0y0z1, ...
_CUBOID_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),     # vertical (z-)pillars
    (0, 2), (1, 3), (4, 6), (5, 7),     # y edges
    (0, 4), (1, 5), (2, 6), (3, 7),     # x edges
]


def _cuboid_corners(world_bbox_xy: list, z_min: float, z_max: float) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in world_bbox_xy[:4]]
    return np.asarray([
        [x0, y0, z_min], [x0, y0, z_max],
        [x0, y1, z_min], [x0, y1, z_max],
        [x1, y0, z_min], [x1, y0, z_max],
        [x1, y1, z_min], [x1, y1, z_max],
    ], dtype=np.float64)


def _infer_bbox_z_range(depth: np.ndarray,
                        extrinsic: np.ndarray, intrinsic: np.ndarray,
                        world_bbox_xy: list,
                        color_image_size: tuple[int, int],
                        ) -> tuple[float, float, int]:
    """Back-project every valid pixel of ``depth`` to world xyz, keep those
    whose world XY falls inside the bbox footprint, and return
    ``(min_world_z, max_world_z, n_hits)``.

    ``min_world_z`` approximates the LOCAL ground at the bbox (not the
    scene-wide mean) — critical for terrain that slopes across a scene:
    placing a cuboid at a scene-wide "average ground z" makes it float
    several metres above actual ground in low areas. ``max_world_z`` is
    the tallest point inside the footprint — the target's height proxy.
    """
    if depth is None or depth.size == 0:
        return 0.0, 0.0, 0
    h, w = depth.shape
    img_w, img_h = color_image_size
    us = np.arange(w, dtype=np.float32) * (img_w / max(1, w))
    vs = np.arange(h, dtype=np.float32) * (img_h / max(1, h))
    uc, vc = np.meshgrid(us, vs)

    fx = float(intrinsic[0, 0]); fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2]); cy = float(intrinsic[1, 2])
    Zc = depth
    valid = Zc > 0.0
    if not valid.any():
        return 0.0, 0.0, 0
    Xc = (uc - cx) * Zc / fx
    Yc = (vc - cy) * Zc / fy

    R = extrinsic[:3, :3].astype(np.float64)
    t = extrinsic[:3, 3].astype(np.float64)
    cam_pts = np.stack([Xc, Yc, Zc], axis=-1).astype(np.float64)
    world = (cam_pts - t) @ R

    x0, y0, x1, y1 = [float(v) for v in world_bbox_xy[:4]]
    wx = world[..., 0]; wy = world[..., 1]; wz = world[..., 2]
    mask = valid & (wx >= x0) & (wx <= x1) & (wy >= y0) & (wy <= y1)
    n = int(mask.sum())
    if n == 0:
        return 0.0, 0.0, 0
    zs = wz[mask]
    return float(zs.min()), float(zs.max()), n


def _draw_cuboid_occluded(
    img: Image.Image, corners_world: np.ndarray,
    extrinsic: np.ndarray, intrinsic: np.ndarray,
    depth: np.ndarray | None,
    *,
    color_visible=(255, 0, 0),
    color_hidden=(255, 0, 0),
    width_visible: int = 3,
    width_hidden: int = 2,
    n_samples_per_edge: int = 32,
    dash_period: int = 4,
    occlusion_eps_m: float = 0.8,
) -> Image.Image:
    """Draw the 12 cuboid edges with per-segment depth culling.

    Each edge is sampled at ``n_samples_per_edge + 1`` points in WORLD
    space (linear-in-world so the projection stays correct under
    perspective). For every adjacent pair the midpoint's visibility is
    tested against the depth buffer:

        • visible  → solid bright-red stroke (width 3)
        • occluded → thin light-grey stroke, only every ``dash_period``th
                     sub-segment (dashed line)

    The dashed-vs-solid contrast makes the "behind-geometry" edges look
    clearly distinct from the front-facing ones, which was the point of
    occlusion-aware drawing. ``depth=None`` disables the test and every
    sub-segment is drawn as visible.
    """
    out = img.copy()
    draw = ImageDraw.Draw(out)
    W, H = out.size

    for a, b in _CUBOID_EDGES:
        ts = np.linspace(0.0, 1.0, n_samples_per_edge + 1)
        pts = (corners_world[a][None, :] * (1 - ts[:, None]) +
               corners_world[b][None, :] *      ts[:, None])
        uv, zc = _project(pts, extrinsic, intrinsic)
        for i in range(n_samples_per_edge):
            za, zb = float(zc[i]), float(zc[i + 1])
            if za <= 0 and zb <= 0:
                continue
            ua, va = float(uv[i,     0]), float(uv[i,     1])
            ub, vb = float(uv[i + 1, 0]), float(uv[i + 1, 1])
            if ((ua < -W and ub < -W) or (ua > 2 * W and ub > 2 * W) or
                (va < -H and vb < -H) or (va > 2 * H and vb > 2 * H)):
                continue
            if depth is not None:
                mu, mv = 0.5 * (ua + ub), 0.5 * (va + vb)
                mz     = 0.5 * (za + zb)
                scene_z = _sample_depth(depth, mu, mv,
                                        image_size=out.size, window=1)
                occluded = (scene_z > 0.0) and (mz > scene_z + occlusion_eps_m)
            else:
                occluded = False
            if occluded:
                continue
            draw.line([(ua, va), (ub, vb)],
                      fill=color_visible, width=width_visible)
    return out


def _project(points_world: np.ndarray, extrinsic: np.ndarray,
             intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply 4x4 extrinsic + 3x3 intrinsic; return (uv (N,2), z (N,))."""
    N = points_world.shape[0]
    homo = np.concatenate([points_world, np.ones((N, 1))], axis=1)     # (N, 4)
    cam  = homo @ extrinsic.T                                          # (N, 4)
    z    = cam[:, 2]
    safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
    img  = (cam[:, :3] / safe[:, None]) @ intrinsic.T                  # (N, 3)
    return img[:, :2], z


def _sample_depth(depth: np.ndarray | None, u: float, v: float,
                  image_size: tuple[int, int] | None = None,
                  window: int = 2) -> float:
    """Return the MINIMUM point-cloud depth in a small (2*window+1)² patch
    around (u, v). Taking the min (closest) makes occlusion tests more
    forgiving — a target at a point-cloud hole between rooftop tiles still
    counts as 'behind the roof' rather than 'visible through the gap'.
    ``image_size`` is (W, H) of the colour image — if it differs from the
    depth shape, (u, v) is scaled accordingly (depth may be downsampled).
    Returns 0 if depth is unavailable or the sample is off-canvas or the
    patch has no rendered geometry."""
    if depth is None:
        return 0.0
    H, W = depth.shape
    if image_size is not None:
        img_w, img_h = image_size
        u = u * W / max(1, img_w)
        v = v * H / max(1, img_h)
    iu, iv = int(round(u)), int(round(v))
    if iu < 0 or iu >= W or iv < 0 or iv >= H:
        return 0.0
    u0 = max(0, iu - window); u1 = min(W, iu + window + 1)
    v0 = max(0, iv - window); v1 = min(H, iv + window + 1)
    patch = depth[v0:v1, u0:u1]
    pos = patch[patch > 0.0]
    if pos.size == 0:
        return 0.0
    return float(pos.min())


def _draw_point_marker(img: Image.Image, uv: np.ndarray, z: np.ndarray,
                       depth: np.ndarray | None = None,
                       color=(255, 0, 0), outline=(255, 255, 255),
                       occluded_color=(120, 120, 120),
                       occluded_outline=(240, 240, 240),
                       radius: int = 10, footprint: bool = False,
                       occlusion_eps_m: float = 0.5) -> Image.Image:
    """Draw a filled dot at the bbox centre plus (optional) a thin footprint
    rectangle connecting the 4 XY corners projected at ground height.

    ``uv``/``z`` shape (5, …): row 0 is centre, rows 1-4 are footprint
    corners [x0y0, x0y1, x1y0, x1y1].
    """
    out = img.copy()
    if z[0] <= 0:
        return out  # centre behind camera — no usable marker
    W, H = out.size
    draw = ImageDraw.Draw(out)

    # Footprint quad (thin line, good for showing bbox XY extent without the
    # visual noise of a 3D wireframe). Skip if centre only is wanted.
    if footprint:
        quad = [1, 2, 4, 3, 1]  # loop around x0y0 → x0y1 → x1y1 → x1y0 → back
        for i in range(len(quad) - 1):
            a, b = quad[i], quad[i + 1]
            if z[a] <= 0 and z[b] <= 0:
                continue
            xa, ya = float(uv[a, 0]), float(uv[a, 1])
            xb, yb = float(uv[b, 0]), float(uv[b, 1])
            if ((xa < -W and xb < -W) or (xa > 2 * W and xb > 2 * W) or
                (ya < -H and yb < -H) or (ya > 2 * H and yb > 2 * H)):
                continue
            draw.line([(xa, ya), (xb, yb)], fill=color, width=2)

    cx, cy = float(uv[0, 0]), float(uv[0, 1])
    cz = float(z[0])
    r = int(radius)

    # Occlusion test: if the scene point cloud at (cx, cy) is CLOSER than
    # the marker by more than eps, the marker sits behind geometry. Draw
    # it in a greyed-out colour so the VLM reads "target is hidden by the
    # building at this spot". `depth` may be stored at a lower resolution
    # than the colour image (downsampled at save time); _sample_depth
    # scales uv accordingly using the colour-image size.
    scene_z = _sample_depth(depth, cx, cy, image_size=out.size) \
              if depth is not None else 0.0
    occluded = (scene_z > 0.0) and (cz > scene_z + occlusion_eps_m)
    dot_fill    = occluded_color if occluded else color
    halo_fill   = occluded_outline if occluded else outline
    # White halo so the dot stays visible over dark point clouds.
    draw.ellipse([cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2],
                 fill=halo_fill)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=dot_fill)
    return out


class BboxCache:
    """Per-scene region cache; lazy image + cam load on each lookup.

    One instance per (cache_root, scene_id) is cached in a process-level
    registry so Ray workers don't reload the KDTree every episode.
    """

    _global_registry: dict[tuple[str, str], "BboxCache"] = {}

    def __init__(self, cache_root: str | Path, scene_id: str,
                 dist_threshold: float = 60.0):
        self.root = Path(cache_root) / scene_id
        self.scene_id = scene_id
        self.dist_threshold = float(dist_threshold)
        self._loaded = False
        self._tree = None
        self._feats: Optional[np.ndarray] = None
        self._regions: Optional[np.ndarray] = None
        self._ids: Optional[np.ndarray] = None
        self._scene_meta: dict = {}

    @classmethod
    def get(cls, cache_root: str | Path, scene_id: str,
            dist_threshold: float = 60.0) -> "BboxCache":
        key = (str(cache_root), scene_id)
        inst = cls._global_registry.get(key)
        if inst is None:
            inst = cls(cache_root, scene_id, dist_threshold=dist_threshold)
            cls._global_registry[key] = inst
        return inst

    def _load(self) -> bool:
        if self._loaded:
            return self._tree is not None
        self._loaded = True
        idx_p   = self.root / "index.pkl"
        reg_p   = self.root / "regions.npy"
        meta_p  = self.root / "scene_meta.json"
        if not (idx_p.exists() and reg_p.exists() and meta_p.exists()):
            logger.info("BboxCache: no cache for scene %s at %s",
                        self.scene_id, self.root)
            return False
        try:
            with open(idx_p, "rb") as f:
                payload = pickle.load(f)
            self._tree = payload["tree"]
            self._feats = np.asarray(payload["feats"], dtype=np.float32)
            regions = np.asarray(np.load(reg_p))
            self._regions = regions
            self._ids = np.arange(int(regions.shape[0]), dtype=np.int32)
            self._scene_meta = json.loads(meta_p.read_text())
        except Exception as e:
            logger.warning("BboxCache: load %s failed: %s", self.root, e)
            self._tree = None
            self._regions = None
            return False
        logger.info("BboxCache: scene=%s loaded %d regions",
                    self.scene_id, int(regions.shape[0]))
        return True

    def lookup(self, world_bbox: list,
               only_direction: Optional[str] = None,
              ) -> tuple[Optional[list], Optional[float]]:
        """Return (PIL Images with the candidate bbox drawn, dist_m).

        With ``only_direction=None`` returns all 4 directions
        [east, north, west, south] — kept for backward compat. When the
        caller only needs ONE direction (e.g. tool_render_oblique post-
        refactor), pass ``only_direction="east"|"north"|"west"|"south"``
        and we'll skip 3/4 of the disk IO + image decode + wireframe
        annotation. The returned list still has length 4 with `None` in
        non-requested slots so callers indexing by DIRECTION_INDEX[card]
        keep working.

        Queries the k=3 nearest regions and returns the first one whose
        on-disk images/cams actually exist. Builder records ALL sampled
        regions in the KDTree (including ones where rendering failed for
        scene-edge / empty areas), so nearest-neighbour can return a "no
        assets" entry — falling back to the next-nearest avoids turning
        those into silent cache misses.

        Returns ``(None, dist_of_nearest)`` when every candidate is either
        too far or missing on disk, and ``(None, None)`` when the cache
        itself isn't present for this scene.
        """
        if not self._load():
            return None, None
        assert self._tree is not None
        q = _region_feat(world_bbox)
        dists, idxs = self._tree.query(q, k=3)
        dists = np.atleast_1d(dists)
        idxs  = np.atleast_1d(idxs)
        first_dist = float(dists[0])
        for d, i in zip(dists, idxs):
            d_f = float(d)
            if d_f > self.dist_threshold:
                break
            imgs = self._load_and_annotate(
                int(i), list(world_bbox), only_direction=only_direction,
            )
            if imgs is not None:
                return imgs, d_f
        return None, first_dist

    def _load_depth_stack(self, entry_id: int, fmt: str,
                          scale: float) -> Optional[np.ndarray]:
        """Return a (4, H, W) float32 array of per-view camera-space depth
        in metres, or None if any expected file is missing / unreadable.
        Supports npz (uint16 stacked), png (uint16 per view), jpg (uint8
        per view, lossy with fixed 150m clip)."""
        try:
            if fmt == "npz":
                p = self.root / "depths" / f"{entry_id:06d}.npz"
                if not p.exists():
                    return None
                return np.load(p)["depths"].astype(np.float32) * scale
            arr = []
            for card in ("e", "n", "w", "s"):
                p = self.root / "depths" / f"{entry_id:06d}_{card}.{fmt}"
                if not p.exists():
                    return None
                im = Image.open(p)
                if fmt == "png":
                    # uint16 grayscale (PIL mode "I" or "I;16"); convert to
                    # numpy then scale by depth_scale m/LSB.
                    d = np.asarray(im, dtype=np.float32) * scale
                else:  # jpg
                    # 8-bit, fixed 0..150m range used at save time.
                    d = np.asarray(im.convert("L"), dtype=np.float32) \
                        * (150.0 / 255.0)
                arr.append(d)
            return np.stack(arr, axis=0)
        except Exception as e:
            logger.warning("BboxCache: depth load failed fmt=%s entry=%d: %s",
                           fmt, entry_id, e)
            return None

    # Cardinal direction → fixed slot in the returned 4-tuple. tool_env's
    # DIRECTION_INDEX uses the same order. Kept here so this module is
    # self-contained.
    _CARD_ORDER = ("e", "n", "w", "s")
    _CARD_FULL  = {"e": "east", "n": "north", "w": "west", "s": "south"}
    _FULL_TO_CARD = {v: k for k, v in _CARD_FULL.items()}

    def _load_and_annotate(self, entry_id: int,
                           world_bbox: list,
                           only_direction: Optional[str] = None,
                          ) -> Optional[list]:
        # Decide which card slots to actually load+annotate. None means
        # all 4. The returned list always has length 4 with None in
        # skipped slots so DIRECTION_INDEX-based indexing keeps working.
        if only_direction is None:
            wanted_cards = set(self._CARD_ORDER)
        else:
            card = self._FULL_TO_CARD.get(only_direction.lower())
            if card is None:
                logger.warning("BboxCache: bad direction %r, "
                               "loading all 4", only_direction)
                wanted_cards = set(self._CARD_ORDER)
            else:
                wanted_cards = {card}

        # Images. Existence-check ALL 4 first (cheap stat) so we keep the
        # original "fall back to next nearest if any image is missing"
        # semantics — partial caches shouldn't silently survive.
        for card in self._CARD_ORDER:
            p = self.root / "imgs" / f"{entry_id:06d}_{card}.jpg"
            if not p.exists():
                logger.warning("BboxCache: missing image %s", p)
                return None
        # Decode only the wanted ones; placeholder None for skipped.
        imgs: list[Optional[Image.Image]] = []
        for card in self._CARD_ORDER:
            if card not in wanted_cards:
                imgs.append(None)
                continue
            p = self.root / "imgs" / f"{entry_id:06d}_{card}.jpg"
            imgs.append(Image.open(p).convert("RGB"))

        # Cameras.
        cam_p = self.root / "cams" / f"{entry_id:06d}.npz"
        if not cam_p.exists():
            logger.warning("BboxCache: missing cams %s", cam_p)
            return imgs  # return raw views without annotation
        try:
            z = np.load(cam_p)
            extrinsics = z["extrinsics"]           # (4, 4, 4)
            intrinsics = z["intrinsics"]           # (4, 3, 3)
        except Exception as e:
            logger.warning("BboxCache: cams load %s failed: %s", cam_p, e)
            return imgs

        # Optional depth buffer (per-pixel point-cloud z). Lets the marker
        # grey out when sitting behind scene geometry. Absent → the draw
        # path skips the occlusion test entirely. Supports three on-disk
        # formats chosen at build time (``depth_format`` in scene_meta):
        #   • npz (legacy): single (4, H, W) uint16 per entry
        #   • png          : 4 × uint16 grayscale PNG (lossless, smallest lossless)
        #   • jpg          : 4 × 8-bit grayscale JPG (lossy, ~17× smaller than npz)
        depth_stack: np.ndarray | None = None
        if self._scene_meta.get("save_depth"):
            fmt = str(self._scene_meta.get("depth_format", "npz"))
            scale = float(self._scene_meta.get("depth_scale", 0.05))
            depth_stack = self._load_depth_stack(entry_id, fmt, scale)

        # Two overlay modes:
        #   • depth buffer present → infer the cuboid's top-z from depth
        #     (method B: back-project every depth pixel inside the bbox
        #     footprint, take max world z), then draw the full 3D cuboid
        #     wireframe with per-segment occlusion culling.
        #   • depth absent → fall back to a flat centre dot (cleaner than a
        #     flat cuboid with no occlusion signal).
        scene_ground_z = float(self._scene_meta.get("ground_z", 0.0))
        default_height = float(self._scene_meta.get("default_bbox_height", 3.0))
        min_h = float(self._scene_meta.get("min_bbox_height", 0.5))
        max_h = float(self._scene_meta.get("max_bbox_height", 60.0))
        has_depth = depth_stack is not None

        out: list[Optional[Image.Image]] = []
        # Image size for projection — pull from any decoded slot (they
        # all share size by construction; we only loaded 1 if filtered).
        first_img_size = next(
            (im.size for im in imgs if im is not None), None
        )
        if has_depth and first_img_size is not None:
            assert depth_stack is not None
            # Per-view (ground_z_local, top_z, n_hits). Use the MIN of ground
            # across views (the actual surface, not a floating point cloud
            # outlier) and MEDIAN of top across views (robust to stray
            # trees / overlapping buildings behind the target).
            zr = [_infer_bbox_z_range(
                    depth_stack[i], extrinsics[i], intrinsics[i],
                    world_bbox, color_image_size=first_img_size,
                  ) for i in range(len(imgs))]
            good = [z for z in zr if z[2] > 0]
            if good:
                ground_local = float(np.min([z[0] for z in good]))
                top_local    = float(np.median([z[1] for z in good]))
                height = float(np.clip(top_local - ground_local, min_h, max_h))
                base_z = ground_local
            else:
                # No depth samples inside footprint — fall back to scene
                # ground + default height.
                base_z = scene_ground_z
                height = default_height
            logger.debug(
                "BboxCache z-range: per-view=%s  base=%.1fm height=%.1fm",
                [(round(z[0],1), round(z[1],1), z[2]) for z in zr],
                base_z, height,
            )
            corners = _cuboid_corners(world_bbox, base_z, base_z + height)
            # Skip non-loaded slots — they stay None in `out` so callers
            # indexing by DIRECTION_INDEX[card] still get the right slot.
            for i, (img, ext, K) in enumerate(
                    zip(imgs, extrinsics, intrinsics)):
                if img is None:
                    out.append(None)
                    continue
                out.append(_draw_cuboid_occluded(
                    img, corners, ext, K, depth_stack[i],
                ))
        else:
            corners = _marker_points(world_bbox, scene_ground_z)
            for img, ext, K in zip(imgs, extrinsics, intrinsics):
                if img is None:
                    out.append(None)
                    continue
                uv, zc = _project(corners, ext, K)
                out.append(_draw_point_marker(img, uv, zc, depth=None))
        return out

    @property
    def size(self) -> int:
        if not self._load() or self._regions is None:
            return 0
        return int(self._regions.shape[0])
