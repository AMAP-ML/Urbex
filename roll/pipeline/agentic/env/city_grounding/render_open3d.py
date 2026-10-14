"""Open3D-backed live oblique rendering.

Fallback for when the offline cache has no region covering the requested box.
Mirrors the camera / AABB / PinholeCameraParameters setup used by the cache
builder so live and cached views are interchangeable. A single hidden Visualizer
is kept per process (lazy-init); the full scene PointCloud is uploaded once per
scene_id via ``set_pcd`` and only the red AABB LineSet is swapped per call.

Requires a GL context (real DISPLAY or xvfb-run). If Open3D cannot open a
window (e.g. headless with no xvfb), ``render_region`` returns an empty
list and the caller should treat that as "no oblique views this turn".
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# 4 oblique directions, matching precompute_candidate_views._DIRECTIONS
DIRECTIONS = [("east", 0), ("north", 90), ("west", 180), ("south", 270)]

_BBOX_EDGES = [
    (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7),
]


def _aabb_corners(mn, mx):
    xs, ys, zs = [mn[0], mx[0]], [mn[1], mx[1]], [mn[2], mx[2]]
    return np.array([[x, y, z] for x in xs for y in ys for z in zs])


def _eye_from_azel(center, distance, azimuth_deg, elevation_deg):
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    offset = distance * np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])
    return center + offset


def _build_aabb_lineset(corners_world, color=(1.0, 0.0, 0.0)):
    import open3d as o3d
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(corners_world, dtype=np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.asarray(_BBOX_EDGES, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, dtype=np.float64), (len(_BBOX_EDGES), 1))
    )
    return ls


def _make_pinhole_params(eye, target, world_up, output_size: int, box_size: float):
    import open3d as o3d

    eye      = np.asarray(eye,      dtype=np.float64)
    target   = np.asarray(target,   dtype=np.float64)
    world_up = np.asarray(world_up, dtype=np.float64)

    forward = target - eye
    d = float(np.linalg.norm(forward))
    forward = forward / (d + 1e-9)

    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)

    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[0, :3] = right
    extrinsic[1, :3] = down
    extrinsic[2, :3] = forward
    extrinsic[:3, 3] = -extrinsic[:3, :3] @ eye

    fy = output_size * d / box_size
    fx = fy
    cx = output_size / 2.0 - 0.5
    cy = output_size / 2.0 - 0.5

    intr = o3d.camera.PinholeCameraIntrinsic(
        output_size, output_size, fx, fy, cx, cy,
    )
    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic = intr
    params.extrinsic = extrinsic
    return params


class _SceneRenderer:
    """Hidden o3d Visualizer; one scene PCD uploaded, AABB swapped per call."""

    def __init__(self, size: int, point_size: float,
                 background=(1.0, 1.0, 1.0)):
        import open3d as o3d
        self._o3d = o3d
        self.size = size
        self.background = np.asarray(background, dtype=np.float64)
        self.vis = o3d.visualization.Visualizer()  # type: ignore[attr-defined]
        self.vis.create_window(width=size, height=size, visible=False)
        opt = self.vis.get_render_option()
        opt.point_size       = point_size
        opt.line_width       = 3.0
        opt.background_color = self.background
        self._scene_key: Optional[str] = None
        self._pcd = None
        self._aabb = None

    def set_pcd(self, pts: np.ndarray, colors_uint8: np.ndarray,
                scene_key: str) -> None:
        if self._scene_key == scene_key:
            return
        o3d = self._o3d
        self.vis.clear_geometries()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(
            colors_uint8.astype(np.float64) / 255.0
        )
        self.vis.add_geometry(pcd, reset_bounding_box=True)
        self._pcd = pcd
        self._aabb = None
        self._scene_key = scene_key

    def _capture(self, eye, target, box_size: float,
                 return_depth: bool = False):
        params = _make_pinhole_params(
            eye, target,
            world_up=[0.0, 0.0, 1.0],
            output_size=self.size,
            box_size=box_size,
        )
        ctr = self.vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        self.vis.poll_events()
        self.vis.update_renderer()
        if return_depth:
            # Open3D's legacy Visualizer clobbers the colour FBO after a
            # depth capture on some drivers (we observed solid-black JPEGs
            # when using do_render=False on the second call). Force a
            # separate draw pass for each buffer — slower by ~0.3s/view but
            # the two captures are guaranteed correct.
            color_buf = self.vis.capture_screen_float_buffer(do_render=True)
            depth_buf = self.vis.capture_depth_float_buffer(do_render=True)
            depth = np.asarray(depth_buf, dtype=np.float32)
        else:
            color_buf = self.vis.capture_screen_float_buffer(do_render=True)
        arr = (np.clip(np.asarray(color_buf), 0.0, 1.0) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        if return_depth:
            return img, depth
        return img

    def render_bbox(self, corners_world: Optional[np.ndarray], target: np.ndarray,
                    box_size: float, elevation_deg: float,
                    camera_dist_factor: float,
                    draw_bbox: bool = True,
                    return_depth: bool = False) -> dict:
        if self._aabb is not None:
            self.vis.remove_geometry(self._aabb, reset_bounding_box=False)
            self._aabb = None
        if draw_bbox and corners_world is not None:
            aabb = _build_aabb_lineset(corners_world)
            self.vis.add_geometry(aabb, reset_bounding_box=False)
            self._aabb = aabb

        cam_dist = camera_dist_factor * box_size
        out: dict = {}
        depths: dict = {}
        for name, az in DIRECTIONS:
            eye = _eye_from_azel(target, cam_dist, az, elevation_deg)
            cap = self._capture(eye=eye, target=target, box_size=box_size,
                                return_depth=return_depth)
            if return_depth:
                assert isinstance(cap, tuple)
                out[name], depths[name] = cap
            else:
                assert not isinstance(cap, tuple)
                out[name] = cap
        if return_depth:
            return {"views": out, "depths": depths}
        return out

    def close(self) -> None:
        try:
            self.vis.destroy_window()
        except Exception:
            pass


# ── Per-process singleton on a dedicated thread ──────────────────────────────
#
# Open3D's legacy Visualizer binds its GL context to the thread that called
# create_window(). Cross-thread calls (even serialized with a lock) segfault.
# We pin all Visualizer operations to one dedicated worker thread and dispatch
# render requests through a queue; concurrent callers simply wait on a
# per-request Event.

_RENDERER_DEAD = False  # True if we failed to create a Visualizer (no display)
_RENDERER_JOBS: "queue.Queue" = queue.Queue()
_RENDERER_THREAD: Optional[threading.Thread] = None
_RENDERER_START_LOCK = threading.Lock()


def _renderer_thread_main(size: int, point_size: float) -> None:
    """Owns the Visualizer. Reads jobs from _RENDERER_JOBS forever."""
    global _RENDERER_DEAD
    try:
        renderer = _SceneRenderer(size=size, point_size=point_size)
    except Exception as e:
        logger.warning(
            "Open3D Visualizer init failed (DISPLAY set? try xvfb-run): %s", e,
        )
        _RENDERER_DEAD = True
        # Drain pending jobs with empty results so callers don't block.
        while True:
            try:
                job = _RENDERER_JOBS.get(timeout=1.0)
            except queue.Empty:
                return
            job["result"]["views"] = []
            job["done"].set()

    while True:
        job = _RENDERER_JOBS.get()
        try:
            renderer.set_pcd(
                job["pts"], job["colors"], scene_key=job["scene_key"],
            )
            want_depth = bool(job.get("return_depth", False))
            out = renderer.render_bbox(
                corners_world      = job["corners"],
                target             = job["target"],
                box_size           = job["box_size"],
                elevation_deg      = job["elevation_deg"],
                camera_dist_factor = job["camera_dist_factor"],
                draw_bbox          = job.get("draw_bbox", True),
                return_depth       = want_depth,
            )
            if want_depth:
                views = out["views"]; depths = out["depths"]
                job["result"]["views"] = [
                    views[name] for name, _ in DIRECTIONS if name in views
                ]
                job["result"]["depths"] = [
                    depths[name] for name, _ in DIRECTIONS if name in depths
                ]
            else:
                job["result"]["views"] = [
                    out[name] for name, _ in DIRECTIONS if name in out
                ]
        except Exception as e:
            logger.warning("Open3D render failed: %s", e)
            job["result"]["views"] = []
            if job.get("return_depth"):
                job["result"]["depths"] = []
        finally:
            job["done"].set()


def _ensure_renderer_thread(size: int, point_size: float) -> bool:
    """Start the dedicated render thread on first use. Returns False if
    initialization already failed."""
    global _RENDERER_THREAD
    if _RENDERER_DEAD:
        return False
    if _RENDERER_THREAD is not None and _RENDERER_THREAD.is_alive():
        return True
    with _RENDERER_START_LOCK:
        if _RENDERER_DEAD:
            return False
        if _RENDERER_THREAD is None or not _RENDERER_THREAD.is_alive():
            t = threading.Thread(
                target=_renderer_thread_main,
                args=(size, point_size),
                name="o3d-render",
                daemon=True,
            )
            t.start()
            _RENDERER_THREAD = t
    return True


# ── Public entry point ────────────────────────────────────────────────────────

def compute_region_cameras(
    world_bbox_xy: list,
    scene_z_mid: float,
    *,
    output_size: int,
    scale: float              = 1.0,
    min_size: float           = 60.0,
    max_size: float           = 100.0,
    elevation_deg: float      = 30.0,
    camera_dist_factor: float = 2.0,
) -> list[dict]:
    """Pure-Python: compute the 4 camera specs that would be used for
    ``render_region`` over this XY region. Cached by the offline builder
    alongside the rendered images so the runtime can project a candidate
    bbox onto each oblique without touching Open3D.

    Returns a list of 4 dicts (east/north/west/south) each with:
        direction, eye (3,), target (3,), box_size,
        intrinsic (3,3), extrinsic (4,4), output_size.
    """
    x0, y0, x1, y1 = [float(v) for v in world_bbox_xy]
    target = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0, float(scene_z_mid)],
                      dtype=np.float64)
    box_size = float(np.clip(max(x1 - x0, y1 - y0) * scale, min_size, max_size))
    cam_dist = camera_dist_factor * box_size
    out: list[dict] = []
    for name, az in DIRECTIONS:
        eye = _eye_from_azel(target, cam_dist, az, elevation_deg)
        params = _make_pinhole_params(
            eye=eye, target=target, world_up=[0.0, 0.0, 1.0],
            output_size=output_size, box_size=box_size,
        )
        intrinsic = np.asarray(params.intrinsic.intrinsic_matrix, dtype=np.float64)
        extrinsic = np.asarray(params.extrinsic,                dtype=np.float64)
        out.append({
            "direction":   name,
            "eye":         np.asarray(eye, dtype=np.float64).tolist(),
            "target":      target.tolist(),
            "box_size":    box_size,
            "output_size": int(output_size),
            "intrinsic":   intrinsic.tolist(),
            "extrinsic":   extrinsic.tolist(),
        })
    return out


def render_region(
    ply_path: str,
    world_bbox: list,
    *,
    scene_key: Optional[str] = None,
    output_size: int        = 512,
    scale: float            = 3.0,
    min_size: float         = 20.0,
    max_size: float         = 100.0,
    elevation_deg: float    = 30.0,
    point_size: float       = 4.0,
    camera_dist_factor: float = 2.0,
    min_points: int         = 5,
    draw_bbox: bool         = True,
    return_depth: bool      = False,
    target_z: Optional[float] = None,
) -> list:
    """Render 4 oblique views (east/north/west/south) over the XY world bbox.

    Reuses the same extrinsic/intrinsic recipe as the offline precompute
    script so verify_3d shows the same framing the offline cache would.

    Returns a list of PIL.Image in DIRECTIONS order, or ``[]`` on failure
    (missing PLY, too few points in region, headless without xvfb, etc.).
    """
    from .render import _load_ply_cached

    try:
        pts, colors = _load_ply_cached(ply_path)
    except Exception as e:
        logger.warning("PLY load failed for %s: %s", ply_path, e)
        return []

    x_min, y_min, x_max, y_max = [float(v) for v in world_bbox]
    # Clip points to the requested XY region, plus a little padding so the
    # context crop doesn't look starved.
    r = max(x_max - x_min, y_max - y_min) / 2.0
    pad = r * 0.3
    mask = (
        (pts[:, 0] >= x_min - pad) & (pts[:, 0] <= x_max + pad) &
        (pts[:, 1] >= y_min - pad) & (pts[:, 1] <= y_max + pad)
    )
    pts_r = pts[mask]
    if len(pts_r) < min_points:
        return []

    # target_z defaults to the centre of the region's own z-range; the
    # builder passes a scene-wide ``target_z`` so the cameras it logs via
    # ``compute_region_cameras`` match the cameras actually used here.
    if target_z is None:
        z_mid = float((pts_r[:, 2].min() + pts_r[:, 2].max()) / 2.0)
    else:
        z_mid = float(target_z)
    # Actual AABB from the points inside the bbox (matches precompute).
    inst_mn = pts_r.min(axis=0)
    inst_mx = pts_r.max(axis=0)
    corners = _aabb_corners(inst_mn, inst_mx)
    target  = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, z_mid],
                       dtype=np.float64)

    box_size = float(np.clip(max(x_max - x_min, y_max - y_min) * scale,
                              min_size, max_size))

    if not _ensure_renderer_thread(size=output_size, point_size=point_size):
        return []

    key = scene_key or ply_path
    done = threading.Event()
    result: dict = {}
    _RENDERER_JOBS.put({
        "pts":                pts,
        "colors":             colors,
        "scene_key":          key,
        "corners":            corners,
        "target":             target,
        "box_size":           box_size,
        "elevation_deg":      elevation_deg,
        "camera_dist_factor": camera_dist_factor,
        "draw_bbox":          draw_bbox,
        "return_depth":       return_depth,
        "done":               done,
        "result":             result,
    })
    done.wait()
    if return_depth:
        return {"views": result.get("views", []),
                "depths": result.get("depths", [])}  # type: ignore[return-value]
    return result.get("views", [])
