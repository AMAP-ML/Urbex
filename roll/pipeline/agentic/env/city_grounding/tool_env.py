"""CityToolEnv — stateless-tool, answer-terminated city-grounding env.

Design rationale
================
Three invariants keep the env compatible with generic multi-turn RL drivers:

  1. tools are logically pure functions of their arguments
  2. inter-tool communication flows through the chat history (tool
     responses → next user turn), never through env-private state
  3. termination is decided by the driver, not by tool side effects
     (an assistant turn without a tool_call ⇒ TERMINATED)

Tool set (3 tools, all stateless from the model's point of view)
---------------------------------------------------------------
  • `get_landmark_xy({"names":[...]})`
        Pure lookup. Returns landmark world XY + projection onto a given
        view (optional `view_id`). No side effects on any view.

  • `zoom_in({"view_id": "...", "bbox_2d":[x0,y0,x1,y1]})`
        Crop `view_id`'s image by `bbox_2d` (in that view's coordinate
        frame). Creates a new view (auto-named "Z<n>") in the registry
        and returns its id, world bounds, and the cropped image. Capped
        at depth 2.

  • `render_oblique({"view_id":"...", "bbox_2d":[x0,y0,x1,y1]})`
        Render 4 oblique views (E/N/W/S) for inspection of a candidate
        bbox. Returns the 4 PILs. No commitment, no side effects.

Termination
-----------
The agent commits an answer by emitting an assistant turn that contains
NO tool_call and embeds a JSON answer:
    <answer>{"bbox_2d":[x0,y0,x1,y1], "view_id":"Z1"}</answer>
The driver/AgentLoop sees no tool_call → episode TERMINATED. Reward fn
extracts the `<answer>` payload, projects bbox_2d on `view_id` to world
coords, computes IoU against GT.

If the assistant emits neither a tool_call nor a parseable `<answer>`,
that's an empty commit → IoU=0.

View registry vs. hidden state
------------------------------
The env still holds a `view_registry: dict[view_id → view_data]`. This is
NOT inter-tool hidden state because:
  • every view_id is announced to the model in the tool response that
    creates it (zoom_in result text + image); the conversation history is
    the source of truth for "which view_ids exist"
  • the registry is content-addressable: tools accept `view_id` as input,
    they don't read an implicit "current view"
  • get_landmark_xy doesn't paint anchors onto any stored view (v1 did);
    if the model wants landmarks visualized, it can zoom_in around them
The registry is therefore an addressable cache, not hidden state.

Budget
------
Tracked but advisory: every tool response carries
"budget remaining: X/MAX". No automatic termination on exhaustion. The
driver's outer loop (max_assistant_turns) is the hard cap.

Public methods returns
----------------------
Each tool method returns (resp_text, images, info_dict) — uniform shape
the SFT driver / AgentLoop can map straight onto a ToolResponse-like
record. No (obs, reward, done, info) tuple, no internal `done` flag.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
from typing import Optional

import yaml
from PIL import Image

from .iou import iou_2d
from .reward import compute_render_reward
from .render import (
    compute_display_size,
    crop_bev_fit,
    load_bev,
    pixel_to_world,
    world_to_pixel,
)

logger = logging.getLogger(__name__)

LANDMARK_ROBUSTNESS_MODES = ("clean", "missing", "noisy", "wrong")


# ── Coord helpers (lifted unchanged from v1) ─────────────────────────────────


def _norm_to_world_bbox(bbox_norm, mapping, img_w, img_h, norm_max=1000.0):
    """Convert a normalized bbox in the source view's coord frame to a
    world-axis-aligned [wx0,wy0,wx1,wy1] in metres.

    `norm_max` is the convention by which the model expresses bbox
    coords:
      • 1000.0  — Qwen-VL family, InternVL, MiniCPM-V, Gemini (default)
      • 1.0     — DeepSeek-VL2 and 0-1 normalized models
      • img_w / img_h ('pixels') — LLaVA / GPT-4V / Claude (pixel coords);
        callers should pass `norm_max=img_w` for x and `norm_max=img_h`
        for y separately, or set norm_max=None and use `_valid_bbox`'s
        pixel branch. We keep the simpler scalar interface here and let
        callers pre-multiply if they need pixel coords.
    """
    x0, y0, x1, y1 = bbox_norm
    px0 = x0 / norm_max * img_w
    px1 = x1 / norm_max * img_w
    py0 = y0 / norm_max * img_h
    py1 = y1 / norm_max * img_h
    wx0, wy_top = pixel_to_world(mapping, px0, py0)
    wx1, wy_bot = pixel_to_world(mapping, px1, py1)
    return [min(wx0, wx1), min(wy_top, wy_bot),
            max(wx0, wx1), max(wy_top, wy_bot)]


def _world_to_norm_xy(mapping, wx, wy, img_w, img_h, norm_max=1000.0):
    """Project world (wx, wy) onto a view's normalized coord frame.
    Returns None if the point falls outside the view's pixel canvas.

    Outputs integers when `norm_max` is integer-like (1000, 999), or
    floats when `norm_max` is fractional (e.g. 1.0 for 0-1 coords).
    """
    px, py = world_to_pixel(mapping, wx, wy)
    if px < 0 or py < 0 or px > img_w or py > img_h:
        return None
    nx = px / img_w * norm_max
    ny = py / img_h * norm_max
    if isinstance(norm_max, int) or norm_max >= 100.0:
        return [round(nx), round(ny)]
    return [round(nx, 4), round(ny, 4)]


def _bbox_iou_norm(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    aw = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    bw = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = aw + bw - inter
    return inter / union if union > 1e-6 else 0.0


# ── The env ──────────────────────────────────────────────────────────────────

class CityToolEnv:
    """Stateless-tool city grounding controller.

    Holds a per-episode view_registry but never reads "current view".
    All tool methods take a `view_id` argument explicitly.
    """

    MAX_TOOL_CALLS  = 7
    MAX_ZOOM_DEPTH  = 2
    ZOOM_DEDUP_IOU  = 0.8
    MIN_ZOOM_SIDE_M = 10.0
    MAX_ZOOM_SIDE_M = 1200.0
    # Default bbox-coord convention: Qwen-VL family / InternVL / Gemini /
    # MiniCPM-V all output bbox in [0, 1000] integers. Override at
    # construction time for other model families:
    #   • DeepSeek-VL2, 0-1 normalized            → bbox_norm_max=1.0
    #   • LLaVA / GPT-4V / Claude (pixel coords)  → bbox_norm_max=None
    #     ("None" means "raw pixel coords on the SOURCE view's image,
    #      no normalization"; _norm_to_world_bbox callers must then
    #      pass `norm_max=img_w` for x and we use the max-side value
    #      separately — see tool_zoom_in for the runtime branch).
    BBOX_NORM_MAX = 1000.0

    def __init__(
        self,
        display_max_side: int   = 768,
        display_patch: int      = 32,
        display_max_pixels: int = 589_824,      # 768 × 768
        multiview_cache_dir: Optional[str] = None,
        max_tool_calls: Optional[int] = None,
        bbox_norm_max: Optional[float] = None,
        landmark_mode: str = "clean",
        landmark_noise_std_m: float = 10.0,
        landmark_wrong_min_distance_m: float = 50.0,
        landmark_seed: int = 42,
    ):
        if landmark_mode not in LANDMARK_ROBUSTNESS_MODES:
            raise ValueError(
                f"unsupported landmark_mode={landmark_mode!r}; "
                f"expected one of {LANDMARK_ROBUSTNESS_MODES}"
            )
        self.display_max_side    = display_max_side
        self.display_patch       = display_patch
        self.display_max_pixels  = display_max_pixels
        self.multiview_cache_dir = multiview_cache_dir
        self.landmark_mode = landmark_mode
        self.landmark_noise_std_m = float(landmark_noise_std_m)
        self.landmark_wrong_min_distance_m = float(landmark_wrong_min_distance_m)
        self.landmark_seed = int(landmark_seed)
        if max_tool_calls is not None:
            self.MAX_TOOL_CALLS = max_tool_calls
        if bbox_norm_max is not None:
            self.BBOX_NORM_MAX = float(bbox_norm_max)
        self._reset_state()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        self.sample: dict = {}
        self._bev_src_img = None
        self._bev_src_mapping: dict = {}
        self.view_registry: dict[str, dict] = {}
        self.tool_calls_used: int = 0
        self._zoom_counter: int = 0
        self._last_zoom_dist: float | None = None

    def reset(self, sample: dict) -> dict:
        """Initialize episode. Returns the global view dict for the driver
        to render the opening user-turn image."""
        self._reset_state()
        self.sample = sample

        bev_img, bev_mapping = load_bev(sample["bev_png"])
        assert bev_img is not None
        bev_w, bev_h = bev_img.size
        dw, dh = compute_display_size(
            bev_w, bev_h,
            max_side=self.display_max_side,
            patch=self.display_patch,
            max_pixels=self.display_max_pixels,
        )
        global_img = bev_img.resize((dw, dh), Image.Resampling.LANCZOS)
        global_mapping = {
            "x_min": bev_mapping["x_min"],
            "x_max": bev_mapping["x_max"],
            "y_min": bev_mapping["y_min"],
            "y_max": bev_mapping["y_max"],
            "res_x": (bev_mapping["x_max"] - bev_mapping["x_min"]) / dw,
            "res_y": (bev_mapping["y_max"] - bev_mapping["y_min"]) / dh,
        }
        self._bev_src_img = bev_img
        self._bev_src_mapping = bev_mapping
        self.view_registry["global"] = {
            "view_id":     "global",
            "kind":        "global",
            "image":       global_img,
            "mapping":     global_mapping,
            "img_w":       dw,
            "img_h":       dh,
            "world_bounds": [bev_mapping["x_min"], bev_mapping["y_min"],
                             bev_mapping["x_max"], bev_mapping["y_max"]],
            "depth":       0,
            "parent_id":   None,
        }
        return self.view_registry["global"]

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _budget_str(self) -> str:
        # Reserve the final assistant turn for an answer. One remaining
        # tool call is still available before that final answer turn.
        remaining = self.MAX_TOOL_CALLS - self.tool_calls_used
        base = f"budget {self.tool_calls_used}/{self.MAX_TOOL_CALLS}"
        if remaining <= 0:
            return (
                f"{base} — BUDGET EXHAUSTED. Your next assistant turn "
                f"MUST emit "
                f"<answer>{{\"bbox_2d\":[x0,y0,x1,y1],\"view_id\":\"...\"}}</answer> "
                f"with NO tool_call. Pick the best candidate from the "
                f"evidence gathered so far; an empty or missing answer "
                f"ends the episode without a predicted box."
            )
        if remaining == 1:
            return (
                f"{base} — ONE TOOL CALL LEFT. You may inspect once more, "
                f"then use the following assistant turn to emit "
                f"<answer>{{\"bbox_2d\":[x0,y0,x1,y1],\"view_id\":\"...\"}}</answer> "
                f"with NO tool_call. You may also submit the answer now."
            )
        return base

    def _coord_frame_str(self) -> str:
        """Short human-readable description of the bbox-coord convention,
        embedded in tool responses so the model can self-adapt across
        teachers without changing the system prompt."""
        m = self.BBOX_NORM_MAX
        if not m or m <= 0:
            return "coord frame: pixel coords (use the view's reported width/height)"
        if m == 1.0:
            return "coord frame: [0,1] floats"
        return f"coord frame: [0,{int(m)}] integers"

    def _valid_bbox(self, bbox) -> bool:
        """Validate a bbox in the configured normalization convention.

        Acceptable input ranges:
          • [0, BBOX_NORM_MAX] — the normalized convention
            (1000 for Qwen-VL, 1.0 for 0-1 models)
          • exception: BBOX_NORM_MAX is None / 0 → pixel coords;
            we accept any [0, max(img_w, img_h)] but we don't have the
            view here, so we just clamp the range loosely. Callers that
            care should set BBOX_NORM_MAX explicitly.
        """
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False
        try:
            x0, y0, x1, y1 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return False
        if x0 >= x1 or y0 >= y1:
            return False
        if self.BBOX_NORM_MAX and self.BBOX_NORM_MAX > 0:
            m = self.BBOX_NORM_MAX
            return 0 <= x0 and x1 <= m and 0 <= y0 and y1 <= m
        # pixel-coord branch (BBOX_NORM_MAX = 0 / None): permissive
        return 0 <= x0 and 0 <= y0

    def _get_view(self, view_id) -> Optional[dict]:
        if not isinstance(view_id, str):
            return None
        return self.view_registry.get(view_id)

    def _bump_budget(self) -> None:
        self.tool_calls_used += 1

    def list_view_ids(self) -> list[str]:
        return list(self.view_registry.keys())

    def view_image(self, view_id: str) -> Optional[Image.Image]:
        v = self._get_view(view_id)
        return v["image"] if v else None

    def project_bbox_to_world(self, view_id: str, bbox_norm: list) -> Optional[list]:
        """Used by reward fn to convert <answer> bbox_2d → world coords."""
        v = self._get_view(view_id)
        if v is None or not self._valid_bbox(bbox_norm):
            return None
        return _norm_to_world_bbox(
            [float(x) for x in bbox_norm], v["mapping"], v["img_w"], v["img_h"],
            norm_max=self.BBOX_NORM_MAX,
        )

    def score_answer(self, bbox_norm: list, view_id: str) -> dict:
        """Compute IoU + center-distance score for a final <answer>.
        Returns {"world_bbox", "iou_2d", "center_dist", "success"}."""
        world_bbox = self.project_bbox_to_world(view_id, bbox_norm)
        if world_bbox is None:
            return {"world_bbox": None, "iou_2d": 0.0, "center_dist": float("inf"), "success": False}
        gt = self.sample.get("gt_bbox")
        if not gt:
            return {"world_bbox": world_bbox, "iou_2d": 0.0, "center_dist": float("inf"), "success": False}
        cx = 0.5 * (world_bbox[0] + world_bbox[2])
        cy = 0.5 * (world_bbox[1] + world_bbox[3])
        dx = max(0.0, world_bbox[2] - world_bbox[0])
        dy = max(0.0, world_bbox[3] - world_bbox[1])
        pred = [cx, cy, 0.0, dx, dy, 1.0]
        iou = float(iou_2d(pred, gt))
        from .iou import center_dist
        dist = float(center_dist(pred, gt))
        return {"world_bbox": world_bbox, "iou_2d": iou, "center_dist": dist,
                "success": iou >= 0.25}

    # ── Tools ────────────────────────────────────────────────────────────────

    def tool_get_landmark_xy(self, args: dict) -> tuple[str, list, dict]:
        """Resolve proper-noun landmarks. Optional `view_id` projects the
        results into that view's coordinate frame."""
        self._bump_budget()
        names_arg = args.get("names") if "names" in args else args.get("name")
        if isinstance(names_arg, str):
            names = [names_arg.strip()] if names_arg.strip() else []
        elif isinstance(names_arg, list):
            names = [str(n).strip() for n in names_arg if str(n).strip()]
        else:
            names = []
        if not names:
            return (f"error: get_landmark_xy needs non-empty 'names'. "
                    f"{self._budget_str()}", [], {"error": "empty_names"})

        proj_view_id = args.get("view_id", "global")
        proj_view = self._get_view(proj_view_id) or self.view_registry["global"]

        entries: list[dict] = []
        lines: list[str] = []
        for name in names:
            hit = self._lookup_landmark(name)
            if hit is None:
                lines.append(f"  - '{name}': no match")
                continue
            wx, wy = float(hit["bbox"][0]), float(hit["bbox"][1])
            nxy = _world_to_norm_xy(
                proj_view["mapping"], wx, wy,
                proj_view["img_w"], proj_view["img_h"],
                norm_max=self.BBOX_NORM_MAX,
            )
            entry = {
                "name":        name,
                "world_xy":    [wx, wy],
                "in_view":     proj_view["view_id"] if nxy else None,
                "norm_xy":     nxy,
                "match_name":  hit.get("object_name", ""),
                "score":       float(hit.get("score", 0.0)),
                "landmark_mode": self.landmark_mode,
            }
            entries.append(entry)
            # Reverted: stripping the world-meter coords regressed
            # s@0.25 by 8pp on val N=50. Keep both world + normalized.
            if nxy is not None:
                lines.append(
                    f"  - '{name}' → world ({wx:.1f},{wy:.1f}); "
                    f"on view {proj_view['view_id']} at ({nxy[0]},{nxy[1]})"
                )
            else:
                lines.append(
                    f"  - '{name}' → world ({wx:.1f},{wy:.1f}); "
                    f"OFF view {proj_view['view_id']}"
                )
        text = ("get_landmark_xy results:\n" + "\n".join(lines)
                + f"\n{self._budget_str()}")
        # Landmark lookup is neutral in the paper's tool-shaping objective.
        landmark_resolve_reward = 0.0
        return text, [], {"landmarks": entries,
                          "n_resolved": len(entries),
                          "landmark_mode": self.landmark_mode,
                          "landmark_resolve_reward": landmark_resolve_reward}

    def tool_zoom_in(self, args: dict) -> tuple[str, list, dict]:
        view_id = args.get("view_id", "global")
        bbox = args.get("bbox_2d", args.get("bbox"))
        self._bump_budget()
        view = self._get_view(view_id)
        if view is None:
            return (f"error: zoom_in unknown view_id={view_id!r}. "
                    f"Known: {self.list_view_ids()}. {self._budget_str()}",
                    [], {"error": "unknown_view"})
        if not self._valid_bbox(bbox):
            return (f"error: zoom_in needs bbox_2d=[x0,y0,x1,y1] with "
                    f"x0<x1 and y0<y1, expressed in view_id's coordinate "
                    f"frame. {self._budget_str()}",
                    [], {"error": "invalid_bbox"})
        bbox_norm = [float(v) for v in bbox]

        parent_depth = view["depth"]
        if parent_depth >= self.MAX_ZOOM_DEPTH:
            return (f"error: zoom_in cannot nest deeper than depth "
                    f"{self.MAX_ZOOM_DEPTH} (view {view_id} is at depth "
                    f"{parent_depth}). Pick a different parent view_id "
                    f"(e.g. 'global') and try again. {self._budget_str()}",
                    [], {"error": "max_depth"})

        # Dedup against zooms taken from the same parent
        for vid, v in self.view_registry.items():
            if v["parent_id"] == view_id:
                iou = _bbox_iou_norm(bbox_norm, v["bbox_norm_in_parent"])
                if iou >= self.ZOOM_DEDUP_IOU:
                    return (f"error: zoom region too similar to existing "
                            f"view {vid} (IoU={iou:.2f}); pick a different "
                            f"area or reuse {vid} via view_id arg. "
                            f"{self._budget_str()}",
                            [], {"error": "duplicate_zoom",
                                 "duplicate_of": vid})

        world_bbox = _norm_to_world_bbox(
            bbox_norm, view["mapping"], view["img_w"], view["img_h"],
            norm_max=self.BBOX_NORM_MAX,
        )
        side = max(world_bbox[2] - world_bbox[0], world_bbox[3] - world_bbox[1])
        if side < self.MIN_ZOOM_SIDE_M:
            return (f"error: zoom region too small ({side:.1f}m < "
                    f"{self.MIN_ZOOM_SIDE_M:.0f}m). {self._budget_str()}",
                    [], {"error": "too_small"})
        if side > self.MAX_ZOOM_SIDE_M:
            return (f"error: zoom region too large ({side:.1f}m > "
                    f"{self.MAX_ZOOM_SIDE_M:.0f}m). {self._budget_str()}",
                    [], {"error": "too_large"})

        assert self._bev_src_img is not None, "reset() not called"
        crop_img, crop_mapping = crop_bev_fit(
            self._bev_src_img, self._bev_src_mapping,
            world_bbox[0], world_bbox[1], world_bbox[2], world_bbox[3],
            max_side=self.display_max_side,
            patch=self.display_patch,
            max_pixels=self.display_max_pixels,
            padding_frac=0.0,
        )
        cw, ch = crop_img.size
        self._zoom_counter += 1
        new_id = f"Z{self._zoom_counter}"
        self.view_registry[new_id] = {
            "view_id":     new_id,
            "kind":        "zoom",
            "image":       crop_img,
            "mapping":     crop_mapping,
            "img_w":       cw,
            "img_h":       ch,
            "world_bounds": [crop_mapping["x_min"], crop_mapping["y_min"],
                             crop_mapping["x_max"], crop_mapping["y_max"]],
            "depth":       parent_depth + 1,
            "parent_id":   view_id,
            "bbox_norm_in_parent": [round(v) for v in bbox_norm],
        }
        # Reverted to original full anchor (world meters + pixel dims):
        # eval N=50 showed both the strip variant (14%) and the
        # global-anchor [0,1000] back-projection variant (8.2%) regress
        # vs the original (22%). Within ±10pp N=50 noise, but lower mean
        # signals the metadata genuinely helps the model anchor the
        # cropped region's scale.
        wb = self.view_registry[new_id]["world_bounds"]
        text = (
            f"zoom_in ok → new view '{new_id}' (parent={view_id}, "
            f"depth={parent_depth + 1})\n"
            f"  world_bounds: x∈[{wb[0]:.1f},{wb[2]:.1f}]m  "
            f"y∈[{wb[1]:.1f},{wb[3]:.1f}]m  "
            f"(side ≈ {side:.1f}m)\n"
            f"  image frame: {cw}×{ch} px; {self._coord_frame_str()}\n"
            f"{self._budget_str()}"
        )

        # Directional step reward: reward zooming *closer* to the GT.
        zoom_cx = 0.5 * (world_bbox[0] + world_bbox[2])
        zoom_cy = 0.5 * (world_bbox[1] + world_bbox[3])
        gt = self.sample.get("gt_bbox")
        zoom_dist_reward = 0.0
        if gt:
            gt_cx, gt_cy = float(gt[0]), float(gt[1])
            curr_dist = ((zoom_cx - gt_cx) ** 2 + (zoom_cy - gt_cy) ** 2) ** 0.5
            if self._last_zoom_dist is not None:
                delta = (self._last_zoom_dist - curr_dist) / 50.0  # +1 per 50m closer
                zoom_dist_reward = max(-0.05, min(0.05, 0.05 * delta))
            self._last_zoom_dist = curr_dist

        return text, [crop_img], {
            "view_id": new_id,
            "world_bbox": world_bbox,
            "parent_id": view_id,
            "zoom_dist_reward": zoom_dist_reward,
        }

    # Cache returns 4 images in fixed order [east, north, west, south];
    # render_oblique exposes the choice as a required `direction` arg so
    # the model picks ONE per call (1 image, 256 vision tokens vs 4×256
    # for the old all-four behaviour). Multiple directions = multiple
    # render_oblique calls — counts against tool budget so the model
    # is forced to be selective.
    DIRECTION_INDEX = {"east": 0, "north": 1, "west": 2, "south": 3}

    def tool_render_oblique(self, args: dict) -> tuple[str, list, dict]:
        """Render ONE oblique inspection view (chosen cardinal direction)
        for a candidate bbox on `view_id`. No commitment — purely visual
        evidence. Call again with a different `direction` to see another
        angle; each call costs one budget unit."""
        view_id = args.get("view_id", "global")
        bbox = args.get("bbox_2d", args.get("bbox"))
        direction = str(args.get("direction") or "").lower()
        self._bump_budget()
        view = self._get_view(view_id)
        if view is None:
            return (f"error: render_oblique unknown view_id={view_id!r}. "
                    f"{self._budget_str()}",
                    [], {"error": "unknown_view"})
        if not self._valid_bbox(bbox):
            return (f"error: render_oblique needs bbox_2d=[x0,y0,x1,y1] "
                    f"with x0<x1 and y0<y1, in view_id's coordinate frame. "
                    f"{self._budget_str()}",
                    [], {"error": "invalid_bbox"})
        if direction not in self.DIRECTION_INDEX:
            return (f"error: render_oblique needs `direction` to be one of "
                    f"{sorted(self.DIRECTION_INDEX)}, got {direction!r}. "
                    f"{self._budget_str()}",
                    [], {"error": "invalid_direction"})
        bbox_norm = [float(v) for v in bbox]
        world_bbox = _norm_to_world_bbox(
            bbox_norm, view["mapping"], view["img_w"], view["img_h"],
            norm_max=self.BBOX_NORM_MAX,
        )
        # Pass direction so BboxCache only loads + annotates the requested
        # view (4× cheaper on the cache hot path). Falls back to all-4 if
        # cache miss → Open3D render still produces 4 views (rare path).
        oblique = self._render_oblique(world_bbox, only_direction=direction)
        if not oblique:
            return (
                f"error: render_oblique: no oblique views available for region "
                f"[{world_bbox[0]:.1f},{world_bbox[1]:.1f},"
                f"{world_bbox[2]:.1f},{world_bbox[3]:.1f}] (cache miss / "
                f"no scene PLY). Decide from BEV alone. {self._budget_str()}",
                [], {"error": "no_oblique_views", "world_bbox": world_bbox, "n_views": 0},
            )
        idx = self.DIRECTION_INDEX[direction]
        if idx >= len(oblique) or oblique[idx] is None:
            return (
                f"error: render_oblique: {direction} view missing for region "
                f"[{world_bbox[0]:.1f},{world_bbox[1]:.1f},"
                f"{world_bbox[2]:.1f},{world_bbox[3]:.1f}]. "
                f"{self._budget_str()}",
                [], {"error": "missing_oblique_direction", "world_bbox": world_bbox, "n_views": 0,
                     "missing_direction": direction},
            )
        img = oblique[idx]
        # Reverted to original world-meter region (option B's
        # [0,1000] global-anchor variant scored worst on val N=50).
        text = (
            f"render_oblique ok → 1 oblique view from the {direction.upper()} "
            f"over world [{world_bbox[0]:.1f},{world_bbox[1]:.1f},"
            f"{world_bbox[2]:.1f},{world_bbox[3]:.1f}] (view_id={view_id}). "
            f"The image shows the proposed bbox as a RED 3D wireframe. If "
            f"it lands on the queried object, commit by emitting an "
            f"assistant turn with NO tool_call and an "
            f"<answer>{{\"bbox_2d\":[...], \"view_id\":\"{view_id}\"}}</answer> "
            f"tag. To inspect another angle, call render_oblique again "
            f"with a different `direction` (each call costs 1 tool budget). "
            f"Otherwise revise via zoom_in / new bbox.\n{self._budget_str()}"
        )
        # Privileged tool shaping is returned in info, never in the model's
        # text/image observation. Each successful render is graded by its
        # candidate-region overlap, independent of the next action.
        gt = self.sample.get("gt_bbox")
        iou = 0.0
        if gt and len(gt) >= 5:
            # iou_2d expects center + full dimensions, not XYXY corners.
            candidate = [
                0.5 * (world_bbox[0] + world_bbox[2]),
                0.5 * (world_bbox[1] + world_bbox[3]),
                0.0,
                world_bbox[2] - world_bbox[0],
                world_bbox[3] - world_bbox[1],
                1.0,
            ]
            try:
                iou = float(iou_2d(candidate, gt))
            except Exception:
                iou = 0.0
        return text, [img], {"world_bbox": world_bbox, "n_views": 1,
                             "view_id": view_id, "direction": direction,
                             "render_iou_reward": compute_render_reward(iou),
                             "render_iou": iou}

    # ── Internals (oblique render + landmark lookup) ─────────────────────────

    def _landmark_rng(self, query: str) -> random.Random:
        """Return a deterministic per-sample RNG for landmark corruption."""
        identity = "\x1f".join([
            str(self.landmark_seed),
            str(self.sample.get("dataset", "cityrefer")),
            str(self.sample.get("scene_id", "")),
            str(self.sample.get("query", "")),
            query,
        ])
        digest = hashlib.sha256(identity.encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def _perturb_landmark_hit(
        self,
        query: str,
        hit: Optional[dict],
    ) -> Optional[dict]:
        """Apply the configured evaluation-only landmark perturbation."""
        if self.landmark_mode == "missing":
            return None
        if hit is None or self.landmark_mode == "clean":
            return hit

        perturbed = dict(hit)
        perturbed["bbox"] = list(hit.get("bbox") or [])
        if len(perturbed["bbox"]) < 2:
            return None

        if self.landmark_mode == "noisy":
            rng = self._landmark_rng(query)
            perturbed["bbox"][0] += rng.gauss(0.0, self.landmark_noise_std_m)
            perturbed["bbox"][1] += rng.gauss(0.0, self.landmark_noise_std_m)
            perturbed["perturbation"] = {
                "mode": "noisy",
                "std_m": self.landmark_noise_std_m,
            }
            return perturbed

        from .tools.retrieve import list_landmarks_for_scene

        scene_id = str(self.sample.get("scene_id", ""))
        dataset = str(self.sample.get("dataset", "cityrefer"))
        correct_name = str(hit.get("object_name") or hit.get("landmark") or "").casefold()
        correct_x, correct_y = float(perturbed["bbox"][0]), float(perturbed["bbox"][1])
        alternatives: list[tuple[float, dict]] = []
        fallback: list[tuple[float, dict]] = []
        for candidate in list_landmarks_for_scene(scene_id, dataset):
            candidate_name = str(
                candidate.get("object_name") or candidate.get("landmark") or ""
            ).casefold()
            if candidate_name == correct_name:
                continue
            candidate_bbox = candidate.get("bbox") or []
            if len(candidate_bbox) < 2:
                continue
            distance = math.hypot(
                float(candidate_bbox[0]) - correct_x,
                float(candidate_bbox[1]) - correct_y,
            )
            fallback.append((distance, candidate))
            if distance >= self.landmark_wrong_min_distance_m:
                alternatives.append((distance, candidate))

        pool = alternatives or fallback
        if not pool:
            return None
        pool.sort(key=lambda item: item[0])
        shortlist = pool[:min(5, len(pool))]
        _, wrong_hit = self._landmark_rng(query).choice(shortlist)
        perturbed = dict(wrong_hit)
        perturbed["bbox"] = list(wrong_hit["bbox"])
        perturbed["perturbation"] = {
            "mode": "wrong",
            "requested_name": query,
            "correct_match_name": hit.get("object_name", ""),
        }
        return perturbed

    def _lookup_landmark(self, query: str) -> Optional[dict]:
        try:
            from .tools.retrieve import find_landmark_by_name
            if self.landmark_mode == "missing":
                return None
            matches = find_landmark_by_name(
                self.sample.get("scene_id", ""), query, top_k=1,
                dataset=self.sample.get("dataset", "cityrefer"),
            )
            hit = matches[0] if matches else None
            return self._perturb_landmark_hit(query, hit)
        except Exception as e:
            logger.warning("landmark lookup failed for %r: %s", query, e)
            return None

    def _render_oblique(self, world_bbox: list,
                        only_direction: Optional[str] = None) -> list:
        """Returns a length-4 list (e/n/w/s order) with PIL or None.
        With `only_direction` set, the BboxCache fast-path skips load +
        annotation for the 3 unwanted slots — same image quality at the
        chosen slot but ~4× cheaper. Open3D fallback still renders all 4
        because it's only used when the cache misses (rare on training
        hosts; CITY_TOOL_REQUIRE_CACHE=1 disables it entirely)."""
        scene_id = self.sample.get("scene_id", "")
        dataset  = self.sample.get("dataset", "cityrefer")
        if self.multiview_cache_dir:
            from .verify_cache import BboxCache
            cache = BboxCache.get(self.multiview_cache_dir, scene_id)
            imgs, _ = cache.lookup(world_bbox, only_direction=only_direction)
            if imgs is not None:
                return imgs
            if os.environ.get("CITY_TOOL_REQUIRE_CACHE", "") == "1":
                return []
        from .render_open3d import render_region
        from .tools.retrieve import ply_path_for_scene
        ply_path = ply_path_for_scene(scene_id, dataset)
        if not ply_path or not os.path.exists(ply_path):
            return []
        try:
            return render_region(ply_path, world_bbox,
                                  scene_key=scene_id, output_size=384)
        except Exception as e:
            logger.warning("open3d render_region failed: %s", e)
            return []


# ── Single source of truth: system prompt ────────────────────────────────────
# Used by BOTH the SFT data collector (sft_collect/collect.py) and the GRPO
# parquet builder (dataset/build_parquet.py) — keeping them imported from
# one place guarantees train/inference distribution match. Editing the
# tool protocol? Edit this string only.

# Stage-3 minimalist system prompt (Think-with-Map style). All
# protocol details (view_id naming, [0,1000] frame, depth cap, scale
# hints, the <answer> commit format) have been pushed into:
#   • the per-tool schema descriptions in config/tool_config.yaml
#     (which the chat template auto-injects under "# Tools"), and
#   • the per-sample user prompt's "commit protocol" line
#     (USER_PROMPT_TEMPLATE in dataset/build_parquet.py and
#     _build_initial_user_content in sft_collect/collect.py).
# Editing protocol = edit those two sources, NOT this string.
SYSTEM_PROMPT = "You are a helpful visual grounding assistant for 3D city scenes."


# ── Tool dispatch + answer parsing ───────────────────────────────────────────

TOOL_METHODS = {
    "get_landmark_xy": "tool_get_landmark_xy",
    "zoom_in":         "tool_zoom_in",
    "render_oblique":  "tool_render_oblique",
}

# ── Tool schemas ─────────────────────────────────────────────────────────────
# tool_config.yaml is the single source of truth: the same list is handed to the
# chat template as `tools=[...]` during training, rollout and API baselines.

_DEFAULT_TOOL_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "tool_config.yaml",
)


def load_openai_tool_schemas(path: Optional[str] = None) -> list[dict]:
    """Return the tool schemas in the shape OpenAI / hermes chat templates
    expect for their ``tools=[...]`` argument."""
    with open(path or _DEFAULT_TOOL_CONFIG_PATH) as f:
        return yaml.safe_load(f).get("tools", [])


OPENAI_TOOL_SCHEMAS = load_openai_tool_schemas()

_ANSWER_RE = re.compile(r"<answer>\s*(\{.*?\})\s*</answer>", re.DOTALL)


def parse_answer(text: str) -> Optional[dict]:
    """Extract `<answer>{"bbox_2d":[...],"view_id":"..."}</answer>` payload.

    Returns the parsed dict or None if missing / malformed.
    """
    if not text:
        return None
    m = _ANSWER_RE.search(text)
    if not m:
        return None
    try:
        payload = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if "bbox_2d" not in payload or "view_id" not in payload:
        return None
    bbox = payload.get("bbox_2d")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    return payload
