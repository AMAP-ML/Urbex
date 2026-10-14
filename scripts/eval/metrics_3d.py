#!/usr/bin/env python3
"""3D evaluation metrics for local rollout predictions.

The model emits 2D world bboxes (`world_bbox = [x_min, y_min, x_max, y_max]`
in world metres). We "lift" this 2D prediction to a 3D box by:
  1. Loading the scene point cloud (.pth from cityrefer/sensaturban).
  2. Cropping points whose XY falls inside the predicted 2D bbox.
  3. Taking the axis-aligned bounding box of the cropped points → pred 3D
     bbox (predicted Z extent comes from observed point Z, not assumed).
Then we compare against the GT 3D bbox from the dataset JSONL
(``gt_bbox = [cx, cy, cz, dx, dy, dz, class_id, object_id]``).

Reported metrics
----------------
Per-sample:
  - iou_2d_pred:   the IoU the env reported (kept for sanity)
  - iou_3d:        axis-aligned 3D IoU between pred 3D bbox and GT
  - center_dist_3d: Euclidean distance between centers (metres)
  - point_recall:  fraction of GT-bbox points that fall inside the pred 3D bbox
                   (0 → completely missed, 1 → fully covers GT)
  - point_precision: fraction of pred-cropped points that fall inside GT bbox
  - n_points_cropped: how many points the prediction actually contains

Aggregate:
  - mean / median for each metric
  - Acc@IoU={0.25, 0.5} (success rate at standard 3D thresholds)
  - n_zero_iou (predictions that completely miss GT)

Multi-GT per (scene_id, query): the dataset can have several GT rows
sharing the same query. The script computes metrics against EACH
candidate GT and reports the BEST match (max iou_3d) — standard
Hit@1 evaluation.

Usage
-----
    python scripts/eval/metrics_3d.py \\
        --predictions /path/to/urbex/output/rollout/records.dedup.jsonl \\
        --gt          /path/to/agent_data/resplit/cityrefer_test_NO.jsonl \\
        --scene_dir   /path/to/cityanchor/data_cityrefer/sensaturban/pointgroup_data/balance_split/random-50_crop-250 \\
        --output      ./output/rollout/eval_3d.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Geometry helpers ───────────────────────────────────────────────────────


def gt_bbox_to_3d_minmax(gt_bbox: list) -> np.ndarray:
    """Convert dataset gt_bbox=[cx, cy, cz, dx, dy, dz, ...] to
    [x_min, y_min, z_min, x_max, y_max, z_max]."""
    cx, cy, cz = float(gt_bbox[0]), float(gt_bbox[1]), float(gt_bbox[2])
    dx, dy, dz = float(gt_bbox[3]), float(gt_bbox[4]), float(gt_bbox[5])
    return np.array([
        cx - dx / 2, cy - dy / 2, cz - dz / 2,
        cx + dx / 2, cy + dy / 2, cz + dz / 2,
    ], dtype=np.float64)


def iou_2d(a: np.ndarray, b: np.ndarray) -> float:
    """Axis-aligned 2D IoU. a, b are [x_min, y_min, x_max, y_max]."""
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return float(inter / union) if union > 1e-9 else 0.0


def giou_3d(a: np.ndarray, b: np.ndarray) -> float:
    """Generalized 3D IoU — distance-aware; range [-1, 1]; equals IoU when
    boxes overlap, drops below 0 as boxes separate."""
    inter_min = np.maximum(a[:3], b[:3])
    inter_max = np.minimum(a[3:], b[3:])
    inter = np.maximum(0.0, inter_max - inter_min).prod()
    vol_a = max(0.0, a[3] - a[0]) * max(0.0, a[4] - a[1]) * max(0.0, a[5] - a[2])
    vol_b = max(0.0, b[3] - b[0]) * max(0.0, b[4] - b[1]) * max(0.0, b[5] - b[2])
    union = vol_a + vol_b - inter
    if union < 1e-9:
        return 0.0
    iou = inter / union
    enc_min = np.minimum(a[:3], b[:3])
    enc_max = np.maximum(a[3:], b[3:])
    enc = max(1e-9, np.maximum(0.0, enc_max - enc_min).prod())
    return float(iou - (enc - union) / enc)


def points_to_3d_minmax(points: np.ndarray) -> Optional[np.ndarray]:
    """Axis-aligned bbox of an (N, 3) point set. Returns None if empty."""
    if points.shape[0] == 0:
        return None
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    return np.concatenate([mn, mx]).astype(np.float64)


def crop_points_xy(points: np.ndarray, bbox_2d_minmax: np.ndarray) -> np.ndarray:
    """Keep points whose XY lies inside the 2D bbox [x_min, y_min, x_max, y_max]."""
    x_min, y_min, x_max, y_max = bbox_2d_minmax
    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
    )
    return points[mask]


def iou_3d(a: np.ndarray, b: np.ndarray) -> float:
    """Axis-aligned 3D IoU. a, b are [x_min, y_min, z_min, x_max, y_max, z_max]."""
    inter_min = np.maximum(a[:3], b[:3])
    inter_max = np.minimum(a[3:], b[3:])
    inter_dims = np.maximum(0.0, inter_max - inter_min)
    inter = inter_dims.prod()
    vol_a = max(0.0, a[3] - a[0]) * max(0.0, a[4] - a[1]) * max(0.0, a[5] - a[2])
    vol_b = max(0.0, b[3] - b[0]) * max(0.0, b[4] - b[1]) * max(0.0, b[5] - b[2])
    union = vol_a + vol_b - inter
    return float(inter / union) if union > 1e-9 else 0.0


def center_dist_3d(a: np.ndarray, b: np.ndarray) -> float:
    ca = (a[:3] + a[3:]) / 2
    cb = (b[:3] + b[3:]) / 2
    return float(np.linalg.norm(ca - cb))


def points_in_bbox(points: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Boolean mask of points inside the 3D bbox."""
    return (
        (points[:, 0] >= bbox[0]) & (points[:, 0] <= bbox[3]) &
        (points[:, 1] >= bbox[1]) & (points[:, 1] <= bbox[4]) &
        (points[:, 2] >= bbox[2]) & (points[:, 2] <= bbox[5])
    )


# ── Scene point-cloud loader ───────────────────────────────────────────────


_SCENE_CACHE: Dict[str, np.ndarray] = {}


def _resolve_scene_path(scene_dir: str, scene_id: str) -> Optional[str]:
    """Search scene_dir (and its train/ test/ subdirs) for {scene_id}.ply
    or {scene_id}.pth. Returns the first match, or None."""
    candidates = []
    for ext in (".ply", ".pth"):
        candidates.append(os.path.join(scene_dir, f"{scene_id}{ext}"))
        for sub in ("train", "test", "val"):
            candidates.append(os.path.join(scene_dir, sub, f"{scene_id}{ext}"))
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _load_ply_xyz(ply_path: str) -> Optional[np.ndarray]:
    """Load XYZ from a .ply file, trying plyfile then open3d."""
    # Try plyfile first (lightweight, handles ASCII + binary).
    try:
        from plyfile import PlyData
        ply = PlyData.read(ply_path)
        v = ply["vertex"].data
        return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    except ImportError:
        pass
    except Exception as e:
        print(f"[warn] plyfile failed on {ply_path}: {e}", file=sys.stderr)
    # Fallback: open3d.
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(ply_path)
        return np.asarray(pcd.points, dtype=np.float32)
    except ImportError:
        print(f"[err] neither `plyfile` nor `open3d` available; "
              f"`pip install plyfile` to load {ply_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[warn] open3d failed on {ply_path}: {e}", file=sys.stderr)
        return None


def _load_pth_xyz(pth_path: str) -> Optional[np.ndarray]:
    """Load XYZ from a .pth (torch.save'd) file."""
    try:
        import torch
        d = torch.load(pth_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[warn] failed to load {pth_path}: {e}", file=sys.stderr)
        return None

    if isinstance(d, dict):
        for key in ("coords", "xyz", "points", "coord"):
            if key in d:
                arr = np.asarray(d[key])
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    return arr[:, :3].astype(np.float32)
    elif isinstance(d, (tuple, list)) and len(d) > 0:
        first = np.asarray(d[0])
        if first.ndim == 2 and first.shape[1] >= 3:
            return first[:, :3].astype(np.float32)
    else:
        arr = np.asarray(d)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, :3].astype(np.float32)
    print(f"[warn] could not extract XYZ from {pth_path} "
          f"(type={type(d).__name__})", file=sys.stderr)
    return None


def load_scene_points(scene_dir: str, scene_id: str) -> Optional[np.ndarray]:
    """Load XYZ points for a scene from .ply or .pth. Cached per-process.
    Returns (N, 3) float32 or empty array if not found / load fails."""
    if scene_id in _SCENE_CACHE:
        return _SCENE_CACHE[scene_id]

    path = _resolve_scene_path(scene_dir, scene_id)
    if path is None:
        print(f"[warn] scene file missing: {scene_id} "
              f"(searched .ply/.pth under {scene_dir} and train/test/val/)",
              file=sys.stderr)
        _SCENE_CACHE[scene_id] = np.empty((0, 3), dtype=np.float32)
        return _SCENE_CACHE[scene_id]

    if path.endswith(".ply"):
        points = _load_ply_xyz(path)
    elif path.endswith(".pth"):
        points = _load_pth_xyz(path)
    else:
        points = None

    if points is None or points.shape[0] == 0:
        _SCENE_CACHE[scene_id] = np.empty((0, 3), dtype=np.float32)
    else:
        _SCENE_CACHE[scene_id] = np.ascontiguousarray(points[:, :3], dtype=np.float32)
        print(f"[load] {scene_id}: {_SCENE_CACHE[scene_id].shape[0]:,} points "
              f"from {os.path.basename(path)} "
              f"(x=[{_SCENE_CACHE[scene_id][:,0].min():.1f},"
              f"{_SCENE_CACHE[scene_id][:,0].max():.1f}] "
              f"y=[{_SCENE_CACHE[scene_id][:,1].min():.1f},"
              f"{_SCENE_CACHE[scene_id][:,1].max():.1f}] "
              f"z=[{_SCENE_CACHE[scene_id][:,2].min():.1f},"
              f"{_SCENE_CACHE[scene_id][:,2].max():.1f}])")
    return _SCENE_CACHE[scene_id]


# ── Per-sample evaluation ──────────────────────────────────────────────────


def evaluate_one(
    pred_world_bbox: list,
    gt_bbox: list,
    scene_points: np.ndarray,
) -> Dict[str, float]:
    """Compute 3D metrics for one (pred 2D bbox, GT 3D bbox) pair."""
    bbox_2d = np.array([float(v) for v in pred_world_bbox[:4]], dtype=np.float64)
    cropped = crop_points_xy(scene_points, bbox_2d)
    pred_3d = points_to_3d_minmax(cropped)
    gt_3d = gt_bbox_to_3d_minmax(gt_bbox)

    # GT XY (BEV) bbox extracted directly from gt_bbox (no z).
    gt_xy = np.array([gt_3d[0], gt_3d[1], gt_3d[3], gt_3d[4]], dtype=np.float64)
    bev_iou = iou_2d(bbox_2d, gt_xy)
    # 2D center distance — XY only, ignores Z mismatch noise.
    pred_cx_xy = 0.5 * (bbox_2d[0] + bbox_2d[2])
    pred_cy_xy = 0.5 * (bbox_2d[1] + bbox_2d[3])
    gt_cx, gt_cy, gt_cz = (gt_3d[0]+gt_3d[3])/2, (gt_3d[1]+gt_3d[4])/2, (gt_3d[2]+gt_3d[5])/2
    cd_2d = float(((pred_cx_xy - gt_cx) ** 2 + (pred_cy_xy - gt_cy) ** 2) ** 0.5)

    if pred_3d is None:
        # Pred 2D bbox doesn't enclose any scene points → empty pred volume.
        return {
            "iou_3d": 0.0,
            "iou_3d_zclip": 0.0,
            "iou_bev": bev_iou,
            "giou_3d": -1.0,
            "center_dist_3d": float("inf"),
            "center_dist_2d": cd_2d,
            "point_recall": 0.0,
            "point_precision": 0.0,
            "point_f1": 0.0,
            "n_points_cropped": 0,
            "pred_3d_bbox": None,
        }

    iou = iou_3d(pred_3d, gt_3d)
    cd = center_dist_3d(pred_3d, gt_3d)
    giou = giou_3d(pred_3d, gt_3d)

    # Z-clipped 3D IoU: pretend the model's Z range is identical to GT's
    # (the urban-scene "tall column" Z bias is then cancelled, isolating
    # XY localization quality). Equivalent to BEV IoU × ground-truth Z.
    pred_3d_zclip = pred_3d.copy()
    pred_3d_zclip[2] = gt_3d[2]
    pred_3d_zclip[5] = gt_3d[5]
    iou_zclip = iou_3d(pred_3d_zclip, gt_3d)

    # Point-level metrics: which scene points are inside each box.
    gt_mask = points_in_bbox(scene_points, gt_3d)
    n_gt = int(gt_mask.sum())
    if n_gt > 0:
        pred_mask_in_gt = points_in_bbox(scene_points[gt_mask], pred_3d)
        recall = float(pred_mask_in_gt.sum()) / n_gt
    else:
        recall = 0.0
    n_pred = cropped.shape[0]
    if n_pred > 0:
        in_gt = points_in_bbox(cropped, gt_3d)
        precision = float(in_gt.sum()) / n_pred
    else:
        precision = 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 1e-9 else 0.0

    return {
        "iou_3d": iou,
        "iou_3d_zclip": iou_zclip,
        "iou_bev": bev_iou,
        "giou_3d": giou,
        "center_dist_3d": cd,
        "center_dist_2d": cd_2d,
        "point_recall": recall,
        "point_precision": precision,
        "point_f1": f1,
        "n_points_cropped": n_pred,
        "pred_3d_bbox": pred_3d.tolist(),
    }


# ── Main eval loop ─────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="3D metrics for local rollout predictions.")
    parser.add_argument(
        "--predictions",
        required=True,
        help="JSONL produced by scripts/eval/rollout.py (typically records.dedup.jsonl). "
             "Each row needs scene_id, query, world_bbox.",
    )
    parser.add_argument(
        "--gt",
        required=True,
        help="GT dataset JSONL, e.g. resplit/cityrefer_test_NO.jsonl. "
             "Each row needs scene_id, query, gt_bbox.",
    )
    parser.add_argument(
        "--scene_dir",
        required=True,
        help="Directory containing per-scene point cloud files, e.g. "
             "$URBEX_RAW_DATA/sensaturban/ply. Searches "
             "{dir}/{scene_id}.{ply,pth} as well as the train/, test/, val/ "
             "subdirs (sensaturban PLYs are partitioned that way). Expects "
             "raw PLYs in un-normalized world-metre coordinates.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to write per-sample detail JSONL.",
    )
    parser.add_argument(
        "--iou_thresholds",
        default="0.25,0.5",
        help="Comma-separated 3D IoU thresholds for Acc@K reporting.",
    )
    args = parser.parse_args()

    iou_thresholds = [float(s.strip()) for s in args.iou_thresholds.split(",") if s.strip()]

    # Load predictions.
    if not os.path.exists(args.predictions):
        print(f"[err] predictions not found: {args.predictions}", file=sys.stderr)
        return 1
    preds = [json.loads(l) for l in open(args.predictions, "r", encoding="utf-8")
             if l.strip()]
    print(f"[load] {len(preds)} predictions from {args.predictions}")

    # Load GT, group by (scene_id, query) for multi-GT lookup.
    if not os.path.exists(args.gt):
        print(f"[err] GT not found: {args.gt}", file=sys.stderr)
        return 1
    gt_rows = [json.loads(l) for l in open(args.gt, "r", encoding="utf-8")
               if l.strip()]
    gt_lookup: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in gt_rows:
        key = (r["scene_id"], r.get("query") or r.get("description"))
        gt_lookup[key].append(r)
    print(f"[load] {len(gt_rows)} GT rows from {args.gt} "
          f"({len(gt_lookup)} unique (scene_id, query) pairs)")

    # Iterate predictions.
    out_records: List[dict] = []
    metric_lists: Dict[str, List[float]] = {
        "iou_3d": [], "iou_3d_zclip": [], "iou_bev": [], "giou_3d": [],
        "center_dist_3d": [], "center_dist_2d": [],
        "point_recall": [], "point_precision": [], "point_f1": [],
    }
    n_no_gt = n_no_world_bbox = n_zero_iou = 0
    sanity_checked: set = set()
    t0 = time.time()

    for i, pred in enumerate(preds):
        scene_id = pred.get("scene_id")
        query = pred.get("query")
        wb = pred.get("world_bbox")
        if wb is None or len(wb) < 4:
            n_no_world_bbox += 1
            continue
        gts = gt_lookup.get((scene_id, query))
        if not gts:
            n_no_gt += 1
            continue
        points = load_scene_points(args.scene_dir, scene_id)
        if points is None or points.shape[0] == 0:
            continue

        # Coordinate-system sanity check: scene points and gt_bbox MUST be
        # in the same world-metre frame. If they're not (e.g., scene loaded
        # from a normalized .pth), the crop/iou would silently produce 0 →
        # bail loudly so the user knows.
        if scene_id not in sanity_checked:
            sanity_checked.add(scene_id)
            gt_centers = np.array([
                [g["gt_bbox"][0], g["gt_bbox"][1], g["gt_bbox"][2]]
                for g in gts
            ])
            pmin = points.min(axis=0)
            pmax = points.max(axis=0)
            inside_x = ((gt_centers[:, 0] >= pmin[0]) & (gt_centers[:, 0] <= pmax[0])).mean()
            inside_y = ((gt_centers[:, 1] >= pmin[1]) & (gt_centers[:, 1] <= pmax[1])).mean()
            if inside_x < 0.5 or inside_y < 0.5:
                print(
                    f"[ERR] coordinate mismatch on scene {scene_id}: "
                    f"point cloud xy=[{pmin[0]:.1f},{pmax[0]:.1f}]×"
                    f"[{pmin[1]:.1f},{pmax[1]:.1f}] but GT centers in "
                    f"x=[{gt_centers[:,0].min():.1f},{gt_centers[:,0].max():.1f}] "
                    f"y=[{gt_centers[:,1].min():.1f},{gt_centers[:,1].max():.1f}]. "
                    f"Likely a normalized point cloud — switch --scene_dir to "
                    f"raw world-metre PLY/PCD.",
                    file=sys.stderr,
                )
                return 2

        # Multi-GT: pick best match by BEV IoU (more aligned with what the
        # model actually predicts). Falls back to iou_3d_zclip if all-zero.
        best = None
        for gt in gts:
            res = evaluate_one(wb, gt["gt_bbox"], points)
            if best is None or res["iou_bev"] > best["iou_bev"]:
                best = res
                best["matched_object_id"] = gt.get("object_id")

        for k in metric_lists:
            metric_lists[k].append(best[k])
        if best["iou_bev"] == 0.0:
            n_zero_iou += 1

        out_records.append({
            "scene_id":           scene_id,
            "query":              query,
            "matched_object_id":  best.get("matched_object_id"),
            "n_gt_candidates":    len(gts),
            "iou_2d_pred":        pred.get("iou_2d"),
            "success_pred":       pred.get("success"),
            "iou_3d":             best["iou_3d"],
            "iou_3d_zclip":       best["iou_3d_zclip"],
            "iou_bev":            best["iou_bev"],
            "giou_3d":            best["giou_3d"],
            "center_dist_3d":     best["center_dist_3d"],
            "center_dist_2d":     best["center_dist_2d"],
            "point_recall":       best["point_recall"],
            "point_precision":    best["point_precision"],
            "point_f1":           best["point_f1"],
            "n_points_cropped":   best["n_points_cropped"],
            "pred_world_bbox":    list(wb),
            "pred_3d_bbox":       best.get("pred_3d_bbox"),
        })
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(preds)}] processed, "
                  f"running mean BEV-IoU={np.mean(metric_lists['iou_bev']):.4f}",
                  flush=True)

    elapsed = time.time() - t0

    # Aggregate report.
    print()
    print("=" * 72)
    print("3D evaluation report")
    print("=" * 72)
    n_eval = len(metric_lists["iou_bev"])
    print(f"  predictions in:       {len(preds)}")
    print(f"  no world_bbox (skip): {n_no_world_bbox}")
    print(f"  no GT match (skip):   {n_no_gt}")
    print(f"  evaluated:            {n_eval}")
    print(f"  zero-BEV-IoU:         {n_zero_iou} ({100 * n_zero_iou / max(1, n_eval):.1f}%)")
    print()
    if n_eval > 0:
        def _stat(name: str, vals: List[float], unit: str = "") -> None:
            arr = np.array(vals)
            finite = arr[np.isfinite(arr)]
            if len(finite) == 0:
                print(f"  {name:<18} (all NaN/inf)")
                return
            print(f"  {name:<18} mean={finite.mean():.4f}{unit}  "
                  f"median={np.median(finite):.4f}{unit}  "
                  f"max={finite.max():.4f}{unit}")

        # Geometric IoU family
        _stat("IoU_BEV (XY):",      metric_lists["iou_bev"])
        _stat("IoU_3D_zclip:",      metric_lists["iou_3d_zclip"])
        _stat("IoU_3D (raw):",      metric_lists["iou_3d"])
        _stat("GIoU_3D:",           metric_lists["giou_3d"])
        print()
        # Distance
        _stat("center_dist_2d:",    metric_lists["center_dist_2d"], "m")
        _stat("center_dist_3d:",    metric_lists["center_dist_3d"], "m")
        print()
        # Point-level
        _stat("point_recall:",      metric_lists["point_recall"])
        _stat("point_precision:",   metric_lists["point_precision"])
        _stat("point_F1:",          metric_lists["point_f1"])
        print()
        # Acc thresholds.
        bev = np.array(metric_lists["iou_bev"])
        for thr in iou_thresholds:
            n_hit = int((bev >= thr).sum())
            print(f"  Acc@BEV-IoU>={thr}:    {n_hit}/{n_eval} = {100 * n_hit / n_eval:.2f}%")
        cd2 = np.array(metric_lists["center_dist_2d"])
        cd2 = cd2[np.isfinite(cd2)]
        for d in (5, 10, 20):
            n_hit = int((cd2 <= d).sum())
            print(f"  Acc@dist_2d<={d}m:    {n_hit}/{len(cd2)} = "
                  f"{100 * n_hit / max(1, len(cd2)):.2f}%")
    print()
    print(f"  elapsed: {elapsed:.1f}s ({len(preds) / elapsed:.1f} preds/s)")

    if args.output and out_records:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for r in out_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  per-sample detail: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
