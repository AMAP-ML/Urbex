#!/usr/bin/env python3
"""Evaluate Hit@k metrics for city grounding predictions.

For each prediction, compare the predicted world_bbox with all candidate
bboxes from the scene's box3d/pth file. Sort candidates by the selected
center-distance, IoU, or CIoU rule and check if the GT object is in the top-k.

Candidate sources (mutually exclusive):
  --box3d_dir   Original bbox JSONs (~232 instances / scene, world coords).
  --pth_dir     Pre-processed .pth files (~172 instances / scene,
                cropped coords + offset in pth[9]).

Denominator semantics
---------------------
The denominator counts every model attempt that has a usable GT row +
box3d candidates for the scene. A pred whose ``world_bbox`` is None
(model never committed an answer / commit was unparseable) is counted as
0 hits across all k. Excluding no-answer attempts inflates Hit@k for
weak baselines that frequently exhaust the tool budget without
committing.

Eval-pipeline misses (no GT row, no box3d for the scene) are tracked
separately as ``skipped_no_gt`` / ``skipped_no_box3d`` and NOT counted in
the denominator — they're our data-pipeline gaps, not model failures.

Usage::

    # Original box3d JSON source
    python scripts/eval/hit_k.py \\
        --pred records.jsonl --dataset cityrefer_val_NO.jsonl \\
        --box3d_dir /path/to/cityrefer/box3d

    # PTH source (fewer candidates, cropped coords)
    python scripts/eval/hit_k.py \\
        --pred records.jsonl --dataset cityrefer_val_NO.jsonl \\
        --pth_dir /path/to/cityanchor/data_cityrefer/sensaturban/pointgroup_data/balance_split/random-50_crop-250
"""

import argparse
import json
import math
import os
from typing import Dict, List


def load_jsonl(path: str) -> List[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_box3d_cache(box3d_dir: str) -> Dict[str, List[dict]]:
    """Load from *_bbox.json files."""
    cache = {}
    for fname in os.listdir(box3d_dir):
        if not fname.endswith("_bbox.json"):
            continue
        with open(os.path.join(box3d_dir, fname)) as f:
            data = json.load(f)
        cache[data["scene_id"]] = data["bboxes"]
    return cache


def load_pth_cache(pth_dir: str) -> Dict[str, List[dict]]:
    """Load instance bboxes from .pth files.

    Each .pth is a tuple where:
      pth[6] : dict {instance_id -> bbox_array}
      pth[9] : [offset_x, offset_y, scale]  (world = pth + offset)
    """
    try:
        import torch
    except ImportError:
        raise ImportError("torch is required for --pth_dir. `pip install torch`")

    cache = {}
    for fname in sorted(os.listdir(pth_dir)):
        if not fname.endswith(".pth"):
            continue
        scene_id = fname.replace(".pth", "")
        pth_path = os.path.join(pth_dir, fname)
        try:
            d = torch.load(pth_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[warn] failed to load {pth_path}: {e}")
            continue

        # d is a tuple; index 6 = instance bbox dict, index 9 = metadata
        instance_bboxes = d[6]
        metadata = d[9]
        offset_x, offset_y = float(metadata[0]), float(metadata[1])

        bboxes = []
        for obj_id, bbox_arr in instance_bboxes.items():
            # bbox_arr: [cx, cy, cz, dx, dy, dz, class_id, instance_id]
            cx = float(bbox_arr[0]) + offset_x
            cy = float(bbox_arr[1]) + offset_y
            cz = float(bbox_arr[2])
            dx = float(bbox_arr[3])
            dy = float(bbox_arr[4])
            dz = float(bbox_arr[5])
            class_id = float(bbox_arr[6])
            instance_id = int(bbox_arr[7])
            bboxes.append({
                "object_id": instance_id,
                "object_name": "",  # pth doesn't provide names
                "bbox": [cx, cy, cz, dx, dy, dz, class_id, instance_id],
            })
        cache[scene_id] = bboxes
    return cache


def _bbox3d_to_world_bbox(bbox3d: List[float]) -> List[float]:
    """Convert box3d [cx, cy, cz, dx, dy, dz, ...] to [x_min, y_min, x_max, y_max]."""
    cx, cy, dx, dy = bbox3d[0], bbox3d[1], bbox3d[3], bbox3d[4]
    return [cx - dx / 2, cy - dy / 2, cx + dx / 2, cy + dy / 2]


def _iou_2d(a: List[float], b: List[float]) -> float:
    """Compute 2D IoU between two axis-aligned bboxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    aw = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    bw = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = aw + bw - inter
    return inter / union if union > 1e-6 else 0.0


def _center_distance_score(a: List[float], b: List[float]) -> float:
    """Return negative center distance so larger values rank first."""
    a_cx = 0.5 * (a[0] + a[2])
    a_cy = 0.5 * (a[1] + a[3])
    b_cx = 0.5 * (b[0] + b[2])
    b_cy = 0.5 * (b[1] + b[3])
    return -math.hypot(a_cx - b_cx, a_cy - b_cy)


def _ciou_2d(a: List[float], b: List[float]) -> float:
    """Compute Complete IoU for two axis-aligned 2D boxes."""
    iou = _iou_2d(a, b)
    a_w = max(0.0, a[2] - a[0])
    a_h = max(0.0, a[3] - a[1])
    b_w = max(0.0, b[2] - b[0])
    b_h = max(0.0, b[3] - b[1])
    a_cx = 0.5 * (a[0] + a[2])
    a_cy = 0.5 * (a[1] + a[3])
    b_cx = 0.5 * (b[0] + b[2])
    b_cy = 0.5 * (b[1] + b[3])
    rho_sq = (a_cx - b_cx) ** 2 + (a_cy - b_cy) ** 2

    enclosing_w = max(a[2], b[2]) - min(a[0], b[0])
    enclosing_h = max(a[3], b[3]) - min(a[1], b[1])
    enclosing_diag_sq = enclosing_w ** 2 + enclosing_h ** 2
    distance_penalty = rho_sq / enclosing_diag_sq if enclosing_diag_sq > 1e-12 else 0.0

    if min(a_w, a_h, b_w, b_h) <= 1e-12:
        aspect_penalty = 0.0
    else:
        v = 4.0 / math.pi ** 2 * (
            math.atan(b_w / b_h) - math.atan(a_w / a_h)
        ) ** 2
        alpha = v / max(1e-12, 1.0 - iou + v)
        aspect_penalty = alpha * v
    return iou - distance_penalty - aspect_penalty


def _candidate_score(
    pred_box: List[float],
    candidate_box: List[float],
    score_rule: str,
) -> float:
    if score_rule == "center_distance":
        return _center_distance_score(pred_box, candidate_box)
    if score_rule == "ciou":
        return _ciou_2d(pred_box, candidate_box)
    return _iou_2d(pred_box, candidate_box)


def compute_hit_k(
    preds: List[dict],
    dataset: List[dict],
    candidate_cache: Dict[str, List[dict]],
    k_values=(1, 3, 5),
    score_rule: str = "iou",
    require_positive_overlap: bool = False,
) -> dict:
    # Build GT lookup
    gt_lookup = {}
    for s in dataset:
        key = (s["scene_id"], s["query"])
        gt_lookup[key] = s

    hits = {k: 0 for k in k_values}
    total = 0
    no_answer = 0
    skipped_no_gt = 0
    skipped_no_box3d = 0
    stats = []

    for pred in preds:
        key = (pred["scene_id"], pred["query"])
        gt = gt_lookup.get(key)
        if gt is None:
            skipped_no_gt += 1
            continue

        scene_id = pred["scene_id"]
        candidates = candidate_cache.get(scene_id)
        if not candidates:
            skipped_no_box3d += 1
            continue

        # Counted from here on: every model attempt with usable GT.
        total += 1
        pred_wb = pred.get("world_bbox")
        if pred_wb is None:
            # Model never committed -> 0 hit across all k, but still in denom.
            no_answer += 1
            stats.append({
                "scene_id": scene_id,
                "query": pred["query"][:80],
                "gt_obj_id": str(gt["object_id"]),
                "pred_gt_iou": 0.0,
                "top1_obj_id": None,
                "top1_score": None,
                "top1_iou": 0.0,
                "top1_is_gt": False,
                "score_rule": score_rule,
                "success": pred.get("success", False),
                "iou_2d": pred.get("iou_2d", 0.0),
                "reason": pred.get("reason", "no_answer"),
            })
            continue

        gt_obj_id = str(gt["object_id"])

        # Compute 2D IoU with each candidate
        scores = []
        for cand in candidates:
            cand_wb = _bbox3d_to_world_bbox(cand["bbox"])
            iou = _iou_2d(pred_wb, cand_wb)
            if require_positive_overlap and iou <= 0.0:
                continue
            score = _candidate_score(pred_wb, cand_wb, score_rule)
            scores.append(
                (score, iou, str(cand["object_id"]), cand.get("object_name", ""))
            )

        # All rules are represented as "higher is better".
        scores.sort(key=lambda x: x[0], reverse=True)

        # Check hits
        for k in k_values:
            top_k_ids = [oid for _, _, oid, _ in scores[:k]]
            if gt_obj_id in top_k_ids:
                hits[k] += 1

        # Per-sample detail
        if scores:
            top1_score, top1_iou, top1_oid, _ = scores[0]
        else:
            top1_score, top1_iou, top1_oid = float("-inf"), 0.0, None
        gt_iou = _iou_2d(pred_wb, _bbox3d_to_world_bbox(gt["gt_bbox"]))
        stats.append({
            "scene_id": scene_id,
            "query": pred["query"][:80],
            "gt_obj_id": gt_obj_id,
            "pred_gt_iou": gt_iou,
            "top1_obj_id": top1_oid,
            "top1_score": top1_score,
            "top1_iou": top1_iou,
            "top1_is_gt": top1_oid == gt_obj_id,
            "score_rule": score_rule,
            "success": pred.get("success", False),
            "iou_2d": pred.get("iou_2d", 0.0),
            "reason": pred.get("reason", "answer"),
        })

    return {
        "total": total,
        "no_answer": no_answer,
        "skipped_no_gt": skipped_no_gt,
        "skipped_no_box3d": skipped_no_box3d,
        "hits": hits,
        "rates": {k: hits[k] / total if total else 0.0 for k in k_values},
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hit@k for city grounding")
    parser.add_argument("--pred", required=True, help="Path to prediction JSONL")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSONL (with gt_bbox)")
    parser.add_argument(
        "--box3d_dir",
        default="",
        help="Directory containing *_bbox.json files",
    )
    parser.add_argument(
        "--pth_dir",
        default="",
        help="Directory containing {scene_id}.pth files (uses pth[6] instance bboxes + pth[9] offset)",
    )
    parser.add_argument(
        "--score_rule",
        choices=["center_distance", "iou", "ciou"],
        default="iou",
        help="Geometric rule used to rank candidate boxes.",
    )
    parser.add_argument(
        "--require_positive_overlap",
        action="store_true",
        help="Exclude zero-IoU candidates before ranking. Useful for checking "
             "whether plain-IoU Hit@k benefits from arbitrary all-zero ties.",
    )
    parser.add_argument("--save_detail", default="", help="Path to save per-sample detail JSONL")
    args = parser.parse_args()

    if not args.box3d_dir and not args.pth_dir:
        parser.error(
            "one of --box3d_dir or --pth_dir is required (candidate pool "
            "source; see README.md)"
        )

    preds = load_jsonl(args.pred)
    dataset = load_jsonl(args.dataset)

    if args.pth_dir:
        candidate_cache = load_pth_cache(args.pth_dir)
        source_name = "pth"
    else:
        candidate_cache = load_box3d_cache(args.box3d_dir)
        source_name = "box3d"

    # Count total candidates
    total_cands = sum(len(v) for v in candidate_cache.values())
    avg_cands = total_cands / len(candidate_cache) if candidate_cache else 0

    print(f"Predictions: {len(preds)}")
    print(f"Dataset: {len(dataset)}")
    print(f"Scenes with {source_name}: {len(candidate_cache)}")
    print(f"Total candidates: {total_cands} (avg {avg_cands:.1f}/scene)")
    print()

    result = compute_hit_k(
        preds,
        dataset,
        candidate_cache,
        score_rule=args.score_rule,
        require_positive_overlap=args.require_positive_overlap,
    )

    total = result["total"]
    no_ans = result["no_answer"]
    print(f"Evaluated:  {total} samples (denominator includes no-answer attempts)")
    print(
        f"Ranking:    {args.score_rule}"
        + (" (positive-overlap candidates only)" if args.require_positive_overlap else "")
    )
    print(f"  no_answer: {no_ans}  (model never committed -> 0 hit, still in denom)")
    print(f"  scored:    {total - no_ans}  (model committed and was ranked vs candidates)")
    if result["skipped_no_gt"] or result["skipped_no_box3d"]:
        print(f"  skipped:   no_gt={result['skipped_no_gt']}  "
              f"no_{source_name}={result['skipped_no_box3d']}  "
              f"(eval-pipeline gaps, NOT counted in denom)")
    print()
    for k in sorted(result["hits"].keys()):
        n = result["hits"][k]
        rate = result["rates"][k]
        print(f"  Hit@{k}: {n}/{total} = {rate * 100:.2f}%")

    # Save detail if requested
    if args.save_detail:
        with open(args.save_detail, "w") as f:
            for s in result["stats"]:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"\nPer-sample detail saved to: {args.save_detail}")


if __name__ == "__main__":
    main()
