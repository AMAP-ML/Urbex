"""Axis-aligned 3D IoU for bounding boxes in [cx, cy, cz, dx, dy, dz] format."""

import numpy as np


def iou_3d(box1, box2) -> float:
    """
    Compute axis-aligned 3D IoU between two boxes.
    box format: [cx, cy, cz, dx, dy, dz]  (center + full dimensions)
    """
    box1, box2 = np.asarray(box1, dtype=np.float64), np.asarray(box2, dtype=np.float64)

    min1 = box1[:3] - box1[3:6] / 2
    max1 = box1[:3] + box1[3:6] / 2
    min2 = box2[:3] - box2[3:6] / 2
    max2 = box2[:3] + box2[3:6] / 2

    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_dims = np.maximum(0.0, inter_max - inter_min)
    inter_vol = float(np.prod(inter_dims))

    vol1 = float(np.prod(box1[3:6]))
    vol2 = float(np.prod(box2[3:6]))
    union_vol = vol1 + vol2 - inter_vol

    if union_vol < 1e-9:
        return 0.0
    return inter_vol / union_vol


def iou_2d(box1, box2) -> float:
    """
    Compute axis-aligned 2D IoU (XY plane only), ignoring Z.
    box format: [cx, cy, cz, dx, dy, dz]
    """
    box1, box2 = np.asarray(box1, dtype=np.float64), np.asarray(box2, dtype=np.float64)

    min1 = box1[:2] - box1[3:5] / 2
    max1 = box1[:2] + box1[3:5] / 2
    min2 = box2[:2] - box2[3:5] / 2
    max2 = box2[:2] + box2[3:5] / 2

    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_dims = np.maximum(0.0, inter_max - inter_min)
    inter_area = float(np.prod(inter_dims))

    area1 = float(np.prod(box1[3:5]))
    area2 = float(np.prod(box2[3:5]))
    union_area = area1 + area2 - inter_area

    if union_area < 1e-9:
        return 0.0
    return inter_area / union_area


def center_dist(box1, box2) -> float:
    """Euclidean distance between box centers (XY only)."""
    b1, b2 = np.asarray(box1), np.asarray(box2)
    return float(np.linalg.norm(b1[:2] - b2[:2]))


def region_hit(gt_box, region) -> bool:
    """
    Check whether the GT box center falls within a 2D world region.
    region: [x_min, y_min, x_max, y_max]
    gt_box: [cx, cy, cz, dx, dy, dz]
    """
    cx, cy = gt_box[0], gt_box[1]
    x_min, y_min, x_max, y_max = region
    return x_min <= cx <= x_max and y_min <= cy <= y_max
