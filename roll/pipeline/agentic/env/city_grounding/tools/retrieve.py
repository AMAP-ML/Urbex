"""Scene asset lookup: landmark resolution and point-cloud paths.

The three online tools only need the landmark database; ``ply_path_for_scene``
is used by the live oblique renderer fallback and by offline scripts.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Root for all agent-side artifacts (landmark DBs, query jsonls, BEV images,
# oblique-render caches). See the Data section of README.md.
_CITY_AGENT_DATA = os.environ.get("CITY_AGENT_DATA", "")

# STPLS3D-Refer uses a parallel artifact root; defaults to the same root.
_CITY_AGENT_DATA_STPLS3D = os.environ.get("CITY_AGENT_DATA_STPLS3D", _CITY_AGENT_DATA)

# Root of the RAW source datasets (SensatUrban / STPLS3D / UrbanBIS point
# clouds). Only the live oblique renderer and scripts/eval/metrics_3d.py read these.
_RAW_DATA = os.environ.get("URBEX_RAW_DATA", "")


def _raw(*parts: str) -> str:
    return os.path.join(_RAW_DATA, *parts)


_LANDMARK_DB = {
    "cityrefer": os.path.join(_CITY_AGENT_DATA, "landmarks_cityrefer.json"),
    "stpls3d": os.path.join(_CITY_AGENT_DATA_STPLS3D, "landmarks_stpls3d.json"),
}

# SensatUrban PLYs are partitioned into train/ and test/.
_SENSATURBAN_PLY_DIRS = [_raw("sensaturban", "ply", "train"),
                         _raw("sensaturban", "ply", "test")]


def ply_path_for_scene(scene_id: str, dataset: str) -> Optional[str]:
    """Return the raw point-cloud path for a scene, or None if not found."""
    if dataset == "cityrefer":
        candidates = [os.path.join(d, f"{scene_id}.ply") for d in _SENSATURBAN_PLY_DIRS]
    elif dataset == "stpls3d":
        candidates = [_raw("STPLS3D", "RealWorldData", filename)
                      for filename in (f"{scene_id}_points.ply", f"{scene_id}.ply")]
    elif dataset == "cityanchor":
        candidates = [_raw("STPLS3D", ver, f"{scene_id}.ply")
                      for ver in ("Synthetic_v3", "Synthetic_v2", "Synthetic_v1")]
    elif dataset == "urbanbis":
        # scene_id "Lihu_Area10" -> "Lihu/Area10.ply"
        candidates = [_raw("UrbanBIS", "mesh", f"{scene_id.replace('_', '/')}.ply")]
    else:
        return None
    return next((p for p in candidates if os.path.exists(p)), None)


# ── Landmark database ────────────────────────────────────────────────────────

_LANDMARKS_CACHE: dict = {}   # {dataset: {scene_id: [items]}}


def _load_landmarks(dataset: str = "cityrefer") -> dict:
    """Lazy-load and cache the landmark JSON. Returns {} when unavailable."""
    if dataset in _LANDMARKS_CACHE:
        return _LANDMARKS_CACHE[dataset]
    path = _LANDMARK_DB.get(dataset)
    if path is None:
        logger.warning("no landmark DB registered for dataset=%s", dataset)
        _LANDMARKS_CACHE[dataset] = {}
    else:
        try:
            with open(path) as f:
                _LANDMARKS_CACHE[dataset] = json.load(f)
        except FileNotFoundError:
            logger.warning("landmark DB for %s not found at %s; lookups will "
                           "return no matches.", dataset, path)
            _LANDMARKS_CACHE[dataset] = {}
    return _LANDMARKS_CACHE[dataset]


def _as_record(item: dict, score: float) -> Optional[dict]:
    """Convert a raw landmark DB entry into a lookup result, or None if unusable."""
    name = str(item.get("name") or "").strip()
    xy = item.get("xy") or []
    if not name or len(xy) < 2:
        return None
    return {
        "object_id": None,
        "object_name": name,
        "landmark": name,
        "bbox": [float(xy[0]), float(xy[1])],
        "score": float(score),
        "source": item.get("source", ""),
    }


def list_landmarks_for_scene(scene_id: str, dataset: str = "cityrefer") -> list[dict]:
    """Return every named landmark in a scene, in lookup-result format.

    Used by the landmark-robustness ablation, which replaces a correctly
    resolved landmark with a plausible but wrong one from the same scene.
    """
    records = (_as_record(item, 1.0) for item in _load_landmarks(dataset).get(scene_id, []))
    return [r for r in records if r is not None]


_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def _norm(s: str) -> str:
    return _PUNCT_RE.sub(" ", s.lower()).strip()


def _toks(s: str) -> set[str]:
    return {t for t in s.split() if len(t) >= 2}


def find_landmark_by_name(
    scene_id: str,
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
    dataset: str = "cityrefer",
) -> list[dict]:
    """Resolve a landmark name against a scene's landmark DB.

    Substring matches score 1.0; otherwise the score is token overlap
    normalized by the longer token set. ``bbox`` is ``[cx, cy]`` in world
    metres — the DB has no full 3D extent.
    """
    q = (query or "").strip().lower()
    if not q:
        return []
    q_norm = _norm(q)
    q_tokens = _toks(q_norm)

    scored: list[dict] = []
    for item in _load_landmarks(dataset).get(scene_id, []):
        name = item.get("name") or ""
        if not name:
            continue
        lm_norm = _norm(name)
        if q in name.lower() or (q_norm and q_norm in lm_norm):
            score = 1.0
        else:
            lm_tokens = _toks(lm_norm)
            overlap = len(q_tokens & lm_tokens)
            if not overlap:
                continue
            score = overlap / max(len(q_tokens), len(lm_tokens))
        if score < min_score:
            continue
        record = _as_record(item, score)
        if record is not None:
            scored.append(record)

    scored.sort(key=lambda x: -x["score"])
    return scored[:top_k]
