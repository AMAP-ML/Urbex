"""Query difficulty filter for SFT data collection.

Tracks per-query rollout quality and blacklists queries that are
consistently too hard for the current model (8 consecutive failures
with IoU=0 and center_dist > 50m).

Also tracks the **best** result across all rollouts:
  - ``max_iou``       – highest IoU seen for this query
  - ``min_dist``      – smallest center distance seen
  - ``success_count`` – how many rollouts reached IoU >= 0.25

Multi-process safe: each Ray EnvironmentWorker writes to its own
per-process file (``<base>.<host>-pid<nnnn>.json``).  A separate
merge step is run offline to concatenate all shards.

Usage::

    tracker = QueryFilterTracker("/path/to/blacklist.json")

    # Check before rollout
    if tracker.is_blacklisted(scene_id, query):
        skip_this_query()

    # Update after rollout
    tracker.update(scene_id, query, iou_2d=iou, center_dist=dist)
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Thresholds
CONSECUTIVE_FAIL_THRESHOLD = 8
IOU_FAIL_THRESHOLD = 0.0
DIST_FAIL_THRESHOLD = 50.0


def _per_worker_path(base_path: str) -> str:
    """Insert hostname + pid + thread-id into a per-shard subdirectory.

    Each env thread inside the same Ray worker process gets its own
    file so concurrent writes do not clobber each other.

    /path/to/blacklist.json
        -> /path/to/blacklist_shards/blacklist.<host>-pid12345-tid1400123456.json
    """
    root, ext = os.path.splitext(base_path)
    shard_dir = f"{root}_shards"
    os.makedirs(shard_dir, exist_ok=True)
    host = socket.gethostname().split(".")[0]
    pid = os.getpid()
    tid = threading.current_thread().ident
    basename = os.path.basename(root)
    return os.path.join(
        shard_dir,
        f"{basename}.{host}-pid{pid}-tid{tid}{ext or '.json'}"
    )


class QueryFilterTracker:
    """Track per-query rollout quality and maintain a blacklist.

    Thread-safe for use in multi-threaded API workers.
    Process-safe for use in multi-process ROLL workers (each process
    gets its own file; no shared writers).
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._worker_path = _per_worker_path(path) if path else None
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        if self._worker_path:
            self._load()

    # -- persistence --------------------------------------------------------

    def _load(self) -> None:
        assert self._worker_path
        # 1. Try per-process shard first (warm restart within same worker).
        if os.path.exists(self._worker_path):
            try:
                with open(self._worker_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.info(
                    "[query_filter] loaded %d entries from %s",
                    len(self._data), self._worker_path,
                )
                return
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load %s: %s", self._worker_path, e)
        # 2. Fall back to the base file (first-time start).
        if self.path and os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.info(
                    "[query_filter] loaded %d entries from %s",
                    len(self._data), self.path,
                )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load %s: %s", self.path, e)
                self._data = {}

    def save(self) -> None:
        if not self._worker_path:
            return
        tmp = self._worker_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._worker_path)
        except OSError as e:
            logger.warning("Failed to save blacklist %s: %s", self._worker_path, e)

    # -- key ----------------------------------------------------------------

    @staticmethod
    def _key(scene_id: str, query: str) -> str:
        return f"{scene_id}::{query}"

    # -- public API ---------------------------------------------------------

    def is_blacklisted(self, scene_id: str, query: str) -> bool:
        with self._lock:
            entry = self._data.get(self._key(scene_id, query))
            return entry is not None and entry.get("blacklisted", False)

    def update(
        self,
        scene_id: str,
        query: str,
        iou_2d: float,
        center_dist: float,
    ) -> bool:
        """Record one rollout result.

        Returns True if the query is newly blacklisted after this update.
        """
        key = self._key(scene_id, query)
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                entry = {
                    "scene_id": scene_id,
                    "query": query,
                    "consecutive_fails": 0,
                    "last_iou": None,
                    "last_dist": None,
                    "max_iou": None,
                    "min_dist": None,
                    "success_count": 0,
                    "blacklisted": False,
                    "blacklisted_at": None,
                    "total_rollouts": 0,
                }
                self._data[key] = entry

            entry["total_rollouts"] += 1
            entry["last_iou"] = float(iou_2d)
            entry["last_dist"] = float(center_dist)

            # Track best results across all rollouts
            _iou = float(iou_2d)
            _dist = float(center_dist)
            if entry["max_iou"] is None or _iou > entry["max_iou"]:
                entry["max_iou"] = _iou
            if entry["min_dist"] is None or _dist < entry["min_dist"]:
                entry["min_dist"] = _dist
            if _iou >= 0.25:
                entry["success_count"] += 1

            # Determine if this rollout is a "hard fail"
            is_hard_fail = (
                float(iou_2d) <= IOU_FAIL_THRESHOLD
                and float(center_dist) > DIST_FAIL_THRESHOLD
            )

            if is_hard_fail:
                entry["consecutive_fails"] += 1
            else:
                entry["consecutive_fails"] = 0

            newly_blacklisted = False
            if entry["consecutive_fails"] >= CONSECUTIVE_FAIL_THRESHOLD:
                if not entry["blacklisted"]:
                    entry["blacklisted"] = True
                    entry["blacklisted_at"] = time.strftime(
                        "%Y-%m-%dT%H:%M:%S"
                    )
                    newly_blacklisted = True
                    logger.info(
                        "[query_filter] BLACKLISTED %s "
                        "(consecutive_fails=%d, last_iou=%.3f, last_dist=%.1fm)",
                        key, entry["consecutive_fails"],
                        entry["last_iou"], entry["last_dist"],
                    )

        return newly_blacklisted

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._data)
            blacklisted = sum(
                1 for e in self._data.values() if e.get("blacklisted")
            )
        return {
            "total_queries_tracked": total,
            "blacklisted": blacklisted,
            "active": total - blacklisted,
        }

    def get_blacklist_queries(self) -> list[str]:
        """Return list of blacklisted query keys."""
        with self._lock:
            return [
                k for k, e in self._data.items()
                if e.get("blacklisted", False)
            ]

    @classmethod
    def merge_shards(cls, base_path: str, output_path: str) -> int:
        """Merge all per-worker shards into a single blacklist file.

        Returns the number of blacklisted queries in the merged result.
        """
        root, ext = os.path.splitext(base_path)
        shard_dir = f"{root}_shards"
        pattern = os.path.join(shard_dir, f"*-pid*-tid*{ext or '.json'}")
        import glob

        shards = glob.glob(pattern)
        if not shards:
            logger.warning("No shard files found matching %s", pattern)
            return 0

        logger.info("Merging %d shard files into %s", len(shards), output_path)

        merged: Dict[str, Dict[str, Any]] = {}
        for shard in sorted(shards):
            try:
                with open(shard, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Skipping bad shard %s: %s", shard, e)
                continue

            for key, entry in data.items():
                if key not in merged:
                    merged[key] = entry
                else:
                    # Same query seen by different workers.
                    # Merge total_rollouts and consecutive_fails carefully.
                    old = merged[key]
                    new = entry
                    # Cumulative rollouts
                    old["total_rollouts"] = old.get("total_rollouts", 0) + new.get("total_rollouts", 0)
                    # If either shard blacklisted it, mark as blacklisted
                    if new.get("blacklisted") and not old.get("blacklisted"):
                        old["blacklisted"] = True
                        old["blacklisted_at"] = new.get("blacklisted_at")
                    # consecutive_fails: take the max seen across shards
                    old["consecutive_fails"] = max(
                        old.get("consecutive_fails", 0),
                        new.get("consecutive_fails", 0),
                    )
                    # Best results across shards: max iou, min dist, sum success_count
                    old["max_iou"] = max(
                        old.get("max_iou") if old.get("max_iou") is not None else float("-inf"),
                        new.get("max_iou") if new.get("max_iou") is not None else float("-inf"),
                    )
                    old["min_dist"] = min(
                        old.get("min_dist") if old.get("min_dist") is not None else float("inf"),
                        new.get("min_dist") if new.get("min_dist") is not None else float("inf"),
                    )
                    old["success_count"] = old.get("success_count", 0) + new.get("success_count", 0)
                    # Last values from whichever shard has more total rollouts
                    if new.get("total_rollouts", 0) > old.get("total_rollouts", 0):
                        old["last_iou"] = new.get("last_iou")
                        old["last_dist"] = new.get("last_dist")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        blacklisted_count = sum(1 for e in merged.values() if e.get("blacklisted"))
        logger.info(
            "Merged %d unique queries (%d blacklisted) -> %s",
            len(merged), blacklisted_count, output_path,
        )
        return blacklisted_count
