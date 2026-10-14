from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

SUPPORTED_DATASETS = ("cityrefer", "stpls3d")


def load_scene_ids(
    jsonl_path: str | Path | Sequence[str | Path],
    dataset: str,
    selected_scenes: Sequence[str] | None = None,
) -> list[str]:
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset {dataset!r}; expected one of {SUPPORTED_DATASETS}")

    paths = [Path(jsonl_path)] if isinstance(jsonl_path, (str, Path)) else [Path(path) for path in jsonl_path]
    if not paths:
        raise ValueError("no annotation JSONL paths provided")
    scene_ids: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}:{line_number}: {exc.msg}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"expected an object in {path}:{line_number}")
                scene_id = record.get("scene_id")
                if not isinstance(scene_id, str) or not scene_id.strip():
                    raise ValueError(f"missing scene_id in {path}:{line_number}")
                record_dataset = record.get("dataset")
                if record_dataset and record_dataset != dataset:
                    raise ValueError(
                        f"dataset mismatch in {path}:{line_number}: "
                        f"expected {dataset!r}, got {record_dataset!r}"
                    )
                scene_ids.add(scene_id.strip())

    sources = ", ".join(str(path) for path in paths)
    if not scene_ids:
        raise ValueError(f"no scenes found in {sources}")

    if selected_scenes:
        selected = set(selected_scenes)
        missing = sorted(selected - scene_ids)
        if missing:
            raise ValueError(f"requested scenes are absent from {sources}: {', '.join(missing)}")
        scene_ids &= selected

    return sorted(scene_ids)


def ply_candidates(scene_id: str, dataset: str, ply_root: str | Path) -> list[Path]:
    root = Path(ply_root).expanduser()
    if dataset == "cityrefer":
        relative_paths: Iterable[Path] = (
            Path(f"{scene_id}.ply"),
            Path("train") / f"{scene_id}.ply",
            Path("test") / f"{scene_id}.ply",
            Path("ply") / "train" / f"{scene_id}.ply",
            Path("ply") / "test" / f"{scene_id}.ply",
            Path("sensaturban") / "ply" / "train" / f"{scene_id}.ply",
            Path("sensaturban") / "ply" / "test" / f"{scene_id}.ply",
        )
    elif dataset == "stpls3d":
        # Original STPLS3D releases use <scene>.ply; some processed exports
        # use <scene>_points.ply. Accept both without renaming source data.
        relative_paths = tuple(
            directory / filename
            for filename in (f"{scene_id}_points.ply", f"{scene_id}.ply")
            for directory in (Path(), Path("RealWorldData"), Path("STPLS3D") / "RealWorldData")
        )
    else:
        raise ValueError(f"unsupported dataset {dataset!r}; expected one of {SUPPORTED_DATASETS}")
    return [root / relative_path for relative_path in relative_paths]


def resolve_ply_path(scene_id: str, dataset: str, ply_root: str | Path) -> Path:
    candidates = ply_candidates(scene_id, dataset, ply_root)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    attempted = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"no PLY found for scene {scene_id!r} ({dataset}); tried:\n  {attempted}"
    )
