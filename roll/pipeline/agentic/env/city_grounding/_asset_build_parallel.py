from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

SceneJob = dict[str, object]
SceneResult = dict[str, object]


def _failed_outcome(scene_id: str, exc: Exception) -> SceneResult:
    return {
        "scene_id": scene_id,
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
    }


def run_scene_jobs(
    jobs: list[SceneJob],
    job_fn: Callable[[SceneJob], dict],
    workers: int,
) -> list[SceneResult]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if not jobs:
        return []

    if workers == 1:
        outcomes = []
        for job in jobs:
            scene_id = str(job["scene_id"])
            try:
                outcomes.append({"scene_id": scene_id, "ok": True, "result": job_fn(job)})
            except Exception as exc:
                outcomes.append(_failed_outcome(scene_id, exc))
        return outcomes

    executor = ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)),
        mp_context=multiprocessing.get_context("spawn"),
    )
    future_to_job: dict = {}
    outcomes: list[SceneResult | None] = [None] * len(jobs)
    try:
        for index, job in enumerate(jobs):
            future = executor.submit(job_fn, job)
            future_to_job[future] = (index, str(job["scene_id"]))
        for future in as_completed(future_to_job):
            index, scene_id = future_to_job[future]
            try:
                outcomes[index] = {
                    "scene_id": scene_id,
                    "ok": True,
                    "result": future.result(),
                }
            except Exception as exc:
                outcomes[index] = _failed_outcome(scene_id, exc)
    except KeyboardInterrupt:
        for future in future_to_job:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    return [outcome for outcome in outcomes if outcome is not None]
