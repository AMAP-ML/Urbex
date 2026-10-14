#!/usr/bin/env python3
"""Urbex test-set rollout.

Runs ROLL's AgenticRolloutPipeline with a vLLM backend and
CityVLRolloutEnvManager: group_size=1, greedy decoding, sequential_mode so the
split is covered exactly once. Per-worker JSONL shards are merged and
de-duplicated at the end into records.dedup.jsonl, which is what the scorers
consume.

Usage::

    python scripts/eval/rollout.py \
        exp_name=cityrefer_ND \
        output_dir=./output/eval_ND \
        custom_envs.CityTool.env_config.data_path=$CITY_AGENT_DATA/meta/cityrefer_test_ND.jsonl
"""

import argparse

from dacite import from_dict
from hydra import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init as roll_init
from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.pipeline.agentic.agentic_rollout_pipeline import AgenticRolloutPipeline


def main():
    parser = argparse.ArgumentParser(description="Urbex test-set rollout")
    # hydra.initialize resolves config_path relative to THIS file, not the cwd.
    parser.add_argument("--config_path", default="../configs", help="Config dir, relative to this file")
    parser.add_argument("--config_name", default="rollout", help="Config file name without extension")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. rollout_batch_size=64")
    args = parser.parse_args()

    initialize(config_path=args.config_path, job_name="urbex_rollout", version_base=None)
    cfg = compose(config_name=args.config_name, overrides=args.overrides)

    print("=" * 60)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("=" * 60)

    if cfg.max_steps == -1:
        _infer_max_steps(cfg)

    config = from_dict(data_class=AgenticConfig, data=OmegaConf.to_container(cfg, resolve=True))

    # The manager opens per-worker shards in append mode, so stale files from a
    # previous attempt would pile up in this run's dump.
    _clean_old_shards(cfg)

    roll_init()
    AgenticRolloutPipeline(pipeline_config=config).run()
    _dedup_dump(cfg)


def _infer_max_steps(cfg) -> None:
    data_path = cfg.custom_envs.CityTool.env_config.data_path
    try:
        with open(data_path) as f:
            n_samples = sum(1 for line in f if line.strip())
    except OSError as e:
        print(f"[warn] cannot size {data_path}: {e}; pass max_steps=N explicitly")
        return
    steps = (n_samples + cfg.rollout_batch_size - 1) // cfg.rollout_batch_size
    OmegaConf.set_struct(cfg, False)
    cfg.max_steps = steps
    OmegaConf.set_struct(cfg, True)
    print(f"[auto] {n_samples} samples / batch {cfg.rollout_batch_size} -> max_steps={steps}")


def _clean_old_shards(cfg) -> None:
    """Remove leftover per-worker shards from previous runs so the
    current run's dump is not contaminated by append-mode accumulation.
    Also removes the prior merged/dedup outputs."""
    import glob
    import os

    base = cfg.custom_envs.CityTool.get("rollout_dump_jsonl", None)
    if not base:
        return
    root, ext = os.path.splitext(base)
    patterns = [f"{root}.*{ext or '.jsonl'}", base]
    removed = []
    for p in patterns:
        for f in glob.glob(p):
            try:
                os.remove(f)
                removed.append(f)
            except OSError:
                pass
    if removed:
        print(f"[clean] removed {len(removed)} stale dump file(s) under {root}.*")


_TOOL_CALL_RE_STR = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"


def _count_tool_calls(rec: dict, counter: dict) -> int:
    """Walk a dump record's `messages` and tally each <tool_call> by name.
    `unknown` covers calls the env couldn't dispatch (name not in
    TOOL_METHODS) — env replies with role="tool" + text "error: unknown
    tool: ...", which we infer from the assistant text's parsed name not
    being one of the canonical tools.
    Returns the number of tool calls found in this record."""
    import re
    canonical = {"get_landmark_xy", "zoom_in", "render_oblique"}
    n = 0
    for msg in rec.get("messages") or []:
        if msg.get("role") != "assistant":
            continue
        text = msg.get("content")
        if not isinstance(text, str):
            continue
        for m in re.finditer(_TOOL_CALL_RE_STR, text, re.DOTALL):
            try:
                payload = __import__("json").loads(m.group(1))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            name = payload.get("name")
            if name not in canonical:
                name = "unknown"
            counter[name] = counter.get(name, 0) + 1
            n += 1
    return n


def _dedup_dump(cfg) -> None:
    import glob
    import json
    import os

    base = cfg.custom_envs.CityTool.get("rollout_dump_jsonl", None)
    if not base:
        return
    root, ext = os.path.splitext(base)
    shards = sorted(glob.glob(f"{root}.*{ext or '.jsonl'}"))
    if not shards:
        print(f"[dedup] no per-worker shards under {root}.*{ext or '.jsonl'}, skipping")
        return

    merged_path = f"{root}.merged{ext or '.jsonl'}"
    dedup_path = f"{root}.dedup{ext or '.jsonl'}"
    seen: set = set()
    n_in = n_out = 0
    iou_sum = 0.0
    success_count = 0
    iou_buckets = [0, 0, 0, 0, 0]   # [0,0.1) [0.1,0.25) [0.25,0.5) [0.5,0.75) [0.75,1]
    tool_counter: dict = {}          # tool_name -> total invocations across dedup'd records
    n_with_tool_call = 0             # episodes that issued >=1 tool call
    with open(merged_path, "w", encoding="utf-8") as fmerge, \
         open(dedup_path, "w", encoding="utf-8") as fdedup:
        for shard in shards:
            with open(shard, encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    n_in += 1
                    fmerge.write(line)
                    rec = json.loads(line)
                    key = (rec.get("scene_id"), rec.get("query"), rec.get("dataset"))
                    if key in seen:
                        continue
                    seen.add(key)
                    fdedup.write(line)
                    n_out += 1
                    iou = float(rec.get("iou_2d") or 0.0)
                    iou_sum += iou
                    if rec.get("success"):
                        success_count += 1
                    if iou < 0.1:
                        iou_buckets[0] += 1
                    elif iou < 0.25:
                        iou_buckets[1] += 1
                    elif iou < 0.5:
                        iou_buckets[2] += 1
                    elif iou < 0.75:
                        iou_buckets[3] += 1
                    else:
                        iou_buckets[4] += 1
                    n_calls = _count_tool_calls(rec, tool_counter)
                    if n_calls > 0:
                        n_with_tool_call += 1
    print(f"[dedup] {n_in} records in {len(shards)} shards -> {n_out} unique")
    print(f"[dedup] merged: {merged_path}")
    print(f"[dedup] dedup:  {dedup_path}")
    if n_out > 0:
        b = iou_buckets
        print(f"[score] dedup samples: {n_out}")
        print(f"[score] mean IoU:      {iou_sum / n_out:.4f}")
        print(f"[score] success_rate:  {100 * success_count / n_out:.2f}%  "
              f"({success_count}/{n_out})")
        print(f"[score] IoU buckets:   "
              f"[0,0.1)={b[0]}  "
              f"[0.1,0.25)={b[1]}  "
              f"[0.25,0.5)={b[2]}  "
              f"[0.5,0.75)={b[3]}  "
              f"[0.75,1]={b[4]}")
        # Per-tool call statistics across dedup'd records.
        total_calls = sum(tool_counter.values())
        print(f"[tools] total calls:  {total_calls}  "
              f"(avg {total_calls / n_out:.2f}/episode, "
              f"{100 * n_with_tool_call / n_out:.1f}% of episodes used >=1 tool)")
        # Stable order: canonical tools first, then unknown / others.
        canonical_order = ["get_landmark_xy", "zoom_in", "render_oblique", "unknown"]
        for name in canonical_order:
            if name in tool_counter:
                cnt = tool_counter[name]
                pct = 100 * cnt / total_calls if total_calls else 0
                print(f"[tools]   {name:18} {cnt:6}  ({pct:5.1f}%)")
        # Any extras not in canonical (shouldn't happen with current env).
        for name, cnt in sorted(tool_counter.items()):
            if name in canonical_order:
                continue
            pct = 100 * cnt / total_calls if total_calls else 0
            print(f"[tools]   {name:18} {cnt:6}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
