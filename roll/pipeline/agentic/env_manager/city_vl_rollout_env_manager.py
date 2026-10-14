"""Rollout-only env_manager for CityGroundingEnv: collect SFT teacher data.

Extends ``CityVLNativeEnvManager`` to dump per-trajectory records to a
JSONL file for trajectories that pass an IoU threshold. Used with
``AgenticRolloutPipeline`` (no training) to harvest high-quality
trajectories from a stronger model (e.g. Qwen3-VL-72B) for SFT.

Per-row JSONL schema::

    {
      "scene_id":       "birmingham_block_12",
      "query":          "the gray car ...",
      "iou_2d":         0.42,
      "success":        true,
      "assistant_turns": 4,
      "world_bbox":     [1226.3, 1141.9, 1230.7, 1166.0],
      "answer":         {"bbox_2d": [389, 115, ...], "view_id": "global"},
      "messages":       [ {role, content}, ... ],   # full conversation
      "tools":          [ openai_schema, ... ],
      "tag":            "CityTool",
      "group_id":       0,
      "env_id":         3,
      "episode_id":     7,
      "exp_name":       "city_tool_rollout_qwen72b",
      "timestamp":      "20260427_173005",
    }

Configure via yaml::

    train_env_manager:
      env_manager_cls: roll.pipeline.agentic.env_manager.city_vl_rollout_env_manager.CityVLRolloutEnvManager
      ...
      env_config:
        ...
      # the manager reads these two from env_config:
      rollout_dump_jsonl: ${output_dir}/sft_teacher_data.jsonl
      rollout_iou_threshold: 0.2

Note: image bytes are NOT dumped (would explode JSONL size). The env
sample's ``bev_png`` path + ``scene_id`` is enough to reconstruct images
on demand at SFT time.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import socket
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from roll.pipeline.agentic.env_manager.base_env_manager import RolloutCache
from roll.pipeline.agentic.env_manager.city_vl_native_env_manager import (
    CityVLNativeEnvManager,
)


# Process-local lock for threads inside the SAME worker. Cross-process
# synchronization is handled by giving each worker its own file (see
# _per_worker_path below) so different processes never share a writer.
_DUMP_LOCK = Lock()


def _per_worker_path(base_path: str) -> str:
    """Insert hostname + pid before the file extension so each worker
    process writes to its own file. Concatenate offline.

    /path/to/sft_teacher_data.jsonl
        -> /path/to/sft_teacher_data.host42-pid12345.jsonl
    """
    root, ext = os.path.splitext(base_path)
    host = socket.gethostname().split(".")[0]
    pid = os.getpid()
    return f"{root}.{host}-pid{pid}{ext or '.jsonl'}"


# ── Per-turn vis recorder ──────────────────────────────────────────────────
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _json_safe(obj: Any) -> Any:
    """Recursively coerce non-JSON-serialisable values into safe forms.
    Drops PIL images entirely (replaced with the literal "[IMAGE]")."""
    try:
        from PIL import Image  # type: ignore
        _PIL_TYPE = Image.Image
    except Exception:
        _PIL_TYPE = type(None)
    if isinstance(obj, _PIL_TYPE):
        return "[IMAGE]"
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # numpy scalars, sets, etc.
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return repr(obj)


def _parse_tool_calls(assistant_text: str) -> List[Dict[str, Any]]:
    """Pull every <tool_call>{...}</tool_call> JSON from assistant text."""
    out: List[Dict[str, Any]] = []
    for m in _TOOL_CALL_RE.finditer(assistant_text or ""):
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "name" in payload:
            out.append(payload)
    return out


def _save_pil(im, path: Path) -> None:
    try:
        im.save(path)
    except Exception:
        pass


def _flatten_messages_for_log(messages: List[Dict[str, Any]]) -> str:
    """Render the message list as a readable text dump (images -> [IMAGE])."""
    lines: List[str] = []
    for msg in messages:
        role = msg.get("role", "?")
        name = msg.get("name")
        header = f"=== {role}" + (f" ({name})" if name else "") + " ==="
        lines.append(header)
        content = msg.get("content")
        if isinstance(content, str):
            lines.append(content)
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    lines.append(repr(item))
                    continue
                t = item.get("type")
                if t == "text":
                    lines.append(item.get("text", ""))
                elif t == "image":
                    lines.append("[IMAGE]")
                else:
                    lines.append(json.dumps(item, ensure_ascii=False))
        else:
            lines.append(repr(content))
        lines.append("")
    return "\n".join(lines)


def _collect_tool_images_from_obs(messages: List[Dict[str, Any]]) -> List[Any]:
    """Walk a message list and collect every PIL image attached to a
    role='tool' message (one tool reply may have multiple images)."""
    imgs: List[Any] = []
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                im = item.get("image")
                if im is not None:
                    imgs.append(im)
    return imgs


def _strip_images_for_dump(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove PIL.Image objects from messages content.

    Replaces ``{"type": "image", "image": <PIL>}`` with
    ``{"type": "image"}`` (just the placeholder marker). Keeps everything
    else verbatim. PIL bytes would balloon the dump and aren't needed
    for SFT (images are reconstructed from the scene_id at training
    time).
    """
    out: List[Dict[str, Any]] = []
    for msg in messages:
        msg_copy: Dict[str, Any] = {"role": msg.get("role")}
        if "name" in msg:
            msg_copy["name"] = msg["name"]
        content = msg.get("content")
        if isinstance(content, str):
            msg_copy["content"] = content
        elif isinstance(content, list):
            new_content: List[Dict[str, Any]] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "image":
                        new_content.append({"type": "image"})
                    else:
                        new_content.append({k: v for k, v in item.items() if k != "image"})
                else:
                    new_content.append(item)
            msg_copy["content"] = new_content
        else:
            msg_copy["content"] = content
        out.append(msg_copy)
    return out


class CityVLRolloutEnvManager(CityVLNativeEnvManager):
    """Rollout-only collector. Dumps successful trajectories to JSONL."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Read rollout-specific knobs from env_config (kept off the
        # standard env init signature so they don't leak into CityToolEnv).
        cfg = self.env_config or {}
        base_path: Optional[str] = cfg.get("rollout_dump_jsonl")
        self.iou_threshold: float = float(cfg.get("rollout_iou_threshold", 0.2))

        if base_path:
            # Each worker process writes to its OWN file -- avoids OSS /
            # FUSE concurrent-append corruption that produces interleaved
            # bytes across processes. Merge offline:
            #   cat sft_teacher_data.*.jsonl > sft_teacher_data.merged.jsonl
            self.dump_jsonl_path: Optional[str] = _per_worker_path(base_path)
            os.makedirs(os.path.dirname(self.dump_jsonl_path), exist_ok=True)
            base_root, base_ext = os.path.splitext(base_path)
            merge_glob = f"{base_root}.*{base_ext or '.jsonl'}"
            self.logger.info(
                f"[rollout-collect] dump path={self.dump_jsonl_path} "
                f"(per-worker; merge later with `cat {merge_glob} > merged.jsonl`), "
                f"iou_threshold={self.iou_threshold}"
            )
        else:
            self.dump_jsonl_path = None
            self.logger.warning(
                "[rollout-collect] rollout_dump_jsonl is empty; "
                "trajectories will NOT be persisted"
            )

        # Per-turn visualization recorder. Mirrors what
        # scripts/eval/api_baseline.py:StepRecorder does for the API
        # path. Disabled if `vis_dir` is not set in env_config; pass
        # `vis_iou_min: -1` (default) to record every episode regardless
        # of outcome, or set a threshold to record only successes.
        self.vis_dir: Optional[Path] = None
        vis_dir_str = cfg.get("vis_dir") or cfg.get("rollout_vis_dir")
        if vis_dir_str:
            self.vis_dir = Path(vis_dir_str)
            self.vis_dir.mkdir(parents=True, exist_ok=True)
            self.vis_iou_min: float = float(cfg.get("vis_iou_min", -1.0))
            self.logger.info(
                f"[rollout-vis] per-episode dump enabled at {self.vis_dir} "
                f"(record episodes with iou >= {self.vis_iou_min})"
            )
        else:
            self.vis_iou_min = -1.0

    # --------------------------------------------------------------

    def formulate_rollouts(self, rollout_cache: RolloutCache):
        # Inspect the per-episode terminal info BEFORE the parent
        # mutates rollout_cache.history (parent pops history[-1]).
        self._maybe_dump_trajectory(rollout_cache)
        self._maybe_dump_vis(rollout_cache)
        return super().formulate_rollouts(rollout_cache)

    # --------------------------------------------------------------

    def _maybe_dump_trajectory(self, rollout_cache: RolloutCache) -> None:
        if not self.dump_jsonl_path:
            return

        # Walk history backwards for the terminal step's metrics.
        terminal: Optional[Dict[str, Any]] = None
        for entry in reversed(rollout_cache.history):
            metrics = entry.get("metrics") or {}
            if "city_iou_2d" in metrics:
                terminal = entry
                break
        if terminal is None:
            return

        iou = float(terminal.get("city_iou_2d", 0.0) or 0.0)
        if iou < self.iou_threshold:
            return

        # Pull the full message list from the latest observation. Our env
        # accumulates messages on self._messages and returns it as
        # observation each step; the most recent observation IS the
        # complete conversation.
        messages = None
        for entry in reversed(rollout_cache.history):
            obs = entry.get("observation")
            if isinstance(obs, list) and obs:
                messages = obs
                break
        if not messages:
            return

        # Nested: the top level holds city_iou_2d / city_success, while
        # city_assistant_turns lives under metrics["metrics"].
        metrics = terminal.get("metrics") or {}
        record = {
            "scene_id":         self.env.env_info.get("scene_id", ""),
            "query":            self.env._cur_sample.get("query", ""),
            "dataset":          self.env.env_info.get("dataset", "cityrefer"),
            "iou_2d":           iou,
            "success":          bool(terminal.get("city_success", False)),
            "assistant_turns":  int(metrics.get("city_assistant_turns", 0) or 0),
            "world_bbox":       terminal.get("city_world_bbox"),
            "answer":           terminal.get("city_answer"),
            "reason":           terminal.get("city_reason", "unknown"),
            "messages":         _strip_images_for_dump(messages),
            "tools":            self.tools,
            "tag":              self.env_config.get("tag", ""),
            "group_id":         self.env_config.get("group_id"),
            "env_id":           self.env_config.get("env_id"),
            "episode_id":       self.episode_id,
            "exp_name":         self.pipeline_config.exp_name,
            "timestamp":        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        }

        line = json.dumps(record, ensure_ascii=False, default=str)
        with _DUMP_LOCK:
            # Explicit utf-8: some container images default to a C locale.
            with open(self.dump_jsonl_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    # --------------------------------------------------------------

    def _maybe_dump_vis(self, rollout_cache: RolloutCache) -> None:
        """Drop one directory per episode with full per-turn artifacts.
        Mirrors scripts/eval/api_baseline.py:StepRecorder layout.

        Layout:
          {vis_dir}/w{worker}_g{group}_e{env}_ep{episode}_{scene}_{oid}/
            sample.json           original dataset row
            summary.json          terminal info (iou, success, reason, ...)
            turn00_{tools}/
              prompt.txt          flattened messages snapshot at turn start
              assistant.txt       model output this turn
              tool_call_{i}.json  parsed <tool_call> payload
              tool_image_{j}.png  images returned by the tool that turn
            turn01_.../
              ...
            trajectory.png        global BEV with GT / zoom / answer overlays
        """
        if self.vis_dir is None:
            return

        # Find terminal entry for the IoU gate.
        terminal: Optional[Dict[str, Any]] = None
        for entry in reversed(rollout_cache.history):
            metrics = entry.get("metrics") or {}
            if "city_iou_2d" in metrics:
                terminal = entry
                break
        iou = float((terminal or {}).get("city_iou_2d", 0.0) or 0.0)
        if iou < self.vis_iou_min:
            return

        sample = getattr(self.env, "_cur_sample", {}) or {}
        env_info = getattr(self.env, "env_info", {}) or {}
        scene = str(sample.get("scene_id") or env_info.get("scene_id") or "scene")
        oid = str(sample.get("object_id") or "oid")
        group_id = self.env_config.get("group_id", "x")
        env_id = self.env_config.get("env_id", "x")
        worker_id = socket.gethostname().split(".")[0] + f"-pid{os.getpid()}"

        ep_dir = self.vis_dir / (
            f"w{worker_id}_g{group_id}_e{env_id}"
            f"_ep{self.episode_id:04d}_{scene}_{oid}"
        )
        try:
            ep_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.warning(f"[rollout-vis] mkdir failed {ep_dir}: {e}")
            return

        # sample.json
        try:
            (ep_dir / "sample.json").write_text(
                json.dumps(_json_safe(sample), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            self.logger.warning(f"[rollout-vis] sample.json: {e}")

        # The env returns ``self._messages`` (a SHARED reference) every
        # step, so all entries' ``observation`` actually point to the same
        # final messages list. Counting "what's new this turn" by length
        # diff would be 0 for every turn → silently never saves images.
        # Instead, walk the final messages list once, pair each assistant
        # message with the role="tool" reply(s) immediately after it, and
        # use that pairing to dump per-turn artifacts.
        history = rollout_cache.history
        # Pick the longest-observed messages list in history as the
        # canonical "final" conversation (some entries may have None obs).
        final_messages: List[Dict[str, Any]] = []
        for entry in reversed(history):
            obs = entry.get("observation")
            if isinstance(obs, list) and len(obs) > len(final_messages):
                final_messages = obs
        # Walk final_messages, pair (assistant_msg, [tool_replies]).
        turn_payloads: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
        prefix_messages_per_turn: List[List[Dict[str, Any]]] = []
        i = 0
        while i < len(final_messages):
            msg = final_messages[i]
            if msg.get("role") == "assistant":
                tool_replies: List[Dict[str, Any]] = []
                j = i + 1
                while j < len(final_messages) and final_messages[j].get("role") == "tool":
                    tool_replies.append(final_messages[j])
                    j += 1
                turn_payloads.append((msg, tool_replies))
                # Prompt for this turn = everything strictly before this assistant.
                prefix_messages_per_turn.append(final_messages[:i])
                i = j
            else:
                i += 1

        for turn_idx, (assistant_msg, tool_replies) in enumerate(turn_payloads):
            llm_response = assistant_msg.get("content", "") if isinstance(
                assistant_msg.get("content"), str
            ) else _flatten_messages_for_log([assistant_msg])
            tool_calls = _parse_tool_calls(llm_response)
            tool_names = [tc.get("name", "?") for tc in tool_calls] or ["commit"]
            suffix = "_".join(n[:16] for n in tool_names)
            sdir = ep_dir / f"turn{turn_idx:02d}_{suffix}"
            try:
                sdir.mkdir(parents=True, exist_ok=True)
                (sdir / "prompt.txt").write_text(
                    _flatten_messages_for_log(prefix_messages_per_turn[turn_idx]),
                    encoding="utf-8",
                )
                (sdir / "assistant.txt").write_text(
                    llm_response, encoding="utf-8",
                )
                for i, tc in enumerate(tool_calls):
                    (sdir / f"tool_call_{i}.json").write_text(
                        json.dumps(_json_safe(tc), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                # tool_response_<i>.txt + tool_image_<j>.png from the
                # role="tool" replies that immediately followed this
                # assistant turn.
                tool_response_idx = 0
                for msg in tool_replies:
                    content = msg.get("content")
                    if isinstance(content, str):
                        response_text = content
                    elif isinstance(content, list):
                        parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                parts.append(item.get("text", ""))
                        response_text = "\n".join(parts)
                    else:
                        response_text = repr(content)
                    (sdir / f"tool_response_{tool_response_idx}.txt").write_text(
                        response_text, encoding="utf-8",
                    )
                    tool_response_idx += 1
                tool_imgs = _collect_tool_images_from_obs(tool_replies)
                for j, im in enumerate(tool_imgs):
                    _save_pil(im, sdir / f"tool_image_{j:02d}.png")
            except Exception as e:
                self.logger.warning(
                    f"[rollout-vis] turn {turn_idx}: {e}"
                )

        # summary.json
        try:
            ep_summary = {
                "scene_id":        scene,
                "object_id":       oid,
                "query":           sample.get("query", ""),
                "iou_2d":          iou,
                "success":         bool((terminal or {}).get("city_success", False)),
                "reason":          (terminal or {}).get("city_reason", "unknown"),
                "world_bbox":      (terminal or {}).get("city_world_bbox"),
                "answer":          (terminal or {}).get("city_answer"),
                "assistant_turns": int(((terminal or {}).get("metrics") or {})
                                       .get("city_assistant_turns", 0) or 0),
                "episode_id":      self.episode_id,
                "group_id":        group_id,
                "env_id":          env_id,
                "exp_name":        self.pipeline_config.exp_name,
                "timestamp":       datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            }
            (ep_dir / "summary.json").write_text(
                json.dumps(_json_safe(ep_summary), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            self.logger.warning(f"[rollout-vis] summary: {e}")

        # trajectory.png — global BEV with GT / zoom views / answer overlay.
        try:
            self._draw_trajectory(ep_dir, sample, terminal or {})
        except Exception as e:
            self.logger.warning(f"[rollout-vis] trajectory: {e}")

    def _draw_trajectory(
        self,
        ep_dir: Path,
        sample: Dict[str, Any],
        terminal: Dict[str, Any],
    ) -> None:
        """Render the global BEV with GT (red), zoom views (lime), and
        the model's final answer (blue) overlaid. Skips silently if the
        env doesn't expose a global view (e.g. reset_failed)."""
        from PIL import ImageDraw  # type: ignore

        env = getattr(self, "env", None)
        if env is None:
            return
        view_registry = getattr(env._env, "view_registry", None) \
            if hasattr(env, "_env") else None
        if not view_registry or "global" not in view_registry:
            return
        global_view = view_registry["global"]
        canvas = global_view["image"].copy()
        mapping = global_view["mapping"]
        img_w = global_view["img_w"]
        img_h = global_view["img_h"]
        draw = ImageDraw.Draw(canvas)

        def w2p(wx: float, wy: float) -> Tuple[float, float]:
            px = (wx - mapping["x_min"]) / mapping["res_x"]
            py = (mapping["y_max"] - wy) / mapping["res_y"]
            return (max(0.0, min(img_w, px)), max(0.0, min(img_h, py)))

        def rect_world(wb: list, color: str, width: int = 2,
                       label: Optional[str] = None) -> None:
            x0, y0 = w2p(wb[0], wb[3])
            x1, y1 = w2p(wb[2], wb[1])
            draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
            if label:
                draw.text((x0 + 3, y0 + 3), label, fill=color)

        # GT (red).
        gt = sample.get("gt_bbox")
        if gt and len(gt) >= 5:
            cx, cy, _, dx, dy = [float(v) for v in gt[:5]]
            rect_world([cx - dx / 2, cy - dy / 2, cx + dx / 2, cy + dy / 2],
                       "red", width=3, label="GT")

        # Zoom views (lime).
        for vid, v in view_registry.items():
            if vid == "global":
                continue
            rect_world(v["world_bounds"], "lime", width=2, label=vid)

        # Model answer (blue).
        wb = terminal.get("city_world_bbox")
        if wb and len(wb) == 4:
            rect_world(wb, "dodgerblue", width=3, label="answer")

        # Footer text.
        try:
            iou = float(terminal.get("city_iou_2d", 0.0) or 0.0)
            reason = terminal.get("city_reason", "")
            turns = int(((terminal.get("metrics") or {})
                         .get("city_assistant_turns", 0)) or 0)
            draw.text((6, max(0, img_h - 30)),
                      f"iou={iou:.3f}  reason={reason}  turns={turns}",
                      fill="white")
        except Exception:
            pass

        canvas.save(ep_dir / "trajectory.png")
