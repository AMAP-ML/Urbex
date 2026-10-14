"""Replay images for a dumped trajectory.

Use this at SFT-data-loading time to reconstruct PIL images from a
JSONL record produced by ``CityVLRolloutEnvManager``. Avoids storing
multi-MB PIL bytes in the dump (300x size reduction) at the cost of a
deterministic replay step at load time.

Usage::

    from roll.pipeline.agentic.env.city_grounding.replay_images import (
        replay_trajectory_images,
    )

    with open("sft_teacher_data.jsonl") as f:
        for line in f:
            record = json.loads(line)
            messages_with_images = replay_trajectory_images(
                record,
                bev_root="/data/agent_data",
                multiview_cache_dir="/data/agent_data/verify_region_cache/cityrefer",
            )
            # messages_with_images is the same list but {"type":"image"}
            # placeholders now have a real "image" PIL field again.

Determinism: all three image sources are pure functions of input args
and env resources -- no temperature, no LLM. Re-running on the same
record always produces byte-identical PIL outputs.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from PIL import Image

from .tool_env import CityToolEnv


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool_calls from an assistant message body."""
    out = []
    for m in _TOOL_CALL_RE.finditer(text or ""):
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "name" in payload:
            args = payload.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            out.append({"name": payload["name"], "arguments": args or {}})
    return out


def replay_trajectory_images(
    record: Dict[str, Any],
    *,
    bev_root: str,
    multiview_cache_dir: Optional[str] = None,
    display_max_side: int = 768,
    display_patch: int = 32,
    display_max_pixels: int = 589_824,
    bbox_norm_max: float = 1000.0,
) -> List[Dict[str, Any]]:
    """Reconstruct PIL images into ``record['messages']``.

    Walks the assistant turns in order, parses each <tool_call>, and
    re-runs the inner CityToolEnv to regenerate the same images that
    were stripped at dump time. The first image (BEV) is loaded directly
    from disk.

    Returns a NEW messages list (does not mutate the input record).
    """
    # 1) Build a fresh inner env and reset it on this record's sample.
    sample = {
        "scene_id": record["scene_id"],
        "dataset":  record.get("dataset", "cityrefer"),
        "query":    record.get("query", ""),
        "bev_png":  f"bev_cityrefer/{record['scene_id']}.png",
        # gt_bbox not strictly needed for image replay, but inner env
        # checks its presence -- pass a placeholder.
        "gt_bbox":  record.get("gt_bbox") or [0.0] * 6,
    }
    # CityToolEnv.render uses os.environ["CITY_AGENT_DATA"] for relative
    # paths; or accept absolute path here.
    import os
    os.environ.setdefault("CITY_AGENT_DATA", bev_root)

    env = CityToolEnv(
        display_max_side    = display_max_side,
        display_patch       = display_patch,
        display_max_pixels  = display_max_pixels,
        multiview_cache_dir = multiview_cache_dir,
        bbox_norm_max       = bbox_norm_max,
    )
    global_view = env.reset(sample)
    bev_image = global_view["image"]

    # 2) Walk messages; for each {"type":"image"} placeholder, fill in
    #    the corresponding PIL via env state.
    messages = record["messages"]
    new_messages: List[Dict[str, Any]] = []
    pending_tool_call: Optional[Dict[str, Any]] = None

    for i, msg in enumerate(messages):
        role = msg.get("role")
        new_msg = {k: v for k, v in msg.items() if k != "content"}
        content = msg.get("content")

        if role == "user" and i == 1:
            # Initial user turn -- inject BEV as the lone image.
            new_content: List[Dict[str, Any]] = []
            for item in (content if isinstance(content, list) else [{"type": "text", "text": content}]):
                if isinstance(item, dict) and item.get("type") == "image":
                    new_content.append({"type": "image", "image": bev_image})
                else:
                    new_content.append(dict(item) if isinstance(item, dict) else item)
            new_msg["content"] = new_content

        elif role == "assistant":
            # Parse tool_call to know what tool will produce next image.
            calls = _parse_tool_calls(content if isinstance(content, str) else "")
            pending_tool_call = calls[0] if calls else None
            new_msg["content"] = content

        elif role == "tool":
            # Re-run the corresponding inner tool to regenerate images.
            new_content_list: List[Dict[str, Any]] = []
            replayed_imgs: List[Image.Image] = []
            if pending_tool_call is not None:
                name = pending_tool_call["name"]
                args = pending_tool_call["arguments"]
                tool_method = getattr(env, f"tool_{name}", None)
                if tool_method is not None:
                    try:
                        _, imgs, _ = tool_method(args)
                        replayed_imgs = [im for im in (imgs or []) if im is not None]
                    except Exception:
                        replayed_imgs = []
            img_iter = iter(replayed_imgs)
            for item in (content if isinstance(content, list) else [{"type": "text", "text": content}]):
                if isinstance(item, dict) and item.get("type") == "image":
                    img = next(img_iter, None)
                    if img is not None:
                        new_content_list.append({"type": "image", "image": img})
                    else:
                        # fallback: keep placeholder if we ran out
                        new_content_list.append({"type": "image"})
                else:
                    new_content_list.append(dict(item) if isinstance(item, dict) else item)
            new_msg["content"] = new_content_list
            pending_tool_call = None

        else:
            new_msg["content"] = content

        new_messages.append(new_msg)

    return new_messages
