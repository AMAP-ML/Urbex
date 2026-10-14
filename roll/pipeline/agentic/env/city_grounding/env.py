"""gem.Env adapter following ROLL's OpenAI native tool protocol.

Contract (matches AgentNativeStepEnvManager):
  reset(seed) -> (messages: List[Dict], info: Dict)
      info MUST include ``tools`` -- a list of OpenAI function schemas
      that the env_manager passes to ``apply_chat_template(tools=...)``.

  step(action: str) -> (messages: List[Dict], reward, terminated, truncated, info)
      ``messages`` is the FULL conversation history rendered each turn.
      The env appends its own user/tool replies; the env_manager appends
      the assistant turn separately.

Why the native protocol matters
-------------------------------
When ``tools=`` is passed, the chat template renders ``<tools>...</tools>``,
``<tool_call>{...}</tool_call>`` and ``<tool_response>...</tool_response>``
from structured fields rather than from hand-written prompt text, so a
tool-use SFT checkpoint stays prompt-compatible during RL.

Vision: tool responses with images use OpenAI multimodal content shape
    [{"type": "text", "text": "..."}, {"type": "image", "image": PIL.Image}, ...]
The VL env_manager extracts PIL images via ``processor.apply_chat_template``
(or by walking the messages tree) and feeds them to the collator.
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Optional
import threading

try:
    from gem import Env
except ModuleNotFoundError as exc:
    if exc.name != "gem":
        raise
    # Standalone inference uses the same reset/step implementation without
    # GEM's training registry or base class.
    Env = object

from .query_filter import QueryFilterTracker
from .reward import compute_terminal_reward, reason_to_id, IOU_THRESHOLD
from .tool_env import (
    CityToolEnv,
    OPENAI_TOOL_SCHEMAS,
    SYSTEM_PROMPT,
    TOOL_METHODS,
    parse_answer,
)

logger = logging.getLogger(__name__)

# Tool shaping follows the paper: signed zoom progress, capped render IoU,
# and a penalty for tool errors. Landmark lookup has no positive bonus.
TOOL_ERROR_PENALTY = 0.05    # failed tool call (unknown name / bad args / crash)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# 30B+ models sometimes hallucinate vision-token literals (e.g. "<image>",
# "<|vision_start|>") in <think> or response text. When such an action
# message is fed back through Qwen3-VL chat template, those literals get
# expanded into real <|vision_start|><|image_pad|><|vision_end|> blocks,
# but no corresponding PIL.Image exists -> downstream processor crashes
# with vision-block / image-count mismatch. Strip them before appending.
_VISION_LITERAL_RE = re.compile(
    r"<image>|<\|vision_start\|>|<\|vision_end\|>|<\|image_pad\|>|<\|video_pad\|>"
)

# vLLM's decoded output for Qwen3-VL often INCLUDES the chat-end token
# (`<|im_end|>`, sometimes followed by a newline) at the tail of the
# generation, because `skip_special_tokens=False` is needed to preserve
# `<tool_call>` / `<|vision_start|>` in legitimate places. If we leave it
# in the assistant message content, the next apply_chat_template pass
# wraps the content with ANOTHER `<|im_end|>` (template invariant:
# `<|im_start|>assistant\n{content}<|im_end|>\n`) producing the OOD
# sequence `<|im_end|><|im_end|>` -- which the model has never seen in
# training and starts to drift on. Strip it once at env-side.
_ASSISTANT_TAIL_RE = re.compile(
    r"\s*(?:<\|im_end\|>|<\|endoftext\|>)\s*$"
)


def _sanitize_action(text: str) -> str:
    """Clean a raw LLM action before it enters self._messages.

    1) Strip vision-token literals that the chat template would
       misinterpret as real image placeholders.
    2) Strip trailing `<|im_end|>` / `<|endoftext|>` (with surrounding
       whitespace) so the chat template's own end-of-turn marker does
       not get duplicated when we re-render the conversation.
    """
    if not text:
        return text
    text = _VISION_LITERAL_RE.sub("", text)
    text = _ASSISTANT_TAIL_RE.sub("", text)
    return text


_METRICS_AGG_MODE = {
    # Episode-level (emitted once at terminal step). "mean" is fine because
    # they only appear in the last history entry, so mean = that value.
    "city_success":          "mean",
    "city_iou_2d":           "mean",
    "city_assistant_turns":  "mean",
    "city_reason_id":        "mean",
    "city_iou_term":         "mean",
    "city_dist_reward":      "mean",
    "city_step_term":        "mean",

    # Per-step flags emitted on every tool call. "mean" → fraction of
    # tool-call steps that were valid / errored, averaged over the
    # trajectory's tool-call steps (steps without a tool call don't emit
    # these keys, so they don't dilute).
    "tool_call_valid":       "mean",
    "tool_error":            "mean",

    # Per-tool invocation counts (sum across the trajectory). One unit is
    # added on each call attempt — including failed ones (crash / bad args).
    # `unknown` covers attempts to call a name not in TOOL_METHODS.
    "city_tool_call_get_landmark_xy": "sum",
    "city_tool_call_zoom_in":         "sum",
    "city_tool_call_render_oblique":  "sum",
    "city_tool_call_unknown":         "sum",

    # Per-tool step-reward sums (signed). Sum across the trajectory tells
    # you which tools the model is being rewarded for using. Train-time
    # signal that tells you whether the reward shaping in tool_env.py is
    # actually firing as intended.
    #   zoom_dist:        signed (±0.05/call), positive = zoomed closer
    #   landmark_resolve: 0 (neutral lookup)
    #   render_iou:       0.05 * candidate IoU, capped at 0.05/call
    "city_step_reward_zoom_dist":         "sum",
    "city_step_reward_render_iou":        "sum",
    "city_step_reward_landmark_resolve":  "sum",
}


_USER_PROMPT_TEMPLATE = (
    "Query: {query}\n\n"
    "This is the Bird's-Eye View (BEV) of the scene; the initial view's id "
    "is \"global\". Coordinates throughout are in the SOURCE view's "
    "[0,1000] x [0,1000] integer frame, origin top-left, x->right, y->down "
    "(\"above\"=smaller y, \"below\"=larger y, \"left\"=smaller x).\n\n"
    "For each step, you must conduct a thought section to reason about the "
    "visual clues and plan your next move before calling any tools or "
    "committing the final answer.\n\n"
    "Commit by emitting ONE assistant turn with NO tool_call containing "
    "an <answer> tag with the final bbox and the view_id you committed "
    "on, e.g.:\n"
    "  <answer>{{\"bbox_2d\":[855,605,985,705], \"view_id\":\"Z1\"}}</answer>"
)


def _parse_first_tool_call(text: str) -> Optional[tuple[str, dict]]:
    """Find the FIRST <tool_call>{...}</tool_call> JSON. Returns (name, args)
    or None.
    """
    m = _TOOL_CALL_RE.search(text or "")
    if not m:
        return None
    try:
        payload = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    name = payload.get("name")
    if not isinstance(name, str):
        return None
    args = payload.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return name, args


class CityGroundingEnv(Env):
    """City-grounding env, OpenAI tool protocol."""

    # Per-process sequential counter for deterministic full-dataset coverage
    # in rollout-only mode (group_size=1). Each env instance in the same
    # process increments this counter on reset().
    _seq_counter: int = 0
    _seq_lock = threading.Lock()

    def __init__(
        self,
        data_path: str,
        mode: str = "train",
        seed: Optional[int] = None,
        multiview_cache_dir: Optional[str] = None,
        max_tool_calls: int = 7,
        display_max_side: int = 768,
        display_patch: int = 32,
        display_max_pixels: int = 589_824,
        bbox_norm_max: float = 1000.0,
        max_assistant_turns: int = 8,
        blacklist_file: Optional[str] = None,
        landmark_mode: str = "clean",
        landmark_noise_std_m: float = 10.0,
        landmark_wrong_min_distance_m: float = 50.0,
        landmark_seed: int = 42,
        **_: dict,
    ):
        self.mode = mode
        with open(data_path) as f:
            self._dataset = [json.loads(line) for line in f if line.strip()]
        if not self._dataset:
            raise ValueError(f"empty dataset at {data_path}")
        if mode == "train" and seed is not None:
            random.Random(seed).shuffle(self._dataset)

        self._env = CityToolEnv(
            display_max_side    = display_max_side,
            display_patch       = display_patch,
            display_max_pixels  = display_max_pixels,
            multiview_cache_dir = multiview_cache_dir,
            max_tool_calls      = max_tool_calls,
            bbox_norm_max       = bbox_norm_max,
            landmark_mode       = landmark_mode,
            landmark_noise_std_m = landmark_noise_std_m,
            landmark_wrong_min_distance_m = landmark_wrong_min_distance_m,
            landmark_seed       = landmark_seed,
        )
        self._max_assistant_turns = int(max_assistant_turns)
        self._cur_sample: dict = {}
        self._assistant_turns: int = 0
        self._messages: list[dict] = []
        # Surfaced for the env_manager to inspect after reset_failed.
        self.env_reset_failed: bool = False
        self.env_info: dict = {}
        # Query difficulty filter (shared across env instances in the same process)
        self._query_filter = QueryFilterTracker(blacklist_file) if blacklist_file else None
        self._is_blacklisted: bool = False
        # Expose dataset size for sequential rollout scheduling.
        self.dataset_size = len(self._dataset)

    # gem.Env API ---------------------------------------------------------

    def reset(self, seed: Optional[int] = None):
        if seed is None:
            seed = 0
        idx = seed % len(self._dataset)
        sample = self._dataset[idx]
        self._cur_sample = sample
        self._assistant_turns = 0
        self._is_blacklisted = False

        # Check query difficulty filter before setting up the episode.
        scene_id = sample.get("scene_id", "")
        query = sample.get("query", "")
        if self._query_filter and self._query_filter.is_blacklisted(scene_id, query):
            self._is_blacklisted = True
            logger.info(
                "[city_env] reset BLACKLISTED query: scene=%s query=%s",
                scene_id, query[:60],
            )

        global_view = self._env.reset(sample)

        self._messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": _USER_PROMPT_TEMPLATE.format(query=sample.get("query", ""))},
                {"type": "image", "image": global_view["image"]},
            ]},
        ]
        info = {
            "tools":    OPENAI_TOOL_SCHEMAS,
            "query":    sample.get("query", ""),
            "scene_id": sample.get("scene_id", ""),
            "dataset":  sample.get("dataset", "cityrefer"),
            "landmark_mode": getattr(self._env, "landmark_mode", "clean"),
        }
        self.env_info = {
            "scene_id": info["scene_id"],
            "dataset":  info["dataset"],
            "landmark_mode": info["landmark_mode"],
        }
        return self._messages, info

    def step(self, action):
        # Blacklisted queries are too hard for the current model.
        # Short-circuit immediately to save inference budget.
        if self._is_blacklisted:
            logger.info(
                "[city_env] step BLACKLISTED short-circuit: scene=%s",
                self._cur_sample.get("scene_id", ""),
            )
            return self._terminal_no_answer()

        action = _sanitize_action(action or "")
        self._assistant_turns += 1

        # Append the assistant turn we just received (env_manager could do
        # this too, but recording it here keeps step input/output symmetric).
        self._messages.append({"role": "assistant", "content": action})

        # Last-turn budget enforcement: env_manager force-terminates at
        # `step >= max_steps` regardless of env-emitted `terminated`. If we
        # let _dispatch_tool fire on the last turn, its reward=0 wins and the
        # no_answer step_cost penalty never gets emitted -> all-zero
        # advantages for "ran out of turns" rollouts. Instead: on the last
        # allowed turn, only commit (<answer>) is honored; tool_call gets
        # treated as no_answer so the negative reward is recorded.
        is_last_turn = self._assistant_turns >= self._max_assistant_turns

        answer = parse_answer(action)
        tool_call = _parse_first_tool_call(action)

        if tool_call is not None and not is_last_turn:
            return self._dispatch_tool(*tool_call)
        if answer is not None:
            return self._terminal_commit(answer)
        return self._terminal_no_answer()

    # Internal: tool dispatch ---------------------------------------------

    def _dispatch_tool(self, name: str, args: dict):
        method_name = TOOL_METHODS.get(name)
        if method_name is None:
            return self._tool_error(
                f"unknown tool: {name!r}. Valid tools: {sorted(TOOL_METHODS)}",
                tool_call_valid=False,
                tool_name="unknown",
            )
        try:
            text, images, tinfo = getattr(self._env, method_name)(args)
        except Exception as e:
            logger.exception("tool %s raised: %s", name, e)
            return self._tool_error(
                f"tool {name} crashed: {type(e).__name__}: {e}",
                tool_call_valid=True,
                tool_name=name,
            )

        images = [im for im in (images or []) if im is not None]
        content = [{"type": "text", "text": text}]
        for im in images:
            content.append({"type": "image", "image": im})
        # role="tool": Qwen hermes chat template wraps in <tool_response>...</tool_response>
        # under a user-role <|im_start|> block.
        self._messages.append({"role": "tool", "content": content, "name": name})

        # Verification log: every dispatch records how many images we
        # actually attached to the tool reply. Cross-reference with
        # downstream "_walk_collect_images" log in env_manager to confirm
        # 1-to-1 binding. Counts must be: 0 for get_landmark_xy, 0/1 for
        # render_oblique (cache miss -> 0), 1 for zoom_in.
        n_text_items = sum(1 for c in content if c.get("type") == "text")
        n_image_items = sum(1 for c in content if c.get("type") == "image")
        total_images_so_far = sum(
            1
            for msg in self._messages
            if isinstance(msg.get("content"), list)
            for item in msg["content"]
            if isinstance(item, dict) and item.get("type") == "image" and item.get("image") is not None
        )
        # also count user's BEV image (which is in the initial user msg)
        logger.debug(
            "[city_tool dispatch] turn=%d tool=%s text_items=%d image_items=%d "
            "tool_returned=%d images cumulative_pil_in_messages=%d",
            self._assistant_turns, name, n_text_items, n_image_items,
            len(images), total_images_so_far,
        )

        has_tool_error = "error" in tinfo
        if has_tool_error:
            step_reward = -TOOL_ERROR_PENALTY
        elif "zoom_dist_reward" in tinfo:
            # Directional reward: positive when zooming closer to GT,
            # negative when zooming away. Capped at +/-0.05.
            step_reward = float(tinfo["zoom_dist_reward"])
        elif "render_iou_reward" in tinfo:
            # Successful rendering earns a small candidate-overlap reward.
            step_reward = float(tinfo["render_iou_reward"])
        elif "landmark_resolve_reward" in tinfo:
            # Landmark lookup is neutral; there is no resolve-success bonus.
            step_reward = float(tinfo["landmark_resolve_reward"])
        else:
            step_reward = 0.0

        # Per-tool counter for monitoring. `name` is guaranteed to be in
        # TOOL_METHODS here (unknown branched to _tool_error above).
        # aggregate_metrics will sum this across the trajectory.
        per_tool_count_key = f"city_tool_call_{name}"

        metrics_dict = {
            "tool_call_valid": 1.0,
            "tool_error": 1.0 if has_tool_error else 0.0,
            per_tool_count_key: 1.0,
        }
        # Surface per-tool reward components so we can see the new shaping
        # actually firing in train metrics. Only emit the key when the
        # reward source actually produced this turn (else it dilutes
        # the trajectory sum with zeros).
        if "zoom_dist_reward" in tinfo:
            metrics_dict["city_step_reward_zoom_dist"] = float(tinfo["zoom_dist_reward"])
        if "render_iou_reward" in tinfo:
            metrics_dict["city_step_reward_render_iou"] = float(tinfo["render_iou_reward"])
        if "landmark_resolve_reward" in tinfo:
            metrics_dict["city_step_reward_landmark_resolve"] = float(
                tinfo["landmark_resolve_reward"]
            )
        info = {
            "metrics": metrics_dict,
            "metrics_agg_mode": _METRICS_AGG_MODE,
            "tool_name": name,
            "step_reward": step_reward,
            **{k: v for k, v in tinfo.items() if k != "metrics"},
        }
        return self._messages, step_reward, False, False, info

    def _tool_error(self, message: str, tool_call_valid: bool,
                    tool_name: str = "unknown"):
        self._messages.append({"role": "tool", "content": f"error: {message}"})
        per_tool_count_key = f"city_tool_call_{tool_name}"
        step_reward = -TOOL_ERROR_PENALTY
        metrics_dict = {
            "tool_call_valid": 1.0 if tool_call_valid else 0.0,
            "tool_error":      1.0,
            per_tool_count_key: 1.0,
        }
        info = {
            "metrics": metrics_dict,
            "metrics_agg_mode": _METRICS_AGG_MODE,
            "error": message,
            "step_reward": step_reward,
        }
        return self._messages, step_reward, False, False, info

    # Internal: query filter ----------------------------------------------

    def _update_query_filter(self, iou_2d: float, center_dist: float) -> None:
        """Record episode result in the query difficulty tracker."""
        if self._query_filter is None or self._is_blacklisted:
            return
        scene_id = self._cur_sample.get("scene_id", "")
        query = self._cur_sample.get("query", "")
        newly_blacklisted = self._query_filter.update(
            scene_id, query, iou_2d=iou_2d, center_dist=center_dist
        )
        if newly_blacklisted:
            logger.warning(
                "[city_env] Query BLACKLISTED after %d consecutive hard fails: %s",
                8, query[:60],
            )
        self._query_filter.save()

    # Internal: terminal branches -----------------------------------------

    def _terminal_commit(self, answer: dict):
        score = self._env.score_answer(answer["bbox_2d"], answer.get("view_id", "global"))
        iou = float(score["iou_2d"])
        cdist = float(score.get("center_dist", float("inf")))
        self._update_query_filter(iou_2d=iou, center_dist=cdist)
        rs = compute_terminal_reward(
            iou_2d=iou,
            center_dist=cdist,
            assistant_turns=self._assistant_turns,
            max_assistant_turns=self._max_assistant_turns,
        )
        info = self._terminal_info(
            iou_2d=iou,
            center_dist=cdist,
            success=(iou >= IOU_THRESHOLD),     # success now derived locally
            iou_term=rs["iou_term"],
            dist_reward=rs["dist_reward"],
            step_term=rs["step_term"],
            reason="answer",
            world_bbox=score["world_bbox"], answer=answer,
        )
        return self._messages, float(rs["reward"]), True, False, info

    def _terminal_no_answer(self):
        self._update_query_filter(iou_2d=0.0, center_dist=float("inf"))
        rs = compute_terminal_reward(
            iou_2d=0.0,
            center_dist=float("inf"),
            assistant_turns=self._assistant_turns,
            max_assistant_turns=self._max_assistant_turns,
        )
        info = self._terminal_info(
            iou_2d=0.0,
            center_dist=float("inf"),
            success=False,
            iou_term=rs["iou_term"],
            dist_reward=rs["dist_reward"],
            step_term=rs["step_term"],
            reason="no_answer",
        )
        return self._messages, float(rs["reward"]), True, True, info

    def _terminal_info(self, *, iou_2d, center_dist, success, iou_term, dist_reward,
                       step_term, reason, world_bbox=None, answer=None):
        # Per-tool counters: emit 0.0 for every tool on the terminal step
        # so every episode contributes a value to the cross-trajectory mean.
        # Without this, list_of_dict_to_dict_of_list (protocol.py:93) would
        # silently drop episodes that never invoked a given tool — making
        # the tensorboard mean compute over "episodes that used the tool"
        # instead of "all episodes". A model with 0.5% render_oblique usage
        # would then show ~1.0 on the chart (mean of the 0.5% that used it,
        # each contributing exactly 1) instead of the true 0.005.
        # The "sum" agg_mode adds these zeros to whatever was emitted by
        # tool dispatches earlier in the trajectory, leaving the true count.
        return {
            "metrics": {
                "city_success":          1.0 if success else 0.0,
                "city_iou_2d":           float(iou_2d),
                "city_assistant_turns":  float(self._assistant_turns),
                "city_reason_id":        reason_to_id(reason),
                "city_iou_term":         float(iou_term),
                "city_dist_reward":      float(dist_reward),
                "city_step_term":        float(step_term),
                "tool_call_valid":       1.0,
                "tool_error":            0.0,
                "city_tool_call_get_landmark_xy": 0.0,
                "city_tool_call_zoom_in":         0.0,
                "city_tool_call_render_oblique":  0.0,
                "city_tool_call_unknown":         0.0,
            },
            "metrics_agg_mode": _METRICS_AGG_MODE,
            "city_iou_2d":     float(iou_2d),
            "city_center_dist": float(center_dist),
            "city_success":    bool(success),
            "city_world_bbox": world_bbox,
            "city_answer":     answer,
            "city_reason":     reason,
        }
