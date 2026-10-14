#!/usr/bin/env python3
"""Rollout city-grounding trajectories via Alibaba Cloud DashScope API.

Uses Qwen3-VL (e.g. qwen3-vl-plus / qwen3-vl-235b-a22b-instruct) to collect
SFT teacher data. Local CityToolEnv executes tools so the API model only
generates assistant turns.

Outputs per-trajectory JSONL records compatible with
CityVLRolloutEnvManager's dump format.

Usage::

    # Qwen baseline (DashScope)
    export DASHSCOPE_API_KEY=sk-xxx
    python scripts/eval/api_baseline.py \
        --data_path /data/agent_data/cityrefer_val_ND.jsonl \
        --output_dir ./output/api_baseline_qwen_val_ND \
        --model qwen3-vl-plus \
        --max_workers 4

    # GPT-5 baseline (OpenAI native endpoint)
    export OPENAI_API_KEY=sk-xxx
    python scripts/eval/api_baseline.py \
        --provider openai \
        --base_url https://api.openai.com/v1 \
        --model gpt-5 \
        --reasoning_effort medium \
        --no_thinking \
        --bbox_order xyxy \
        --max_workers 2 \
        --data_path /data/agent_data/cityrefer_val_ND.jsonl \
        --output_dir ./output/api_baseline_gpt5_val_ND

    # Gemini-3 baseline (Google OpenAI-compat shim)
    export GEMINI_API_KEY=...
    python scripts/eval/api_baseline.py \
        --provider gemini \
        --base_url https://generativelanguage.googleapis.com/v1beta/openai/ \
        --model gemini-3-pro \
        --bbox_order yxyx \
        --no_thinking \
        --max_workers 2 \
        --data_path /data/agent_data/cityrefer_val_ND.jsonl \
        --output_dir ./output/api_baseline_gemini3_val_ND

    Defaults are aligned with scripts/configs/rollout.yaml so the API
    baseline and the local-model GRPO baseline are scored under matching
    inference settings:
        --max_turns 8 --max_tool_calls 7 --temperature 0.0
        --iou_threshold 0.0  --bbox_order xyxy
        --multiview_cache_dir <verify_region_cache/cityrefer>
    Override any of them with --flag if you need a different sweep.

Checkpoint / resume:
    Each invocation writes unique append-only worker shards. On restart, the
    script skips (scene_id, query) pairs already present in any shard and
    rebuilds merged.jsonl and dedup.jsonl from all valid records.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import random
import re
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

try:
    from openai import OpenAI
except ImportError as exc:
    raise ImportError(
        "openai package is required. Install: pip install openai"
    ) from exc

# ── Import ROLL's CityToolEnv without dragging in heavy deps ────────────────
import importlib.util as _imp_util
import types

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_ENV_DIR = os.path.join(_REPO_ROOT, "roll", "pipeline", "agentic", "env")

# 1) Stubs for heavy side-effect deps
_sys_mods = sys.modules
if "gem" not in _sys_mods:
    _gem = types.ModuleType("gem")
    _gem.Env = object  # type: ignore[attr-defined]
    _sys_mods["gem"] = _gem

# 2) Make city_grounding a real package so relative imports work
_cg_dir = os.path.join(_ENV_DIR, "city_grounding")
_cg_pkg = types.ModuleType("city_grounding")
_cg_pkg.__path__ = [_cg_dir]  # type: ignore[attr-defined]
_sys_mods["city_grounding"] = _cg_pkg

# 3) Load intra-package deps in order (iou → render → tool_env)
for _sub in ("iou", "render", "tool_env"):
    _spec = _imp_util.spec_from_file_location(
        f"city_grounding.{_sub}", os.path.join(_cg_dir, f"{_sub}.py")
    )
    _mod = _imp_util.module_from_spec(_spec)
    _sys_mods[f"city_grounding.{_sub}"] = _mod
    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]

CityToolEnv = _sys_mods["city_grounding.tool_env"].CityToolEnv  # type: ignore[attr-defined]
OPENAI_TOOL_SCHEMAS = _sys_mods["city_grounding.tool_env"].OPENAI_TOOL_SCHEMAS  # type: ignore[attr-defined]
SYSTEM_PROMPT = _sys_mods["city_grounding.tool_env"].SYSTEM_PROMPT  # type: ignore[attr-defined]
TOOL_METHODS = _sys_mods["city_grounding.tool_env"].TOOL_METHODS  # type: ignore[attr-defined]
parse_answer = _sys_mods["city_grounding.tool_env"].parse_answer  # type: ignore[attr-defined]

# Load query filter tracker (same package, so already on sys.path)
_qf_spec = _imp_util.spec_from_file_location(
    "city_grounding.query_filter",
    os.path.join(_cg_dir, "query_filter.py"),
)
_qf_mod = _imp_util.module_from_spec(_qf_spec)
sys.modules["city_grounding.query_filter"] = _qf_mod
_qf_spec.loader.exec_module(_qf_mod)
QueryFilterTracker = _qf_mod.QueryFilterTracker

# Tool descriptions injected into system prompt for native-text protocol
# (API tool role does not support images, so we don't use API tools= parameter)
_TOOL_DESC_TEXT = "\n".join(
    f"- {s['function']['name']}: {s['function']['description']}"
    for s in OPENAI_TOOL_SCHEMAS
)

logger = logging.getLogger(__name__)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_USER_PROMPT_TEMPLATE = (
    "Query: {query}\n\n"
    "This is the Bird's-Eye View (BEV) of the scene; the initial view's id "
    'is "global". Coordinates throughout are in the SOURCE view\'s '
    "[0,1000] x [0,1000] integer frame, origin top-left, x->right, y->down "
    '("above"=smaller y, "below"=larger y, "left"=smaller x).\n\n'
    "For each step, you must conduct a thought section to reason about the "
    "visual clues and plan your next move before calling any tools or "
    "committing the final answer.\n\n"
    "Commit by emitting ONE assistant turn with NO tool_call containing "
    "an <answer> tag with the final bbox and the view_id you committed "
    'on, e.g.:\n'
    '  <answer>{{"bbox_2d":[855,605,985,705], "view_id":"Z1"}}</answer>'
)


# ── API helpers ────────────────────────────────────────────────────────────


def _pil_to_base64(img: Image.Image) -> str:
    """Convert PIL.Image to base64 data URL."""
    buf = BytesIO()
    fmt = "PNG" if img.mode in ("RGBA", "P") else "JPEG"
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _build_api_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert internal messages (with PIL.Image objects) to OpenAI API format.

    The API uses:
      - role=system / user / assistant / tool
      - tool responses need tool_call_id
      - images as image_url blocks
    """
    api_messages: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            api_messages.append({"role": "system", "content": content})
            continue

        if role == "user":
            api_content: List[Dict[str, Any]] = []
            if isinstance(content, str):
                api_content.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            api_content.append(
                                {"type": "text", "text": item.get("text", "")}
                            )
                        elif item.get("type") == "image":
                            img = item.get("image")
                            if isinstance(img, Image.Image):
                                api_content.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": _pil_to_base64(img),
                                            "detail": "high",
                                        },
                                    }
                                )
            api_messages.append({"role": "user", "content": api_content})
            continue

        if role == "assistant":
            api_messages.append({"role": "assistant", "content": content or ""})
            continue

        if role == "tool":
            # OpenAI tool response format
            tool_call_id = msg.get("tool_call_id", "")
            tool_content = ""
            if isinstance(content, str):
                tool_content = content
            elif isinstance(content, list):
                # Extract text from mixed content
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif isinstance(item, dict) and item.get("type") == "image":
                        # API tool role cannot include images directly;
                        # images are handled by converting tool response
                        # to user message in the next round.
                        pass
                tool_content = "\n".join(parts)
            api_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_content,
                }
            )
            continue

    return api_messages


def _parse_first_tool_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
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


def _tool_call_to_text(name: str, arguments: Dict[str, Any]) -> str:
    """Convert a parsed tool call back to the <tool_call> text format."""
    return (
        f'<tool_call>{{"name": "{name}", '
        f'"arguments": {json.dumps(arguments, ensure_ascii=False)}}}</tool_call>'
    )


def _swap_bbox(bbox: Any, bbox_order: str) -> Any:
    """Reorder a 4-element bbox into the env's canonical [x0,y0,x1,y1].

    Why this exists
    ---------------
    Different VLM families emit bbox coordinates in different conventions:
      * Qwen-VL / GPT family — ``[x0, y0, x1, y1]``
      * **Gemini family** —    ``[y0, x0, y1, x1]``  (Y FIRST!)
    The CityToolEnv assumes xyxy throughout. If we feed Gemini's yxyx
    output straight into the env, ``_norm_to_world_bbox`` swaps x and y,
    bbox lands in the wrong region, IoU is silently near zero, and we
    blame the model. This helper swaps once, at the model→env boundary,
    so the env stays provider-agnostic.

    Returns the bbox unchanged if it's not a length-4 list/tuple of
    numerics, so a malformed payload still hits the same env-side
    "invalid bbox" error path.
    """
    if bbox_order != "yxyx":
        return bbox
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return bbox
    try:
        y0, x0, y1, x1 = bbox
    except (TypeError, ValueError):
        return bbox
    return [x0, y0, x1, y1]


def _swap_args_bbox(args: Dict[str, Any], bbox_order: str) -> Dict[str, Any]:
    """Apply ``_swap_bbox`` to the ``bbox_2d`` / ``bbox`` field of a
    tool-call argument dict, returning a NEW dict (caller may pass the
    original args dict to the env / log without mutation surprise)."""
    if bbox_order != "yxyx" or not isinstance(args, dict):
        return args
    out = dict(args)
    for k in ("bbox_2d", "bbox"):
        if k in out:
            out[k] = _swap_bbox(out[k], bbox_order)
    return out


def _is_openai_reasoning_model(model: str) -> bool:
    """True for models that accept the OpenAI ``reasoning_effort`` kwarg.

    Covers GPT-5 family and the o-series (o1/o3/o4...). Conservative match:
    if a future provider name slips through, the API returns 400 and the
    user gets a clear error rather than silent drop.
    """
    m = (model or "").lower()
    return (
        m.startswith("gpt-5")
        or m.startswith("o1")
        or m.startswith("o3")
        or m.startswith("o4")
    )


def _is_gemini_model(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith("gemini")


_PROVIDER_API_KEY_VARS = {
    "dashscope": ("DASHSCOPE_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "local": (),
}


def _resolve_api_key(provider: str) -> str:
    env_vars = _PROVIDER_API_KEY_VARS[provider]
    for env_var in env_vars:
        api_key = os.environ.get(env_var)
        if api_key:
            return api_key
    if provider == "local":
        return "EMPTY"
    joined = " or ".join(env_vars)
    raise ValueError(f"No API key found for provider {provider!r}. Set {joined}.")


class DashScopeClient:
    """Thin wrapper around an OpenAI-compatible chat-completions endpoint.

    Despite the historical name, this client works against any provider
    that speaks the OpenAI chat-completions protocol:

      * Qwen on DashScope (``compatible-mode/v1``)
      * OpenAI native (``api.openai.com/v1``)            — GPT-4o, GPT-5
      * Gemini OpenAI compat shim                         — gemini-2.x / 3
        (``generativelanguage.googleapis.com/v1beta/openai/``)
      * Any other OpenAI-compat gateway                   — vLLM, OpenRouter

    Provider-specific knobs are auto-routed by model name:
      * Qwen → ``extra_body={"enable_thinking": ...}``
      * GPT-5 / o-series → ``reasoning_effort=...``
      * Other models → no extra args (server defaults)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen3-vl-plus",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        enable_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
        provider: str = "dashscope",
    ):
        self.api_key = api_key or _resolve_api_key(provider)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            max_retries=5,
            timeout=120,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.enable_thinking = enable_thinking
        self.last_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        # "none" / None means: do not pass reasoning_effort — let the server
        # use its default (typically "medium" for GPT-5 / o-series).
        if reasoning_effort and reasoning_effort.lower() not in {"none", ""}:
            self.reasoning_effort: Optional[str] = reasoning_effort.lower()
        else:
            self.reasoning_effort = None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        forced_tool_name: Optional[str] = None,
    ) -> Any:
        """Call the API with standard OpenAI function-calling.

        Token-limit kwarg
        -----------------
        We always pass the legacy ``max_tokens``. OpenAI's native API
        deprecates this in favor of ``max_completion_tokens`` for GPT-5 /
        o-series, but most internal proxies (e.g. the Amap gateway) still
        speak the legacy contract and 500 on the new kwarg. If you hit an
        OpenAI-native endpoint that demands the new name, set
        ``self.max_tokens = None`` upstream and add ``max_completion_tokens``
        to ``EXTRA_ARGS`` via the wrapper script — kept narrow to avoid
        breaking the common case.

        Provider-specific reasoning knobs are routed by model name:
          * Qwen → ``extra_body.enable_thinking``
          * GPT-5 / o-series → optional ``reasoning_effort``
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if tools:
            kwargs["tools"] = tools
            if forced_tool_name:
                kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": forced_tool_name},
                }
            else:
                kwargs["tool_choice"] = "auto"
        # Qwen3.x: server-side toggle for reasoning_content.
        if self.model.lower().startswith("qwen"):
            kwargs["extra_body"] = {"enable_thinking": self.enable_thinking}
        # GPT-5 / o-series: optional reasoning_effort. Only pass when the
        # user explicitly opts in (default "none") so we don't trip
        # gateways that silently 500 on unrecognized kwargs.
        if self.reasoning_effort and _is_openai_reasoning_model(self.model):
            kwargs["reasoning_effort"] = self.reasoning_effort

        completion = self.client.chat.completions.create(**kwargs)
        usage = getattr(completion, "usage", None)
        self.last_usage = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }

        # Defensive: some gateways (Amap internal, OpenAI-compat shims) can
        # return 200-OK with ``choices=None`` or ``choices=[]`` when the
        # request was almost-valid (e.g. content-filtered, refused, or hit
        # an internal token cap). Surface a clear error instead of letting
        # ``choices[0]`` raise ``'NoneType' object is not subscriptable``.
        choices = getattr(completion, "choices", None)
        if not choices:
            err_dump = ""
            try:
                err_dump = completion.model_dump_json()[:500]  # type: ignore[attr-defined]
            except Exception:
                try:
                    err_dump = repr(completion)[:500]
                except Exception:
                    err_dump = "<unprintable>"
            raise RuntimeError(
                f"API returned no choices (likely content-filter / refusal "
                f"/ gateway issue). Raw response prefix: {err_dump}"
            )
        return choices[0].message


# ── Rollout logic ──────────────────────────────────────────────────────────


def run_episode(
    sample: Dict[str, Any],
    client: DashScopeClient,
    env: CityToolEnv,
    max_assistant_turns: int = 8,
    recorder=None,
    bbox_order: str = "xyxy",
    forced_tool_sequence: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Run one episode with the API model.

    ``bbox_order`` selects how the model emits bbox coordinates:
      * ``"xyxy"`` (default) — Qwen-VL / GPT family convention; bbox is
        ``[x0, y0, x1, y1]`` and passed straight through to the env.
      * ``"yxyx"`` — Gemini convention; we swap to ``[x0, y0, x1, y1]``
        before feeding the env so CityToolEnv stays provider-agnostic.

    Returns a trajectory record dict or None if it failed before termination.
    """
    scene_id = sample.get("scene_id", "")
    query = sample.get("query", "")

    # Reset env
    global_view = env.reset(sample)
    bev_image = global_view["image"]

    # Build conversation — identical structure to the GRPO rollout.
    # SYSTEM_PROMPT is short; the chat template injects <tools>...</tools>
    # when ``tools=`` is passed to ``apply_chat_template`` (GRPO) or to
    # the API server (here).
    user_prompt = _USER_PROMPT_TEMPLATE.format(query=query)
    if forced_tool_sequence:
        schedule = " -> ".join(forced_tool_sequence)
        user_prompt += (
            "\n\nControlled fixed-policy evaluation: call exactly one tool per "
            f"turn in this order: {schedule}. After the final scheduled tool, "
            "submit the answer immediately without another tool call. You must "
            "still infer every tool argument from the query and observations."
        )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt,
                },
                {"type": "image", "image": bev_image},
            ],
        },
    ]

    answer: Optional[Dict[str, Any]] = None
    assistant_turns = 0
    token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    schedule_violation = False

    while assistant_turns < max_assistant_turns:
        assistant_turns += 1
        is_last_turn = assistant_turns >= max_assistant_turns
        schedule_index = assistant_turns - 1
        forced_tool_name = None
        turn_tools = OPENAI_TOOL_SCHEMAS
        if forced_tool_sequence:
            if schedule_index < len(forced_tool_sequence):
                forced_tool_name = forced_tool_sequence[schedule_index]
                turn_tools = [
                    schema for schema in OPENAI_TOOL_SCHEMAS
                    if schema["function"]["name"] == forced_tool_name
                ]
            else:
                turn_tools = None

        # Snapshot prompt for recorder
        prompt_snapshot = list(messages) if recorder is not None else None

        # Call API with native OpenAI function-calling (tools=).
        # Qwen3 on DashScope returns reasoning_content when enable_thinking
        # is set via extra_body.
        api_messages = _build_api_messages(messages)
        try:
            msg = client.chat(
                api_messages,
                tools=turn_tools,
                forced_tool_name=forced_tool_name,
            )
        except Exception as e:
            logger.warning("API call failed for scene_id=%s: %s", scene_id, e)
            return None
        for key in token_usage:
            token_usage[key] += client.last_usage[key]

        # Qwen3 on DashScope returns thinking in reasoning_content when tools= is used.
        # msg.content is often empty on tool-call turns; merge both fields.
        reasoning = (getattr(msg, "reasoning_content", "") or "").strip()
        content_raw = (msg.content or "").strip()
        parts = [p for p in [reasoning, content_raw] if p]
        assistant_content = "\n".join(parts)

        # --- Commit turn (no tool_calls) ---
        if not getattr(msg, "tool_calls", None):
            if forced_tool_name:
                schedule_violation = True
            messages.append({"role": "assistant", "content": assistant_content})
            answer = parse_answer(assistant_content)
            if answer is not None:
                reason = "answer"
            else:
                reason = "no_answer"
            if recorder is not None:
                recorder.log_turn(prompt_snapshot or [], assistant_content, [], [])
            break

        # --- Tool turn: standard OpenAI tool_calls ---
        tool_calls_api = [
            {
                "id": getattr(tc, "id", ""),
                "name": getattr(getattr(tc, "function", None), "name", ""),
                "arguments": getattr(getattr(tc, "function", None), "arguments", "{}"),
            }
            for tc in msg.tool_calls
        ]
        # Convert to <tool_call> text tags for storage (GRPO format).
        tc_text = "\n".join(
            f'<tool_call>{{"name": {json.dumps(tc["name"])}, '
            f'"arguments": {tc["arguments"]}}}</tool_call>'
            for tc in tool_calls_api
        )
        assistant_full = (assistant_content + "\n" + tc_text).strip()
        messages.append({"role": "assistant", "content": assistant_full})

        # Only one tool call per turn for our env.
        if tool_calls_api and not is_last_turn:
            tc = tool_calls_api[0]
            name = tc["name"]
            if forced_tool_name and name != forced_tool_name:
                schedule_violation = True
                logger.warning(
                    "Fixed schedule violation for scene_id=%s: expected %s, got %s",
                    scene_id,
                    forced_tool_name,
                    name,
                )
                break
            try:
                args = json.loads(tc["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}

            method_name = TOOL_METHODS.get(name)
            if method_name is None:
                err_msg = (
                    f"error: unknown tool: {name!r}. "
                    f"Valid tools: {sorted(TOOL_METHODS)}"
                )
                messages.append({"role": "user", "content": err_msg})
                if recorder is not None:
                    recorder.log_turn(
                        prompt_snapshot or [], assistant_full,
                        [{"name": name, "args": args}],
                        [(err_msg, [], {"error": "unknown_tool"})],
                    )
                continue

            # Provider bbox-order swap (Gemini's [y,x,y,x] -> env's [x,y,x,y]).
            # No-op when bbox_order=="xyxy".
            env_args = _swap_args_bbox(args, bbox_order)
            try:
                text, images, tinfo = getattr(env, method_name)(env_args)
            except Exception as e:
                logger.exception("Tool %s crashed: %s", name, e)
                err_msg = f"error: tool {name} crashed: {type(e).__name__}: {e}"
                messages.append({"role": "user", "content": err_msg})
                if recorder is not None:
                    recorder.log_turn(
                        prompt_snapshot or [], assistant_full,
                        [{"name": name, "args": args}],
                        [(err_msg, [], {"error": "exception", "detail": str(e)})],
                    )
                continue

            images = [im for im in (images or []) if im is not None]
            content = [{"type": "text", "text": text}]
            for im in images:
                content.append({"type": "image", "image": im})
            messages.append({"role": "user", "content": content})

            if recorder is not None:
                recorder.log_turn(
                    prompt_snapshot or [], assistant_full,
                    [{"name": name, "args": args}],
                    [(text, images, tinfo)],
                )
        else:
            # Last turn with tool_call -> treat as no_answer
            if recorder is not None:
                recorder.log_turn(prompt_snapshot or [], assistant_full, [], [])
            break

    # Score answer. Apply the same bbox-order swap we used for tool args
    # so a Gemini-style [y,x,y,x] commit is graded against the right region.
    if answer is None:
        score = {
            "world_bbox": None,
            "iou_2d": 0.0,
            "center_dist": float("inf"),
            "success": False,
        }
        reason = "no_answer"
    else:
        scored_bbox = _swap_bbox(answer["bbox_2d"], bbox_order)
        score = env.score_answer(scored_bbox, answer.get("view_id", "global"))
        reason = "answer"

    # Strip PIL images for serialization
    stripped_messages = _strip_images(messages)

    ep_result = {
        "scene_id": scene_id,
        "query": query,
        "dataset": sample.get("dataset", "cityrefer"),
        "iou_2d": float(score["iou_2d"]),
        "center_dist": float(score.get("center_dist", float("inf"))),
        "success": bool(score.get("success", False)),
        "assistant_turns": assistant_turns,
        "world_bbox": score.get("world_bbox"),
        "answer": answer,
        "reason": reason,
        "policy_mode": "fixed" if forced_tool_sequence else "free",
        "forced_tool_sequence": forced_tool_sequence or [],
        "fixed_schedule_violation": schedule_violation,
        "landmark_condition": {
            "mode": env.landmark_mode,
            "noise_std_m": env.landmark_noise_std_m,
            "wrong_min_distance_m": env.landmark_wrong_min_distance_m,
            "seed": env.landmark_seed,
        },
        "token_usage": token_usage,
        "messages": stripped_messages,
        "tools": OPENAI_TOOL_SCHEMAS,
        "tag": "CityTool",
    }

    if recorder is not None:
        recorder.log_end(ep_result, env)

    return ep_result


def _strip_images(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove PIL.Image objects from messages content."""
    out: List[Dict[str, Any]] = []
    for msg in messages:
        msg_copy: Dict[str, Any] = {"role": msg.get("role")}
        for k in ("name", "tool_call_id"):
            if k in msg:
                msg_copy[k] = msg[k]
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
                        new_content.append(
                            {k: v for k, v in item.items() if k != "image"}
                        )
                else:
                    new_content.append(item)
            msg_copy["content"] = new_content
        else:
            msg_copy["content"] = content
        out.append(msg_copy)
    return out


# ── Visualization recorder ────────────────────────────────────────────────

from pathlib import Path


def _flatten_messages_for_log(messages: list) -> str:
    """Convert message list to human-readable text for prompt.txt."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content")
        if "tool_calls" in msg:
            tcs = "\n".join(
                f"  tool_call: {tc.get('function', {}).get('name')} "
                f"args={tc.get('function', {}).get('arguments')}"
                for tc in (msg.get("tool_calls") or [])
            )
        else:
            tcs = ""
        if isinstance(content, str):
            parts.append(f"=== {role} ===\n{content}\n{tcs}".rstrip())
            continue
        sub: list[str] = []
        for p in (content or []):
            t = p.get("type")
            if t in ("text",):
                sub.append(p.get("text", ""))
            elif t in ("image", "image_url"):
                sub.append("(image inlined — saved separately)")
            else:
                sub.append(f"({t}: {p})")
        parts.append(
            f"=== {role} ===\n" + "\n".join(sub) + ("\n" + tcs if tcs else "")
        )
    return "\n\n".join(parts)


def _save_pil(img: Image.Image, dst: Path) -> None:
    try:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(dst, "PNG", optimize=True)
    except Exception as e:
        logger.warning("recorder: save %s failed: %s", dst, e)


def _json_safe(obj):
    """JSON-friendly fallback for env objects, numpy scalars, PIL etc."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Image.Image):
        return f"<PIL.Image {obj.size}>"
    try:
        import numpy as _np
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
    except Exception:
        pass
    return repr(obj)[:200]


class StepRecorder:
    """Drops one directory per episode + one subdir per assistant turn.

    Layout when active:
      <vis_dir>/ep{NNNN}_{scene}_{oid}/
        sample.json
        turn{NN}_{tool[_tool2...]}/
          prompt.txt          full messages snapshot
          assistant.txt       model output this turn
          tool_call_{i}.json  parsed args
          tool_response_{i}.txt
          tool_image_{i}_{j}.png
          tool_info_{i}.json
        summary.json
        trajectory.png      global BEV with GT / zoom / answer overlays
    """

    def __init__(self, vis_root: Path, ep_idx: int, sample: dict,
                 worker_id: int = 0):
        scene = str(sample.get("scene_id", "scene"))
        oid = str(sample.get("object_id", "oid"))
        self.dir = vis_root / f"w{worker_id}_ep{ep_idx:04d}_{scene}_{oid}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sample = sample
        try:
            (self.dir / "sample.json").write_text(
                json.dumps(_json_safe(sample), ensure_ascii=False, indent=2)
            )
        except Exception as e:
            logger.warning("recorder: sample.json write failed: %s", e)
        self.turn_index = 0

    def log_turn(
        self,
        prompt_messages: list,
        assistant_text: str,
        tool_calls: list,
        tool_results: list,
    ) -> None:
        """tool_results: list of (text, images_list, info_dict)."""
        names = [tc.get("name", "?") for tc in tool_calls] or ["commit"]
        suffix = "_".join(n[:16] for n in names)
        sdir = self.dir / f"turn{self.turn_index:02d}_{suffix}"
        sdir.mkdir(parents=True, exist_ok=True)
        try:
            (sdir / "prompt.txt").write_text(
                _flatten_messages_for_log(prompt_messages)
            )
            (sdir / "assistant.txt").write_text(assistant_text or "")
            for i, tc in enumerate(tool_calls):
                (sdir / f"tool_call_{i}.json").write_text(
                    json.dumps(_json_safe(tc), ensure_ascii=False, indent=2)
                )
            for i, (text, images, info) in enumerate(tool_results):
                (sdir / f"tool_response_{i}.txt").write_text(text or "")
                (sdir / f"tool_info_{i}.json").write_text(
                    json.dumps(_json_safe(info), ensure_ascii=False, indent=2)
                )
                for j, im in enumerate(images or []):
                    _save_pil(im, sdir / f"tool_image_{i}_{j:02d}.png")
        except Exception as e:
            logger.warning("recorder turn%d: %s", self.turn_index, e)
        self.turn_index += 1

    def log_end(self, ep_result: dict, env: CityToolEnv) -> None:
        try:
            (self.dir / "summary.json").write_text(
                json.dumps(
                    {k: _json_safe(v) for k, v in ep_result.items()
                     if k != "messages"},
                    ensure_ascii=False, indent=2,
                )
            )
        except Exception as e:
            logger.warning("recorder summary: %s", e)
        try:
            self._draw_trajectory(env, ep_result)
        except Exception as e:
            logger.warning("recorder trajectory: %s", e)

    # ── Trajectory overlay on the global BEV ─────────────────────────────────
    _GT_COLOUR = "red"
    _ZOOM_COLOUR = "lime"
    _OBLIQUE_COLOUR = "yellow"
    _LANDMARK_COLOUR = "orange"
    _ANSWER_COLOUR = "dodgerblue"

    def _draw_trajectory(self, env: CityToolEnv, ep_result: dict) -> None:
        from PIL import ImageDraw
        global_view = env.view_registry.get("global")
        if global_view is None:
            return
        canvas = global_view["image"].copy()
        mapping = global_view["mapping"]
        img_w, img_h = global_view["img_w"], global_view["img_h"]
        draw = ImageDraw.Draw(canvas)

        def w2p(wx: float, wy: float) -> tuple[float, float]:
            px = (wx - mapping["x_min"]) / mapping["res_x"]
            py = (mapping["y_max"] - wy) / mapping["res_y"]
            return max(0.0, min(img_w, px)), max(0.0, min(img_h, py))

        def rect_world(wb: list, color: str, width: int = 2,
                       label: str | None = None) -> None:
            x0, y0 = w2p(wb[0], wb[3])
            x1, y1 = w2p(wb[2], wb[1])
            draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
            if label:
                draw.text((x0 + 3, y0 + 3), label, fill=color)

        # GT
        gt = self.sample.get("gt_bbox")
        if gt and len(gt) >= 5:
            cx, cy, _, dx, dy = [float(v) for v in gt[:5]]
            rect_world([cx - dx / 2, cy - dy / 2, cx + dx / 2, cy + dy / 2],
                       self._GT_COLOUR, width=3, label="GT")

        # All zoom views
        for vid, v in env.view_registry.items():
            if vid == "global":
                continue
            rect_world(v["world_bounds"], self._ZOOM_COLOUR,
                       width=2, label=vid)

        # Final answer
        wb = ep_result.get("world_bbox")
        if wb and len(wb) == 4:
            rect_world(wb, self._ANSWER_COLOUR, width=3, label="answer")

        # Legend
        try:
            iou = ep_result.get("iou_2d", 0.0)
            reason = ep_result.get("reason", "")
            draw.text((6, max(0, img_h - 30)),
                      f"iou_2d={iou:.3f}  reason={reason}  "
                      f"turns={ep_result.get('assistant_turns', 0)}",
                      fill="white")
        except Exception:
            pass

        canvas.save(self.dir / "trajectory.png")


def make_recorder(vis_root: Path | None, ep_idx: int,
                  sample: dict, worker_id: int = 0) -> "StepRecorder | None":
    """Return None when vis is disabled."""
    if vis_root is None:
        return None
    try:
        return StepRecorder(vis_root, ep_idx, sample, worker_id)
    except Exception as e:
        logger.warning("recorder construction failed: %s", e)
        return None


# ── Main ───────────────────────────────────────────────────────────────────


def _get_done_keys(output_dir: str) -> set[Tuple[str, str]]:
    """Scan output files and collect (scene_id, query) pairs already processed.

    Keying by scene_id alone was wrong: a single scene typically has many
    queries (one per object), so on resume we'd skip every other query in a
    scene the moment one of them landed in the dump. Pairing scene_id with
    query restores per-sample resume semantics, matching how the dedup pass
    keys records.
    """
    done: set[Tuple[str, str]] = set()
    if not os.path.isdir(output_dir):
        return done
    for fname in os.listdir(output_dir):
        if not fname.startswith("api_rollout_worker") or not fname.endswith(".jsonl"):
            continue
        path = os.path.join(output_dir, fname)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "Skipping invalid JSON in %s:%d: %s",
                            path,
                            line_number,
                            exc,
                        )
                        continue
                    sid = rec.get("scene_id")
                    q = rec.get("query")
                    if sid is not None and q is not None:
                        done.add((sid, q))
        except OSError as exc:
            logger.warning("Failed to scan %s: %s", path, exc)
    return done


_TOOL_CALL_RE_STR = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"


def _count_tool_calls(rec: dict, counter: dict) -> int:
    """Walk a dump record's `messages` and tally each <tool_call> by name.
    `unknown` covers calls the env couldn't dispatch.

    Mirrors scripts/eval/rollout.py:_count_tool_calls so the API
    baseline and local rollout report the same per-tool stats.
    """
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
                payload = json.loads(m.group(1))
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


def _dedup_dump(output_dir: str) -> None:
    """Merge per-worker shards into one merged jsonl + a dedup'd jsonl
    (keyed by (scene_id, query, dataset)) and print IoU + per-tool stats.

    Mirrors scripts/eval/rollout.py:_dedup_dump so both baselines
    print directly comparable summaries without manual post-processing.
    """
    import glob

    shards = sorted(glob.glob(os.path.join(output_dir, "api_rollout_worker*.jsonl")))
    if not shards:
        print(f"[dedup] no per-worker shards under {output_dir}/api_rollout_worker*.jsonl, skipping")
        return

    merged_path = os.path.join(output_dir, "merged.jsonl")
    dedup_path = os.path.join(output_dir, "dedup.jsonl")
    seen: set = set()
    n_in = n_out = 0
    iou_sum = 0.0
    success_count = 0
    iou_buckets = [0, 0, 0, 0, 0]   # [0,0.1) [0.1,0.25) [0.25,0.5) [0.5,0.75) [0.75,1]
    tool_counter: dict = {}
    n_with_tool_call = 0
    token_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    schedule_violations = 0
    with open(merged_path, "w", encoding="utf-8") as fmerge, \
         open(dedup_path, "w", encoding="utf-8") as fdedup:
        for shard in shards:
            with open(shard, encoding="utf-8", errors="replace") as fin:
                for line_number, line in enumerate(fin, start=1):
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "Skipping invalid JSON in %s:%d: %s",
                            shard,
                            line_number,
                            exc,
                        )
                        continue
                    if not line.endswith("\n"):
                        line += "\n"
                    n_in += 1
                    fmerge.write(line)
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
                    usage = rec.get("token_usage") or {}
                    for token_key in token_totals:
                        token_totals[token_key] += int(usage.get(token_key, 0) or 0)
                    if rec.get("fixed_schedule_violation"):
                        schedule_violations += 1
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
        total_calls = sum(tool_counter.values())
        print(f"[tools] total calls:  {total_calls}  "
              f"(avg {total_calls / n_out:.2f}/episode, "
              f"{100 * n_with_tool_call / n_out:.1f}% of episodes used >=1 tool)")
        canonical_order = ["get_landmark_xy", "zoom_in", "render_oblique", "unknown"]
        for name in canonical_order:
            if name in tool_counter:
                cnt = tool_counter[name]
                pct = 100 * cnt / total_calls if total_calls else 0
                print(f"[tools]   {name:18} {cnt:6}  ({pct:5.1f}%)")
        for name, cnt in sorted(tool_counter.items()):
            if name in canonical_order:
                continue
            pct = 100 * cnt / total_calls if total_calls else 0
            print(f"[tools]   {name:18} {cnt:6}  ({pct:5.1f}%)")
        if token_totals["total_tokens"]:
            print(
                "[tokens] avg prompt/completion/total per episode: "
                f"{token_totals['prompt_tokens'] / n_out:.1f}/"
                f"{token_totals['completion_tokens'] / n_out:.1f}/"
                f"{token_totals['total_tokens'] / n_out:.1f}"
            )
        if schedule_violations:
            print(
                f"[fixed-policy] schedule violations: {schedule_violations}/{n_out}"
            )


def _worker_loop(
    samples: List[Dict[str, Any]],
    worker_id: int,
    run_id: str,
    output_dir: str,
    args: argparse.Namespace,
) -> int:
    """Process a chunk of samples and write results."""
    client = DashScopeClient(
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=not args.no_thinking,
        reasoning_effort=args.reasoning_effort,
        provider=args.provider,
    )

    # Each worker gets its own CityToolEnv (not thread-safe)
    env = CityToolEnv(
        display_max_side=args.display_max_side,
        display_patch=args.display_patch,
        display_max_pixels=args.display_max_pixels,
        multiview_cache_dir=args.multiview_cache_dir,
        max_tool_calls=args.max_tool_calls,
        bbox_norm_max=args.bbox_norm_max,
        landmark_mode=args.landmark_mode,
        landmark_noise_std_m=args.landmark_noise_std_m,
        landmark_wrong_min_distance_m=args.landmark_wrong_min_distance_m,
        landmark_seed=args.landmark_seed,
    )

    # Query difficulty tracker (shared file, thread-safe dict + atomic save)
    tracker = QueryFilterTracker(args.blacklist_file) if args.blacklist_file else None
    tracker_save_interval = 10  # save every N rollouts
    tracker_counter = 0

    # Visualization recorder
    vis_root: Path | None = None
    if args.vis_dir:
        vis_root = Path(args.vis_dir)
        vis_root.mkdir(parents=True, exist_ok=True)
        logger.info("[worker %d] Vis dir: %s", worker_id, vis_root)

    output_path = os.path.join(
        output_dir, f"api_rollout_worker{worker_id}_{run_id}.jsonl"
    )
    written = 0
    skipped_blacklist = 0

    for i, sample in enumerate(samples):
        scene_id = sample.get("scene_id", "")
        query = sample.get("query", "")

        # Skip blacklisted queries (too hard for current model)
        if tracker and tracker.is_blacklisted(scene_id, query):
            skipped_blacklist += 1
            logger.info(
                "[worker %d] %d/%d scene_id=%s — BLACKLISTED (skipped)",
                worker_id, i + 1, len(samples), scene_id,
            )
            continue

        logger.info(
            "[worker %d] %d/%d scene_id=%s", worker_id, i + 1, len(samples), scene_id
        )

        # Create per-episode recorder if vis_dir is set
        recorder = make_recorder(vis_root, i, sample, worker_id)

        try:
            record = run_episode(
                sample,
                client,
                env,
                max_assistant_turns=args.max_turns,
                recorder=recorder,
                bbox_order=args.bbox_order,
                forced_tool_sequence=args.forced_tool_sequence,
            )
        except Exception as e:
            logger.exception("Episode failed for scene_id=%s: %s", scene_id, e)
            continue

        if record is None:
            logger.info("[worker %d] scene_id=%s — API failure", worker_id, scene_id)
            continue

        iou = float(record.get("iou_2d", 0.0))
        center_dist = float(record.get("center_dist", float("inf")))
        success = bool(record.get("success", False))
        logger.info(
            "[worker %d] scene_id=%s — iou=%.3f dist=%.1fm success=%s turns=%d reason=%s",
            worker_id,
            scene_id,
            iou,
            center_dist,
            success,
            record["assistant_turns"],
            record["reason"],
        )

        # Update query difficulty tracker
        if tracker is not None:
            newly_blacklisted = tracker.update(
                scene_id, query, iou_2d=iou, center_dist=center_dist
            )
            if newly_blacklisted:
                logger.warning(
                    "[worker %d] Query BLACKLISTED after %d consecutive hard fails: %s",
                    worker_id, 8, query[:60],
                )
            tracker_counter += 1
            if tracker_counter % tracker_save_interval == 0:
                tracker.save()
                stats = tracker.stats()
                logger.info(
                    "[worker %d] tracker stats: %d tracked, %d blacklisted",
                    worker_id, stats["total_queries_tracked"], stats["blacklisted"],
                )

        if iou < args.iou_threshold:
            logger.info(
                "[worker %d] scene_id=%s — below threshold (%.3f < %.3f)",
                worker_id,
                scene_id,
                iou,
                args.iou_threshold,
            )
            continue

        line = json.dumps(record, ensure_ascii=False, default=str)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        written += 1

    if tracker:
        tracker.save()
        stats = tracker.stats()
        logger.info(
            "[worker %d] FINAL tracker stats: %d tracked, %d blacklisted, %d skipped",
            worker_id, stats["total_queries_tracked"], stats["blacklisted"],
            skipped_blacklist,
        )

    return written


def main():
    parser = argparse.ArgumentParser(
        description="API rollout for city tool SFT data"
    )
    parser.add_argument("--data_path", required=True, help="Path to dataset JSONL")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument(
        "--provider",
        choices=sorted(_PROVIDER_API_KEY_VARS),
        default="dashscope",
        help="API provider used to select the corresponding credential environment variable",
    )
    parser.add_argument(
        "--model", default="qwen3-vl-plus", help="OpenAI-compatible model name"
    )
    parser.add_argument(
        "--base_url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument(
        "--max_turns", type=int, default=8,
        help="Max assistant turns per episode "
             "(aligned with scripts/configs/rollout.yaml: max_actions_per_traj=8)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024, help="Max new tokens per API call"
    )
    # Greedy decoding by default — aligned with scripts/configs/rollout.yaml
    # (vLLM treats temperature=0 as greedy; top_p/top_k are then ignored).
    # Re-running the same prompt yields the same response for reproducible
    # baseline numbers.
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--no_thinking",
        action="store_true",
        help="Disable Qwen3 thinking/reasoning content in API responses. "
             "No effect on non-Qwen models — for GPT-5 / o-series use "
             "--reasoning_effort instead.",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="none",
        choices=["none", "low", "medium", "high"],
        help="GPT-5 / o-series reasoning_effort level. 'none' (default) "
             "leaves the parameter unset, letting the server pick its own "
             "default (typically 'medium'). Ignored on non-OpenAI models.",
    )
    parser.add_argument(
        "--bbox_order",
        type=str,
        default="xyxy",
        choices=["xyxy", "yxyx"],
        help="How the model emits bbox coordinates. Use 'xyxy' for "
             "Qwen-VL / GPT family (default); use 'yxyx' for the Gemini "
             "family, which trains on [y0,x0,y1,x1]. We swap once at the "
             "model->env boundary so CityToolEnv stays provider-agnostic. "
             "Wrong setting silently produces near-zero IoU — verify by "
             "eyeballing one rollout's trajectory.png.",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.0,
        help="Min IoU to save trajectory. Default 0.0 to dump every "
             "episode — matches scripts/configs/rollout.yaml so the JSONL "
             "denominator is consistent across baselines.",
    )
    parser.add_argument(
        "--max_workers", type=int, default=4, help="Concurrent API workers"
    )
    parser.add_argument("--display_max_side", type=int, default=768)
    parser.add_argument("--display_patch", type=int, default=32)
    parser.add_argument("--display_max_pixels", type=int, default=589_824)
    parser.add_argument(
        "--multiview_cache_dir", type=str,
        default=(os.path.join(os.environ["CITY_AGENT_DATA"], "verify_region_cache", "cityrefer")
                 if os.environ.get("CITY_AGENT_DATA") else ""),
        help="Path to verify_region_cache for render_oblique. Defaults to "
             "$CITY_AGENT_DATA/verify_region_cache/cityrefer, matching "
             "scripts/configs/rollout.yaml so render_oblique quality is "
             "comparable; pass --multiview_cache_dir '' to disable.",
    )
    parser.add_argument(
        "--max_tool_calls", type=int, default=7,
        help="Aligned with scripts/configs/rollout.yaml (max_tool_calls=7 = "
             "max_actions_per_traj - 1).",
    )
    parser.add_argument("--bbox_norm_max", type=float, default=1000.0)
    parser.add_argument(
        "--landmark_mode",
        choices=["clean", "missing", "noisy", "wrong"],
        default="clean",
        help="Controlled Landmark Query condition for robustness evaluation.",
    )
    parser.add_argument(
        "--landmark_noise_std_m",
        type=float,
        default=10.0,
        help="Gaussian coordinate noise in metres when --landmark_mode=noisy.",
    )
    parser.add_argument(
        "--landmark_wrong_min_distance_m",
        type=float,
        default=50.0,
        help="Minimum preferred distance to the substitute landmark in wrong mode.",
    )
    parser.add_argument(
        "--landmark_seed",
        type=int,
        default=42,
        help="Stable seed for noisy/wrong landmark perturbations.",
    )
    parser.add_argument(
        "--forced_tool_sequence",
        type=str,
        default="",
        help="Comma-separated fixed policy, e.g. "
             "'get_landmark_xy,zoom_in,render_oblique'. Empty means free exploration.",
    )
    parser.add_argument("--blacklist_file", type=str, default=None,
                        help="Path to query blacklist JSON. Queries with 8 "
                             "consecutive hard fails (IoU=0, dist>50m) are skipped.")
    parser.add_argument(
        "--vis_dir", type=str, default="",
        help="Per-episode/turn visualization dump root. Empty (default) = disabled. "
             "When set, each episode gets its own subdir with prompts, tool calls, "
             "tool images, and a trajectory.png BEV overlay. Heavy on I/O."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_samples", type=int, default=-1,
        help="Take a deterministic random subset of N samples from "
             "--data_path before sharding across workers. Default -1 = "
             "use the full dataset. Subsampling happens AFTER the resume "
             "filter and uses --seed, so re-running the same script with "
             "the same seed picks the same N. Pass the FULL data file as "
             "--gt to the eval scripts; eval iterates over predictions, "
             "so unmatched GT rows are silently skipped (denominator = "
             "num_samples).",
    )
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    if args.forced_tool_sequence:
        args.forced_tool_sequence = [
            name.strip() for name in args.forced_tool_sequence.split(",")
            if name.strip()
        ]
        valid_tools = set(TOOL_METHODS)
        invalid_tools = [
            name for name in args.forced_tool_sequence if name not in valid_tools
        ]
        if invalid_tools:
            parser.error(
                f"invalid --forced_tool_sequence tool(s): {invalid_tools}; "
                f"valid tools: {sorted(valid_tools)}"
            )
    else:
        args.forced_tool_sequence = []

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    with open(args.data_path, "r", encoding="utf-8") as f:
        all_samples = [json.loads(line) for line in f if line.strip()]
    logger.info("Loaded %d samples from %s", len(all_samples), args.data_path)

    # Skip already done — keyed by (scene_id, query) so multi-query scenes
    # resume sample-by-sample instead of skipping the whole scene as soon
    # as one of its queries has been written.
    done_keys = _get_done_keys(args.output_dir)
    if done_keys:
        logger.info("Skipping %d already-processed (scene_id, query) pairs", len(done_keys))
        all_samples = [
            s for s in all_samples
            if (s.get("scene_id"), s.get("query")) not in done_keys
        ]
        logger.info("Remaining: %d samples", len(all_samples))

    if not all_samples:
        logger.info("Nothing to do — all samples already processed.")
        _dedup_dump(args.output_dir)
        return

    # Shuffle for load distribution across workers. Doubles as the
    # subsampling pass when --num_samples > 0: same seed -> same subset
    # across providers, so qwen / gpt-5 / gemini all score the same N rows.
    rng = random.Random(args.seed)
    rng.shuffle(all_samples)
    if args.num_samples > 0 and args.num_samples < len(all_samples):
        before = len(all_samples)
        all_samples = all_samples[: args.num_samples]
        n_scenes = len({s.get("scene_id") for s in all_samples})
        logger.info(
            "Subsampling %d -> %d (seed=%d, covers %d unique scene_ids)",
            before, len(all_samples), args.seed, n_scenes,
        )

    # Split into worker chunks
    n_workers = min(args.max_workers, len(all_samples))
    chunk_size = (len(all_samples) + n_workers - 1) // n_workers
    chunks = [
        all_samples[i : i + chunk_size]
        for i in range(0, len(all_samples), chunk_size)
    ]

    run_id = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
    logger.info(
        "Starting %d workers, each handling ~%d samples (run_id=%s)",
        len(chunks),
        chunk_size,
        run_id,
    )

    total_written = 0
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = {
            executor.submit(
                _worker_loop, chunk, wid, run_id, args.output_dir, args
            ): wid
            for wid, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            wid = futures[future]
            try:
                written = future.result()
                total_written += written
                logger.info(
                    "[worker %d] finished — wrote %d trajectories", wid, written
                )
            except Exception as e:
                logger.exception("[worker %d] crashed: %s", wid, e)

    logger.info("=" * 60)
    logger.info("All workers done. Total trajectories written: %d", total_written)
    logger.info("Output dir: %s", args.output_dir)

    # Merge per-worker shards + dedup by (scene_id, query, dataset),
    # mirroring scripts/eval/rollout.py:_dedup_dump so the printed
    # IoU / per-tool stats are directly comparable across the two pipelines.
    _dedup_dump(args.output_dir)


if __name__ == "__main__":
    main()
