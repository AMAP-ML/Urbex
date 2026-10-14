"""VL + OpenAI-tool-protocol env_manager for city_grounding.

Bridges two existing env_managers:
  • ``VLTrajEnvManager`` -- multimodal collator wiring (processor, mm_data,
    pixel_values, image_grid_thw).
  • ``AgentNativeStepEnvManager`` -- ``info["tools"]`` capture +
    ``apply_chat_template(tools=...)`` rendering.

Why a new class
---------------
Neither base supports VL + tools together:
  • VLTrajEnvManager builds messages from observation dicts +
    ``agent_system_template`` (no ``tools=`` argument).
  • AgentNativeStepEnvManager calls only ``tokenizer.apply_chat_template``
    (no processor / image plumbing).

This class makes one move: treat the env's emitted observation as a
COMPLETE OpenAI message list (``List[Dict]`` with hermes-style content
items including ``{"type": "image", "image": PIL.Image}``), extract images
in tree order, and render with ``tools=info["tools"]``. The existing VL
collator handles tokenization + pixel_values + image_grid_thw.

Result: prompts rendered here go through the same chat template, the same
way, as a tool-use SFT stage would, so an SFT checkpoint stays
prompt-compatible during RL.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.env_manager.base_env_manager import RolloutCache
from roll.pipeline.agentic.env_manager.vl_traj_env_manager import VLTrajEnvManager


def _walk_collect_images(messages: List[Dict[str, Any]]) -> List[Any]:
    """Walk the messages tree in document order and return all PIL images
    found in ``content[*]["image"]``. Order MUST match the order the chat
    template emits ``<vision_start>...<vision_end>`` placeholders so the
    processor's image_grid_thw aligns with token positions.
    """
    out: List[Any] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    img = item.get("image")
                    if img is not None:
                        out.append(img)
    return out


class CityVLNativeEnvManager(VLTrajEnvManager):
    """VL env_manager that uses the env's full message list + ``tools=``."""

    # Captured at reset, used at every format_messages call.
    tools: List[Dict] = []

    def reset(self) -> RolloutCache:
        # VLTrajEnvManager.reset() builds initial RolloutCache and calls
        # self.env.reset(seed=...). The env returns (messages, info); the
        # parent stores observation + info on history[-1].
        rollout_cache = super().reset()
        if rollout_cache is None:
            return rollout_cache  # type: ignore[return-value]  # parent lies about Optional
        # Pull tools from the env's reset info (now stored on history[0]
        # as part of the per-step record by the parent class).
        first = rollout_cache.history[0] if rollout_cache.history else {}
        self.tools = list(first.get("tools", []) or [])
        return rollout_cache

    # step() inherited from VLTrajEnvManager: it decodes the LLM response
    # and calls self.env.step(action=response). Our env returns the FULL
    # updated messages list as observation -- the parent stores it on
    # history[-1]["observation"] for format_messages to pick up.

    def format_messages(self, history: RolloutCache) -> Tuple[DataProto, List[Dict]]:
        """Render the current full conversation with ``tools=self.tools``.

        Parent VLTrajEnvManager builds messages by walking history and
        wrapping each ``observation`` with templates. Our env emits the
        complete message list each turn, so we just use the latest.
        """
        last = history.history[-1]
        messages: List[Dict[str, Any]] = last.get("observation") or []
        if not messages:
            # Empty obs -> reuse the prior turn's messages so we don't crash.
            for prior in reversed(history.history[:-1]):
                if prior.get("observation"):
                    messages = prior["observation"]
                    break

        # If the latest observation was emitted by env.step() (i.e. the
        # episode has progressed past reset), the env appended an assistant
        # turn for `action`. In that case we DO NOT add a generation
        # prompt -- we are computing tokens to score, not requesting more.
        # On reset, the trailing message is the user turn -> add prompt.
        last_role = messages[-1]["role"] if messages else "user"
        add_generation_prompt = last_role != "assistant"

        # Render to text. ``tokenizer.apply_chat_template`` with ``tools=``
        # produces the same hermes-formatted output as Qwen's official
        # template at SFT time.
        lm_input_texts = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )

        # Collect PIL images in document order; processor consumes them
        # to fill image_grid_thw + pixel_values.
        images = _walk_collect_images(messages)

        # Verification log: per-turn image count fed to collator. Should
        # equal vision_block count in rendered text (asserted below).
        # Cross-reference with env's "[city_tool dispatch]" log to confirm
        # each tool's image survives all the way through.
        if isinstance(lm_input_texts, str):
            n_vis = lm_input_texts.count("<|vision_start|>")
            self.logger.debug(
                "[city_tool format_messages] msgs=%d images_collected=%d "
                "vision_blocks_in_text=%d roles=%s",
                len(messages), len(images), n_vis,
                [m.get("role") for m in messages],
            )

        # Sanity: the chat template emits one ``<|vision_start|>`` per image
        # content item. Counts must match; if not, force them to match by
        # stripping spurious vision-token literals that some models emit in
        # their text responses (sanitization at env-side should usually
        # catch these, but belt-and-braces here so a single bad rollout
        # doesn't crash the whole job).
        if isinstance(lm_input_texts, str):
            n_vision_blocks = lm_input_texts.count("<|vision_start|>")
            if n_vision_blocks != len(images):
                self.logger.warning(
                    f"vision-block / image-count mismatch: "
                    f"template emitted {n_vision_blocks} <|vision_start|> "
                    f"but messages tree has {len(images)} images. "
                    f"Stripping spurious vision tokens from rendered text "
                    f"to recover."
                )
                # Remove all vision-block triples from the rendered text.
                # The matched real images stay attached via the collator.
                # If counts still mismatch (template added MORE blocks than
                # we have images), drop excess; if fewer, fall back to
                # leaving as-is (collator will error meaningfully).
                if n_vision_blocks > len(images):
                    excess = n_vision_blocks - len(images)
                    # Strip trailing `excess` vision-block triples.
                    pattern = "<|vision_start|><|image_pad|><|vision_end|>"
                    parts = lm_input_texts.rsplit(pattern, excess)
                    lm_input_texts = "".join(parts)

        feature: Dict[str, Any] = {self.collator.prompt_key: lm_input_texts}
        if images and self.collator.image_key:
            feature[self.collator.image_key] = images

        if not add_generation_prompt:
            self.collator.return_infer_inputs = False
        inputs = self.collator([feature])
        self.collator.return_infer_inputs = True
        lm_input: DataProto = DataProto.from_single_dict(inputs)

        # Track suffix length the same way VLTrajEnvManager does for
        # response-mask alignment when the final turn is a closed assistant
        # message (no generation prompt).
        if not add_generation_prompt and len(history.history) >= 1:
            history.history[-1]["extra_suffix_length"] = lm_input.batch["input_ids"].shape[1] - (
                history.history[-1].get("input_ids_length", 0)
                + history.history[-1].get("response_ids_length", 0)
            )

        return lm_input, messages
