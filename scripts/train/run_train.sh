#!/usr/bin/env bash
# Urbex GRPO training on a single 8-GPU node.
#
#   CITY_AGENT_DATA=/path/to/agent_data bash scripts/train/run_train.sh
#
# Trailing arguments are forwarded to Hydra, e.g.
#   bash scripts/train/run_train.sh max_steps=100 actor_train.training_args.learning_rate=5e-7
#
# Adding a key that is not already in grpo.yaml needs a "+" prefix:
#   bash scripts/train/run_train.sh +custom_envs.CityTool.env_config.landmark_mode=noisy

set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"
export PATH="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("scripts"))'):$PATH"

# Resolve `roll` from this checkout rather than from a pip-installed copy.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

if [ -z "${CITY_AGENT_DATA:-}" ]; then
    echo "[err] CITY_AGENT_DATA is not set; see the Data section of README.md." >&2
    exit 2
fi
export CITY_AGENT_DATA
export CITY_TOOL_REQUIRE_CACHE="${CITY_TOOL_REQUIRE_CACHE:-1}"

# Transformer Engine also links cuDNN component libraries at runtime.
if [ -z "${CUDNN_PATH:-}" ]; then
    export CUDNN_PATH="$("$PYTHON" - <<'PY'
from importlib.metadata import distribution
print(distribution("nvidia-cudnn-cu12").locate_file("nvidia/cudnn"))
PY
    )"
fi
export LD_LIBRARY_PATH="$CUDNN_PATH/lib${CUDA_HOME:+:$CUDA_HOME/lib64}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# vLLM selects its own backend; a stale value here breaks Qwen3-VL.
unset VLLM_ATTENTION_BACKEND || true

set -x
"$PYTHON" scripts/train/train.py --config_path ../configs --config_name grpo "$@"
