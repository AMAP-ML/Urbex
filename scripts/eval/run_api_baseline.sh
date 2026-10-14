#!/usr/bin/env bash
# Closed-model baseline: give an OpenAI-compatible model the same three tools,
# the same prompt and the same oblique-render cache as Urbex. The tools execute
# locally, so the policy is the only difference.
#
#   OPENAI_API_KEY=...    bash scripts/eval/run_api_baseline.sh openai
#   DASHSCOPE_API_KEY=... bash scripts/eval/run_api_baseline.sh dashscope
#   GEMINI_API_KEY=...    bash scripts/eval/run_api_baseline.sh gemini
#   bash scripts/eval/run_api_baseline.sh local     # a vLLM server on localhost:8000
#
# Optional env:
#   MODEL         provider-specific default, see below
#   SPLIT         test_ND (CityRefer default) | test_NO | test (STPLS3D default)
#   DATASET       cityrefer (default) | stpls3d
#   NUM_SAMPLES   -1 = whole split; N>0 takes a SEED-deterministic subsample
#   MAX_WORKERS   concurrent requests
#   BASE_URL      override the provider endpoint
#   EXTRA_ARGS    appended verbatim to scripts/eval/api_baseline.py
#
# Image tokens dominate the bill: a full split is ~500 episodes x 4-9 turns x
# 1-2 images. Check NUM_SAMPLES=10 first.

set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"

# Resolve `roll` from this checkout rather than from a pip-installed copy.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

PROVIDER="${1:-openai}"
case "$PROVIDER" in
    openai)
        : "${OPENAI_API_KEY:?OPENAI_API_KEY not set}"
        BASE_URL="${BASE_URL:-https://api.openai.com/v1}"
        MODEL="${MODEL:-gpt-5}"
        # GPT-5 has no thinking switch; --no_thinking stops the script from
        # injecting the Qwen-only extra_body.
        PROVIDER_ARGS=( --no_thinking --bbox_order xyxy --reasoning_effort "${REASONING:-none}" )
        ;;
    dashscope)
        : "${DASHSCOPE_API_KEY:?DASHSCOPE_API_KEY not set}"
        BASE_URL="${BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
        MODEL="${MODEL:-qwen3-vl-plus}"
        PROVIDER_ARGS=( --bbox_order xyxy )
        [ "${THINKING:-off}" = "off" ] && PROVIDER_ARGS+=( --no_thinking )
        ;;
    gemini)
        if [ -z "${GEMINI_API_KEY:-}" ] && [ -z "${GOOGLE_API_KEY:-}" ]; then
            echo "[err] GEMINI_API_KEY (or GOOGLE_API_KEY) not set" >&2
            exit 2
        fi
        BASE_URL="${BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}"
        MODEL="${MODEL:-gemini-3-pro}"
        # Gemini emits boxes as [y0,x0,y1,x1].
        PROVIDER_ARGS=( --no_thinking --bbox_order yxyx )
        ;;
    local)
        BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
        MODEL="${MODEL:-${URBEX_MODEL_PATH:?set MODEL or URBEX_MODEL_PATH}}"
        PROVIDER_ARGS=( --bbox_order xyxy --no_thinking )
        ;;
    *)
        echo "[err] unknown provider '$PROVIDER'; expected openai, dashscope, gemini or local" >&2
        exit 2
        ;;
esac

DATASET="${DATASET:-cityrefer}"
if [ "$DATASET" = "stpls3d" ]; then
    SPLIT="${SPLIT:-test}"
    ROOT="${CITY_AGENT_DATA_STPLS3D:?CITY_AGENT_DATA_STPLS3D not set}"
    DATA_PATH="${ROOT}/meta/stpls3d_${SPLIT}.jsonl"
    CACHE_DIR="${ROOT}/verify_region_cache/stpls3d"
    CANDIDATE_FLAG="--box3d_dir"
else
    SPLIT="${SPLIT:-test_ND}"
    ROOT="${CITY_AGENT_DATA:?CITY_AGENT_DATA not set}"
    DATA_PATH="${ROOT}/meta/cityrefer_${SPLIT}.jsonl"
    CACHE_DIR="${ROOT}/verify_region_cache/cityrefer"
    CANDIDATE_FLAG="--pth_dir"
fi

OUTPUT_DIR="${OUTPUT_ROOT:-./output/api_baseline}/${PROVIDER}_${MODEL//\//_}_${SPLIT}_$(date +%Y%m%d_%H%M%S)"

case " ${EXTRA_ARGS:-} " in
    *" --api_key "*|*" --api_key="*|*" --api-key "*|*" --api-key="*)
        echo "[err] pass API credentials through the provider environment variable, not EXTRA_ARGS" >&2
        exit 2
        ;;
esac

printf '[run] provider=%s model=%s split=%s output=%s\n' \
    "$PROVIDER" "$MODEL" "$SPLIT" "$OUTPUT_DIR"
"$PYTHON" scripts/eval/api_baseline.py \
    --provider "$PROVIDER" \
    --base_url "$BASE_URL" \
    --model "$MODEL" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --multiview_cache_dir "$CACHE_DIR" \
    --vis_dir "${OUTPUT_DIR}/vis" \
    --max_workers "${MAX_WORKERS:-8}" \
    --num_samples "${NUM_SAMPLES:--1}" \
    --seed "${SEED:-42}" \
    "${PROVIDER_ARGS[@]}" \
    ${EXTRA_ARGS:-}

echo
echo "[done] $OUTPUT_DIR"
echo "[next] python scripts/eval/hit_k.py --pred ${OUTPUT_DIR}/dedup.jsonl --dataset $DATA_PATH $CANDIDATE_FLAG \$CANDIDATE_DIR"
