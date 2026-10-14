#!/usr/bin/env bash
# Urbex evaluation: rollout -> Hit@k -> 3D metrics, for one dataset.
#
#   CITY_AGENT_DATA=/path/to/agent_data \
#   URBEX_MODEL_PATH=/path/to/model/Urbex-4B \
#   bash scripts/eval/run_eval.sh cityrefer_ND
#
# Datasets: cityrefer_ND (default) | cityrefer_NO | stpls3d
#
# Optional env:
#   RUNS            repeats, for inference-variance numbers (default 1). With
#                   RUNS>1 the per-run logs are aggregated into summary/.
#   OUT             output root (default ./output/eval_<dataset>_<timestamp>)
#   URBEX_RAW_DATA  raw point-cloud root; the 3D metrics are skipped without it
#   CANDIDATE_DIR   candidate pool for Hit@k (per-scene PointGroup .pth for
#                   CityRefer, box3d json for STPLS3D); Hit@k is skipped
#                   without it. See README.
#   DATA_PATH       query JSONL, overriding the one derived from <dataset>

set -uo pipefail   # -e off on purpose: a failed run must not abort the sweep

cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"

# Resolve `roll` from this checkout rather than from a pip-installed copy.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

DATASET="${1:-cityrefer_ND}"
RUNS="${RUNS:-1}"
OUT="${OUT:-./output/eval_${DATASET}_$(date +%Y%m%d_%H%M%S)}"

for var in CITY_AGENT_DATA URBEX_MODEL_PATH; do
    if [ -z "${!var:-}" ]; then
        echo "[err] \$$var is not set; see README.md." >&2
        exit 2
    fi
done
export CITY_TOOL_REQUIRE_CACHE="${CITY_TOOL_REQUIRE_CACHE:-1}"

case "$DATASET" in
    cityrefer_ND|cityrefer_NO)
        SPLIT="${DATASET#cityrefer_}"
        export CITY_AGENT_DATA
        DATA_PATH="${DATA_PATH:-${CITY_AGENT_DATA}/meta/cityrefer_test_${SPLIT}.jsonl}"
        CACHE_DIR="${CITY_AGENT_DATA}/verify_region_cache/cityrefer"
        CANDIDATE_FLAG=( --pth_dir "${CANDIDATE_DIR:-}" )
        SCENE_DIR="${URBEX_RAW_DATA:-}/sensaturban/ply"
        ;;
    stpls3d)
        export CITY_AGENT_DATA="${CITY_AGENT_DATA_STPLS3D:-$CITY_AGENT_DATA}"
        DATA_PATH="${DATA_PATH:-${CITY_AGENT_DATA}/meta/stpls3d_test.jsonl}"
        CACHE_DIR="${CITY_AGENT_DATA}/verify_region_cache/stpls3d"
        CANDIDATE_FLAG=( --box3d_dir "${CANDIDATE_DIR:-}" )
        SCENE_DIR="${URBEX_RAW_DATA:-}/STPLS3D/RealWorldData"
        ;;
    *)
        echo "[err] unknown dataset '$DATASET'; expected cityrefer_ND, cityrefer_NO or stpls3d" >&2
        exit 2
        ;;
esac

DS_ROOT="${OUT}/${DATASET}"
mkdir -p "${DS_ROOT}/eval"
echo "[eval] dataset : $DATASET ($DATA_PATH)"
echo "[eval] model   : $URBEX_MODEL_PATH"
echo "[eval] runs    : $RUNS"
echo "[eval] output  : $DS_ROOT"

for run in $(seq 1 "$RUNS"); do
    RUN_ID="$(printf '%02d' "$run")"
    RUN_DIR="${DS_ROOT}/run_${RUN_ID}"
    mkdir -p "$RUN_DIR"

    echo "[eval] run $run/$RUNS: rollout"
    "$PYTHON" scripts/eval/rollout.py \
        exp_name="${DATASET}_run${RUN_ID}" \
        output_dir="$RUN_DIR" \
        logging_dir="${RUN_DIR}/logs" \
        rollout_dump_dir="${RUN_DIR}/dump" \
        custom_envs.CityTool.env_config.data_path="$DATA_PATH" \
        custom_envs.CityTool.env_config.multiview_cache_dir="$CACHE_DIR" \
        > "${RUN_DIR}/rollout.log" 2>&1

    RECORDS="${RUN_DIR}/records.dedup.jsonl"
    if [ ! -f "$RECORDS" ]; then
        echo "[eval] run $run/$RUNS FAILED, see ${RUN_DIR}/rollout.log"
        continue
    fi
    echo "[eval] run $run/$RUNS: $(wc -l < "$RECORDS") records"

    if [ -d "${CANDIDATE_DIR:-}" ]; then
        echo "[eval] run $run/$RUNS: Hit@k"
        "$PYTHON" scripts/eval/hit_k.py \
            --pred "$RECORDS" --dataset "$DATA_PATH" "${CANDIDATE_FLAG[@]}" \
            2>&1 | tee "${DS_ROOT}/eval/hit_k_run_${RUN_ID}.log"
    else
        echo "[eval] skipping Hit@k: set CANDIDATE_DIR to the candidate pool (see README)"
    fi

    if [ -d "$SCENE_DIR" ]; then
        echo "[eval] run $run/$RUNS: 3D metrics"
        "$PYTHON" scripts/eval/metrics_3d.py \
            --predictions "$RECORDS" --gt "$DATA_PATH" --scene_dir "$SCENE_DIR" \
            --output "${DS_ROOT}/eval/3d_metrics_run_${RUN_ID}.jsonl" \
            2>&1 | tee "${DS_ROOT}/eval/3d_metrics_run_${RUN_ID}.log"
    else
        echo "[eval] skipping 3D metrics: URBEX_RAW_DATA/$SCENE_DIR not found"
    fi
done

if [ "$RUNS" -gt 1 ]; then
    echo "[eval] aggregating $RUNS runs"
    "$PYTHON" scripts/eval/aggregate.py \
        --sweep_root "$OUT" --output "${OUT}/summary" \
        --num_runs "$RUNS" --datasets "$DATASET"
    echo "[eval] summary: ${OUT}/summary/"
fi

echo "[eval] done: $DS_ROOT"
