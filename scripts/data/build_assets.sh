#!/usr/bin/env bash
# Build BEV PNG/mapping pairs or cached oblique views on demand from raw PLY.
# The release includes prebuilt assets; use this for missing scenes or custom data.
#
#   bash scripts/data/build_assets.sh cityrefer bev
#   xvfb-run -a bash scripts/data/build_assets.sh cityrefer cache
#
# Required env: URBEX_RAW_DATA and CITY_AGENT_DATA (CityRefer) or
#               CITY_AGENT_DATA_STPLS3D (STPLS3D).
# Optional env: DATA_PATH (query JSONL), OUT (asset directory), WORKERS
#               (default 1), PYTHON. Remaining arguments go to the builder.

set -euo pipefail

usage() {
    echo "Usage: bash scripts/data/build_assets.sh <cityrefer|stpls3d> <bev|cache> [builder arguments]"
    echo "CityRefer uses all available annotation splits; DATA_PATH selects a custom JSONL."
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi
if [ "$#" -lt 2 ]; then
    usage >&2
    exit 2
fi

cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

DATASET="$1"
ASSET_KIND="$2"
shift 2

case "$DATASET" in
    cityrefer)
        BUILDER_DATASET="cityrefer"
        DATA_ROOT="${CITY_AGENT_DATA:?CITY_AGENT_DATA not set}"
        DEFAULT_DATA_PATHS=("${DATA_ROOT}"/meta/cityrefer_{train,val,test_ND,test_NO}.jsonl)
        ;;
    stpls3d)
        BUILDER_DATASET="stpls3d"
        DATA_ROOT="${CITY_AGENT_DATA_STPLS3D:?CITY_AGENT_DATA_STPLS3D not set}"
        DEFAULT_DATA_PATHS=("${DATA_ROOT}/meta/stpls3d_test.jsonl")
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

case "$ASSET_KIND" in
    bev)
        BUILDER="roll.pipeline.agentic.env.city_grounding.build_bev"
        DEFAULT_OUT="${DATA_ROOT}/bev_${BUILDER_DATASET}"
        ;;
    cache)
        BUILDER="roll.pipeline.agentic.env.city_grounding.build_verify_cache"
        DEFAULT_OUT="${DATA_ROOT}/verify_region_cache/${BUILDER_DATASET}"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

if [ -n "${DATA_PATH:-}" ]; then
    DATA_PATHS=("$DATA_PATH")
else
    DATA_PATHS=()
    for path in "${DEFAULT_DATA_PATHS[@]}"; do
        [ ! -f "$path" ] || DATA_PATHS+=("$path")
    done
    if [ "${#DATA_PATHS[@]}" -eq 0 ]; then
        echo "No annotation JSONL found under ${DATA_ROOT}/meta; set DATA_PATH for custom data." >&2
        exit 2
    fi
fi

exec "$PYTHON" -m "$BUILDER" \
    --dataset "$BUILDER_DATASET" \
    --jsonl "${DATA_PATHS[@]}" \
    --ply_root "${URBEX_RAW_DATA:?URBEX_RAW_DATA not set}" \
    --out "${OUT:-$DEFAULT_OUT}" \
    --workers "${WORKERS:-1}" \
    "$@"
