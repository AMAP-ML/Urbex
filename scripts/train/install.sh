#!/usr/bin/env bash
# Activate a Python 3.11 environment and set CUDA_HOME to the CUDA 12.8 toolkit.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ "$#" -ne 0 ]; then
    echo "Usage: bash scripts/train/install.sh" >&2
    exit 2
fi
: "${CUDA_HOME:?Set CUDA_HOME to the CUDA 12.8 toolkit.}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-4}"
export NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB:-1}"

python -m pip install 'setuptools>=77,<80' wheel 'packaging>=24.2' ninja pybind11 cmake
# CUDA extension build hooks need PyTorch installed first.
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
python -m pip install -r requirements.txt

# Build against the cuDNN headers and libraries packaged with PyTorch.
export CUDNN_PATH="$(python -c 'from importlib.metadata import distribution; print(distribution("nvidia-cudnn-cu12").locate_file("nvidia/cudnn"))')"
export NVTE_CUDA_INCLUDE_PATH="$CUDA_HOME/include"
export LD_LIBRARY_PATH="$CUDNN_PATH/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m pip install --no-build-isolation -r requirements-train.txt
python -m pip install --no-build-isolation -e .
python -m pip check
