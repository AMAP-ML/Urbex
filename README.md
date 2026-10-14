# <picture><source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.png" /><img src="assets/logo.png" width="40" alt="Urbex logo" /></picture> Urbex

[![arXiv](https://img.shields.io/badge/arXiv-2610.18356-b31b1b.svg)](https://arxiv.org/abs/2610.18356)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/vvangfaye/Urbex-Data)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/vvangfaye/Urbex-4B)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMAP-ML/Urbex/blob/main/notebooks/urbex_colab.ipynb)

Official implementation of the paper "Urbex: Agentic Spatial Grounding for City-Scale 3D Scenes".

<table>
  <tr>
    <td width="33%" align="center"><a href="assets/demo/blue-rooftop-building.gif"><img src="assets/demo/blue-rooftop-building.gif" width="320" alt="USC blue-rooftop building: global BEV, zoom BEV, then 30-degree oblique inspection" /></a></td>
    <td width="33%" align="center"><a href="assets/demo/castle-road-red-van.gif"><img src="assets/demo/castle-road-red-van.gif" width="320" alt="Cambridge Castle Road red van: global BEV, zoom BEV, then 30-degree oblique inspection" /></a></td>
    <td width="33%" align="center"><a href="assets/demo/aldridge-road-parking.gif"><img src="assets/demo/aldridge-road-parking.gif" width="320" alt="Aldridge Road triangular parking: global BEV, zoom BEV, then 30-degree oblique inspection" /></a></td>
  </tr>
</table>

Urbex formulates city-scale 3D grounding as active spatial search, enabling a vision-language agent to progressively gather evidence in an interactive multi-view environment without precomputed instance proposals. Reinforcement learning jointly optimizes localization and search decisions, teaching the agent where to inspect, how to refine its search and when to commit an answer.

## Installation

Requires Linux, Python 3.11 and the CUDA 12.8 toolkit (`nvcc` and a C++ compiler). Run commands from this repository root.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
bash scripts/train/install.sh
```

The installer includes Megatron Core 0.16.0, Transformer Engine 2.9.0 (CUDA 12 / PyTorch bindings), FlashAttention 2.8.1 and NVML. The training launcher locates PyTorch's cuDNN libraries automatically.

## Data and model

Try the [Colab notebook](notebooks/urbex_colab.ipynb) for single-query inference with an interactive 3D view of the prediction. The [model weights](https://huggingface.co/vvangfaye/Urbex-4B) and [data](https://huggingface.co/datasets/vvangfaye/Urbex-Data) are on Hugging Face; the dataset card lists the terms of the upstream datasets.

```bash
bash scripts/data/download.sh model --output /path/to/urbex/model/Urbex-4B
bash scripts/data/download.sh cityrefer --all --output /path/to/urbex/data
bash scripts/data/download.sh stpls3d --all --output /path/to/urbex/data

export CITY_AGENT_DATA=/path/to/urbex/data/cityrefer
export CITY_AGENT_DATA_STPLS3D=/path/to/urbex/data/stpls3d
export URBEX_MODEL_PATH=/path/to/urbex/model/Urbex-4B
```

Use `--scenes <scene_id> ...` for selected scenes. Add `--point-clouds` to download original PLY into the sibling `raw_data/` directory, or `--candidates` for CityRefer Hit@k. Data files are checksum-verified; `--repo-id` or `HF_DATASET_ID` / `HF_MODEL_ID` selects another Hub repository.

BEV images and oblique-view caches are provided. Build them only as needed:

```bash
export URBEX_RAW_DATA=/path/to/urbex/raw_data
bash scripts/data/build_assets.sh cityrefer bev
xvfb-run -a bash scripts/data/build_assets.sh cityrefer cache
```

CityRefer merges scenes across available annotation splits. Use `stpls3d` for STPLS3D, or add `--scenes <scene_id> ...` to build selected scenes.

## Training

```bash
bash scripts/train/run_train.sh
```

Training uses Megatron and vLLM. The default [GRPO configuration](scripts/configs/grpo.yaml) uses 8 GPUs; adjust device mappings and parallel sizes for your hardware.

## Evaluation

```bash
bash scripts/eval/run_eval.sh cityrefer_ND
# Other splits: cityrefer_NO, stpls3d
```

Set `CANDIDATE_DIR` to the candidate pool for Hit@k, and `URBEX_RAW_DATA` to the raw point-cloud directory for 3D metrics. Results are saved under `output/`. The [rollout configuration](scripts/configs/rollout.yaml) controls inference settings.

## Citation

```bibtex
@inproceedings{wang2026urbex,
  title         = {Urbex: Agentic Spatial Grounding for City-Scale 3D Scenes},
  author        = {Yu Wang and Rui Dai and Wei Chen and Yi Wang and Bo Dang and
                   Kaikui Liu and Xiangxiang Chu and Yansheng Li},
  booktitle     = {Advances in Neural Information Processing Systems (NeurIPS 2026)},
  year          = {2026},
  eprint        = {2610.18356},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

## Acknowledgements

Built on [ROLL](https://github.com/alibaba/ROLL). See [LICENSE](LICENSE) and [NOTICE](NOTICE).
