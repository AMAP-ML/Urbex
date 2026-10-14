# Scripts

Run commands from the repository root. See the [README](../README.md) for setup and usage.

| Directory | Purpose | Entry points |
|---|---|---|
| `data/` | Hub download/release and on-demand BEV/cache construction | [download.sh](data/download.sh), [release.py](data/release.py), [build_assets.sh](data/build_assets.sh) |
| `train/` | Megatron installation and GRPO training | [install.sh](train/install.sh), [run_train.sh](train/run_train.sh) |
| `eval/` | Evaluation and API baselines | [run_eval.sh](eval/run_eval.sh), [run_api_baseline.sh](eval/run_api_baseline.sh) |
| `configs/` | Training and rollout settings | [grpo.yaml](configs/grpo.yaml), [rollout.yaml](configs/rollout.yaml) |
| `demo/` | 3D replay rendering; cache-based 3D view for Colab | [build_3d_demo.py](demo/build_3d_demo.py), [view_3d.py](demo/view_3d.py) |
| `tests/` | Unit tests | `python -m pytest scripts/tests -q` |

Download the released model with `bash scripts/data/download.sh model`; dataset downloads use `--all` or `--scenes ...`. The default Hub IDs are `vvangfaye/Urbex-4B` and `vvangfaye/Urbex-Data`. Run `hf auth login` with an authorized account while the repositories are private.

Training accepts Hydra overrides, for example `bash scripts/train/run_train.sh max_steps=100`. Replay rendering takes `--record` and `--output`; use `--point-cloud` and `--bev-meta` for custom asset paths.

For data releases, stage files separately from the source and upload to a Hub dataset (private by default; add `--public` to publish):

```bash
python -m scripts.data.release pack --data-root /path/to/data --output /path/to/staging --candidates /path/to/cityrefer_candidates
python -m scripts.data.release upload --folder /path/to/staging --repo-id owner/Urbex-Data
```

Include original PLY with `pack --raw-root /path/to/raw_data`, or add it to an existing release:

```bash
python -m scripts.data.release add-ply --raw-root /path/to/raw_data --folder /path/to/staging
bash scripts/data/download.sh stpls3d --scenes USC --point-clouds --output /path/to/urbex/data
```
