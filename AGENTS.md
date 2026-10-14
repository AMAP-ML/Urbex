# Urbex agent instructions

Urbex is a Qwen3-VL agent for city-scale grounding in urban point clouds,
built on ROLL with Megatron GRPO training and vLLM rollout. These instructions
apply to this repository, including its bundled `mcore_adapter/`.

All paths below are relative to this Git root. In the original workspace,
this root is `code/`; `../data/`, `../model/`, `../raw_data/`,
`../reproduction/`, and the paper sources are separate local resources.
Public entry points must work without the private reproduction directory.
Read `README.md` and the files relevant to the requested change.

Keep the public release concise. Preserve existing user edits and record the
scope of validation accurately; smoke tests do not establish paper reproduction.
Update this guide when entry points or project conventions change.

## Repository map

| Task | Start here |
| --- | --- |
| Installation | `scripts/train/install.sh`, `requirements.txt`, `requirements-train.txt` |
| GRPO training | `scripts/train/train.py`, `scripts/train/run_train.sh`, `scripts/configs/grpo.yaml` |
| Rollout / evaluation | `scripts/eval/rollout.py`, `scripts/eval/run_eval.sh`, `scripts/configs/rollout.yaml` |
| API baselines | `scripts/eval/api_baseline.py`, `scripts/eval/run_api_baseline.sh` |
| Scoring | `scripts/eval/hit_k.py`, `scripts/eval/metrics_3d.py`, `scripts/eval/aggregate.py` |
| Agent tools / rewards | `roll/pipeline/agentic/env/city_grounding/{env,tool_env,reward,iou}.py`, `tools/retrieve.py` |
| Trajectory handling | `roll/pipeline/agentic/env_manager/city_vl_native_env_manager.py` (training), `city_vl_rollout_env_manager.py` (evaluation) |
| BEV / oblique cache | `roll/pipeline/agentic/env/city_grounding/{build_bev,build_verify_cache,verify_cache,asset_paths}.py`, `scripts/data/build_assets.sh` |
| Hub data release / download | `scripts/data/{release,download,hub_assets}.py`, `scripts/data/download.sh` |
| Model adaptation | `mcore_adapter/`, `roll/models/`, `roll/distributed/` |
| Replay / Colab | `scripts/demo/build_3d_demo.py` (PLY replay), `scripts/demo/view_3d.{py,html}` (cache-based 3D view), `notebooks/urbex_colab.ipynb` |
| Unit tests | `scripts/tests/`; `pyproject.toml` selects this directory |

Keep executable train/eval/test workflows under `scripts/`; the former top-level
`train/`, `eval/`, `tests/`, and `docker/` layout has been removed. Bundled upstream
components can have their own examples; avoid unrelated changes to them.

## Environment and entry points

Use an activated Python 3.11 environment on Linux with a CUDA 12.8 toolkit,
including `nvcc` and a C++ compiler. The installer uses the active `python`,
requires `CUDA_HOME`, and accepts no arguments; `PYTHON` selects the interpreter
in the launchers, not in the installer.

```bash
export CUDA_HOME=/usr/local/cuda-12.8  # use the actual toolkit location
bash scripts/train/install.sh
```

The tested profile pins Torch 2.8.0, vLLM 0.11.0, Megatron Core 0.16.0,
Transformer Engine 2.9.0, and FlashAttention 2.8.1. Preserve the installation
order: build tools and Torch, runtime dependencies, CUDA extensions with
`--no-build-isolation`, then editable project installation. TE needs the cuDNN
headers/libraries supplied by Torch; the installer and training launcher resolve
them. Hydra uses ANTLR 4.9.3; an incidental upgrade to 4.13.2 broke startup.
When changing these pins, verify compatibility and actual CUDA forward/backward
behavior as well as `pip check`.

Configure paths through `CITY_AGENT_DATA`, `CITY_AGENT_DATA_STPLS3D`,
`URBEX_MODEL_PATH`, and `URBEX_RAW_DATA`; see `README.md` for their meanings.
Do not hard-code workstation paths into public code.

```bash
bash scripts/train/run_train.sh
bash scripts/eval/run_eval.sh cityrefer_ND
bash scripts/data/build_assets.sh cityrefer bev --scenes <scene_id>
xvfb-run -a bash scripts/data/build_assets.sh cityrefer cache --scenes <scene_id>
```

Training forwards Hydra overrides, e.g. `max_steps=2 pretrain=/path/to/checkpoint`.
New Hydra keys require `+`. Both distributed configs default to 8 GPUs; adapting
them requires consistent device mappings, parallel sizes, and environment/group
counts. `device_mapping` is evaluated as a Python expression string, such as
`list(range(0,8))`. The rollout manager's `sequential_mode` stays at the CityTool
manager level, outside `env_config`. Preserve Qwen3-VL's explicit context-length
cap and `enforce_eager` unless the replacement has been validated.

## Data and geometry conventions

- Evaluation arguments: `cityrefer_ND`, `cityrefer_NO`, `stpls3d`. Asset building
  and annotation `dataset` values use `cityrefer` or `stpls3d`. The CityRefer
  builder merges and deduplicates scenes from available train/val/ND/NO JSONL
  files; `--scenes` selects a subset and `DATA_PATH` overrides the annotations.
  The API launcher uses `DATASET=cityrefer` / `stpls3d`, with default
  `SPLIT=test_ND` / `test` respectively.
- Canonical annotation files: `meta/cityrefer_{train,val,test_ND,test_NO}.jsonl`
  and `meta/stpls3d_test.jsonl`. Do not reintroduce `_v1` or `sub200` aliases.
- GT boxes start with `[cx, cy, cz, dx, dy, dz]`, using **full dimensions** in
  metres; trailing class/object metadata is not geometry. `iou.py` expects this
  six-coordinate representation, including for BEV IoU.
- Predicted `world_bbox` is `[xmin, ymin, xmax, ymax]`. Tool and answer `bbox_2d`
  is XYXY normalized to 0–1000 in the specified `view_id`; resolve it through
  that view's mapping. Convert representations explicitly before IoU calls.
- Answers use `<answer>{"bbox_2d":[x0,y0,x1,y1],"view_id":"Z1"}</answer>`.
  GT may supply training rewards and evaluation fixtures, never model-visible
  observations, candidate repairs, or displayed prediction geometry.
- BEV images/mappings and oblique caches are supplied in the release; build
  missing scenes or custom assets on demand. `DATA_PATH` chooses annotations and
  `OUT` chooses a separate output directory. Cache construction needs OpenGL;
  use Xvfb on headless hosts. Use separate outputs while changing builders.
- Hub data releases use `manifest.json` format 1: loose annotations/landmarks and
  per-scene BEV files plus checksum-verified `.tar.zst` cache archives. Downloads
  pin one Hub revision and restore the runtime directory layout. Use
  `scripts/data/download.sh model` downloads the model into `model/Urbex-4B`.
  `--repo-id`, `HF_DATASET_ID`, and `HF_MODEL_ID` override release defaults.
  Dataset downloads use `scripts/data/download.sh <dataset> --scenes ...` or `--all`;
  `--metadata-only` avoids BEV/cache downloads, `--candidates` adds CityRefer Hit@k
  pools. Hub repositories are public; `hf auth login` is only needed for uploads. Release
  uploads default to private; `release upload --public` explicitly selects public
  visibility. The uploader refuses a visibility mismatch and never makes an
  existing private repository public automatically. Original PLY is optional and
  covers STPLS3D only: SensatUrban's terms forbid redistribution, so never stage or
  upload SensatUrban PLY; CityRefer users obtain them from the official SensatUrban
  release. Pack with `--raw-root` or extend an existing staging folder with
  `release add-ply`. `download --point-clouds` keeps original filenames and restores
  the sibling `raw_data/` tree; `--all --point-clouds` includes every hosted source
  PLY, including scenes outside the annotation/cache subset. Default inference
  excludes PLY.
- STPLS3D source PLY may be `<scene>.ply` or `<scene>_points.ply`; preserve the
  resolver's support for both rather than renaming raw data.
- Candidate pools are for Hit@k scoring, not required instance proposals for the
  agent. CityRefer uses `--pth_dir`; STPLS3D uses `--box3d_dir`.

## Training invariants

- Default episode budget: 8 assistant turns, at most 7 tool calls, reserving the
  final turn for committing an answer. Keep environment, manager, configs, and
  baseline defaults consistent when changing this budget.
- Terminal reward: `IoU + 0.3 * exp(-XY_center_distance / 80)
  - 0.05 * min(turns, max_turns) / max_turns`.
- Successful oblique-tool shaping: `min(0.05, 0.05 * max(0, candidate_IoU))`.
  It is paid on the tool step; there is no extra render bonus at answer commit.
  Signed zoom-progress shaping is bounded by ±0.05; failed tools receive −0.05.
- Vision freezing currently covers `vision_model.blocks` and
  `vision_model.patch_embed`. Position embeddings and mergers remain trainable;
  do not describe this as a fully frozen vision encoder.
- The default training checkpoint setting `save_optimizer: false` saves model
  weights without complete optimizer/scheduler/RNG state for training resumption.

## Release and demo conventions

Keep the README concise; detailed audits, logs, machine-specific launchers,
gallery selection, and historical validation belong in local `../reproduction/`
when available. Public code must accept explicit inputs and work in a standalone
checkout. Data, models, point clouds, caches, run outputs, and environments stay
outside versioned code. The Hub model and dataset are the public repositories
`vvangfaye/Urbex-4B` and `vvangfaye/Urbex-Data`. Keep their README/Colab links,
downloader defaults and the arXiv link consistent.

Colab uses Hugging Face inference with its preinstalled GPU Torch. Its dependency
list is separate from training: allow NumPy 2 for Python 3.13 and use Transformers
4.57.1 (4.57.0 was yanked). GEM remains required for training; standalone
`CityGroundingEnv` inference can run without GEM and skips its registry when absent.
Its last section shows the prediction in 3D with `scripts/demo/view_3d.py`: a local
point cloud is back-projected from the cached oblique views and their depth maps, so
it needs no PLY, Open3D or OpenGL on the host. It displays a self-contained WebGL
page (`view_3d.html`, inlined into an iframe, no external scripts) that replays
global BEV, the zooms leading to the answer, then an oblique orbit. The display rule matches the replay: XY from the prediction, Z from point
percentiles, never ground truth.

Public assets currently comprise three distinct GIF scenes (building, car,
parking lot) and transparent light/dark logos. The owner removed the method
figure; do not restore it as part of cleanup. README demos use a three-column
table for building, car, and parking, with 320px display widths and no text
captions; the owner accepts horizontal scrolling. Keep original high-quality
media private and use compact GIF exports for the README (448×252, 20 fps).
The current GIFs were composited outside this repository from caption-free
replay renders (128-color palette): category and source at the top left, a
Global BEV / Zoom in / Oblique view tracker with a `Prediction` tag at the top
right, and the query at the bottom. The owner removed the table captions and
the IoU readout; do not restore them. The renderer's own GIF export (96 colors,
built-in labels) optionally uses `gifsicle` on PATH for further compression; it
works with FFmpeg alone and records the actual encoding in its provenance.
For new replay renders, use `--record` and `--output`, and explicit
`--point-cloud` / `--bev-meta` for portable inputs. Prefer `--camera-motion gentle`
with a south presentation camera: global top-down, zoomed top-down, then oblique;
smooth limited motion, infinite looping, and a red 3D box appearing after zoom.
The box's XY comes from the prediction; displayed Z comes from raw-point
percentiles. The camera path and display height are presentation choices, not
an independently predicted 3D box or the original recorded camera trajectory.

## Verification and current limits

Run checks that exercise the changed behavior. Core reward, tool, geometry,
asset-path, and baseline changes have tests under `scripts/tests/`:

```bash
python -m pytest scripts/tests -q
bash -n scripts/train/run_train.sh  # substitute the changed shell script
git diff --check
```

For dependency changes also use `python -m pip check` and appropriate CUDA/module
checks. For documentation/media-only edits verify local links, image decoding or
notebook validity as relevant; avoid installing dependencies or starting training
just to edit documentation. Report checks actually run and their scope. Keep
GPU/Ray smoke outputs separate and clean up only the session created for the task.

The following is a **2026-10-03 snapshot**, not a claim that these issues persist
after future changes. Recheck the relevant source before relying on it:

- Training defaults are 940 steps / 8 GPUs; the current paper states 300 steps /
  16 H20. Some YAML comments still refer to older paper budgets.
- `run_eval.sh` can finish with exit 0 despite failed or incomplete runs. Verify
  child statuses, record coverage, and scoring outputs; file existence is insufficient.
- 3D scoring skips unanswered samples, while Hit@k counts them. Duplicated or
  conflicting annotations are resolved differently across scorers. Report
  denominators and distinguish raw 3D IoU from GT-height-clipped diagnostics.
- Some CityRefer GT IDs are absent from candidate pools; zero-overlap ranking
  ties can affect Hit@k. Preserve benchmark annotations and disclose any scoring
  protocol change rather than silently filtering samples or injecting GT candidates.
- The cache builder's validator rejects some older release-cache formats that
  runtime lookup supports. Runtime center coverage does not prove coverage of
  arbitrary searched regions; do not rebuild or overwrite all downloaded caches blindly.
- Historical local inference and two-step Megatron runs predate reward alignment;
  the released weights have not been retrained for that change. Those smoke runs
  and successful installation checks do not reproduce full paper results.

If the original workspace is available, consult `../reproduction/release-audit/README.md`
for unresolved release metadata/NOTICE/license questions and audit evidence,
`../reproduction/training-install/README.md` for installation verification,
and `../reproduction/reward-alignment/README.md` for reward changes. These are
optional local records, not prerequisites for the public project. Read follow-up
notes before historical snapshots. Preserve pre-existing uncommitted work,
including reorganized files shown as deletions plus untracked replacements.
