"""Download Urbex models or datasets from Hugging Face."""
import argparse
import os
from pathlib import Path
from scripts.data.hub_assets import fetch, load_manifest, unpack

DEFAULT_DATASET_ID = 'vvangfaye/Urbex-Data'
DEFAULT_MODEL_ID = 'vvangfaye/Urbex-4B'

DATASETS = {'cityrefer_ND': 'cityrefer', 'cityrefer_NO': 'cityrefer',
            'cityrefer': 'cityrefer', 'stpls3d': 'stpls3d'}


def download_metadata(repo_id, dataset, output, revision='main'):
    dataset = DATASETS[dataset]
    manifest, revision = load_manifest(repo_id, output, revision)
    for entry in manifest['datasets'][dataset]['files']:
        fetch(repo_id, entry, output, revision)
    return Path(output) / dataset, manifest, revision


def download_release(repo_id, dataset, output, scenes=None, all_scenes=False,
                     candidates=False, revision='main', point_clouds=False):
    dataset = DATASETS[dataset]
    manifest, revision = load_manifest(repo_id, output, revision)
    data = manifest['datasets'][dataset]
    selected = sorted(data['scenes']) if all_scenes else list(dict.fromkeys(scenes or []))
    clouds = manifest.get('point_clouds', {}).get(dataset, {})
    available = set(data['scenes']) | (set(clouds) if point_clouds else set())
    unknown = set(selected) - available
    if unknown:
        raise ValueError(f'Unknown scenes for {dataset}: {sorted(unknown)}')
    if candidates and dataset != 'cityrefer':
        raise ValueError('--candidates is only needed for CityRefer; STPLS3D box3d is included.')
    for entry in data['files']:
        fetch(repo_id, entry, output, revision)
    for scene in selected:
        if scene not in data['scenes']:
            continue
        entry = data['scenes'][scene]
        print(f'Downloading {dataset}/{scene}', flush=True)
        for file in entry['files']:
            fetch(repo_id, file, output, revision)
        archive = fetch(repo_id, entry['archive'], output, revision, install=False)
        unpack(archive, output, entry['extract_prefix'], entry['archive'])
    if point_clouds:
        requested = sorted(clouds) if all_scenes else selected
        if not clouds or any(scene not in clouds for scene in requested):
            raise ValueError('Original point clouds are not released for the requested scenes')
        # Data lives under output/<dataset>; raw_data is its sibling directory.
        for scene in requested:
            for entry in clouds[scene]:
                fetch(repo_id, entry, Path(output).resolve().parent, revision)
    if candidates:
        entry = manifest['optional']['cityrefer_candidates']
        archive = fetch(repo_id, entry, output, revision, install=False)
        unpack(archive, output, 'cityrefer_candidates', entry)
    return Path(output) / dataset


def download_model(repo_id, output, revision='main'):
    from huggingface_hub import HfApi, snapshot_download
    revision = HfApi().model_info(repo_id, revision=revision).sha
    return Path(snapshot_download(repo_id, revision=revision, local_dir=str(output)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=[*DATASETS, 'model'])
    parser.add_argument('--repo-id', help='Override HF_DATASET_ID or HF_MODEL_ID and the release defaults')
    parser.add_argument('--output', type=Path,
                        help='Data parent directory (default: data) or model directory (default: model/Urbex-4B)')
    parser.add_argument('--revision', default='main')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--scenes', nargs='+')
    selection.add_argument('--all', action='store_true', dest='all_scenes')
    selection.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--candidates', action='store_true')
    parser.add_argument('--point-clouds', action='store_true',
                        help='Also download original PLY to output/../raw_data/')
    args = parser.parse_args()
    if args.dataset == 'model':
        if args.scenes or args.all_scenes or args.metadata_only or args.candidates or args.point_clouds:
            parser.error('Scene, candidate and point-cloud options apply to datasets only')
        repo_id = args.repo_id or os.getenv('HF_MODEL_ID') or DEFAULT_MODEL_ID
        root = download_model(repo_id, args.output or Path('model/Urbex-4B'), args.revision)
        print(f'Model root: {root.resolve()}')
        return
    if not (args.scenes or args.all_scenes or args.metadata_only):
        parser.error('Choose --scenes, --all or --metadata-only for a dataset')
    repo_id = args.repo_id or os.getenv('HF_DATASET_ID') or DEFAULT_DATASET_ID
    if args.metadata_only and (args.candidates or args.point_clouds):
        parser.error('--metadata-only cannot be combined with --candidates or --point-clouds')
    root = download_release(repo_id, args.dataset, args.output or Path('data'), args.scenes,
                            args.all_scenes, args.candidates, args.revision, args.point_clouds)
    print(f'Data root: {root.resolve()}')


if __name__ == '__main__':
    main()
