"""Pack and upload a standalone Urbex data release."""
import argparse
import os
import json
from pathlib import Path
import shutil
import tarfile
from scripts.data.hub_assets import describe, relative_path


UPSTREAM_NOTICE = """
## Sources and terms

These assets include processed annotations, rendered views and candidate pools based on upstream urban datasets. Their data terms are separate from the Urbex code license.

- [CityRefer](https://github.com/ATR-DBI/CityRefer): the [dataset datasheet](https://proceedings.neurips.cc/paper_files/paper/2023/file/f4cef76305dcad4efd3537da087ff520-Supplemental-Datasets_and_Benchmarks.pdf) specifies CC BY 4.0 for data and MIT for baseline code.
- [SensatUrban](https://github.com/QingyongHu/SensatUrban): source point clouds for CityRefer scenes. Consult the upstream download terms; this release does not assign a replacement license to those source assets or derivatives.
- [STPLS3D](https://www.stpls3d.com/data): the official data page specifies CC BY-NC-SA 3.0. Retain attribution, noncommercial use and share-alike conditions for STPLS3D-derived assets.

Urbex converts annotations to JSONL, generates BEV mappings and cached oblique renders, and packages those assets for scene-level downloads. Cite Urbex and the applicable upstream datasets. Raw point clouds are optional release assets; when included, preserve their upstream terms and use `--point-clouds` to download them.
"""


def pack_archive(source, output, prefix, archive_path):
    import zstandard
    files = sorted(p for p in source.rglob('*') if p.is_file())
    if not files or any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError(f'Empty or symlinked asset tree: {source}')
    target = output / archive_path
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial')
    size = 0
    with partial.open('wb') as stream:
        with zstandard.ZstdCompressor(level=3, threads=2).stream_writer(stream) as writer:
            with tarfile.open(fileobj=writer, mode='w|', format=tarfile.PAX_FORMAT) as handle:
                for file in files:
                    name = str(relative_path(str(prefix / file.relative_to(source))))
                    info = handle.gettarinfo(str(file), arcname=name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    info.mtime = 0
                    info.mode = 0o644
                    size += info.size
                    with file.open('rb') as reader:
                        handle.addfile(info, reader)
    partial.replace(target)
    entry = describe(target, output)
    entry.update(file_count=len(files), unpacked_bytes=size)
    return entry


def pack_release(data_root, output, candidates=None, raw_root=None):
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    if output.is_relative_to(data_root) or data_root.is_relative_to(output):
        raise ValueError('Use a staging directory separate from the input data tree')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Staging output must be empty to avoid uploading stale files')
    manifest = {'format_version': 1, 'datasets': {}, 'optional': {}}
    for dataset in ('cityrefer', 'stpls3d'):
        source = data_root / dataset
        def copy(file):
            relative = file.relative_to(data_root)
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if file.is_symlink():
                raise ValueError(f'Symlinked input: {file}')
            shutil.copyfile(file, target)
            return describe(target, output)
        annotations = sorted((source / 'meta').glob('*.jsonl'))
        if not annotations:
            raise ValueError(f'Missing annotations: {source}')
        core = annotations + [source / f'landmarks_{dataset}.json']
        if dataset == 'stpls3d':
            core = annotations + [source / f'landmarks_{dataset}.json'] + sorted(
                p for p in (source / 'box3d').iterdir() if p.is_file())
        cache_root = source / 'verify_region_cache' / dataset
        scenes = {p.name: set() for p in cache_root.iterdir() if p.is_dir()}
        for file in annotations:
            for line in file.read_text().splitlines():
                if not line.strip():
                    continue
                sample = json.loads(line)
                scene = sample['scene_id']
                if len(relative_path(scene).parts) != 1 or scene == '.':
                    raise ValueError(f'Invalid scene ID: {scene}')
                scenes.setdefault(scene, set()).add(str(relative_path(sample['bev_png'])))
        data = {'files': [copy(file) for file in core], 'scenes': {}}
        for scene, bevs in sorted(scenes.items()):
            print(f'Packing {dataset}/{scene}', flush=True)
            if not bevs:
                bevs = {f'bev_{dataset}/{scene}.png'}
            files = sorted({source / path for bev in bevs
                            for path in (bev, str(Path(bev).with_suffix('.json')))})
            prefix = Path(dataset) / 'verify_region_cache' / dataset / scene
            cache = data_root / prefix
            for name in ('scene_meta.json', 'regions.npy', 'index.pkl'):
                if not (cache / name).is_file():
                    raise ValueError(f'Missing scene cache file: {cache / name}')
            archive = pack_archive(cache, output, prefix,
                                   Path(dataset) / 'cache_archives' / f'{scene}.tar.zst')
            data['scenes'][scene] = {'files': [copy(file) for file in files],
                                     'extract_prefix': prefix.as_posix(), 'archive': archive}
        manifest['datasets'][dataset] = data
    if candidates:
        manifest['optional']['cityrefer_candidates'] = pack_archive(
            Path(candidates), output, Path('cityrefer_candidates'),
            Path('optional/cityrefer_candidates.tar.zst'))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'README.md').write_text(
        '# Urbex data assets\n\n'
        'Annotations, landmarks, BEV images and scene-level oblique caches for '
        '[Urbex](https://github.com/AMAP-ML/Urbex).\n\n'
        'Download selected scenes with `scripts/data/download.sh` from the code repository. '
        'Set `HF_DATASET_ID` to this repository ID. Archives restore the original '
        '`verify_region_cache/<dataset>/<scene>/` layout. `manifest.json` records '
        'SHA-256 checksums, sizes and extraction paths. CityRefer candidate pools '
        'are optional resources for Hit@k evaluation. Raw point clouds, when included, '
        'are downloaded only with `--point-clouds`.\n' + UPSTREAM_NOTICE)
    if raw_root:
        manifest = add_point_clouds(raw_root, output)
    return manifest


def add_point_clouds(raw_root, folder):
    """Add original PLY files to an already packed data release."""
    raw_root, folder = Path(raw_root).resolve(), Path(folder).resolve()
    if folder.is_relative_to(raw_root) or raw_root.is_relative_to(folder):
        raise ValueError('Staging and raw-data directories must be separate')
    manifest = json.loads((folder / 'manifest.json').read_text())
    point_clouds = {}
    for dataset, directory in (('cityrefer', 'sensaturban'), ('stpls3d', 'STPLS3D')):
        files = sorted((raw_root / directory).rglob('*.ply'))
        if not files:
            raise ValueError(f'No PLY files found in {raw_root / directory}')
        scenes = {}
        for source in files:
            if source.is_symlink():
                raise ValueError(f'Symlinked PLY input: {source}')
            scene = source.stem.removesuffix('_points') if dataset == 'stpls3d' else source.stem
            relative = Path('raw_data') / source.relative_to(raw_root)
            destination = folder / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise ValueError(f'PLY already staged: {destination}')
            # Hard links avoid duplicating large local files; uploads only read them.
            try:
                os.link(source, destination)
            except OSError:
                shutil.copyfile(source, destination)
            entry = describe(destination, folder)
            scenes.setdefault(scene, []).append(entry)
            print(f'Staged {relative}', flush=True)
        point_clouds[dataset] = scenes
    manifest['point_clouds'] = point_clouds
    temporary = folder / 'manifest.json.partial'
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(folder / 'manifest.json')
    return manifest


def upload_release(folder, repo_id, private=True):
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError
    api = HfApi()
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    expected = {'manifest.json', 'README.md'}
    entries = list(manifest['optional'].values())
    for scenes in manifest.get('point_clouds', {}).values():
        for files in scenes.values():
            entries += files
    for data in manifest['datasets'].values():
        entries += data['files']
        for scene in data['scenes'].values():
            entries += scene['files'] + [scene['archive']]
    for entry in entries:
        relative_path(entry['path'])
        actual = describe(folder / entry['path'], folder)
        if any(actual[key] != entry[key] for key in ('path', 'bytes', 'sha256')):
            raise ValueError(f'Changed release file: {entry["path"]}')
        expected.add(entry['path'])
    actual_files = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    if actual_files != expected:
        raise ValueError('Staging folder contains missing or unexpected files')
    try:
        if api.dataset_info(repo_id).private != private:
            raise ValueError('Existing repository visibility differs from the requested mode; '
                             'change it explicitly in Hub settings before uploading.')
    except RepositoryNotFoundError:
        api.create_repo(repo_id, repo_type='dataset', private=private)
    assert api.dataset_info(repo_id).private == private
    visibility = 'private' if private else 'public'
    print(f'Uploading {visibility} dataset {repo_id}', flush=True)
    api.upload_folder(repo_id=repo_id, repo_type='dataset', folder_path=str(folder),
                      allow_patterns=sorted(expected), commit_message='Upload Urbex data release')
    info = api.dataset_info(repo_id, files_metadata=True)
    assert info.private == private
    remote = {p.rfilename: p for p in info.siblings}
    for entry in entries:
        file = remote[entry['path']]
        assert file.size == entry['bytes'], entry['path']
        if file.lfs:
            assert file.lfs.sha256 == entry['sha256'], entry['path']
    return {'repo_id': repo_id, 'revision': info.sha, 'private': info.private,
            'verified_files': len(entries)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    pack = commands.add_parser('pack')
    pack.add_argument('--data-root', required=True, type=Path)
    pack.add_argument('--output', required=True, type=Path)
    pack.add_argument('--candidates', type=Path)
    pack.add_argument('--raw-root', type=Path, help='Optionally include original PLY files')
    raw = commands.add_parser('add-ply', help='Add PLY files to an existing release')
    raw.add_argument('--raw-root', required=True, type=Path)
    raw.add_argument('--folder', required=True, type=Path)
    upload = commands.add_parser('upload')
    upload.add_argument('--folder', required=True, type=Path)
    upload.add_argument('--repo-id', required=True)
    upload.add_argument('--public', action='store_true',
                        help='Create/upload to a public dataset (default: private)')
    args = parser.parse_args()
    if args.command == 'pack':
        pack_release(args.data_root, args.output, args.candidates, args.raw_root)
    elif args.command == 'add-ply':
        add_point_clouds(args.raw_root, args.folder)
    else:
        print(json.dumps(upload_release(args.folder, args.repo_id, private=not args.public)), flush=True)


if __name__ == '__main__':
    main()
