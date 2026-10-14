import io
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile

import pytest
import zstandard

from scripts.data.download import download_release
from scripts.data.hub_assets import describe, fetch, unpack
from scripts.data.release import pack_release


@pytest.fixture
def release(tmp_path, monkeypatch):
    import huggingface_hub
    source = tmp_path / 'source'
    for dataset in ('cityrefer', 'stpls3d'):
        root = source / dataset
        (root / 'meta').mkdir(parents=True)
        (root / 'meta' / f'{dataset}_test.jsonl').write_text(json.dumps(
            {'scene_id': 'scene', 'bev_png': f'bev_{dataset}/scene.png'}) + '\n')
        (root / f'landmarks_{dataset}.json').write_text('{}')
        (root / f'bev_{dataset}').mkdir()
        for extension in ('png', 'json'):
            (root / f'bev_{dataset}' / f'scene.{extension}').write_bytes(b'bev-data')
        cache = root / 'verify_region_cache' / dataset / 'scene'
        cache.mkdir(parents=True)
        for name in ('scene_meta.json', 'regions.npy', 'index.pkl'):
            (cache / name).write_bytes(b'fixture')
        (cache / 'images').mkdir()
        (cache / 'images' / '000.png').write_bytes(b'image')
        if dataset == 'stpls3d':
            (root / 'box3d').mkdir()
            (root / 'box3d' / 'scene.npy').write_bytes(b'boxes')
    candidates = tmp_path / 'candidates'
    candidates.mkdir()
    (candidates / 'scene.pth').write_bytes(b'candidate')
    staged = tmp_path / 'staged'
    manifest = pack_release(source, staged, candidates)
    calls = []
    def fake_download(repo_id, filename, **kwargs):
        assert kwargs['repo_type'] == 'dataset'
        assert kwargs['revision'] == 'fixed-commit'
        calls.append(filename)
        return str(staged / filename)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', fake_download)
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda: SimpleNamespace(
        dataset_info=lambda *args, **kwargs: SimpleNamespace(sha='fixed-commit')))
    return source, staged, manifest, calls


def test_roundtrip_selected_scene_candidates_and_repair(release, tmp_path):
    source, staged, manifest, calls = release
    output = tmp_path / 'download'
    root = download_release('owner/repo', 'cityrefer_ND', output,
                            scenes=['scene'], candidates=True)
    for original in (source / 'cityrefer').rglob('*'):
        if original.is_file():
            assert (root / original.relative_to(source / 'cityrefer')).read_bytes() == original.read_bytes()
    assert (output / 'cityrefer_candidates/scene.pth').read_bytes() == b'candidate'
    assert not (output / 'stpls3d').exists()
    cache = root / 'verify_region_cache/cityrefer/scene'
    (cache / 'images/000.png').unlink()
    download_release('owner/repo', 'cityrefer_NO', output, scenes=['scene'])
    assert (cache / 'images/000.png').read_bytes() == b'image'
    assert set(p.split('/')[0] for p in calls) <= {'manifest.json', 'cityrefer', 'optional'}


def test_metadata_only_and_unknown_scene(release, tmp_path):
    _, _, _, calls = release
    root = download_release('owner/repo', 'stpls3d', tmp_path / 'download')
    assert (root / 'box3d/scene.npy').exists()
    assert not (root / 'bev_stpls3d').exists()
    assert not any('cache_archives' in p for p in calls)
    with pytest.raises(ValueError, match='Unknown scenes'):
        download_release('owner/repo', 'stpls3d', tmp_path / 'unknown', scenes=['typo'])
    assert not (tmp_path / 'unknown/stpls3d').exists()


def test_corrupted_hub_file_is_not_installed(release, tmp_path):
    _, staged, manifest, _ = release
    entry = manifest['datasets']['cityrefer']['files'][0]
    (staged / entry['path']).write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        fetch('owner/repo', entry, tmp_path / 'download', 'fixed-commit')
    assert not (tmp_path / 'download' / entry['path']).exists()


@pytest.mark.parametrize('name,kind', [('../escape', 'file'),
                                       ('data/scene/link', 'link'),
                                       ('other/scene/file', 'file')])
def test_unsafe_archive_never_replaces_existing_scene(tmp_path, name, kind):
    archive = tmp_path / 'unsafe.tar.zst'
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as handle:
        member = tarfile.TarInfo(name)
        if kind == 'link':
            member.type = tarfile.SYMTYPE
            member.linkname = '/tmp/outside'
            handle.addfile(member)
        else:
            member.size = 1
            handle.addfile(member, io.BytesIO(b'x'))
    archive.write_bytes(zstandard.ZstdCompressor().compress(stream.getvalue()))
    output = tmp_path / 'download'
    scene = output / 'data/scene'
    scene.mkdir(parents=True)
    (scene / 'keep').write_text('original')
    entry = describe(archive, tmp_path)
    entry.update(file_count=1, unpacked_bytes=1)
    with pytest.raises(ValueError):
        unpack(archive, output, 'data/scene', entry)
    assert (scene / 'keep').read_text() == 'original'
    assert not (tmp_path / 'escape').exists()


def test_destination_symlink_escape_is_rejected(release, tmp_path):
    _, _, manifest, _ = release
    output = tmp_path / 'download'
    output.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (output / 'cityrefer').symlink_to(outside, target_is_directory=True)
    entry = manifest['datasets']['cityrefer']['files'][0]
    with pytest.raises(ValueError, match='escapes output'):
        fetch('owner/repo', entry, output, 'fixed-commit')
    assert not list(outside.iterdir())


def test_pack_includes_unannotated_scene(release, tmp_path):
    import shutil
    source, _, _, _ = release
    root = source / 'stpls3d'
    shutil.copytree(root / 'verify_region_cache/stpls3d/scene',
                    root / 'verify_region_cache/stpls3d/extra')
    for suffix in ('png', 'json'):
        shutil.copyfile(root / f'bev_stpls3d/scene.{suffix}',
                        root / f'bev_stpls3d/extra.{suffix}')
    manifest = pack_release(source, tmp_path / 'extra-release')
    assert 'extra' in manifest['datasets']['stpls3d']['scenes']


@pytest.mark.parametrize('private', [True, False])
def test_upload_explicit_visibility(release, monkeypatch, private):
    import huggingface_hub
    from huggingface_hub.errors import RepositoryNotFoundError
    from scripts.data.release import upload_release
    _, stage, manifest, _ = release
    created = []
    uploaded = []
    entries = list(manifest['optional'].values())
    for data in manifest['datasets'].values():
        entries += data['files']
        for scene in data['scenes'].values():
            entries += scene['files'] + [scene['archive']]
    def info(*args, **kwargs):
        if not created:
            raise RepositoryNotFoundError('not created')
        return SimpleNamespace(private=created[0], sha='uploaded', siblings=[
            SimpleNamespace(rfilename=e['path'], size=e['bytes'],
                            lfs=SimpleNamespace(sha256=e['sha256'])) for e in entries])
    api = SimpleNamespace(dataset_info=info,
                          create_repo=lambda *a, **kw: created.append(kw['private']),
                          upload_folder=lambda **kw: uploaded.append(kw))
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda: api)
    result = upload_release(stage, 'owner/repo', private=private)
    assert created == [private] and len(uploaded) == 1
    assert result['private'] == private


def test_upload_never_implicitly_changes_visibility(release, monkeypatch):
    import huggingface_hub
    from scripts.data.release import upload_release
    _, stage, _, _ = release
    calls = []
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda: SimpleNamespace(
        dataset_info=lambda *a, **kw: SimpleNamespace(private=True),
        upload_folder=lambda **kw: calls.append(kw)))
    with pytest.raises(ValueError, match='visibility differs'):
        upload_release(stage, 'owner/repo', private=False)
    assert not calls


def test_point_clouds_are_optional_and_keep_original_paths(release, tmp_path):
    from scripts.data.release import add_point_clouds
    _, stage, _, _ = release
    raw = tmp_path / 'raw'
    sensat = raw / 'sensaturban/ply/train'
    stpls = raw / 'STPLS3D/RealWorldData'
    sensat.mkdir(parents=True)
    stpls.mkdir(parents=True)
    (sensat / 'scene.ply').write_bytes(b'city cloud')
    (stpls / 'scene_points.ply').write_bytes(b'stpls cloud')
    (stpls / 'extra.ply').write_bytes(b'extra cloud')
    add_point_clouds(raw, stage)
    # SensatUrban terms forbid redistribution, so its PLY are never staged.
    assert not (stage / 'raw_data/sensaturban').exists()
    output = tmp_path / 'workspace/data'
    download_release('owner/repo', 'cityrefer_ND', output, scenes=['scene'])
    assert not (output.parent / 'raw_data').exists()
    with pytest.raises(ValueError, match='SensatUrban point clouds are not redistributed'):
        download_release('owner/repo', 'cityrefer_ND', output, scenes=['scene'], point_clouds=True)
    download_release('owner/repo', 'stpls3d', output, scenes=['scene'], point_clouds=True)
    result = output.parent / 'raw_data/STPLS3D/RealWorldData'
    assert (result / 'scene_points.ply').read_bytes() == b'stpls cloud'
    assert not (result / 'extra.ply').exists()
    download_release('owner/repo', 'stpls3d', output, all_scenes=True, point_clouds=True)
    assert (result / 'extra.ply').read_bytes() == b'extra cloud'


def test_model_download_pins_revision(release, monkeypatch, tmp_path):
    import huggingface_hub
    from scripts.data.download import download_model
    calls = []
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda: SimpleNamespace(
        model_info=lambda repo, revision: SimpleNamespace(sha='model-commit')))
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', lambda repo, **kw:
                        calls.append((repo, kw)) or kw['local_dir'])
    output = tmp_path / 'model'
    assert download_model('owner/model', output, 'release-tag') == output
    assert calls == [('owner/model', {'revision': 'model-commit', 'local_dir': str(output)})]


@pytest.mark.parametrize('kind,env_name,default_repo,output', [
    ('model', 'HF_MODEL_ID', 'vvangfaye/Urbex-4B', Path('model/Urbex-4B')),
    ('stpls3d', 'HF_DATASET_ID', 'vvangfaye/Urbex-Data', Path('data')),
])
def test_cli_repo_defaults_and_overrides(monkeypatch, kind, env_name, default_repo, output):
    from scripts.data import download
    calls = []
    def model(repo, path, *args):
        calls.append((repo, path))
        return path
    def dataset(repo, dataset, path, *args):
        calls.append((repo, path))
        return path
    monkeypatch.setattr(download, 'download_model', model)
    monkeypatch.setattr(download, 'download_release', dataset)
    monkeypatch.delenv(env_name, raising=False)
    arguments = ['download', kind] + ([] if kind == 'model' else ['--all'])
    monkeypatch.setattr('sys.argv', arguments)
    download.main()
    assert calls[-1] == (default_repo, output)
    monkeypatch.setenv(env_name, 'owner/environment')
    download.main()
    assert calls[-1][0] == 'owner/environment'
    monkeypatch.setattr('sys.argv', arguments + ['--repo-id', 'owner/explicit'])
    download.main()
    assert calls[-1][0] == 'owner/explicit'


def test_model_cli_rejects_dataset_flags(monkeypatch):
    from scripts.data import download
    calls = []
    monkeypatch.setattr(download, 'download_model', lambda *args: calls.append(args))
    monkeypatch.setattr('sys.argv', ['download', 'model', '--point-clouds'])
    with pytest.raises(SystemExit) as error:
        download.main()
    assert error.value.code == 2 and not calls
