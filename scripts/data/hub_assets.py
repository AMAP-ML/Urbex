"""Portable, checksum-verified Hub assets and scene archives."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def relative_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value:
        raise ValueError(f'Unsafe release path: {value!r}')
    return path


def describe(path, root):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected a regular file: {path}')
    return {'path': path.relative_to(root).as_posix(),
            'bytes': path.stat().st_size, 'sha256': sha256(path)}


def fetch(repo_id, entry, destination, revision, install=True):
    from huggingface_hub import hf_hub_download
    relative = relative_path(entry['path'])
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    # HF maintains resumable downloads in its cache. Verify before copying into data.
    source = Path(hf_hub_download(repo_id, str(relative), repo_type='dataset',
                                 revision=revision, cache_dir=str(destination / '.hf-cache')))
    if source.stat().st_size != entry['bytes'] or sha256(source) != entry['sha256']:
        raise ValueError(f'Checksum mismatch: {relative}')
    if not install:
        return source
    target = destination / relative
    if not target.resolve().is_relative_to(destination.resolve()):
        raise ValueError(f'Destination escapes output directory: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError(f'Refusing to overwrite symlink: {target}')
    if target.is_file() and target.stat().st_size == entry['bytes'] and sha256(target) == entry['sha256']:
        return target
    temporary = target.with_name(target.name + '.partial')
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)
    return target


def load_manifest(repo_id, output, revision='main'):
    from huggingface_hub import HfApi, hf_hub_download
    # Pin all files to one immutable revision, including the manifest.
    revision = HfApi().dataset_info(repo_id, revision=revision).sha
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id, 'manifest.json', repo_type='dataset',
                           revision=revision, cache_dir=str(output / '.hf-cache'))
    manifest = json.loads(Path(path).read_text())
    if manifest.get('format_version') != 1:
        raise ValueError('Unsupported Urbex release manifest')
    return manifest, revision


def unpack(archive, destination, prefix, expected):
    """Validate into a temporary tree; replace only the selected scene/package."""
    import zstandard
    destination = Path(destination)
    prefix = relative_path(prefix)
    target = destination / prefix
    if not target.resolve().is_relative_to(destination.resolve()):
        raise ValueError(f'Destination escapes output directory: {target}')
    marker = target / '.urbex-download.json'
    if marker.is_file():
        saved = json.loads(marker.read_text())
        files = [p for p in target.rglob('*') if p.is_file() and p != marker]
        if (saved.get('sha256') == expected['sha256'] and
                len(files) == expected['file_count'] and
                sum(p.stat().st_size for p in files) == expected['unpacked_bytes']):
            return target
    with tempfile.TemporaryDirectory(prefix='.urbex-extract-', dir=destination) as temporary:
        stage = Path(temporary)
        names = set()
        size = 0
        with Path(archive).open('rb') as source:
            with zstandard.ZstdDecompressor().stream_reader(source) as reader:
                with tarfile.open(fileobj=reader, mode='r|') as handle:
                    for member in handle:
                        path = relative_path(member.name)
                        if not path.is_relative_to(prefix) or not member.isfile() or str(path) in names:
                            raise ValueError(f'Invalid archive member: {member.name}')
                        names.add(str(path))
                        size += member.size
                        if len(names) > expected['file_count'] or size > expected['unpacked_bytes']:
                            raise ValueError('Archive exceeds declared contents')
                        handle.extract(member, path=stage, filter='data')
        if len(names) != expected['file_count'] or size != expected['unpacked_bytes']:
            raise ValueError('Archive contents do not match the manifest')
        new = stage / prefix
        (new / marker.name).write_text(json.dumps({'sha256': expected['sha256']}) + '\n')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise ValueError(f'Refusing to replace symlink: {target}')
        backup = stage / 'previous'
        if target.exists():
            target.rename(backup)
        try:
            new.rename(target)
        except BaseException:
            if backup.exists():
                backup.rename(target)
            raise
    return target
