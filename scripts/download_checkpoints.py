#!/usr/bin/env python3
"""Download selected SurfNA V2 weights from versioned GitHub Release assets."""
import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set', choices=['default', 'benchmark-repeats'], default='default')
    parser.add_argument('--archive', type=Path, help='Use an already downloaded archive')
    args = parser.parse_args()
    registry = json.loads((ROOT/'checkpoints/assets.json').read_text())
    asset = next(item for item in registry['assets'] if item['id'] == args.set)
    archive = args.archive or ROOT/'downloads'/asset['file']
    if args.archive is None and not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(archive.suffix+'.part')
        try:
            request = urllib.request.Request(asset['url'], headers={'User-Agent': 'SurfNA-v2-checkpoints'})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as target:
                shutil.copyfileobj(response, target)
            if sha256(temporary) != asset['sha256']:
                raise ValueError('Downloaded archive hash differs from the released manifest')
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(archive) != asset['sha256']:
        raise ValueError('Archive hash differs from the released manifest')
    expected = {item['path']: item for item in asset['files']}
    seen = set()
    with tarfile.open(archive) as container:
        for member in container:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError('Unexpected archive member: '+member.name)
            destination = (ROOT/member.name).resolve()
            if not destination.is_relative_to(ROOT):
                raise ValueError('Archive member is outside the repository')
            if destination.exists() and sha256(destination) != expected[member.name]['sha256']:
                raise FileExistsError('Refusing to overwrite different local weights: '+str(destination))
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix+'.part')
            try:
                with container.extractfile(member) as source, temporary.open('wb') as target:
                    shutil.copyfileobj(source, target)
                if sha256(temporary) != expected[member.name]['sha256']:
                    raise ValueError('Extracted file hash differs: '+member.name)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError('The archive is missing expected model files')
    print('Installed', args.set, 'checkpoint set')


if __name__ == '__main__':
    main()
