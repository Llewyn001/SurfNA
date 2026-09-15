#!/usr/bin/env python3
"""Download the public PDB entries identified by the SurfNA instance manifest."""
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def retrieve(url, destination):
    request = urllib.request.Request(url, headers={'User-Agent': 'SurfNA-v2-data-download'})
    temporary = destination.with_suffix(destination.suffix + '.part')
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        if url.endswith('.gz'):
            payload = gzip.decompress(payload)
        if b'data_' not in payload[:4096]:
            raise ValueError('The server did not return a CIF file')
        temporary.write_bytes(payload)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'datasets/nucleic_acid_instances.csv')
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='test')
    parser.add_argument('--names', type=Path, help='Optional complex IDs, one per line')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ccd', action='store_true', help='Also download the referenced CCD definitions')
    args = parser.parse_args()
    with args.manifest.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if args.split != 'all':
        rows = [row for row in rows if row['split'] == args.split]
    if args.names:
        requested = set(args.names.read_text().split())
        rows = [row for row in rows if row['complex_id'] in requested]
        absent = requested - {row['complex_id'] for row in rows}
        if absent:
            parser.error('Unknown names in selected split: ' + ', '.join(sorted(absent)))
    if not rows:
        parser.error('No instances selected')
    entries = sorted({row['pdb_id'].upper() for row in rows})
    jobs = []
    for pdb in entries:
        if not re.fullmatch(r'[0-9A-Z]{4}', pdb):
            raise ValueError('Invalid PDB identifier: ' + pdb)
        lower = pdb.lower()
        jobs.append((pdb, args.output/'pdb'/(pdb+'.cif'), [
            f'https://files.rcsb.org/download/{pdb}.cif',
            f'https://files.wwpdb.org/pub/pdb/data/structures/obsolete/mmCIF/{lower[1:3]}/{lower}.cif.gz',
        ]))
    if args.ccd:
        codes = sorted({code for row in rows for code in re.split(r'[;,|\s]+', row['ligand_ccd']) if code})
        for code in codes:
            if not re.fullmatch(r'[A-Z0-9]{1,8}', code):
                raise ValueError('Invalid CCD identifier: ' + code)
            jobs.append((code, args.output/'ccd'/(code+'.cif'), [f'https://files.rcsb.org/ligands/download/{code}.cif']))
    records = []
    failures = []
    for identifier, path, urls in jobs:
        path.parent.mkdir(parents=True, exist_ok=True)
        source = None
        existed = path.exists()
        errors = []
        if not existed:
            for url in urls:
                try:
                    retrieve(url, path)
                    source = url
                    break
                except (OSError, ValueError, urllib.error.URLError) as error:
                    errors.append(f'{url}: {error}')
            if not path.exists():
                failures.append({'id': identifier, 'errors': errors})
                continue
        records.append({'id': identifier, 'file': str(path.relative_to(args.output)),
                        'url': source, 'reused_existing': existed,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        print(identifier, 'reused' if existed else 'downloaded', flush=True)
    report = {'retrieved_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'split': args.split, 'files': records, 'failures': failures}
    (args.output/'download_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    if failures:
        raise SystemExit(f'{len(failures)} downloads failed; see download_manifest.json. No replacement PDB was substituted.')


if __name__ == '__main__':
    main()
