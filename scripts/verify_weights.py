#!/usr/bin/env python3
"""Check the exact three frozen weight files without importing PyTorch."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
    for item in json.loads((ROOT/'MODEL_CARD.json').read_text())['weights']:
        p=ROOT/item['path']
        if not p.is_file():raise FileNotFoundError(str(p)+'; see README: model weights')
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        if p.stat().st_size!=item['bytes'] or h.hexdigest()!=item['sha256']:
            raise ValueError('Checkpoint checksum mismatch: '+item['path'])
        print('OK',item['path'],h.hexdigest())
if __name__=='__main__':main()
