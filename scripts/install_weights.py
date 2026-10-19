#!/usr/bin/env python3
"""Install the separately distributed model archive; never extracts arbitrary paths."""
import argparse,hashlib,json,tarfile,tempfile,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--archive',type=Path,help='Local surfna-v2-minimal-weights.tar.gz')
    group.add_argument('--url',help='HTTPS URL of the matching model archive')
    a=p.parse_args()
    if a.url and not a.url.startswith('https://'):p.error('Use an HTTPS download URL')
    with tempfile.TemporaryDirectory(prefix='surfna_weights_') as tmp:
        archive=a.archive
        if a.url:
            archive=Path(tmp)/'weights.tar.gz';urllib.request.urlretrieve(a.url,archive)
        expected={w['path']:w for w in json.loads((ROOT/'MODEL_CARD.json').read_text())['weights']}
        staged=[]
        with tarfile.open(archive,'r:gz') as tf:
            members=tf.getmembers()
            if len(members)!=len(expected) or {m.name for m in members}!=set(expected):
                raise ValueError('Archive does not contain exactly the expected weight files')
            for i,m in enumerate(members):
                info=expected[m.name]
                if not m.isfile() or m.size!=info['bytes']:raise ValueError('Invalid archive member')
                dest=Path(tmp)/str(i);h=hashlib.sha256()
                with tf.extractfile(m) as src,dest.open('wb') as out:
                    for block in iter(lambda:src.read(1024*1024),b''):out.write(block);h.update(block)
                if h.hexdigest()!=info['sha256']:raise ValueError('Checkpoint checksum mismatch: '+m.name)
                staged.append((dest,ROOT/m.name))
        for src,dest in staged:
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                if hashlib.sha256(dest.read_bytes()).hexdigest()!=hashlib.sha256(src.read_bytes()).hexdigest():
                    raise FileExistsError('Refusing to replace a different checkpoint: '+str(dest))
            else:
                import shutil
                shutil.copyfile(src,dest)
            print('Installed',dest.relative_to(ROOT))
if __name__=='__main__':main()
