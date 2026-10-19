#!/usr/bin/env python3
"""Generate and rank 40 poses for a user-prepared nucleic-acid binding site."""
import argparse,csv,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('receptor','ligand','surface','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260903)
    p.add_argument('--batch-size',type=int,default=5)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args()
    if a.batch_size<1 or a.batch_size>40:p.error('--batch-size must be in [1,40]')
    if a.output.exists():p.error('Use a new output directory')
    # Fail before creating outputs if the downloaded weights are missing or changed.
    run=lambda script,*args:subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],check=True)
    run('verify_weights.py')
    a.output.mkdir(parents=True)
    prep=a.output/'prepared'
    run('prepare.py','--receptor',a.receptor.resolve(),'--ligand',a.ligand.resolve(),
        '--surface',a.surface.resolve(),'--output',prep.resolve(),'--seed',a.seed)
    run('generate.py','--cache',prep/'cache','--receptor-root',prep/'input',
        '--output',a.output/'poses','--samples',40,'--seed',a.seed,
        '--batch-size',a.batch_size,'--device',a.device)
    run('rank_poses.py','--manifest',a.output/'poses/query/manifest.json',
        '--output',a.output/'scores','--device',a.device)
    from rdkit import Chem
    manifest=json.loads((a.output/'poses/query/manifest.json').read_text())
    with (a.output/'scores/ranking.csv').open() as f:rows=list(csv.DictReader(f))
    with Chem.SDWriter(str(a.output/'ranked_poses.sdf')) as writer:
        for row in rows:
            path=a.output/'poses/query'/manifest['poses'][int(row['pose_ordinal'])]
            mol=next(Chem.SDMolSupplier(str(path),removeHs=False))
            if mol is None:raise RuntimeError('Unreadable generated pose: '+str(path))
            for key in ('rank','pose_ordinal','score'):mol.SetProp('SurfNA_'+key,row[key])
            writer.write(mol)
    print('Finished:',a.output/'ranked_poses.sdf',flush=True)

if __name__=='__main__':main()
