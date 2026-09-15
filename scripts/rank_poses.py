#!/usr/bin/env python3
"""Extract frozen MDN, Generator and chemistry features, then apply W0 ranking."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cuda'],default='cuda');a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 for family in ['mdn32','interaction','geometry']:
  subprocess.run([sys.executable,str(ROOT/'scripts/extract_features.py'),'--manifest',str(Path(a.manifest).resolve()),'--family',family,'--output',str(a.output/(family+'.npz')),'--device',a.device],check=True)
 import numpy as np
 arrays=[np.load(a.output/(f+'.npz'),allow_pickle=False) for f in ['mdn32','interaction','geometry']]
 if any(not np.array_equal(arrays[0]['pose_ordinals'],x['pose_ordinals']) for x in arrays[1:]):raise ValueError('Feature pose order mismatch')
 tokens=np.concatenate([x['tokens'] for x in arrays],axis=-1)
 np.savez_compressed(a.output/'features.npz',tokens=tokens,baseline=arrays[0]['baseline'])
 subprocess.run([sys.executable,str(ROOT/'scripts/rank.py'),'--features',str(a.output/'features.npz'),'--output',str(a.output/'ranking.csv'),'--device',a.device],check=True)
 print(a.output/'ranking.csv')
if __name__=='__main__':main()
