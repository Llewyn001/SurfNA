#!/usr/bin/env python3
"""Rank one complete K40 candidate ensemble from frozen 711-dimensional features."""
import argparse,csv,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--features',required=True,help='NPZ with tokens[40,atoms,711], baseline[40], optional mask[40,atoms]')
 p.add_argument('--checkpoint',default=str(ROOT/'checkpoints/scorer/model.pt'))
 p.add_argument('--output',required=True)
 p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
 a=p.parse_args()
 import numpy as np,torch
 from surfna.scorer import load_scorer
 data=np.load(a.features,allow_pickle=False)
 tokens=torch.from_numpy(np.asarray(data['tokens'],dtype=np.float32)).to(a.device)
 baseline=torch.from_numpy(np.asarray(data['baseline'],dtype=np.float32)).to(a.device)
 mask=torch.from_numpy(np.asarray(data['mask'],dtype=bool)).to(a.device) if 'mask' in data else torch.ones(tokens.shape[:2],dtype=torch.bool,device=a.device)
 model=load_scorer(a.checkpoint,a.device)
 with torch.inference_mode():score,success,log_rmsd=model(tokens,mask,baseline,groups=1,k=40)
 values=score[0].cpu().tolist();order=sorted(range(40),key=lambda i:(-values[i],i))
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['rank','pose_ordinal','score','success_logit','predicted_log_rmsd'])
  for rank,i in enumerate(order,1):w.writerow([rank,i,values[i],float(success[0,i]),float(log_rmsd[0,i])])
 print('Ranked 40 candidates:',out)
if __name__=='__main__':main()
