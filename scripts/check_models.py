#!/usr/bin/env python3
"""Load each frozen model in an isolated process, or test W0 set permutation invariance."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=['generator','mdn','scorer','all'],default='all')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    if a.kind=='all':
        for kind in ('generator','mdn','scorer'):
            subprocess.run([sys.executable,__file__,'--kind',kind,'--device',a.device],check=True)
        return
    import torch
    torch.set_num_threads(2)
    if a.kind=='scorer':
        from surfna.scorer import load_scorer
        model=load_scorer(ROOT/'checkpoints/scorer/model.pt',a.device)
        torch.manual_seed(17)
        tokens=torch.randn(40,7,711,device=a.device)
        mask=torch.ones(40,7,dtype=torch.bool,device=a.device)
        baseline=torch.randn(40,device=a.device);perm=torch.randperm(40,device=a.device)
        with torch.inference_mode():
            x=model(tokens,mask,baseline,groups=1)[0][0]
            y=model(tokens[perm],mask[perm],baseline[perm],groups=1)[0][0]
        torch.testing.assert_close(y,x[perm],atol=2e-5,rtol=2e-5)
    else:
        from surfna.runtime import load_generator,load_mdn
        model=load_generator(device=a.device)[0] if a.kind=='generator' else load_mdn(a.device)
    print('PASS',a.kind,'parameters',sum(x.numel() for x in model.parameters()),flush=True)
if __name__=='__main__':main()
