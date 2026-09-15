#!/usr/bin/env python3
"""Build a new named-complex graph cache from prepared structures and v2 surfaces."""
import argparse,sys,random,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True,type=Path);p.add_argument('--surfaces',required=True,type=Path);p.add_argument('--names',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 from surfna.runtime import activate,generator_config
 activate('generator')
 import numpy as np,torch
 from datasets.pdbbind import PDBBind
 cfg=generator_config();random.seed(20260826);np.random.seed(20260826);torch.manual_seed(20260826)
 if a.output.exists():raise ValueError('Use a new output directory')
 requested=a.names.read_text().splitlines()
 if not requested or len(requested)!=len(set(requested)):raise ValueError('Empty or duplicate names')
 a.output.mkdir(parents=True)
 data=PDBBind(root=str(a.data.resolve()),surface_path=str(a.surfaces.resolve()),cache_path=str(a.output/'cache'),split_path=str(a.names.resolve()),receptor_radius=15.0,c_alpha_max_neighbors=24,popsize=20,maxiter=20,matching=True,keep_original=True,remove_hs=True,num_workers=1,num_conformers=1,require_ligand=True,max_surface_vertices=512,surface_feature_schema='v2_full8',surface_scaler_json=cfg.surface_scaler_json)
 retained=[str(g.name) for g in data.complex_graphs]
 report=dict(requested=len(requested),retained=len(retained),missing=sorted(set(requested)-set(retained)),cache=data.full_cache_path)
 (a.output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
 if len(retained)!=len(requested) or set(retained)!=set(requested):raise RuntimeError('Not all requested complexes were prepared; inspect preparation.json')
if __name__=='__main__':main()
