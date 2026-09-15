#!/usr/bin/env python3
"""Extract one frozen feature family from a prepared graph and K40 ordered poses."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--manifest',required=True,help='JSON: graph, receptor, reference_ligand, poses (40 SDF paths)')
 p.add_argument('--family',required=True,choices=['mdn32','interaction','geometry']);p.add_argument('--output',required=True)
 p.add_argument('--device',default='cuda',choices=['cpu','cuda']);a=p.parse_args()
 if a.family!='geometry' and a.device!='cuda':
  raise ValueError('Model feature extraction requires CUDA: capped radius neighborhoods differ on CPU. W0 ranking of saved features supports CPU.')
 import numpy as np
 from rdkit import Chem
 from surfna.pose_quality import atom_order_signature,read_molecule
 from surfna.runtime import activate,load_generator,load_mdn
 manifest=Path(a.manifest).resolve();d=json.loads(manifest.read_text());base=manifest.parent
 def path(x):return (base/x).resolve() if not Path(x).is_absolute() else Path(x)
 if len(d['poses'])!=40:raise ValueError('W0 requires all 40 candidate poses')
 mols=[read_molecule(path(x)) for x in d['poses']]
 template_mol=read_molecule(path(d['reference_ligand']));signature=atom_order_signature(template_mol)
 if any(atom_order_signature(m)!=signature for m in mols):raise ValueError('Candidate atom ordering or chemistry differs from the graph reference')
 values=[];baselines=[]
 if a.family=='geometry':
  from surfna.geometry_features import prepare_receptor,candidate_features
  receptor=prepare_receptor(path(d['receptor']))
  values=[candidate_features(m,receptor)[0] for m in mols]
 else:
  import torch
  activate('generator' if a.family=='interaction' else 'mdn')
  from torch_geometric.data import Batch
  payload=torch.load(path(d['graph']),map_location='cpu',weights_only=False)
  graph=payload['graph'] if isinstance(payload,dict) and 'graph' in payload else payload
  if a.family=='interaction':
   from surfna.interaction_features import make_candidate_graph,capture_forward,captured_tokens
   from utils.diffusion_utils import set_time
   model,_,_=load_generator(device=a.device)
   for mol in mols:
    candidate,_=make_candidate_graph(graph,mol.GetConformer().GetPositions(),graph.original_center,
     graph_frame='l2_prepared' if getattr(graph,'native_mdn_prepared',False) else 'l2_raw',
     expected_atom_count=mol.GetNumAtoms(),expected_atom_order_sha256=signature,candidate_atom_order_sha256=signature)
    batch=Batch.from_data_list([candidate]).to(a.device);set_time(batch,0.,0.,0.,1,False,a.device)
    with torch.inference_mode():captured,_=capture_forward(model,batch);tokens,_=captured_tokens(model,batch,captured)
    values.append(tokens.numpy())
  else:
   from surfna.mdn_features import candidate_graph,evidence,clean_graph
   from models.surfna_v2_scorer_components import mixture_log_probability
   model=load_mdn(a.device)
   if not getattr(graph,'native_mdn_prepared',False):graph['receptor'].center_pos-=graph.original_center.reshape(1,3)
   surface=surface_pos=None
   for mol in mols:
    candidate=candidate_graph(graph,mol.GetConformer().GetPositions(),clean_graph)
    with torch.inference_mode():
     if surface is None:
      encoded=model.backbone.encode_surface_static(Batch.from_data_list([candidate]).to(a.device))
      if not bool(encoded.mask[0].all()):raise ValueError('Unexpected surface padding')
      surface,surface_pos=encoded.scalar[0],encoded.position[0]
     x,b,_=evidence(model,candidate,surface,surface_pos,mixture_log_probability,a.device)
    values.append(x.numpy());baselines.append(float(b))
 tokens=np.asarray(values,dtype=np.float32)
 if tokens.shape!=(40,template_mol.GetNumAtoms(),{'mdn32':32,'interaction':576,'geometry':103}[a.family]) or not np.isfinite(tokens).all():raise ValueError('Invalid feature shape or non-finite values')
 result={'tokens':tokens,'pose_ordinals':np.arange(40)}
 if baselines:result['baseline']=np.asarray(baselines,dtype=np.float32)
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('xb') as f:np.savez_compressed(f,**result)
 print(a.family,tokens.shape,out)
if __name__=='__main__':main()
