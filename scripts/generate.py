#!/usr/bin/env python3
"""Generate ligand poses from the released graph cache using the frozen Generator."""
import argparse,copy,csv,json,pickle,random,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--cache',type=Path,required=True,help='Directory with heterographs.pkl and rdkit_ligands.pkl')
 p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path)
 p.add_argument('--config',type=Path);p.add_argument('--names',type=Path,help='Optional ordered complex names, one per line')
 p.add_argument('--samples',type=int,default=40,choices=[10,40]);p.add_argument('--batch-size',type=int,default=10)
 p.add_argument('--seed',type=int,default=20260826);p.add_argument('--protocol',choices=['benchmark','figure3'],default='benchmark')
 p.add_argument('--device',choices=['cuda'],default='cuda');p.add_argument('--receptor-root',type=Path)
 a=p.parse_args()
 if a.protocol=='figure3' and (a.samples!=10 or a.batch_size!=5):raise ValueError('Figure 3 uses K10 and inference batch 5')
 if a.output.exists():raise ValueError('Use a new output directory')
 from surfna.runtime import activate,load_generator
 source=activate('generator')
 import numpy as np,torch
 from rdkit import Chem
 from scipy.spatial.transform import Rotation
 from torch_geometric.data import Batch
 from datasets.pdbbind import PDBBind
 from utils import sampling as sampler
 from utils.diffusion_utils import get_t_schedule
 from surfna.pose_quality import remove_all_hs
 from surfna.sampling_protocol import strict_sampling_ast,seed_for,NoiseTorch
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 with (a.cache/'heterographs.pkl').open('rb') as f:graphs=pickle.load(f)
 with (a.cache/'rdkit_ligands.pkl').open('rb') as f:mols=pickle.load(f)
 if len(graphs)!=len(mols):raise ValueError('Graph/ligand cache size mismatch')
 lookup={str(g.name):i for i,g in enumerate(graphs)}
 if len(lookup)!=len(graphs):raise ValueError('Duplicate graph identities')
 names=a.names.read_text().splitlines() if a.names else list(lookup)
 if not names or len(names)!=len(set(names)) or set(names)-set(lookup):raise ValueError('Invalid requested graph identities')
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 model,cfg,sigma=load_generator(a.checkpoint,a.config,a.device);device=torch.device(a.device)
 def finite_model(batch):
  scores=model(batch)
  if not all(torch.isfinite(v).all() for v in scores):raise FloatingPointError('Nonfinite diffusion score')
  return scores
 def finite_modify(*args,**kwargs):
  graph=sampler.modify_conformer(*args,**kwargs)
  if not torch.isfinite(graph['ligand'].pos).all():raise FloatingPointError('Nonfinite ligand coordinates')
  return graph
 view=types.SimpleNamespace(complex_graphs=graphs,rdkit_ligands=mols,require_ligand=True,transform=None)
 schedule=get_t_schedule(20);a.output.mkdir(parents=True);rows=[]
 for name in names:
  idx=lookup[name];original=Batch.from_data_list([PDBBind.get(view,idx)])
  data=[copy.deepcopy(original) for _ in range(a.samples)]
  context=dict(vars(sampler),modify_conformer=finite_modify)
  if a.protocol=='figure3':
   initial_seed=seed_for(a.seed,name,'initial');diffusion_seed=seed_for(a.seed,name,'diffusion')
   random.seed(initial_seed);np.random.seed(initial_seed);torch.manual_seed(initial_seed)
   initial_rng=NoiseTorch(torch,initial_seed)
   init_globals=dict(vars(sampler),torch=initial_rng)
   initialize=types.FunctionType(sampler.randomize_position.__code__,init_globals,argdefs=sampler.randomize_position.__defaults__)
   initialize(data,cfg.no_torsion,False,cfg.tr_sigma_max,False)
   context['torch']=NoiseTorch(torch,diffusion_seed)
  else:sampler.randomize_position(data,cfg.no_torsion,False,cfg.tr_sigma_max,False)
  exec(compile(strict_sampling_ast((source/'utils/sampling.py').read_text()),'<strict-sampling>','exec'),context)
  result=context['sampling'](data,finite_model,20,schedule,schedule,schedule,device,sigma,cfg,batch_size=a.batch_size,args=None)
  if result is None or len(result[0])!=a.samples:raise RuntimeError('Incomplete generated ensemble')
  folder=a.output/name;folder.mkdir();template=remove_all_hs(copy.deepcopy(mols[idx]))
  center=original.original_center.cpu().numpy().reshape(-1,3)[0]
  heavy=original['ligand'].x[:,0]!=0
  torch.save({'graph':graphs[idx]},folder/'graph.pt')
  with Chem.SDWriter(str(folder/'reference.sdf')) as f:f.write(template)
  paths=[]
  for ordinal,g in enumerate(result[0]):
   xyz=g['ligand'].pos.cpu().numpy()[heavy]+center
   if xyz.shape!=(template.GetNumAtoms(),3) or not np.isfinite(xyz).all():raise ValueError('Invalid generated coordinates')
   mol=copy.deepcopy(template);conformer=Chem.Conformer(mol.GetNumAtoms())
   for j,point in enumerate(xyz):conformer.SetAtomPosition(j,tuple(float(v) for v in point))
   mol.RemoveAllConformers();mol.AddConformer(conformer);dest=folder/f'pose_{ordinal:02d}.sdf'
   with Chem.SDWriter(str(dest)) as f:f.write(mol)
   paths.append(dest.name);rows.append([name,ordinal,str(dest.relative_to(a.output))])
  receptor_root=a.receptor_root or ROOT/'reproducibility/test128/input'
  receptor=receptor_root/name/f'{name}_protein_processed.pdb'
  manifest={'graph':'graph.pt','reference_ligand':'reference.sdf','receptor':str(receptor.resolve()),'poses':paths}
  (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
  print(name,a.samples,'poses',flush=True)
 with (a.output/'poses.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['complex_name','pose_ordinal','pose_sdf']);w.writerows(rows)
 (a.output/'generation.json').write_text(json.dumps({'seed':a.seed,'protocol':a.protocol,'steps':20,'samples':a.samples,'batch_size':a.batch_size,'complexes':len(names),'force_optimize':False},indent=2)+'\n')
if __name__=='__main__':main()
