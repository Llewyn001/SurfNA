"""Portable locations for the separately frozen Generator and MDN implementations."""
from pathlib import Path
import os,sys,json
from functools import partial
from argparse import Namespace
ROOT=Path(__file__).resolve().parents[2]
def activate(kind):
 if kind not in ('generator','mdn'):raise ValueError(kind)
 source=ROOT/'runtime'/kind
 for name,module in tuple(sys.modules.items()):
  if name.split('.')[0] in ('models','datasets','utils'):
   file=getattr(module,'__file__',None)
   if file and not Path(file).resolve().is_relative_to(source):
    raise RuntimeError('Run Generator and MDN feature extraction in separate processes to avoid import conflicts.')
 table_root=ROOT/'reproducibility/diffusion_tables'
 table_root.mkdir(parents=True,exist_ok=True)
 os.environ['precomputed_arrays']=str(table_root)
 os.environ.pop('SURFNA_V2_G22_SOURCE_ROOT',None)
 sys.path.insert(0,str(source))
 return source

def generator_config(path=None):
 import yaml
 d=yaml.safe_load(Path(path or ROOT/'configs/generator.yml').read_text())
 d['surface_scaler_json']=str(ROOT/'configs/scalers/surface_v2_l2_train_only.json')
 return Namespace(**d)

def load_generator(checkpoint=None,config=None,device='cpu'):
 activate('generator')
 import torch
 from utils.utils import get_model
 from utils.diffusion_utils import t_to_sigma
 from .interaction_features import checkpoint_state,strict_load
 args=generator_config(config);sigma=partial(t_to_sigma,args=args)
 model=get_model(args,torch.device(device),sigma,no_parallel=True,model_type='surface_score_model')
 state=checkpoint_state(torch.load(checkpoint or ROOT/'checkpoints/generator_seed0/model.pt',map_location='cpu',weights_only=True))
 strict_load(model,state);model.requires_grad_(False).eval()
 return model,args,sigma

def load_mdn(device='cpu'):
 activate('mdn')
 import torch
 from utils.scorer_factory_v2 import build_surfna_v2_scorer
 from utils.diffusion_utils import t_to_sigma
 from models.surfna_v2_transfer import STATIC_PREFIXES
 cfg=Namespace(**json.loads((ROOT/'configs/mdn_architecture.json').read_text()))
 cfg.surface_scaler_json=str(ROOT/'configs/scalers/surface_v2_l2_train_only.json')
 cfg.mdn_reference_json=str(ROOT/'checkpoints/mdn/native_train_reference.json')
 torch.manual_seed(20260828)
 model=build_surfna_v2_scorer(cfg,torch.device(device),partial(t_to_sigma,args=cfg))
 bundle=torch.load(ROOT/'checkpoints/mdn/native_prior_static_bundle.pt',map_location='cpu',weights_only=False)
 state=bundle['model']
 expected={k for k in model.state_dict() if k.startswith('prior_head.') or (k.startswith('backbone.') and any(k[9:]==p or k[9:].startswith(p+'.') for p in STATIC_PREFIXES))}
 if set(state)!=expected:raise RuntimeError('The MDN static bundle does not cover the exact static feature path')
 model.load_state_dict(state,strict=False)
 for k,v in state.items():
  if not torch.equal(model.state_dict()[k].cpu(),v.cpu()):raise RuntimeError('MDN state mismatch: '+k)
 model.requires_grad_(False).eval()
 return model
