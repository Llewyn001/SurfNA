#!/usr/bin/env python3
"""Train W0 on explicit K40 feature groups and separately supplied RMSD labels."""
import argparse,json,random,sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--features',type=Path,required=True,help='JSON rows: name, split (train/val), path to tokens/baseline NPZ')
 p.add_argument('--labels',type=Path,required=True,help='JSON object: complex name -> 40 RMSD values')
 p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
 p.add_argument('--smoke',action='store_true',help='One optimizer update only; not a completed training run')
 a=p.parse_args()
 import numpy as np,torch
 from surfna.scorer import TopObjectiveScorer,objective
 from surfna.scorer_common import pack_features,validate_feature_group,finite
 from surfna.metrics import ranking_metrics,selection_key
 torch.set_num_threads(2);random.seed(20260831);np.random.seed(20260831);torch.manual_seed(20260831)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 rows=json.loads(a.features.read_text());labels=json.loads(a.labels.read_text());groups={}
 for r in rows:
  if r['split'] not in ('train','val') or r['name'] in groups:raise ValueError('Invalid/duplicate training member')
  if 'families' in r:
   arrays=[np.load(a.features.resolve().parent/r['families'][family]['path'],allow_pickle=False) for family in ['mdn32','interaction','geometry']]
   if any(not np.array_equal(arrays[0]['pose_uids'],v['pose_uids']) for v in arrays[1:]):raise ValueError('Feature family pose identities differ')
   tokens=np.concatenate([v['tokens'] for v in arrays],axis=-1);baseline=arrays[0]['baseline']
  else:
   arr=np.load(a.features.resolve().parent/r['path'],allow_pickle=False);tokens=arr['tokens'];baseline=arr['baseline']
  g=dict(name=r['name'],split=r['split'],tokens=torch.from_numpy(tokens),baseline=torch.from_numpy(baseline))
  validate_feature_group(g,k=40,width=711);groups[g['name']]=g
  y=np.asarray(labels[g['name']],dtype=float)
  if y.shape!=(40,) or not np.isfinite(y).all() or (y<0).any():raise ValueError('Invalid K40 labels')
 names={s:(ROOT/f'reproducibility/splits/original/{s}.txt').read_text().splitlines() for s in ['train','val']}
 if not a.smoke and any({n for n,g in groups.items() if g['split']==s}!=set(names[s]) for s in names):raise ValueError('Full training requires the released train801/val89 membership')
 if a.smoke:names={s:[n for n,g in groups.items() if g['split']==s] for s in names}
 if not names['train']:raise ValueError('No training groups')
 scaler=json.loads((ROOT/'checkpoints/scorer/rank_scaler.json').read_text())
 model=TopObjectiveScorer(scaler['mean'],scaler['std'],scaler['score_scale'],auxiliary=True).to(a.device)
 optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
 a.output.mkdir(parents=True,exist_ok=False)
 def forward(batch):return model(*pack_features(batch,a.device),groups=len(batch))
 def evaluate():
  model.eval();out=[]
  with torch.no_grad():
   for start in range(0,len(names['val']),4):
    batch=[groups[n] for n in names['val'][start:start+4]];scores=forward(batch)[0].cpu().tolist()
    out.extend(dict(name=g['name'],ordinal=j,score=scores[i][j],rmsd=float(labels[g['name']][j])) for i,g in enumerate(batch) for j in range(40))
  return ranking_metrics(out)
 def save(name,epoch,metrics):
  torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=epoch,seed=20260831,val={k:v for k,v in metrics.items() if k!='groups'}),a.output/name)
 best=None;history=[];steps=0
 if not a.smoke:
  initial=evaluate();best=selection_key(initial);save('selected_rank.pt',-1,initial)
 for epoch in range(1 if a.smoke else 50):
  order=list(names['train']);random.Random(20260831+epoch).shuffle(order)
  losses=[]
  for start in range(0,len(order),16):
   batch=[groups[n] for n in order[start:start+16]];model.train();optimizer.zero_grad(set_to_none=True)
   for offset in range(0,len(batch),4):
    micro=batch[offset:offset+4];score,success,pred=forward(micro)
    y=torch.tensor([labels[g['name']] for g in micro],dtype=torch.float32,device=a.device)
    terms=objective('W0',score,success,pred,y);(terms['loss']*len(micro)/len(batch)).backward();losses.append(float(terms['loss']))
   finite(*[p.grad for p in model.parameters() if p.grad is not None]);torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();steps+=1
   if a.smoke:break
  metrics={} if a.smoke else evaluate();selected=not a.smoke and selection_key(metrics)>best
  if selected:best=selection_key(metrics);save('selected_rank.pt',epoch,metrics)
  record=dict(epoch=epoch,optimizer_steps=steps,selected=selected,loss=math.fsum(losses)/len(losses),val={k:v for k,v in metrics.items() if k!='groups'});history.append(record)
  (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n');print(json.dumps(record),flush=True)
 save('smoke_only.pt' if a.smoke else 'last_rank.pt',epoch,metrics)
 (a.output/'status.json').write_text(json.dumps({'status':'SMOKE_ONLY' if a.smoke else 'TRAINING_COMPLETE','optimizer_steps':steps,'original_training_input_replay':False},indent=2)+'\n')
if __name__=='__main__':main()
