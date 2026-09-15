"""Full-K40 training for the LambdaRank@3 x confidence-shrinkage screen."""
from __future__ import annotations

import csv, hashlib, importlib.util, io, json, math, time, traceback
from pathlib import Path

import torch

from setwise_model import ARMS, BASE, make_model, objective

REFERENCE=Path('/root/SurfNA_L2_scorer_mechanism_20260901_autodl_v1/code/training.py')
REFERENCE_SHA='947c05bb38ea5b7b61cb460104b5960b90eac6c98e4f773c40cf460b6067de76'
SEED=20260831


def reference():
    if hashlib.sha256(REFERENCE.read_bytes()).hexdigest()!=REFERENCE_SHA:
        raise RuntimeError('Frozen trainer dependency changed')
    spec=importlib.util.spec_from_file_location('_setwise_frozen_training',REFERENCE)
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api);api._legacy()
    return api


def forward(model,groups,device):
    x,mask,baseline=BASE.pack_features(groups,device)
    return model(x,mask,baseline,groups=len(groups),k=40)


def score_groups(model,groups,device='cpu',batch_groups=4):
    if not groups or len({g['name'] for g in groups})!=len(groups):raise ValueError('Invalid score cohort')
    for g in groups:BASE.validate_feature_group(g,k=40,width=711)
    model.eval();rows=[]
    with torch.no_grad():
        for start in range(0,len(groups),batch_groups):
            batch=groups[start:start+batch_groups]
            scores,success,pred=forward(model,batch,device);BASE.finite(scores,success,pred)
            scores,success,pred=(x.cpu().tolist() for x in (scores,success,pred))
            for gi,g in enumerate(batch):
                for ordinal in range(40):
                    rows.append(dict(name=g['name'],split=g['split'],ordinal=ordinal,
                        baseline=float(g['baseline'][ordinal]),score=scores[gi][ordinal],
                        success_logit=success[gi][ordinal],predicted_log_rmsd=pred[gi][ordinal]))
    if len(rows)!=len(groups)*40:raise ValueError('Incomplete scoring')
    return rows


def evaluate(model,groups,labels,device):
    rows=score_groups(model,groups,device)
    for row in rows:row['rmsd']=float(labels[row['name']][row['ordinal']])
    return reference()._legacy().ranking_metrics(rows),rows


def update(arm,model,groups,labels,optimizer,device):
    model.train();optimizer.zero_grad(set_to_none=True);parts=[];den=len(groups)
    for start in range(0,den,4):
        micro=groups[start:start+4]
        y=torch.tensor([labels[g['name']] for g in micro],dtype=torch.float32,device=device)
        score,success,pred=forward(model,micro,device);terms=objective(arm,score,success,pred,y)
        (terms['loss']*len(micro)/den).backward()
        parts.append((len(micro),{k:float(v) if isinstance(v,torch.Tensor) else v for k,v in terms.items()}))
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    if not gradients:raise ValueError('No gradients')
    BASE.finite(*gradients);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step();BASE.finite(*(p.detach() for p in model.parameters()))
    keys=set.intersection(*(set(x) for _,x in parts));merged={k:math.fsum(n*x[k] for n,x in parts)/den
        for k in keys if isinstance(parts[0][1][k],(int,float))}
    merged['gradient_norm']=float(norm);return merged


def fit(arm,groups40,labels,split_ids,provenance,output,verify_inputs,seed=SEED,device='cuda'):
    if arm not in ARMS or seed!=SEED or not callable(verify_inputs):raise ValueError('Unregistered experiment')
    out=Path(output)
    if out.exists():raise ValueError('No overwrite/retry/resume')
    ref=reference();old=ref._legacy();verify_inputs();old.verify_pins(provenance)
    meta=ref.validate_metadata('R0',labels,split_ids,provenance,seed=seed)
    subsets=ref._feature_population(groups40,split_ids,meta['feature_dimension']);train,val=subsets['train'],subsets['val']
    scaler=ref.fit_scaler(ref.prefix_groups(train),split_ids['train']);match=ref.verify_original_scaler(scaler,provenance)
    if not match['checked']:raise ValueError('Original scaler not checked')
    scaler.update(fit_k=8,values_sha256=ref._scaler_value_digest(scaler),original_scaler_comparison=match)
    snapshot={s:old._numeric_digest(g) for s,g in subsets.items()};labels_sha=hashlib.sha256(old._json_bytes(labels)).hexdigest()
    old._seed(seed,device);model=make_model(arm,scaler['mean'],scaler['std'],scaler['score_scale']).to(device)
    initial=old._state_digest(model.state_dict());out.mkdir(parents=True,exist_ok=False)
    old.atomic_json(out/'STARTED.json',dict(arm=arm,seed=seed,epochs=50,train_k=40,
        optimizer_steps=2550,microbatch_groups=4,optimizer_batch_groups=16,
        fresh_optimizer=True,early_stopping=False,test_opened=False,automatic_retry=False))
    old.atomic_json(out/'rank_scaler.json',scaler);scaler_sha=old.sha(out/'rank_scaler.json')
    epoch=-1;steps=0;started=time.monotonic()
    try:
        def checkpoint(vm):
            return dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                arm=arm,seed=seed,epoch=epoch,val=old.compact(vm),provenance=provenance,
                scaler_sha256=scaler_sha,test_used_for_selection=False,
                recipe=dict(epochs=50,train_k=40,optimizer_steps=2550,lr=.001,
                    weight_decay=.001,auxiliary=True,
                    lambda_top3=arm in ('W1','W3'),score_shrinkage=arm in ('W2','W3'),
                    lambda_weight=.5,shrinkage_weight=.01,unbounded=True))
        current,_=evaluate(model,val,labels,device);best_key=old.selection_key(current);best_epoch=-1
        with torch.no_grad():zero=forward(model,val[:1],device)[0]
        if not torch.equal(zero,torch.zeros_like(zero)):raise ValueError('Initial scores must be zero')
        old._save_torch(out/'initial_rank.pt',checkpoint(current));old._save_torch(out/'selected_rank.pt',checkpoint(current))
        history=[dict(epoch=-1,optimizer_steps=0,selected_epoch=-1,val=old.compact(current))]
        old.atomic_json(out/'history.json',history);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        by_name={g['name']:g for g in train};orders=[]
        for epoch in range(50):
            order=ref.epoch_group_order(split_ids['train'],epoch,seed);orders.append(hashlib.sha256(old._json_bytes(order)).hexdigest())
            losses=[];norms=[]
            for start in range(0,801,16):
                batch=[by_name[n] for n in order[start:start+16]];result=update(arm,model,batch,labels,optimizer,device)
                losses.append((len(batch),result));norms.append(result['gradient_norm']);steps+=1
            if steps!=(epoch+1)*51:raise ValueError('Update count drift')
            current,_=evaluate(model,val,labels,device);key=old.selection_key(current);improved=key>best_key
            if improved:
                best_key,best_epoch=key,epoch;old._save_torch(out/'selected_rank.pt',checkpoint(current),replace=True)
            old._save_torch(out/'last_rank.pt',checkpoint(current),replace=epoch>0)
            record=dict(epoch=epoch,optimizer_steps=steps,selected_epoch=best_epoch,selected=improved,
                objective=math.fsum(n*x['loss'] for n,x in losses)/801,gradient_norm_max=max(norms),
                elapsed_seconds=time.monotonic()-started,group_order_sha256=orders[-1],val=old.compact(current))
            history.append(record);old.atomic_json(out/'history.json',history,replace=True)
            print(json.dumps(dict(event='setwise_epoch',arm=arm,seed=seed,**record),allow_nan=False),flush=True)
        if epoch!=49 or steps!=2550:raise ValueError('Incomplete training')
        state=torch.load(out/'selected_rank.pt',map_location='cpu',weights_only=True);model.load_state_dict(state['model'],strict=True)
        vm,vr=evaluate(model,val,labels,device);tm,tr=evaluate(model,train,labels,device)
        if state['epoch']!=best_epoch or old.selection_key(vm)!=best_key:raise ValueError('Selection not reproducible')
        if any(old._numeric_digest(subsets[s])!=h for s,h in snapshot.items()):raise ValueError('Input mutation')
        verify_inputs();old.verify_pins(provenance)
        old.atomic_json(out/'selected_metrics.json',dict(selected_train40=tm,selected_val=vm))
        text=io.StringIO(newline='');writer=csv.DictWriter(text,fieldnames=list(tr[0]),delimiter='\t');writer.writeheader();writer.writerows(tr+vr)
        old._publish_bytes(out/'selected_pose_scores.tsv',text.getvalue().encode())
        files=('initial_rank.pt','selected_rank.pt','last_rank.pt','rank_scaler.json','selected_metrics.json','history.json','selected_pose_scores.tsv')
        receipt=dict(status='PASS',arm=arm,seed=seed,completed_epochs=50,optimizer_steps=steps,
            selected_epoch=best_epoch,selected_val=old.compact(vm),initial_state_sha256=initial,
            selected_state_sha256=old._state_digest(model.state_dict()),selected_checkpoint_sha256=old.sha(out/'selected_rank.pt'),
            scaler_values_sha256=scaler['values_sha256'],scaler_sha256=scaler_sha,labels_digest=labels_sha,
            original_feature_digests=snapshot,group_order_sha256=hashlib.sha256(old._json_bytes(orders)).hexdigest(),
            output_sha256={f:old.sha(out/f) for f in files},provenance=provenance,
            trainable_parameters=sum(p.numel() for p in model.parameters()),test_opened=False,
            test_used_for_selection=False,fresh_optimizer=True,teacher=False,train_k=40,val_selection_k=40,
            auxiliary=True,lambda_top3=arm in ('W1','W3'),
            score_shrinkage=arm in ('W2','W3'),lambda_weight=.5,
            shrinkage_weight=.01,unbounded=True,automatic_retry=False)
        old.atomic_json(out/'VAL_SELECTION_FROZEN.json',receipt);return receipt
    except BaseException:
        old.atomic_json(out/'FAILED.json',dict(epoch=epoch,steps=steps,error=traceback.format_exc(),automatic_retry=False));raise


def load_model(output,device='cpu'):
    out=Path(output);ref=reference();old=ref._legacy();receipt=json.loads((out/'VAL_SELECTION_FROZEN.json').read_text())
    if (out/'FAILED.json').exists() or receipt['status']!='PASS' or receipt['completed_epochs']!=50:raise ValueError('Training incomplete')
    for name,digest in receipt['output_sha256'].items():old.check_hash(out/name,digest)
    old.verify_pins(receipt['provenance']);scaler=json.loads((out/'rank_scaler.json').read_text())
    model=make_model(receipt['arm'],scaler['mean'],scaler['std'],scaler['score_scale'])
    state=torch.load(out/'selected_rank.pt',map_location='cpu',weights_only=True);model.load_state_dict(state['model'],strict=True)
    if state['epoch']!=receipt['selected_epoch'] or old._state_digest(model.state_dict())!=receipt['selected_state_sha256']:
        raise ValueError('Checkpoint drift')
    return model.to(device).eval().requires_grad_(False),receipt
