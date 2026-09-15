"""Fit the fixed 4225-parameter L2 MDN-B residual head; no backbone calls."""
from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
import random
import time
import traceback

from rank_common import (COUNTS, FEATURE_NAMES, RECIPE, SEED, SPLIT_SHAS, atomic_json,
                         atomic_torch, check_hash, event, key, read_json, reserve_stage,
                         sha, validate_contract, validate_features)
from rank_math import finite, ranking_metrics, state_digest


def seed():
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False


def compact(metrics):
    return {k: v for k, v in metrics.items() if k != 'groups'}


def select_key(metrics):
    return (metrics['top1_lt2'], -metrics['regret'], metrics['ndcg'])


def fit_scaler(groups):
    import torch
    if (len(groups) != COUNTS['train'] or any(g['split'] != 'train' for g in groups)
            or len({g['name'] for g in groups}) != COUNTS['train']):
        raise RuntimeError('Rank moments must fit exactly the 801 distinct new L2 TRAIN groups')
    # Formula-identical complex-balanced moments from the frozen B recipe.
    means = torch.stack([g['tokens'].double().mean((0, 1)) for g in groups])
    seconds = torch.stack([g['tokens'].double().square().mean((0, 1)) for g in groups])
    mean = means.mean(0)
    std = (seconds.mean(0) - mean.square()).clamp_min(1e-6).sqrt()
    scale = torch.stack([g['baseline'].std(unbiased=False) for g in groups]).median().clamp_min(.1)
    finite(mean, std, scale)
    return mean.float(), std.float(), scale.float()


def evaluate(model, groups, device, batch=16):
    import torch
    from head import pack, pair_loss
    model.eval()
    rows = []
    loss_sum = active_groups = 0
    with torch.no_grad():
        for start in range(0, len(groups), batch):
            chunk = groups[start:start+batch]
            x, mask, base, rmsd = pack(chunk, device=device)
            scores, residual = model(x, mask, base)
            finite(scores, residual)
            loss, active = pair_loss(scores.reshape(-1, 8), rmsd)
            loss_sum += float(loss) * int(active.sum())
            active_groups += int(active.sum())
            scores, residual = scores.cpu().tolist(), residual.cpu().tolist()
            for gi, group in enumerate(chunk):
                for i, pose in enumerate(group['pose_metadata']):
                    rows.append(dict(name=group['name'], split=group['split'], ordinal=pose['ordinal'],
                        pose_uid=pose['pose_uid'], rmsd=pose['rmsd'], baseline=float(group['baseline'][i]),
                        score=scores[gi*8+i], residual=residual[gi*8+i]))
    metrics = ranking_metrics(rows, 'score')
    metrics.update(pair_loss=loss_sum/max(1, active_groups), comparable_groups=active_groups)
    return metrics, rows


def update(model, groups, optimizer, regularization, device):
    import torch
    from head import pack, pair_loss
    model.train()
    x, mask, base, rmsd = pack(groups, device=device)
    optimizer.zero_grad(set_to_none=True)
    scores, residual = model(x, mask, base)
    ranking, active = pair_loss(scores.reshape(-1, 8), rmsd)
    loss = ranking + regularization * residual.square().mean()
    finite(scores, residual, loss)
    loss.backward()
    finite(*[p.grad for p in model.parameters() if p.grad is not None])
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    finite(*[p.data for p in model.parameters()])
    return float(loss.detach()), float(norm), int(active.sum())


def smoke(model, train, out, device):
    """Original train-only 8-group memorization/invariance check, then RESET."""
    import torch
    from head import pack
    started = time.monotonic()
    saved = copy.deepcopy(model.state_dict())
    mixed = [g for g in train if 0 < sum(value < 2 for value in g['rmsd']) < 8]
    selected = sorted(mixed, key=lambda group: key('smoke|' + group['name']))[:8]
    if len(selected) != 8:
        raise RuntimeError('Not enough mixed train groups for the unchanged fitting smoke')
    initial, _ = evaluate(model, selected, device)
    x, mask, base, _rmsd = pack(selected, device=device)
    model.eval()
    with torch.no_grad():
        initial_score, residual = model(x, mask, base)
        if not torch.equal(initial_score, base/model.score_scale) or not torch.equal(residual, torch.zeros_like(residual)):
            raise RuntimeError('Zero residual did not exactly preserve direct MDN')
        atom_order = torch.arange(x.shape[1]-1, -1, -1, device=x.device)
        pose_order = torch.arange(len(base)-1, -1, -1, device=x.device)
        permuted, _ = model(x[:, atom_order], mask[:, atom_order], base)
        reordered, _ = model(x[pose_order], mask[pose_order], base[pose_order])
        error = max(float((initial_score-permuted).abs().max()), float((initial_score[pose_order]-reordered).abs().max()))
        noisy = x.clone()
        noisy[~mask] = 10000
        padded, _ = model(noisy, mask, base)
        if not torch.allclose(initial_score, padded, atol=1e-6, rtol=1e-6):
            raise RuntimeError('Padding changed output')
        model.pose[-1].weight.normal_(0, .1)
        actual, _ = model(x, mask, base)
        permuted, _ = model(x[:, atom_order], mask[:, atom_order], base)
        padded, _ = model(noisy, mask, base)
        reordered, _ = model(x[pose_order], mask[pose_order], base[pose_order])
        error = max(error, float((actual-permuted).abs().max()), float((actual[pose_order]-reordered).abs().max()))
        if not torch.allclose(actual, padded, atol=1e-5, rtol=1e-5) or error > 1e-5:
            raise RuntimeError('Nonzero-head permutation/padding invariance failed')
    model.load_state_dict(saved)
    model.pose[2].p = 0.
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0)
    norms = [update(model, selected, optimizer, 0., device)[1] for _ in range(250)]
    final, _ = evaluate(model, selected, device)
    if not final['pair_loss'] < .99*initial['pair_loss'] or not max(norms) > 0:
        raise RuntimeError(f"Train-only smoke failed to learn: {initial['pair_loss']} -> {final['pair_loss']}")
    if state_digest(saved) == state_digest(model.state_dict()):
        raise RuntimeError('Smoke did not update parameters')
    model.load_state_dict(saved)
    model.pose[2].p = .1
    if state_digest(model.state_dict()) != state_digest(saved):
        raise RuntimeError('Smoke weights not reset')
    atomic_json(out/'SMOKE_READY.json', dict(status='PASS', groups=[g['name'] for g in selected], train_only=True,
        initial_pair_loss=initial['pair_loss'], final_pair_loss=final['pair_loss'],
        initial_top1_lt2=initial['top1_lt2'], final_top1_lt2=final['top1_lt2'], finite_gradients=True,
        formal_weights_reset=True, permutation_max_abs=error, padding_mask_passed=True,
        baseline_exact_at_initialization=True, seconds=time.monotonic()-started), new=True)
    event('rank_smoke_ready', initial_loss=initial['pair_loss'], final_loss=final['pair_loss'])


def save(path, model, epoch, validation, root, contract):
    atomic_torch(path, dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        epoch=epoch, val=compact(validation), arm='protein_mdn_transfer', head_class='ContactEvidenceResidual',
        architecture=dict(features=32, hidden=32, dropout=.1, residual_cap=2),
        rank_contract_sha256=sha(root/'rank_contract.json'), root_contract_sha256=sha(root/'contract.json'),
        native_head_sha256=contract['native_head_sha256'], source_bundle_sha256=contract['native_static_bundle_sha256'],
        feature_schema=FEATURE_NAMES, selection=RECIPE['selection'], baseline_selectable=True, test_opened=False))


def load_groups(root, contract, manifest, entries):
    import torch
    expected = {group['name']: group for group in manifest}
    allowed = {'name', 'split', 'tokens', 'baseline', 'pose_uids', 'rank_contract_sha256',
               'native_head_sha256', 'source_bundle_sha256'}
    groups = []
    for entry in entries:
        check_hash(entry['path'], entry['sha256'])
        item = torch.load(entry['path'], map_location='cpu', weights_only=True)
        if set(item) != allowed:
            raise RuntimeError('Unexpected cache field; labels/native coordinates are forbidden')
        group = expected[entry['name']]
        poses = sorted(group['poses'], key=lambda pose: pose['ordinal'])
        if (item['name'] != entry['name'] or item['split'] != group['split']
                or item['pose_uids'] != [pose['pose_uid'] for pose in poses]
                or item['rank_contract_sha256'] != sha(root/'rank_contract.json')
                or item['native_head_sha256'] != contract['native_head_sha256']
                or item['source_bundle_sha256'] != contract['native_static_bundle_sha256']):
            raise RuntimeError('Candidate evidence cache/contract alignment failure')
        if (tuple(item['tokens'].shape) != (8, poses[0]['atom_count'], 32)
                or tuple(item['baseline'].shape) != (8,)
                or item['tokens'].dtype != torch.float32 or item['baseline'].dtype != torch.float32):
            raise RuntimeError('Evidence shape/dtype mismatch')
        finite(item['tokens'], item['baseline'])
        # Labels are attached here, only after loading the independent feature cache.
        item.update(rmsd=[pose['rmsd'] for pose in poses], pose_metadata=poses)
        groups.append(item)
    return groups


def run(args):
    root = Path(args.root).resolve(strict=True)
    contract, manifest, _rows = validate_contract(root)
    feature_ready, entries = validate_features(root, contract, manifest)
    if (root/'rank/model').exists():
        raise RuntimeError('Rank fit attempt already exists; no overwrite or automatic retry')
    if not args.execute:
        event('rank_training_check_only', groups=890, poses=7120, device_not_initialized=True)
        return
    out = reserve_stage(root, 'model')
    current_epoch = None
    try:
        import numpy as np
        import torch
        from head import ContactEvidenceResidual
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is not available; no silent CPU fallback')
        seed()
        groups = load_groups(root, contract, manifest, entries)
        train = [group for group in groups if group['split'] == 'train']
        val = [group for group in groups if group['split'] == 'val']
        if (len(train), len(val)) != (801, 89):
            raise RuntimeError('Training/validation coverage mismatch')
        mean, std, scale = fit_scaler(train)
        atomic_json(out/'rank_scaler.json', dict(train_groups=801, fit_split='train_only', complex_balanced=True,
            train_split_sha256=SPLIT_SHAS['train'], train_names=sorted(group['name'] for group in train),
            feature_names=FEATURE_NAMES, mean=mean.tolist(), std=std.tolist(), score_scale=float(scale),
            feature_index_sha256=feature_ready['feature_index_sha256'], input_clip=8,
            validation_used=False, test_opened=False, old_V3_statistics_used=False), new=True)
        model = ContactEvidenceResidual(mean, std, scale).to(args.device)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        if parameters != RECIPE['trainable_parameters']:
            raise RuntimeError('ContactEvidenceResidual is no longer the fixed 4225-parameter architecture')
        smoke(model, train, out, args.device)
        seed()
        initial_train, _ = evaluate(model, train, args.device)
        initial_val, initial_rows = evaluate(model, val, args.device)
        expected = read_json(root/'rank/features/baseline_metrics.json')['val']
        if [(g['name'], g['top1']) for g in initial_val['groups']] != [(g['name'], g['top1']) for g in expected['groups']]:
            raise RuntimeError('Initial residual head changed baseline validation pose selection')
        for name in {row['name'] for row in initial_rows}:
            chunk = [row for row in initial_rows if row['name'] == name]
            chosen = lambda field: min(chunk, key=lambda row: (-row[field], row['ordinal']))['pose_uid']
            if chosen('score') != chosen('baseline') or any(row['residual'] != 0 for row in chunk):
                raise RuntimeError('Initial baseline candidate identity or zero residual changed')
        save(out/'initial_rank.pt', model, -1, initial_val, root, contract)
        save(out/'selected_rank.pt', model, -1, initial_val, root, contract)
        best_key, best_epoch, stale = select_key(initial_val), -1, 0
        best_trained_key = best_trained_epoch = None
        optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE['lr'], weight_decay=RECIPE['weight_decay'])
        history = [dict(epoch=-1, train=compact(initial_train), val=compact(initial_val), seconds=0, selected=True)]
        atomic_json(out/'history.json', history)
        started = time.monotonic()
        for epoch in range(RECIPE['epochs']):
            current_epoch = epoch
            epoch_start = time.monotonic()
            order = list(train)
            random.Random(SEED+epoch).shuffle(order)
            losses, norms = [], []
            for start in range(0, len(order), RECIPE['batch_groups']):
                loss, norm, _ = update(model, order[start:start+RECIPE['batch_groups']], optimizer,
                                       RECIPE['residual_l2'], args.device)
                losses.append(loss)
                norms.append(norm)
            train_metrics, _ = evaluate(model, train, args.device)
            val_metrics, _ = evaluate(model, val, args.device)
            candidate = select_key(val_metrics)
            improved = candidate > best_key
            if improved:
                best_key, best_epoch, stale = candidate, epoch, 0
                save(out/'selected_rank.pt', model, epoch, val_metrics, root, contract)
            else:
                stale += 1
            if best_trained_key is None or candidate > best_trained_key:
                best_trained_key, best_trained_epoch = candidate, epoch
                save(out/'best_trained_rank.pt', model, epoch, val_metrics, root, contract)
            save(out/'last_rank.pt', model, epoch, val_metrics, root, contract)
            record = dict(epoch=epoch, train=compact(train_metrics), val=compact(val_metrics),
                objective=float(np.mean(losses)), gradient_norm_max=max(norms), seconds=time.monotonic()-epoch_start,
                selected=improved, selected_epoch=best_epoch, stale=stale)
            history.append(record)
            atomic_json(out/'history.json', history)
            event('rank_epoch', **record)
            if epoch+1 >= RECIPE['min_epochs'] and stale >= RECIPE['patience']:
                break
        metrics, final_rows = {}, {}
        for label, filename in (('selected', 'selected_rank.pt'), ('best_trained', 'best_trained_rank.pt')):
            model.load_state_dict(torch.load(out/filename, map_location='cpu', weights_only=True)['model'], strict=True)
            metrics[label], final_rows[label] = {}, []
            for split, subset in (('train', train), ('val', val)):
                metric, scored = evaluate(model, subset, args.device)
                metrics[label][split] = metric
                final_rows[label].extend(scored)
        combined = []
        for selected, trained in zip(final_rows['selected'], final_rows['best_trained']):
            if (selected['name'], selected['pose_uid']) != (trained['name'], trained['pose_uid']):
                raise RuntimeError('Final score alignment failed')
            combined.append(dict(name=selected['name'], split=selected['split'], ordinal=selected['ordinal'],
                pose_uid=selected['pose_uid'], rmsd=selected['rmsd'], baseline=selected['baseline'],
                selected=selected['score'], best_trained=trained['score'], selected_residual=selected['residual']))
        if len(combined) != 7120:
            raise RuntimeError('Incomplete final K8 scoring population')
        with (out/'pose_scores.tsv').open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(combined[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(combined)
        metrics['baseline'] = read_json(root/'rank/features/baseline_metrics.json')
        atomic_json(out/'ranking_metrics.json', metrics, new=True)
        validate_contract(root)
        validate_features(root, contract, manifest)
        result = dict(status='PASS', selected_epoch=best_epoch, best_trained_epoch=best_trained_epoch,
            completed_epochs=epoch+1, stop_reason='patience5' if stale >= 5 else 'max30',
            rank_contract_sha256=sha(root/'rank_contract.json'), root_contract_sha256=sha(root/'contract.json'),
            split_counts=COUNTS, groups=890, poses=7120, trainable_parameters=parameters,
            backbone_calls_in_training=0, prior_frozen=True, fresh_optimizer=True, scheduler=None,
            test_opened=False, new_poses_generated=0, seconds=time.monotonic()-started,
            selected_checkpoint=str(out/'selected_rank.pt'), selected_checkpoint_sha256=sha(out/'selected_rank.pt'),
            best_trained_checkpoint_sha256=sha(out/'best_trained_rank.pt'),
            scaler_sha256=sha(out/'rank_scaler.json'), scores_sha256=sha(out/'pose_scores.tsv'),
            metrics_sha256=sha(out/'ranking_metrics.json'), feature_index_sha256=feature_ready['feature_index_sha256'],
            selected_val=compact(metrics['selected']['val']), baseline_val=compact(metrics['baseline']['val']))
        atomic_json(out/'TRAIN_READY.json', result, new=True)
        event('rank_train_ready', **result)
    except Exception:
        atomic_json(out/'FAILED.json', dict(error=traceback.format_exc(), epoch=current_epoch,
                    time=time.time(), automatic_retry=False, test_opened=False), new=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--execute', action='store_true')
    run(parser.parse_args())
