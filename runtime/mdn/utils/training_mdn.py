# import copy
import numpy as np
from tqdm import tqdm
import torch
import gc
from torch.distributions import Normal
from loguru import logger
# from training import AverageMeter
# from mdn_utils import mdn_loss_fn 
def mdn_loss_fn(pi, sigma, mu, y,dist_threhold=7.0,eps = 1e-10):
    mu = torch.clip(torch.nan_to_num(mu,0.0),min=1e-6)
    sigma = torch.clip(torch.nan_to_num(sigma,0.0),min=1e-6)
    pi = torch.clip(torch.nan_to_num(pi,0.0),min=1e-6)
    y = torch.nan_to_num(y, nan=dist_threhold + 1.0, posinf=dist_threhold + 1.0, neginf=0.0)
    
    """calculate the mdn """
    normal = Normal(mu.real, sigma.real)
    loglik = normal.log_prob(y.expand_as(normal.loc))
    loss = -torch.logsumexp(torch.log(pi.real + eps) + loglik, dim=1)
    contact_mask = y.reshape(-1) <= dist_threhold
    if not torch.any(contact_mask):
        return loss.sum() * 0.0
    loss = loss[contact_mask]
    loss = loss.mean()
    if not torch.isfinite(loss):
        return pi.sum() * 0.0
    return loss
# def mdn_loss_fn(pi, sigma, mu, y, eps=1e-10):
#     normal = Normal(mu, sigma)
#     #loss = th.exp(normal.log_prob(y.expand_as(normal.loc)))
#     #loss = th.sum(loss * pi, dim=1)
#     #loss = -th.log(loss)
#     loglik = normal.log_prob(y.expand_as(normal.loc))
#     loss = -th.logsumexp(th.log(pi + eps) + loglik, dim=1)
#     return loss
import torch as th

def mdn_loss_fn_min_diatance_atom(pi, sigma, mu, y,dist_threhold=7.0,eps = 1e-10,topN = 1):
    mu = torch.clip(torch.nan_to_num(mu,0.0),min=1e-6)
    sigma = torch.clip(torch.nan_to_num(sigma,0.0),min=1e-6)
    pi = torch.clip(torch.nan_to_num(pi,0.0),min=1e-6)
    y = torch.nan_to_num(y, nan=dist_threhold + 1.0, posinf=dist_threhold + 1.0, neginf=0.0)

    """use ca- pose to calculate the mdn """
    normal = Normal(mu.real, sigma.real)
    loglik = normal.log_prob(y.expand_as(normal.loc))
    loss = -torch.logsumexp(torch.log(pi.real + eps) + loglik, dim=1)
    contact_mask = y.reshape(-1) <= dist_threhold
    if not torch.any(contact_mask):
        return loss.sum() * 0.0
    loss = loss[contact_mask].mean()
    if not torch.isfinite(loss):
        return pi.sum() * 0.0
    return loss
def calculate_probablity(pi, sigma, mu, y,dist_threhold=5.0,eps = 1e-10):
    mu = torch.clip(torch.nan_to_num(mu,0.0),min=1e-6)
    sigma = torch.clip(torch.nan_to_num(sigma,0.0),min=1e-6)
    pi = torch.clip(torch.nan_to_num(pi,0.0),min=1e-6)
    normal = Normal(mu.real, sigma.real)
    logprob = normal.log_prob(y.expand_as(normal.loc))
    logprob += torch.log(pi.real + eps )
    prob = logprob.exp().sum(1)
    prob[torch.where(y > dist_threhold)[0]] = 0.
    return prob

class AverageMeter():
    def __init__(self, types, unpooled_metrics=False, intervals=1):
        self.types = types
        self.intervals = intervals
        self.count = 0 if intervals == 1 else torch.zeros(len(types), intervals)
        self.acc = {t: torch.zeros(intervals) for t in types}
        self.unpooled_metrics = unpooled_metrics

    def add(self, vals, interval_idx=None):
        if self.intervals == 1:
            self.count += 1 if vals[0].dim() == 0 else len(vals[0])
            for type_idx, v in enumerate(vals):
                self.acc[self.types[type_idx]] += v.sum() if self.unpooled_metrics else v
        else:
            for type_idx, v in enumerate(vals):
                # logger.info(interval_idx[type_idx])
                # logger.info(v)
                # logger.info(interval_idx[type_idx], torch.ones(len(v)))
                self.count[type_idx].index_add_(0, interval_idx[type_idx], torch.ones(len(v)))
                if not torch.allclose(v, torch.tensor(0.0)):
                    self.acc[self.types[type_idx]].index_add_(0, interval_idx[type_idx], v)

    def summary(self):
        if self.intervals == 1:
            if self.count == 0:
                return {k: float('inf') for k in self.acc}
            out = {k: v.item() / self.count for k, v in self.acc.items()}
            return out
        else:
            out = {}
            for i in range(self.intervals):
                for type_idx, k in enumerate(self.types):
                    out['int' + str(i) + '_' + k] = (
                            list(self.acc.values())[type_idx][i] / self.count[type_idx][i]).item()
            return out


def _model_args(model):
    return getattr(model.module if hasattr(model, 'module') else model, 'args', None)


def _is_decoy_supervised(model):
    args = _model_args(model)
    return getattr(args, 'mdn_rerank_mode', 'native_mdn') == 'decoy_supervised'


def _decoy_targets(data, device):
    rmsd = data.decoy_rmsd.float().view(-1).to(device)
    group = data.decoy_group_id.long().view(-1).to(device)
    label = torch.exp(-rmsd / 2.0).clamp(0.0, 1.0)
    return rmsd, group, label


def _pairwise_ranking_loss(scores, rmsd, group, margin=0.2):
    losses = []
    for gid in torch.unique(group):
        mask = group == gid
        if int(mask.sum()) < 2:
            continue
        s = scores[mask]
        r = rmsd[mask]
        better = r[:, None] + 1e-6 < r[None, :]
        if not torch.any(better):
            continue
        score_delta = s[:, None] - s[None, :]
        losses.append(torch.relu(float(margin) - score_delta[better]).mean())
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def _rerank_metrics(scores, rmsd, group):
    rows = []
    for gid in torch.unique(group):
        mask = group == gid
        if int(mask.sum()) == 0:
            continue
        s = scores[mask].detach().float().cpu()
        r = rmsd[mask].detach().float().cpu()
        order = torch.argsort(s, descending=True)
        top1 = r[order[0]]
        topk = min(5, int(r.numel()))
        top5 = torch.min(r[order[:topk]])
        oracle = torch.min(r)
        rows.append((float(top1), float(top5), float(oracle)))
    if not rows:
        return {}
    top1 = torch.tensor([r[0] for r in rows])
    top5 = torch.tensor([r[1] for r in rows])
    oracle = torch.tensor([r[2] for r in rows])
    return {
        'rerank_top1_lt2': float((top1 < 2.0).float().mean().item() * 100.0),
        'rerank_top1_lt5': float((top1 < 5.0).float().mean().item() * 100.0),
        'rerank_top5_lt2': float((top5 < 2.0).float().mean().item() * 100.0),
        'rerank_top5_lt5': float((top5 < 5.0).float().mean().item() * 100.0),
        'rerank_oracle_lt2': float((oracle < 2.0).float().mean().item() * 100.0),
        'rerank_oracle_lt5': float((oracle < 5.0).float().mean().item() * 100.0),
        'rerank_top1_median': float(torch.median(top1).item()),
        'rerank_n_complexes': float(len(rows)),
    }


def _decoy_losses(model, data, outputs, device):
    args = _model_args(model)
    mdn_loss_interaction, mdn_loss_ligand, atom_types_loss, bond_types_loss, residue_types_loss, pose_score = outputs
    pose_score = pose_score.view(-1)
    rmsd, group, label = _decoy_targets(data, device)
    pose_bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(pose_score, label)
    pairwise_loss = _pairwise_ranking_loss(pose_score, rmsd, group, margin=getattr(args, 'pairwise_margin', 0.2))
    aux_loss_weight = getattr(args, 'mdn_aux_loss_weight', 0.001)
    loss = (
        getattr(args, 'mdn_nll_weight', 0.2) * (mdn_loss_interaction + mdn_loss_ligand)
        + getattr(args, 'pose_loss_weight', 1.0) * pose_bce_loss
        + getattr(args, 'pairwise_loss_weight', 0.5) * pairwise_loss
        + aux_loss_weight * (atom_types_loss + bond_types_loss + residue_types_loss)
    )
    metrics = {
        'loss': loss,
        'mdn_loss_interaction': mdn_loss_interaction,
        'mdn_loss_ligand': mdn_loss_ligand,
        'atom_types_loss': atom_types_loss,
        'bond_types_loss': bond_types_loss,
        'residue_types_loss': residue_types_loss,
        'pose_bce_loss': pose_bce_loss,
        'pairwise_loss': pairwise_loss,
    }
    return loss, metrics, pose_score, rmsd, group

def train_mdn_epoch(model, loader, optimizer, device,accelerator,ema_weights, gradient_accumulation_steps=1,
                    aux_loss_weight=0.001):
    model.train()
    decoy_mode = _is_decoy_supervised(model)
    metric_names = ['loss','mdn_loss_interaction', 'mdn_loss_ligand', 'atom_types_loss', 'bond_types_loss', 'residue_types_loss']
    if decoy_mode:
        metric_names += ['pose_bce_loss', 'pairwise_loss']
    meter = AverageMeter(metric_names, unpooled_metrics=True)
    gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
    optimizer.zero_grad()
    accumulated_steps = 0

    for data in loader:
        if data is None:
            logger.info('| WARNING: skipping empty MDN train batch')
            continue
    # for data in tqdm(loader, total=len(loader),disable=not accelerator.is_local_main_process):
        if device.type == 'cuda' and len(data) == 1 or device.type == 'cpu' and data.num_graphs == 1:
            logger.info("Skipping batch of size 1 since otherwise batchnorm would not work.")
        try:
            # pi, sigma, mu, dist = model(data)
            with accelerator.autocast():
                outputs = model(data)#mdn_loss_fn(pi, sigma, mu, dist)
                if decoy_mode:
                    loss, decoy_metrics, _, _, _ = _decoy_losses(model, data, outputs, device)
                    mdn_loss_interaction = decoy_metrics['mdn_loss_interaction']
                    mdn_loss_ligand = decoy_metrics['mdn_loss_ligand']
                    atom_types_loss = decoy_metrics['atom_types_loss']
                    bond_types_loss = decoy_metrics['bond_types_loss']
                    residue_types_loss = decoy_metrics['residue_types_loss']
                    pose_bce_loss = decoy_metrics['pose_bce_loss']
                    pairwise_loss = decoy_metrics['pairwise_loss']
                else:
                    mdn_loss_interaction , mdn_loss_ligand , atom_types_loss , bond_types_loss , residue_types_loss = outputs
                    loss = mdn_loss_interaction + mdn_loss_ligand + aux_loss_weight*atom_types_loss + aux_loss_weight*bond_types_loss + aux_loss_weight*residue_types_loss
                if not torch.isfinite(loss).all():
                    logger.info(
                        '| WARNING: skipping non-finite MDN train batch: '
                        f'loss={loss.detach().float().cpu().item() if loss.numel() == 1 else loss}'
                    )
                    optimizer.zero_grad()
                    accumulated_steps = 0
                    continue
                accelerator.backward(loss / gradient_accumulation_steps)
                accumulated_steps += 1
                if accumulated_steps == gradient_accumulation_steps:
                    optimizer.step()
                    optimizer.zero_grad()
                    ema_weights.update(model.parameters())
                    accumulated_steps = 0
            # gather all loss for plot
            mdn_loss_interaction= accelerator.gather(mdn_loss_interaction)
            mdn_loss_ligand= accelerator.gather(mdn_loss_ligand)
            atom_types_loss= accelerator.gather(atom_types_loss)
            bond_types_loss= accelerator.gather(bond_types_loss)
            residue_types_loss= accelerator.gather(residue_types_loss)
            loss= accelerator.gather(loss)
            # logger.info('loss val: ',loss.mean().cpu().detach())
            metrics = [loss.mean().cpu().detach(),mdn_loss_interaction.mean().cpu().detach() , mdn_loss_ligand.mean().cpu().detach() , atom_types_loss.mean().cpu().detach() , bond_types_loss.mean().cpu().detach() , residue_types_loss.mean().cpu().detach()]
            if decoy_mode:
                metrics.extend([accelerator.gather(pose_bce_loss).mean().cpu().detach(), accelerator.gather(pairwise_loss).mean().cpu().detach()])
            meter.add(metrics)
            # logger.info('loss train: ',loss.mean().cpu().detach())
            # meter.add([loss.mean().cpu().detach()])
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.info('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                optimizer.zero_grad()
                accumulated_steps = 0
                del data
                # loss = 0.0*sum([p.sum() for p in model.parameters() if p.requires_grad])
                # accelerator.backward(loss)
                # optimizer.step()
                gc.collect()
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                logger.info('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                optimizer.zero_grad()
                accumulated_steps = 0
                del data
                gc.collect()
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    if accumulated_steps > 0:
        optimizer.step()
        optimizer.zero_grad()
        ema_weights.update(model.parameters())
    return meter.summary()


def test_mdn_epoch(model, loader, device,accelerator, test_sigma_intervals=False, aux_loss_weight=0.001):
    model.eval()
    decoy_mode = _is_decoy_supervised(model)
    metric_names = ['loss','mdn_loss_interaction', 'mdn_loss_ligand', 'atom_types_loss', 'bond_types_loss', 'residue_types_loss']
    if decoy_mode:
        metric_names += ['pose_bce_loss', 'pairwise_loss']
    meter = AverageMeter(metric_names, unpooled_metrics=True)
    all_scores, all_rmsd, all_group = [], [], []

    if test_sigma_intervals:
        meter_all = AverageMeter(
            ['loss'],
            unpooled_metrics=True, intervals=10)

    for data in loader:
        if data is None:
            logger.info('| WARNING: skipping empty MDN validation batch')
            continue
        try:
            with torch.no_grad():
                    # pi, sigma, mu, dist,_ = model(data)
                with accelerator.autocast():
                    outputs = model(data)#mdn_loss_fn(pi, sigma, mu, dist)
                    if decoy_mode:
                        loss, decoy_metrics, pose_score, rmsd, group = _decoy_losses(model, data, outputs, device)
                        mdn_loss_interaction = decoy_metrics['mdn_loss_interaction']
                        mdn_loss_ligand = decoy_metrics['mdn_loss_ligand']
                        atom_types_loss = decoy_metrics['atom_types_loss']
                        bond_types_loss = decoy_metrics['bond_types_loss']
                        residue_types_loss = decoy_metrics['residue_types_loss']
                        pose_bce_loss = decoy_metrics['pose_bce_loss']
                        pairwise_loss = decoy_metrics['pairwise_loss']
                        all_scores.append(pose_score.detach().float().cpu())
                        all_rmsd.append(rmsd.detach().float().cpu())
                        all_group.append(group.detach().long().cpu())
                    else:
                        mdn_loss_interaction , mdn_loss_ligand , atom_types_loss , bond_types_loss , residue_types_loss,_ = outputs
                        loss = mdn_loss_interaction + mdn_loss_ligand + aux_loss_weight*atom_types_loss + aux_loss_weight*bond_types_loss + aux_loss_weight*residue_types_loss
                    if not torch.isfinite(loss).all():
                        logger.info(
                            '| WARNING: skipping non-finite MDN validation batch: '
                            f'loss={loss.detach().float().cpu().item() if loss.numel() == 1 else loss}'
                        )
                        continue
            mdn_loss_interaction= accelerator.gather(mdn_loss_interaction)
            mdn_loss_ligand= accelerator.gather(mdn_loss_ligand)
            atom_types_loss= accelerator.gather(atom_types_loss)
            bond_types_loss= accelerator.gather(bond_types_loss)
            residue_types_loss= accelerator.gather(residue_types_loss)
            loss= accelerator.gather(loss)
            # logger.info('loss val: ',loss.mean().cpu().detach())
            metrics = [loss.mean().cpu().detach(),mdn_loss_interaction.mean().cpu().detach() , mdn_loss_ligand.mean().cpu().detach() , atom_types_loss.mean().cpu().detach() , bond_types_loss.mean().cpu().detach() , residue_types_loss.mean().cpu().detach()]
            if decoy_mode:
                metrics.extend([accelerator.gather(pose_bce_loss).mean().cpu().detach(), accelerator.gather(pairwise_loss).mean().cpu().detach()])
            meter.add(metrics)

        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.info('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                del data
                gc.collect()
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                logger.info('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                del data
                gc.collect()
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    out = meter.summary()
    if decoy_mode and all_scores:
        out.update(_rerank_metrics(torch.cat(all_scores), torch.cat(all_rmsd), torch.cat(all_group)))
    # if test_sigma_intervals > 0: out.update(meter_all.summary())
    return out
