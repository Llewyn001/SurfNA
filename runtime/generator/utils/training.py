import copy
import random

import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils import so3, torus
from utils.sampling import randomize_position, sampling
import torch
from utils.diffusion_utils import get_t_schedule
from torch_geometric.data import Dataset,Data
from loguru import logger
import gc
import time


MAX_REASONABLE_LOSS = 1e6


def _batch_names(data):
    names = getattr(data, 'name', None)
    if isinstance(names, (list, tuple)):
        return ','.join(str(name) for name in names)
    return str(names)


def _batch_name_list(data):
    names = getattr(data, 'name', None)
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    if names is None:
        num_graphs = int(getattr(data, 'num_graphs', 1))
        return [f'graph_{idx}' for idx in range(num_graphs)]
    return [str(names)]


def _clear_grads(model, optimizer=None):
    if optimizer is not None:
        optimizer.zero_grad()
    for p in model.parameters():
        if p.grad is not None:
            del p.grad


def _empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _has_bad_loss(tensors, max_abs=MAX_REASONABLE_LOSS):
    for tensor in tensors:
        detached = tensor.detach()
        if not torch.isfinite(detached).all():
            return True
        if detached.numel() > 0 and torch.max(torch.abs(detached.float())).item() > max_abs:
            return True
    return False


def _loss_good_mask(tensors, num_graphs, device, max_abs=MAX_REASONABLE_LOSS):
    good = torch.ones(num_graphs, dtype=torch.bool, device=device)
    for tensor in tensors:
        detached = tensor.detach()
        if detached.dim() == 0:
            tensor_good = bool(torch.isfinite(detached).item()) and torch.abs(detached.float()).item() <= max_abs
            if not tensor_good:
                good &= torch.zeros_like(good)
            continue
        if detached.shape[0] != num_graphs:
            if not torch.isfinite(detached).all() or torch.max(torch.abs(detached.float())).item() > max_abs:
                good &= torch.zeros_like(good)
            continue
        flat = detached.reshape(num_graphs, -1).float()
        finite = torch.isfinite(flat).all(dim=1)
        bounded = torch.amax(torch.abs(flat), dim=1) <= max_abs
        good &= finite & bounded
    return good


def _filter_losses(tensors, good_mask):
    filtered = []
    for tensor in tensors:
        if tensor.dim() > 0 and tensor.shape[0] == good_mask.shape[0]:
            filtered.append(tensor[good_mask])
        else:
            filtered.append(tensor)
    return tuple(filtered)


def _sync_skip(accelerator, device, should_skip):
    flag = torch.tensor([1 if should_skip else 0], device=device)
    flags = accelerator.gather(flag)
    return int(flags.sum().item()) > 0


class ListDataset(Dataset):
    def __init__(self, list):
        super().__init__()
        self.data_list = list
    def len(self) -> int:
        return len(self.data_list)
    def get(self, idx: int) -> Data:
            return self.data_list[idx]


class RepeatedInferenceDataset(Dataset):
    """Lazily repeat validation graphs while retaining target/pose identity."""

    def __init__(self, data_list, repeats, padded_length):
        super().__init__()
        self.data_list = data_list
        self.repeats = int(repeats)
        self.real_length = len(data_list) * self.repeats
        self.padded_length = int(padded_length)

    def len(self) -> int:
        return self.padded_length

    def get(self, idx: int) -> Data:
        is_real = idx < self.real_length
        real_idx = idx if is_real else 0
        target_idx = real_idx // self.repeats
        pose_idx = real_idx % self.repeats
        graph = copy.deepcopy(self.data_list[target_idx])
        graph.val_target_idx = torch.tensor([target_idx], dtype=torch.long)
        graph.val_pose_idx = torch.tensor([pose_idx], dtype=torch.long)
        graph.val_sample_real = torch.tensor([1 if is_real else 0], dtype=torch.long)
        return graph

def loss_function(tr_pred, rot_pred, tor_pred, data, t_to_sigma, device,tr_weight=1, rot_weight=1,
                  tor_weight=1, apply_mean=True, no_torsion=False,
                  rot_low_noise_boost=0.0, tor_low_noise_boost=0.0,
                  low_noise_decay=4.0):
    tr_sigma, rot_sigma, tor_sigma = t_to_sigma(
        *[data.complex_t[noise_type]
          for noise_type in ['tr', 'rot', 'tor']])
    mean_dims = (0, 1) if apply_mean else 1


    tr_score = data.tr_score
    tr_sigma = tr_sigma.unsqueeze(-1)

    tr_loss = ((tr_pred - tr_score) ** 2 * tr_sigma ** 2).mean(dim=mean_dims)
    tr_base_loss = (tr_score ** 2 * tr_sigma ** 2).mean(dim=mean_dims)
    # rotation component
    rot_score = data.rot_score
    rot_score_norm = so3.score_norm(rot_sigma.cpu()).unsqueeze(-1).to(device)
    rot_loss = (((rot_pred - rot_score) / rot_score_norm) ** 2).mean(dim=mean_dims)
    rot_base_loss = ((rot_score / rot_score_norm) ** 2).mean(dim=mean_dims)
    # torsion component
    if not no_torsion:

        edge_tor_sigma = torch.from_numpy(
            np.concatenate(data.tor_sigma_edge))
        
        tor_score = data.tor_score
        tor_score_norm2 = torch.tensor(torus.score_norm(edge_tor_sigma.cpu().numpy())).float().to(device)

        tor_loss = ((tor_pred - tor_score) ** 2 / tor_score_norm2)
        tor_base_loss = ((tor_score ** 2 / tor_score_norm2))
        if apply_mean:
            tor_loss, tor_base_loss = tor_loss.mean() * torch.ones(1, dtype=torch.float,device = device), tor_base_loss.mean() * torch.ones(1, dtype=torch.float,device = device)
        else:
            index = data['ligand'].batch[
                data['ligand', 'ligand'].edge_index[0][data['ligand'].edge_mask]]
            num_graphs = data.num_graphs
            t_l, t_b_l, c = torch.zeros(num_graphs,device = device), torch.zeros(num_graphs,device = device), torch.zeros(num_graphs,device = device)

            c.index_add_(0, index, torch.ones(tor_loss.shape,device = device))
            c = c + 0.0001
            t_l.index_add_(0, index, tor_loss)
            t_b_l.index_add_(0, index, tor_base_loss)
            tor_loss, tor_base_loss = t_l / c, t_b_l / c
    else:
        if apply_mean:
            tor_loss, tor_base_loss = torch.zeros(1, dtype=torch.float,device = device), torch.zeros(1, dtype=torch.float,device = device)
        else:
            tor_loss, tor_base_loss = torch.zeros(len(rot_loss), dtype=torch.float,device = device), torch.zeros(len(rot_loss), dtype=torch.float,device = device)

    if not apply_mean:
        decay = max(0.0, float(low_noise_decay))
        if float(rot_low_noise_boost) != 0.0:
            rot_time_weight = 1.0 + float(rot_low_noise_boost) * torch.exp(
                -decay * data.complex_t['rot'].to(rot_loss)
            )
            rot_time_weight = rot_time_weight / rot_time_weight.mean().clamp_min(1e-8)
            rot_loss = rot_loss * rot_time_weight
            rot_base_loss = rot_base_loss * rot_time_weight
        if not no_torsion and float(tor_low_noise_boost) != 0.0:
            tor_time_weight = 1.0 + float(tor_low_noise_boost) * torch.exp(
                -decay * data.complex_t['tor'].to(tor_loss)
            )
            tor_time_weight = tor_time_weight / tor_time_weight.mean().clamp_min(1e-8)
            tor_loss = tor_loss * tor_time_weight
            tor_base_loss = tor_base_loss * tor_time_weight

    loss = tr_loss * tr_weight + rot_loss * rot_weight + tor_loss * tor_weight
  
    return loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss

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

                self.count[type_idx].index_add_(0, interval_idx[type_idx], torch.ones(len(v)))
                if not torch.allclose(v, torch.tensor(0.0)):
                    self.acc[self.types[type_idx]].index_add_(0, interval_idx[type_idx], v)
    def summary(self):
        if self.intervals == 1:
            if self.count == 0:
                return {k: float('nan') for k in self.acc.keys()}
            out = {k: v.item() / self.count for k, v in self.acc.items()}
            return out
        else:
            out = {}
            for i in range(self.intervals):
                for type_idx, k in enumerate(self.types):
                    count = self.count[type_idx][i]
                    out['int' + str(i) + '_' + k] = (
                            list(self.acc.values())[type_idx][i] / count).item() if count.item() > 0 else float('nan')
            return out


def train_epoch(model, loader, optimizer, device, t_to_sigma, loss_fn,accelerator,ema_weights):
    model.train()
    # if mdn_mode:
        # 
    meter = AverageMeter([
        'loss', 'tr_loss', 'rot_loss', 'tor_loss',
        'tr_base_loss', 'rot_base_loss', 'tor_base_loss',
        'task_aux_loss', 'optimized_loss'
    ])
    pbar = tqdm(loader, total=len(loader),disable=not accelerator.is_local_main_process)
    for batch_idx, data in enumerate(pbar, start=1):
        local_skip = False
        skip_reason = ''
        losses = None
        task_aux_loss = None
        task_aux_weight = 0.0
        if device.type == 'cuda' and len(data) == 1 or device.type == 'cpu' and data.num_graphs == 1:
            local_skip = True
            skip_reason = 'batch size 1'
        optimizer.zero_grad()
        if not local_skip:
            try:
                tr_pred, rot_pred, tor_pred = model(data)
                with accelerator.autocast():
                    raw_losses = loss_fn(
                        tr_pred, rot_pred, tor_pred, data=data, t_to_sigma=t_to_sigma,
                        apply_mean=False, device=device
                    )
                good_mask = _loss_good_mask(raw_losses, data.num_graphs, device)
                if not torch.all(good_mask):
                    names = _batch_name_list(data)
                    bad_names = [names[idx] for idx, ok in enumerate(good_mask.detach().cpu().tolist()) if not ok]
                    logger.info(
                        f'| WARNING: masking bad train complexes batch {batch_idx}/{len(loader)} '
                        f'bad={",".join(bad_names)} data={_batch_names(data)}'
                    )
                if not torch.any(good_mask):
                    local_skip = True
                    skip_reason = f'all complexes non-finite or >{MAX_REASONABLE_LOSS:g} loss'
                else:
                    losses = _filter_losses(raw_losses, good_mask)
                    unwrapped_model = accelerator.unwrap_model(model)
                    task_aux_loss = getattr(unwrapped_model, 'last_task_aux_loss', None)
                    task_aux_weight = float(getattr(unwrapped_model, 'task_aux_weight', 0.0))
                    if task_aux_loss is not None and _has_bad_loss([task_aux_loss]):
                        local_skip = True
                        skip_reason = 'non-finite task-aligned auxiliary loss'
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    local_skip = True
                    skip_reason = 'out of memory'
                    _empty_cache()
                elif 'Input mismatch' in str(e):
                    local_skip = True
                    skip_reason = 'torch_cluster input mismatch'
                    _empty_cache()
                else:
                    raise e

        if _sync_skip(accelerator, device, local_skip):
            if local_skip or accelerator.is_local_main_process:
                logger.info(f'| WARNING: skipping train batch {batch_idx}/{len(loader)} reason={skip_reason or "other rank requested skip"} data={_batch_names(data)}')
            _clear_grads(model, optimizer)
            _empty_cache()
            continue

        loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss = losses
        diffusion_loss = loss.mean()
        if task_aux_loss is None:
            task_aux_loss = diffusion_loss * 0.0
        train_loss = diffusion_loss + task_aux_weight * task_aux_loss
        accelerator.backward(train_loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        #gather all loss for plot
        loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss = \
            loss.mean(), tr_loss.mean(), rot_loss.mean(), tor_loss.mean(), \
                tr_base_loss.mean(), rot_base_loss.mean(), tor_base_loss.mean()
        loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss = \
            accelerator.gather(loss),accelerator.gather(tr_loss), accelerator.gather(rot_loss), \
                accelerator.gather(tor_loss), accelerator.gather(tr_base_loss), accelerator.gather(rot_base_loss), accelerator.gather(tor_base_loss)
        task_aux_loss = accelerator.gather(task_aux_loss.detach())
        optimized_loss = accelerator.gather(train_loss.detach())

        ema_weights.update(model.parameters())
        meter.add([
            loss.mean().cpu().detach(), tr_loss.mean().cpu().detach(), rot_loss.mean().cpu().detach(),
            tor_loss.mean().cpu().detach(), tr_base_loss.mean().cpu().detach(),
            rot_base_loss.mean().cpu().detach(), tor_base_loss.mean().cpu().detach(),
            task_aux_loss.mean().cpu().detach(), optimized_loss.mean().cpu().detach()
        ])
    logger.info('clear last train batch data and model grad')
    _clear_grads(model)
    if 'data' in locals():
        del data
    _empty_cache()
    return meter.summary()


def test_epoch(model, loader, device, t_to_sigma, loss_fn,accelerator, test_sigma_intervals=False,model_type = 'energy_score_model'):
    if not model_type == 'energy_score_model':
        model.eval()
    meter = AverageMeter(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss'],
                         unpooled_metrics=True)

    if test_sigma_intervals:
        meter_all = AverageMeter(
            ['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss'],
            unpooled_metrics=True, intervals=10)
    
    for batch_idx, data in enumerate(tqdm(loader, total=len(loader),disable=not accelerator.is_local_main_process), start=1):
        start_time = time.time()
        local_skip = False
        skip_reason = ''
        losses = None
        if accelerator.is_local_main_process:
            logger.info(f'validation batch {batch_idx}/{len(loader)} start data={_batch_names(data)}')
        try:
            if not model_type == 'energy_score_model':
                with torch.no_grad():
                    tr_pred, rot_pred, tor_pred = model(data)
            else:
                tr_pred, rot_pred, tor_pred = model(data)
            with accelerator.autocast():
                raw_losses = loss_fn(
                    tr_pred, rot_pred, tor_pred, data=data, t_to_sigma=t_to_sigma,
                    apply_mean=False, device=device
                )
            good_mask = _loss_good_mask(raw_losses, data.num_graphs, device)
            if not torch.all(good_mask):
                names = _batch_name_list(data)
                bad_names = [names[idx] for idx, ok in enumerate(good_mask.detach().cpu().tolist()) if not ok]
                logger.info(
                    f'| WARNING: masking bad validation complexes batch {batch_idx}/{len(loader)} '
                    f'bad={",".join(bad_names)} data={_batch_names(data)}'
                )
            if not torch.any(good_mask):
                local_skip = True
                skip_reason = f'all complexes non-finite or >{MAX_REASONABLE_LOSS:g} loss'
            else:
                losses = _filter_losses(raw_losses, good_mask)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                local_skip = True
                skip_reason = 'out of memory'
                _empty_cache()
            elif 'Input mismatch' in str(e):
                local_skip = True
                skip_reason = 'torch_cluster input mismatch'
                _empty_cache()
            else:
                raise e

        if _sync_skip(accelerator, device, local_skip):
            if local_skip or accelerator.is_local_main_process:
                logger.info(f'| WARNING: skipping validation batch {batch_idx}/{len(loader)} reason={skip_reason or "other rank requested skip"} data={_batch_names(data)}')
            _clear_grads(model)
            _empty_cache()
            continue

        loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss = losses
        loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss = \
            loss.mean(), tr_loss.mean(), rot_loss.mean(), tor_loss.mean(), \
                tr_base_loss.mean(), rot_base_loss.mean(), tor_base_loss.mean()
        loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss = \
            accelerator.gather(loss), accelerator.gather(tr_loss), accelerator.gather(rot_loss), \
                accelerator.gather(tor_loss), accelerator.gather(tr_base_loss), \
                accelerator.gather(rot_base_loss), accelerator.gather(tor_base_loss)

        metrics = [loss.mean().cpu().detach(), tr_loss.mean().cpu().detach(), \
                   rot_loss.mean().cpu().detach(), tor_loss.mean().cpu().detach(), \
                    tr_base_loss.mean().cpu().detach(), rot_base_loss.mean().cpu().detach(), tor_base_loss.mean().cpu().detach()]
        meter.add(metrics)

        if accelerator.is_local_main_process:
            elapsed = time.time() - start_time
            if elapsed > 5:
                logger.info(f'validation batch {batch_idx}/{len(loader)} finished in {elapsed:.1f}s data={_batch_names(data)}')

        if test_sigma_intervals > 0:
            complex_t_tr, complex_t_rot, complex_t_tor = [data.complex_t[noise_type] for
                                                          noise_type in ['tr', 'rot', 'tor']]
            sigma_index_tr = torch.round(complex_t_tr.cpu() * (10 - 1)).long().to(device)
            sigma_index_rot = torch.round(complex_t_rot.cpu() * (10 - 1)).long().to(device)
            sigma_index_tor = torch.round(complex_t_tor.cpu() * (10 - 1)).long().to(device)
            sigma_index_tr = accelerator.gather(sigma_index_tr).cpu().detach()
            sigma_index_rot = accelerator.gather(sigma_index_rot).cpu().detach()
            sigma_index_tor = accelerator.gather(sigma_index_tor).cpu().detach()
            meter_all.add(
                metrics,
                [sigma_index_tr, sigma_index_tr, sigma_index_rot, sigma_index_tor, sigma_index_tr, sigma_index_rot,
                 sigma_index_tor, sigma_index_tr])
    logger.info('clear val batch data and model grad')
    _clear_grads(model)
    if 'data' in locals():
        del data
    _empty_cache()
    out = meter.summary()
    if test_sigma_intervals > 0: out.update(meter_all.summary())
    return out


def inference_epoch(model, complex_graphs, device, t_to_sigma, args,accelerator):
    
    t_schedule = get_t_schedule(inference_steps=args.inference_steps)
    tr_schedule, rot_schedule, tor_schedule = t_schedule, t_schedule, t_schedule

    dataset = ListDataset(complex_graphs)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    loader = accelerator.prepare(loader)
    rmsds = []
    logger.info(f'dataset size {len(dataset)}')
    for orig_complex_graph in tqdm(loader,disable=not accelerator.is_local_main_process):

        data_list = [copy.deepcopy(orig_complex_graph)]
        randomize_position(data_list, args.no_torsion, False, args.tr_sigma_max)

        # unwrap model safely (Accelerate/DDP/vanilla)
        _model = model
        try:
            if accelerator is not None:
                _model = accelerator.unwrap_model(model)
        except Exception:
            _model = model.module if hasattr(model, 'module') else model

        predictions_list = None
        confidences = None
        failed_convergence_counter = 0
        while predictions_list == None:
            try:
                predictions_list, confidences = sampling(input_data_list=data_list, model=_model,
                                                         inference_steps=args.inference_steps,
                                                         tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                                         tor_schedule=tor_schedule,
                                                         device=device, t_to_sigma=t_to_sigma, model_args=args)
                # predictions_list, confidences = sampling(input_data_list=data_list, model=model,
                #                                          inference_steps=args.inference_steps,
                #                                          tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                #                                          tor_schedule=tor_schedule,
                #                                          device=device, t_to_sigma=t_to_sigma, model_args=args)
            except Exception as e:
                if 'failed to converge' in str(e):
                    failed_convergence_counter += 1
                    if failed_convergence_counter > 5:
                        logger.info('| WARNING: SVD failed to converge 5 times - skipping the complex')
                        break
                    logger.info('| WARNING: SVD failed to converge - trying again with a new sample')
                else:
                    raise e
        if failed_convergence_counter > 5: continue
        if args.no_torsion:
            orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos.cpu().numpy() +
                                                     orig_complex_graph.original_center.cpu().numpy())

        filterHs = torch.not_equal(predictions_list[0]['ligand'].x[:, 0], 0).cpu().numpy()

        if isinstance(orig_complex_graph['ligand'].orig_pos, list):
            orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]

        ligand_pos = np.asarray(
            [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in predictions_list])
        orig_ligand_pos = np.expand_dims(
            orig_complex_graph['ligand'].orig_pos[filterHs] - orig_complex_graph.original_center.cpu().numpy(), axis=0)
        rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))
        rmsds.append(rmsd)
    rmsds = np.array(rmsds)
    logger.info(f'rmsd: {rmsds}')
    losses = {'rmsds_lt2': (100 * (rmsds < 2).sum() / len(rmsds)),
              'rmsds_lt5': (100 * (rmsds < 5).sum() / len(rmsds))}
    del dataset, loader,predictions_list, confidences,ligand_pos, orig_ligand_pos, rmsd,filterHs
    gc.collect()
    torch.cuda.empty_cache()
    return losses
def inference_epoch_parallel(model, complex_graphs, device, t_to_sigma, args,accelerator):
    t_schedule = get_t_schedule(inference_steps=args.inference_steps)
    tr_schedule, rot_schedule, tor_schedule = t_schedule, t_schedule, t_schedule

    samples_per_complex = max(1, int(getattr(args, 'val_samples_per_complex', 1)))
    inference_seed = int(getattr(args, 'val_inference_seed', 20260824))
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    rank_seed = inference_seed + int(accelerator.process_index)
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)

    real_length = len(complex_graphs) * samples_per_complex
    shard_multiple = max(1, int(accelerator.num_processes) * int(args.batch_size))
    padded_length = ((real_length + shard_multiple - 1) // shard_multiple) * shard_multiple
    dataset = RepeatedInferenceDataset(complex_graphs, samples_per_complex, padded_length)
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)
    loader = accelerator.prepare(loader)
    rmsds = []
    target_indices = []
    pose_indices = []
    real_sample_mask = []
    for orig_complex_graph in tqdm(loader,disable=not accelerator.is_local_main_process):
        orig_complex_graph_list = orig_complex_graph.to_data_list()
        data_list = [copy.deepcopy(graph) for graph in orig_complex_graph_list ]
        randomize_position(data_list, args.no_torsion, False, args.tr_sigma_max)

        # unwrap model safely (Accelerate/DDP/vanilla)
        _model = model
        try:
            if accelerator is not None:
                _model = accelerator.unwrap_model(model)
        except Exception:
            _model = model.module if hasattr(model, 'module') else model

        predictions_list = None
        confidences = None
        failed_convergence_counter = 0
        while predictions_list == None:
            try:
                predictions_list, confidences = sampling(input_data_list=data_list, model=_model,
                                                         inference_steps=args.inference_steps,
                                                         tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                                         tor_schedule=tor_schedule,
                                                         device=device, t_to_sigma=t_to_sigma, model_args=args)
            except Exception as e:
                if 'failed to converge' in str(e):
                    failed_convergence_counter += 1
                    if failed_convergence_counter > 5:
                        logger.info('| WARNING: SVD failed to converge 5 times - skipping the complex')
                        break
                    logger.info('| WARNING: SVD failed to converge - trying again with a new sample')
                else:
                    raise e
        if failed_convergence_counter > 5: continue
        for pos_idx ,(predict_graph,orig_graph) in enumerate(zip(predictions_list,orig_complex_graph_list)):

            filterHs = torch.not_equal(predict_graph['ligand'].x[:, 0], 0)
            ligand_pos = predict_graph['ligand'].pos[filterHs].to(model.device)
            orig_ligand_pos = orig_graph['ligand'].pos[filterHs].to(model.device)
            rmsd = torch.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum()/(ligand_pos.shape[0]))
            rmsds.append(rmsd)
            target_indices.append(orig_graph.val_target_idx.reshape(-1)[0].to(model.device))
            pose_indices.append(orig_graph.val_pose_idx.reshape(-1)[0].to(model.device))
            real_sample_mask.append(orig_graph.val_sample_real.reshape(-1)[0].to(model.device))
    rmsds = torch.stack(rmsds)
    target_indices = torch.stack(target_indices).long()
    pose_indices = torch.stack(pose_indices).long()
    real_sample_mask = torch.stack(real_sample_mask).bool()

    rmsds = accelerator.gather(rmsds)
    target_indices = accelerator.gather(target_indices)
    pose_indices = accelerator.gather(pose_indices)
    real_sample_mask = accelerator.gather(real_sample_mask)
    rmsds = rmsds[real_sample_mask]
    target_indices = target_indices[real_sample_mask]
    pose_indices = pose_indices[real_sample_mask]

    pair_ids = target_indices * samples_per_complex + pose_indices
    expected_pairs = len(complex_graphs) * samples_per_complex
    if len(pair_ids) != expected_pairs or torch.unique(pair_ids).numel() != expected_pairs:
        raise RuntimeError(
            f'Validation K-pose identity audit failed: observed={len(pair_ids)} '
            f'unique={torch.unique(pair_ids).numel()} expected={expected_pairs}'
        )

    best_rmsds = []
    for target_idx in range(len(complex_graphs)):
        target_rmsds = rmsds[target_indices == target_idx]
        if target_rmsds.numel() != samples_per_complex:
            raise RuntimeError(
                f'Validation target {target_idx} has {target_rmsds.numel()} poses; '
                f'expected {samples_per_complex}'
            )
        best_rmsds.append(torch.min(target_rmsds))
    best_rmsds = torch.stack(best_rmsds)

    losses = {
        'rmsds_lt2': (100 * (best_rmsds < 2).sum() / len(best_rmsds)),
        'rmsds_lt5': (100 * (best_rmsds < 5).sum() / len(best_rmsds)),
        'median_best_rmsd': torch.median(best_rmsds),
        'pose_density_lt2': (100 * (rmsds < 2).sum() / len(rmsds)),
        'pose_density_lt5': (100 * (rmsds < 5).sum() / len(rmsds)),
        'samples_per_complex': torch.tensor(float(samples_per_complex), device=rmsds.device),
    }

    random.setstate(python_rng_state)
    np.random.set_state(numpy_rng_state)
    torch.set_rng_state(torch_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state)

    del dataset, loader,predictions_list, confidences,ligand_pos, orig_ligand_pos, rmsd,filterHs,complex_graphs
    gc.collect()
    torch.cuda.empty_cache()
    return losses
