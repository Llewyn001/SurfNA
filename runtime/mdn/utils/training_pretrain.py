import copy

import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils import so3, torus
from utils.sampling import randomize_position, sampling
import torch
from utils.diffusion_utils import get_t_schedule
from torch_geometric.data import Dataset,Data
from loguru import logger
class ListDataset(Dataset):
    def __init__(self, list):
        super().__init__()
        self.data_list = list
    def len(self) -> int:
        return len(self.data_list)
    def get(self, idx: int) -> Data:
        return self.data_list[idx]
import gc

HUGE_LOSS_WARN_THRESHOLD = 100.0

def unpack_score_model_outputs(outputs, device):
    if len(outputs) == 4:
        return outputs
    if len(outputs) == 3:
        tr_pred, rot_pred, tor_pred = outputs
        return tr_pred, rot_pred, tor_pred, torch.zeros((), device=device)
    raise ValueError(f"Expected model to return 3 or 4 outputs, got {len(outputs)}")

def tensors_are_finite(*tensors):
    for tensor in tensors:
        if tensor is None:
            continue
        if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
            return False
    return True


def gradients_are_finite(parameters):
    """Return False before an optimizer step can poison the model state."""
    for parameter in parameters:
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return False
    return True

def format_batch_names(data):
    names = getattr(data, 'name', None)
    if names is None:
        return '<unknown>'
    return names

def loss_function(tr_pred, rot_pred, tor_pred, data, t_to_sigma, device,tr_weight=1, rot_weight=1,
                  tor_weight=1, apply_mean=True, no_torsion=False):
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
    # A valid batch can contain only rigid ligands.  In that case the torsion
    # tensors are empty and ``mean()`` would otherwise turn a well-defined
    # zero torsion contribution into NaN.
    if not no_torsion and tor_pred.numel() > 0 and data.tor_score.numel() > 0:

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
                return {k: float('inf') for k in self.acc.keys()}
            out = {k: v.item() / self.count for k, v in self.acc.items()}
            return out
        else:
            out = {}
            for i in range(self.intervals):
                for type_idx, k in enumerate(self.types):
                    out['int' + str(i) + '_' + k] = (
                            list(self.acc.values())[type_idx][i] / self.count[type_idx][i]).item()
            return out


def train_epoch(model, loader, optimizer, device, t_to_sigma, loss_fn,accelerator,ema_weights):
    model.train()
    # if mdn_mode:
        # 
    meter = AverageMeter(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'mask_loss'])
    pbar = tqdm(loader, total=len(loader),disable=not accelerator.is_local_main_process)
    for data in pbar:
        if device.type == 'cuda' and len(data) == 1 or device.type == 'cpu' and data.num_graphs == 1:
            logger.info("Skipping batch of size 1 since otherwise batchnorm would not work.")
            continue
        optimizer.zero_grad()
        try:
            tr_pred, rot_pred, tor_pred, mask_loss = unpack_score_model_outputs(model(data), device)
            with accelerator.autocast():
                loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss = \
                    loss_fn(tr_pred, rot_pred, tor_pred, data=data, t_to_sigma=t_to_sigma, device=device)
                loss = loss + mask_loss*0.33
            finite_status = torch.tensor(
                0 if tensors_are_finite(loss, tr_loss, rot_loss, tor_loss, tr_pred, rot_pred, tor_pred, mask_loss) else 1,
                device=device
            )
            global_finite_status = accelerator.reduce(finite_status, reduction="sum")
            if global_finite_status.item() > 0:
                logger.info('| WARNING: non-finite loss/prediction detected, skipping batch.')
                logger.info(f'forward data: {format_batch_names(data)}')
                accelerator.wait_for_everyone()  # 同步所有设备
                optimizer.zero_grad()
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory  
                
                del data, loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss
                gc.collect()
                torch.cuda.empty_cache()
                continue
            if torch.max(torch.abs(loss.detach())).item() > HUGE_LOSS_WARN_THRESHOLD:
                logger.info(
                    f'| WARNING: huge train loss {loss.detach().cpu().flatten().tolist()} '
                    f'tr={tr_loss.detach().cpu().flatten().tolist()} '
                    f'rot={rot_loss.detach().cpu().flatten().tolist()} '
                    f'tor={tor_loss.detach().cpu().flatten().tolist()} '
                    f'data={format_batch_names(data)}'
                )
            accelerator.backward(loss)
            grad_finite_status = torch.tensor(
                0 if gradients_are_finite(model.parameters()) else 1,
                device=device,
            )
            global_grad_finite_status = accelerator.reduce(grad_finite_status, reduction="sum")
            if global_grad_finite_status.item() > 0:
                logger.info('| WARNING: non-finite gradient detected, skipping optimizer step.')
                logger.info(f'backward data: {format_batch_names(data)}')
                optimizer.zero_grad()
                continue
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            #gather all loss for plot
            loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss, mask_loss = \
                accelerator.gather(loss),accelerator.gather(tr_loss), accelerator.gather(rot_loss), \
                    accelerator.gather(tor_loss), accelerator.gather(tr_base_loss), accelerator.gather(rot_base_loss), accelerator.gather(tor_base_loss), accelerator.gather(mask_loss)

            ema_weights.update(model.parameters())
            meter.add([loss.mean().cpu().detach(), tr_loss.mean().cpu().detach(), rot_loss.mean().cpu().detach(), tor_loss.mean().cpu().detach(), tr_base_loss.mean().cpu().detach(), rot_base_loss.mean().cpu().detach(), tor_base_loss.mean().cpu().detach(), mask_loss.mean().cpu().detach()])
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.info('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                optimizer.zero_grad()
                del data

                gc.collect()
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                logger.info('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                optimizer.zero_grad()
                del data
                gc.collect()
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    logger.info('clear last train batch data and model grad')
    for p in model.parameters():
        if p.grad is not None:
            del p.grad  # free some memory
    if 'data' in locals():
        del data
    gc.collect()
    torch.cuda.empty_cache()
    return meter.summary()


def test_epoch(model, loader, device, t_to_sigma, loss_fn,accelerator, test_sigma_intervals=False,model_type = 'energy_score_model'):
    if not model_type == 'energy_score_model':
        model.eval()
    meter = AverageMeter(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'mask_loss'],
                         unpooled_metrics=True)

    if test_sigma_intervals:
        meter_all = AverageMeter(
            ['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'mask_loss'],
            unpooled_metrics=True, intervals=10)
    
    for data in tqdm(loader, total=len(loader),disable=not accelerator.is_local_main_process):
        try:
            if not model_type == 'energy_score_model':
                with torch.no_grad():
                    tr_pred, rot_pred, tor_pred, mask_loss = unpack_score_model_outputs(model(data), device)
            else:
                tr_pred, rot_pred, tor_pred, mask_loss = unpack_score_model_outputs(model(data), device)
            with accelerator.autocast():
                loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss = \
                    loss_fn(tr_pred, rot_pred, tor_pred, data=data, t_to_sigma=t_to_sigma, apply_mean=False, device=device)

                loss = loss + mask_loss*0.33
            finite_status = torch.tensor(
                0 if tensors_are_finite(loss, tr_loss, rot_loss, tor_loss, tr_pred, rot_pred, tor_pred, mask_loss) else 1,
                device=device
            )
            global_finite_status = accelerator.reduce(finite_status, reduction="sum")
            if global_finite_status.item() > 0:
                logger.info(f'| WARNING: non-finite validation loss/prediction detected, skipping batch: {format_batch_names(data)}')
                del data, loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss
                gc.collect()
                torch.cuda.empty_cache()
                continue
            local_loss_max = torch.max(torch.abs(loss.detach())).item()
            if local_loss_max > HUGE_LOSS_WARN_THRESHOLD:
                logger.info(
                    f'| WARNING: huge validation loss max={local_loss_max:.6g} '
                    f'loss={loss.detach().cpu().flatten().tolist()} '
                    f'tr={tr_loss.detach().cpu().flatten().tolist()} '
                    f'rot={rot_loss.detach().cpu().flatten().tolist()} '
                    f'tor={tor_loss.detach().cpu().flatten().tolist()} '
                    f'data={format_batch_names(data)}'
                )
            loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss, mask_loss = \
                accelerator.gather(loss),accelerator.gather(tr_loss), accelerator.gather(rot_loss), \
                    accelerator.gather(tor_loss), accelerator.gather(tr_base_loss), accelerator.gather(rot_base_loss), accelerator.gather(tor_base_loss), accelerator.gather(mask_loss)

            metrics = [loss.mean().cpu().detach(), tr_loss.mean().cpu().detach(), \
                       rot_loss.mean().cpu().detach(), tor_loss.mean().cpu().detach(), \
                        tr_base_loss.mean().cpu().detach(), rot_base_loss.mean().cpu().detach(), tor_base_loss.mean().cpu().detach(), mask_loss.mean().cpu().detach()]
            meter.add(metrics)


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
    logger.info('clear val batch data and model grad')
    for p in model.parameters():
        if p.grad is not None:
            del p.grad  # free some memory
    del data
    gc.collect()
    torch.cuda.empty_cache()
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

        predictions_list = None
        confidences = None
        failed_convergence_counter = 0
        while predictions_list == None:
            try:
                predictions_list, confidences = sampling(input_data_list=data_list, model=model.module if device.type=='cuda' else model,
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

    dataset = ListDataset(complex_graphs)
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)
    loader = accelerator.prepare(loader)
    rmsds = []
    for orig_complex_graph in tqdm(loader,disable=not accelerator.is_local_main_process):
        orig_complex_graph_list = orig_complex_graph.to_data_list()
        data_list = [copy.deepcopy(graph) for graph in orig_complex_graph_list ]
        randomize_position(data_list, args.no_torsion, False, args.tr_sigma_max)

        predictions_list = None
        confidences = None
        failed_convergence_counter = 0
        while predictions_list == None:
            try:
                predictions_list, confidences = sampling(input_data_list=data_list, model=model.module if device.type=='cuda' else model,
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
    rmsds = torch.stack(rmsds)

    rmsds = accelerator.gather(rmsds)

    losses = {'rmsds_lt2': (100 * (rmsds < 2).sum() / len(rmsds)),
              'rmsds_lt5': (100 * (rmsds < 5).sum() / len(rmsds))}
    del dataset, loader,predictions_list, confidences,ligand_pos, orig_ligand_pos, rmsd,filterHs,complex_graphs
    gc.collect()
    torch.cuda.empty_cache()
    return losses
