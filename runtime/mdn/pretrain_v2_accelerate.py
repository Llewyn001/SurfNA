#!/usr/bin/env python
"""Dedicated SurfNA V2 pretraining runner.

Unlike the legacy ``pretrain_accelerate.py`` score-loss runner, this trains the
V2 contact-field/patch/anchor/reconstruction objectives exposed by
``surface_pretrain_v2``.
"""
import copy
import os
from functools import partial

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from loguru import logger

from datasets.pdbbind import construct_loader
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from utils.parsing import parse_train_args
from utils.utils import ExponentialMovingAverage, get_model, get_optimizer_and_scheduler, save_yaml_file


def run_epoch(model, loader, optimizer, accelerator, train):
    model.train(train)
    totals = torch.zeros(3, device=accelerator.device)
    steps = 0
    for data in loader:
        if getattr(data, 'num_graphs', 1) < 2:
            continue
        with torch.set_grad_enabled(train):
            interaction, reconstruction = model(data)
            loss = interaction + reconstruction
        finite = torch.isfinite(loss) and torch.isfinite(interaction) and torch.isfinite(reconstruction)
        finite_all = accelerator.reduce(torch.tensor(0 if finite else 1, device=accelerator.device), reduction='sum')
        if finite_all.item() > 0:
            logger.warning(f'Skipping non-finite V2 pretraining batch: {getattr(data, "name", "unknown")}')
            if train:
                optimizer.zero_grad(set_to_none=True)
            continue
        if train:
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        gathered = accelerator.gather(torch.stack([loss.detach(), interaction.detach(), reconstruction.detach()]))
        totals += gathered.view(-1, 3).mean(dim=0)
        steps += 1
    if steps == 0:
        raise RuntimeError('No usable pretraining batches were produced.')
    return (totals / steps).detach().cpu().tolist()


def main():
    args = parse_train_args()
    if not args.pretrain_v2_mode:
        raise ValueError('This runner requires --pretrain_v2_mode.')
    if args.model_type != 'surface_pretrain_v2':
        raise ValueError('This runner requires --model_type surface_pretrain_v2.')
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    set_seed(42)
    run_dir = os.path.join(args.log_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger.add(os.path.join(run_dir, 'LogFile.log'), rotation='100 MB')
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    train_loader, val_loader = construct_loader(args, t_to_sigma)
    model = get_model(args, accelerator.device, t_to_sigma, model_type=args.model_type)
    optimizer, _ = get_optimizer_and_scheduler(args, model, accelerator, scheduler_mode='min')
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    if accelerator.is_local_main_process:
        save_yaml_file(os.path.join(run_dir, 'model_parameters.yml'), args.__dict__)
        logger.info(f'V2 pretraining model parameters: {sum(p.numel() for p in model.parameters())}')
    best_val = float('inf')
    for epoch in range(args.n_epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, accelerator, train=True)
        ema.update(model.parameters())
        ema.store(model.parameters())
        if args.use_ema:
            ema.copy_to(model.parameters())
        val_metrics = run_epoch(model, val_loader, optimizer, accelerator, train=False)
        ema.restore(model.parameters())
        if accelerator.is_local_main_process:
            logger.info(
                f'Epoch {epoch}: train total={train_metrics[0]:.4f} interaction={train_metrics[1]:.4f} '
                f'reconstruction={train_metrics[2]:.4f}; val total={val_metrics[0]:.4f} '
                f'interaction={val_metrics[1]:.4f} reconstruction={val_metrics[2]:.4f}'
            )
            state = accelerator.unwrap_model(model).state_dict()
            torch.save({'epoch': epoch, 'model': state, 'optimizer': optimizer.state_dict(),
                        'ema_weights': ema.state_dict()}, os.path.join(run_dir, 'last_model.pt'))
            if val_metrics[0] <= best_val:
                best_val = val_metrics[0]
                torch.save(state, os.path.join(run_dir, 'best_model.pt'))
    if accelerator.is_local_main_process and accelerator.device.type == 'cuda':
        peak = torch.cuda.max_memory_allocated(accelerator.device) / 1024 ** 2
        logger.info(f'Best V2 pretraining validation loss {best_val:.4f}; peak CUDA allocated {peak:.1f} MiB')


if __name__ == '__main__':
    main()
