#!/usr/bin/env python
"""V2 pretraining with explicit protein-to-NA transfer and exact resume semantics."""
from __future__ import annotations

import os
import random
from functools import partial

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from loguru import logger

from datasets.pdbbind import construct_loader
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from utils.parsing import parse_train_args
from utils.utils import ExponentialMovingAverage, get_model, get_optimizer_and_scheduler, save_yaml_file


SURFACE_RESTART_KEYS = (
    "surface_node_embedding", "surface_edge_embedding", "surface_rec_cross_edge_embedding",
    "surface_distance_expansion", "surface_conv_layers", "lig_to_surface_conv_layers",
    "surface_to_lig_conv_layers", "residue_to_surface_conv_layers", "ligand_patch_tokenizer",
    "nusurf_fusion",
)

# These buffers are a transient memory bank, not transferable representation
# parameters.  A restart_file initializes NA from protein weights and must
# begin its negative queue in the target domain; a restart_dir is exact resume
# and deliberately preserves them.
TRANSFER_RESET_BUFFER_KEYS = {
    "pretrain_surface_queue",
    "pretrain_ligand_queue",
    "pretrain_contrastive_queue_count",
    "pretrain_contrastive_queue_ptr",
}


def resolve_restart_path(args) -> str | None:
    if args.restart_file:
        return args.restart_file if os.path.isabs(args.restart_file) else os.path.join(args.restart_dir or "", args.restart_file)
    return os.path.join(args.restart_dir, "last_model.pt") if args.restart_dir else None


def extract_state(checkpoint: object) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError("restart checkpoint must be a state dict or a dict containing 'model'")
    return state


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict | None) -> None:
    if state is None:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # ``torch.load(..., map_location=device)`` moves every tensor in the
    # checkpoint, including RNG byte states, onto CUDA.  PyTorch's RNG setter
    # deliberately accepts a CPU ByteTensor for the host generator.
    torch.set_rng_state(state["torch"].detach().cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all([item.detach().cpu() for item in state["cuda"]])


def capture_dataloader_rng_state(train_loader, val_loader) -> dict:
    return {
        "train": train_loader.generator.get_state().detach().cpu(),
        "val": val_loader.generator.get_state().detach().cpu(),
    }


def restore_dataloader_rng_state(train_loader, val_loader, state: dict | None) -> None:
    if state is None:
        return
    train_loader.generator.set_state(state["train"].detach().cpu())
    val_loader.generator.set_state(state["val"].detach().cpu())


def load_restart(model, optimizer, ema, args, device) -> tuple[int, float, int, dict | None, dict | None]:
    """Return start epoch and historical best state for exact resume."""
    path = resolve_restart_path(args)
    if path is None:
        return 0, float("inf"), -1, None, None
    checkpoint = torch.load(path, map_location=device)
    source, target = extract_state(checkpoint), model.state_dict()
    key_filter = getattr(args, "restart_key_filter", "all")
    compatible, skipped_filter, skipped_transfer_buffers, skipped_shape = {}, 0, 0, []
    for key, value in source.items():
        normalized = key[7:] if key.startswith("module.") else key
        if args.restart_file and normalized in TRANSFER_RESET_BUFFER_KEYS:
            skipped_transfer_buffers += 1
            continue
        if key_filter == "surface" and not any(token in normalized for token in SURFACE_RESTART_KEYS):
            skipped_filter += 1
            continue
        if normalized not in target:
            continue
        if not isinstance(value, torch.Tensor) or target[normalized].shape != value.shape:
            skipped_shape.append(normalized)
            continue
        compatible[normalized] = value
    if not compatible:
        raise ValueError(f"No compatible parameters found in restart checkpoint: {path}")
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    logger.info(
        f"V2 restart initialization path={path} filter={key_filter} source={len(source)} "
        f"loaded={len(compatible)}/{len(target)} missing={len(missing)} unexpected={len(unexpected)} "
        f"skipped_filter={skipped_filter} skipped_transfer_buffers={skipped_transfer_buffers} "
        f"skipped_shape={len(skipped_shape)}"
    )
    # A restart directory is an exact continuation contract; an explicit
    # restart_file is transfer initialization, e.g. protein best_model -> NA.
    if not args.restart_dir:
        return 0, float("inf"), -1, None, None
    if key_filter != "all" or len(compatible) != len(target) or missing or unexpected:
        raise ValueError("Exact V2 resume requires an architecture-compatible all-weights checkpoint")
    if (not isinstance(checkpoint, dict) or "optimizer" not in checkpoint
            or "ema_weights" not in checkpoint or "rng_state" not in checkpoint
            or "dataloader_rng_state" not in checkpoint):
        raise ValueError("Exact V2 resume requires optimizer, ema_weights, rng_state and dataloader_rng_state")
    optimizer.load_state_dict(checkpoint["optimizer"])
    ema.load_state_dict(checkpoint["ema_weights"], device=device)
    epoch = int(checkpoint.get("epoch", -1)) + 1
    best_val = float(checkpoint.get("best_val", float("inf")))
    best_epoch = int(checkpoint.get("best_epoch", -1))
    logger.info(
        f"Resuming exact V2 pretraining at epoch {epoch}; "
        f"historical best_val={best_val:.6f} best_epoch={best_epoch}"
    )
    return (epoch, best_val, best_epoch, checkpoint["rng_state"],
            checkpoint["dataloader_rng_state"])


def run_epoch(model, loader, optimizer, accelerator, train):
    model.train(train)
    totals = torch.zeros(3, device=accelerator.device)
    steps = 0
    for data in loader:
        if getattr(data, "num_graphs", 1) < 2:
            continue
        with torch.set_grad_enabled(train):
            interaction, reconstruction = model(data)
            loss = interaction + reconstruction
        finite = torch.isfinite(loss) and torch.isfinite(interaction) and torch.isfinite(reconstruction)
        finite_all = accelerator.reduce(torch.tensor(0 if finite else 1, device=accelerator.device), reduction="sum")
        if finite_all.item() > 0:
            logger.warning(f"Skipping non-finite V2 pretraining batch: {getattr(data, 'name', 'unknown')}")
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
        raise RuntimeError("No usable pretraining batches were produced.")
    return (totals / steps).detach().cpu().tolist()


def main():
    args = parse_train_args()
    if not args.pretrain_v2_mode or args.model_type != "surface_pretrain_v2":
        raise ValueError("This runner requires --model_type surface_pretrain_v2 --pretrain_v2_mode.")
    if os.environ.get("SURFNA_DETERMINISTIC", "0") == "1":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    set_seed(42)
    run_dir = os.path.join(args.log_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger.add(os.path.join(run_dir, "LogFile.log"), rotation="100 MB")
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    train_loader, val_loader = construct_loader(args, t_to_sigma)
    model = get_model(args, accelerator.device, t_to_sigma, model_type=args.model_type)
    optimizer, _ = get_optimizer_and_scheduler(args, model, accelerator, scheduler_mode="min")
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)
    start_epoch, best_val, best_epoch, resume_rng_state, resume_dataloader_rng_state = load_restart(
        model, optimizer, ema, args, accelerator.device
    )
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    restore_rng_state(resume_rng_state)
    restore_dataloader_rng_state(train_loader, val_loader, resume_dataloader_rng_state)
    if accelerator.is_local_main_process:
        save_yaml_file(os.path.join(run_dir, "model_parameters.yml"), args.__dict__)
        logger.info(f"V2 pretraining model parameters: {sum(p.numel() for p in model.parameters())}")
    for epoch in range(start_epoch, args.n_epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, accelerator, train=True)
        ema.update(model.parameters())
        ema.store(model.parameters())
        if args.use_ema:
            ema.copy_to(model.parameters())
        val_metrics = run_epoch(model, val_loader, optimizer, accelerator, train=False)
        validated_state = None
        if accelerator.is_local_main_process:
            validated_state = {
                key: value.detach().cpu().clone()
                for key, value in accelerator.unwrap_model(model).state_dict().items()
            }
        ema.restore(model.parameters())
        if accelerator.is_local_main_process:
            logger.info(
                f"Epoch {epoch}: train total={train_metrics[0]:.4f} interaction={train_metrics[1]:.4f} "
                f"reconstruction={train_metrics[2]:.4f}; val total={val_metrics[0]:.4f} "
                f"interaction={val_metrics[1]:.4f} reconstruction={val_metrics[2]:.4f}"
            )
            raw_state = {
                key: value.detach().cpu().clone()
                for key, value in accelerator.unwrap_model(model).state_dict().items()
            }
            if val_metrics[0] <= best_val:
                best_val = val_metrics[0]
                best_epoch = epoch
                torch.save(validated_state, os.path.join(run_dir, "best_model.pt"))
                torch.save(raw_state, os.path.join(run_dir, "best_raw_model.pt"))
            torch.save({"epoch": epoch, "model": raw_state, "optimizer": optimizer.state_dict(),
                        "ema_weights": ema.state_dict(), "best_val": best_val,
                        "best_epoch": best_epoch, "rng_state": capture_rng_state(),
                        "dataloader_rng_state": capture_dataloader_rng_state(train_loader, val_loader)},
                       os.path.join(run_dir, "last_model.pt"))
    if accelerator.is_local_main_process and accelerator.device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(accelerator.device) / 1024 ** 2
        logger.info(f"Best V2 pretraining validation loss {best_val:.4f}; peak CUDA allocated {peak:.1f} MiB")


if __name__ == "__main__":
    main()
