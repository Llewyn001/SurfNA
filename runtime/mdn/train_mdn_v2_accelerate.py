#!/usr/bin/env python3
"""Leakage-safe grouped training entrypoint for the SurfNA V2 pose scorer.

This entrypoint is intentionally separate from the legacy MDN trainer.  It
shards complete complexes across ranks, computes full-group listwise gradients
with GPU pose microbatches, and selects checkpoints only on strict validation
ranking metrics.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import time
from functools import partial
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from datasets.group_sampler import (
    DistributedGroupBatchSampler,
    DistributionAwareDistributedGroupBatchSampler,
    GroupBatchSampler,
)
from datasets.mdn_rerank_v2 import DecoyPDBBindV2, stable_group_integer
from models.surfna_v2_transfer import (
    load_protein_mdn_scorer_transfer,
    load_v2_diffusion_backbone,
)
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from utils.scorer_config_v2 import load_scorer_model_args
from utils.scorer_factory_v2 import build_surfna_v2_scorer
from utils.training_mdn_v2 import (
    assert_complete_group_records,
    checkpoint_selection_key,
    gather_pose_records,
    grouped_rerank_metrics,
    near_native_prior_nll_sum_count,
    ranking_output_gradients,
)


STATIC_TRANSFER_PREFIXES = (
    "backbone.lig_node_embedding",
    "backbone.lig_edge_embedding",
    "backbone.lig_conv_layers",
    "backbone.rec_node_embedding",
    "backbone.rec_edge_embedding",
    "backbone.rec_conv_layers",
    "backbone.surface_node_embedding",
    "backbone.surface_edge_embedding",
    "backbone.surface_rec_cross_edge_embedding",
    "backbone.residue_to_surface_conv_layers",
    "backbone.surface_conv_layers",
)
DYNAMIC_TRANSFER_PREFIXES = (
    "backbone.cross_edge_embedding",
    "backbone.surface_to_lig_conv_layers",
    "backbone.lig_to_surface_conv_layers",
    "backbone.ligand_patch_tokenizer",
)
NA_PREFIXES = (
    "backbone.nusurf_fusion",
    "backbone.nusurf_refine_blocks",
    "backbone.nuc_feat_proj",
    "backbone.nuc_feat_gate",
    "na_residual",
)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if isinstance(checkpoint.get(key), dict):
                return checkpoint[key]
    return checkpoint


def configure_phase(model: torch.nn.Module, phase: str) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad_(phase == "joint")
    if phase == "prior":
        allowed = STATIC_TRANSFER_PREFIXES + ("prior_head",)
    elif phase == "ranker":
        allowed = DYNAMIC_TRANSFER_PREFIXES + ("cross_ranker", "combiner")
        if getattr(model, "scorer_variant", None) in {"full_v2_b", "gated_na_s2"}:
            allowed = allowed + NA_PREFIXES
    elif phase == "joint":
        allowed = ("",)
    else:
        raise ValueError(f"unknown phase {phase!r}")
    if phase != "joint":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(allowed))
    elif getattr(model, "scorer_variant", None) == "aligned_a":
        for name, parameter in model.named_parameters():
            if name.startswith(NA_PREFIXES):
                parameter.requires_grad_(False)
    return {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }


def configure_epoch_trainability(
    model: torch.nn.Module,
    phase: str,
    epoch: int,
    head_warmup_epochs: int,
    freeze_backbone: bool = False,
) -> dict[str, int]:
    """Freeze the transferred backbone during the short head-only warm-up."""
    configure_phase(model, phase)
    if freeze_backbone or int(epoch) < int(head_warmup_epochs):
        for name, parameter in model.named_parameters():
            if name.startswith("backbone."):
                parameter.requires_grad_(False)
    return {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }


def validate_gate(path: str, manifest: str, *, allow_nonformal: bool) -> dict:
    report = json.loads(Path(path).read_text())
    if report.get("status") != "pass":
        raise ValueError(f"manifest gate did not pass: {path}")
    if report.get("manifest_sha256") != sha256_file(manifest):
        raise ValueError(f"manifest changed after gate: {manifest}")
    if not allow_nonformal and report.get("formal_provenance") is not True:
        raise ValueError(f"formal scorer training rejects non-formal provenance: {path}")
    return report


def manifest_complexes(path: str) -> set[str]:
    import csv

    with open(path, newline="") as handle:
        return {row["complex_name"] for row in csv.DictReader(handle, delimiter="\t")}


def difficulty_bin(n_lt2: int) -> str:
    if n_lt2 == 0:
        return "zero"
    if n_lt2 <= 3:
        return "one_to_three"
    if n_lt2 <= 9:
        return "four_to_nine"
    if n_lt2 <= 19:
        return "ten_to_nineteen"
    return "twenty_plus"


def load_distribution_policy(path: str, train_manifest: str, val_manifest: str) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("status") != "pass":
        raise ValueError("distribution policy is not pass")
    if payload.get("test_inputs_used") is not False:
        raise ValueError("distribution policy must explicitly forbid test inputs")
    if payload.get("train_manifest_sha256") != sha256_file(train_manifest):
        raise ValueError("distribution policy train manifest hash is stale")
    if payload.get("matched_val_manifest_sha256") != sha256_file(val_manifest):
        raise ValueError("distribution policy matched-val manifest hash is stale")
    target = payload.get("final_group_target_probabilities")
    if not isinstance(target, dict) or not target:
        raise ValueError("distribution policy lacks final group targets")
    pose_weights = payload.get("pose_bin_importance_weights")
    expected_bins = ("lt2", "2to5", "5to10", "ge10")
    if not isinstance(pose_weights, dict) or set(pose_weights) != set(expected_bins):
        raise ValueError("distribution policy has invalid pose-bin weights")
    payload["pose_bin_weight_tuple"] = tuple(
        float(pose_weights[key]) for key in expected_bins
    )
    if any(not math.isfinite(w) or w <= 0 for w in payload["pose_bin_weight_tuple"]):
        raise ValueError("pose-bin weights must be finite and positive")
    if payload.get("sampling_mode") == "full_coverage_group_loss_weighted":
        weights = payload.get("group_loss_weights", {})
        if set(weights) != set(target) or any(
            not math.isfinite(float(w)) or w < 0 for w in weights.values()
        ):
            raise ValueError("invalid full-coverage group loss weights")
        if not math.isclose(payload["group_loss_weight_mean"], 1.0, abs_tol=1e-8):
            raise ValueError("group weights must have mean one")
        if payload.get("pose_importance_denominator") != "group_weighted_pose_bin_probabilities":
            raise ValueError("pose weights must use group-weighted exposure")
    return payload


def group_objective_weight(group_batch, weights_by_uid: dict[str, float]) -> float:
    if not weights_by_uid:
        return 1.0
    uids = group_batch.decoy_pose_group_uid
    uids = {uids} if isinstance(uids, str) else set(uids)
    if len(uids) != 1:
        raise ValueError("group loss weighting requires exactly one complete group")
    weight = float(weights_by_uid[next(iter(uids))])
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("observed group weight must be finite and positive")
    return weight


def group_difficulty_from_dataset(dataset) -> dict[str, str]:
    values: dict[str, list[float]] = {}
    for row in dataset.rows:
        values.setdefault(row["pose_group_uid"], []).append(float(row["rmsd"]))
    return {
        uid: difficulty_bin(sum(rmsd < 2.0 for rmsd in group_rmsd))
        for uid, group_rmsd in values.items()
    }


def dataset_kwargs(args, model_args) -> dict:
    return {
        "root": args.data_dir,
        "cache_path": args.cache_path,
        "receptor_radius": getattr(model_args, "receptor_radius", 30),
        "num_workers": args.dataset_workers,
        "c_alpha_max_neighbors": getattr(model_args, "c_alpha_max_neighbors", None),
        "matching": not getattr(model_args, "no_torsion", False),
        "keep_original": True,
        "remove_hs": args.remove_hs,
        "num_conformers": 1,
        "all_atoms": getattr(model_args, "all_atoms", False),
        "atom_radius": getattr(model_args, "atom_radius", 5),
        "atom_max_neighbors": getattr(model_args, "atom_max_neighbors", None),
        "esm_embeddings_path": getattr(model_args, "esm_embeddings_path", None),
        "surface_path": args.surface_path,
        "per_complex_timeout_sec": args.per_complex_timeout_sec,
        "max_surface_vertices": args.max_surface_vertices,
        "surface_feature_schema": "v2_full8",
        "surface_scaler_json": args.surface_scaler_json,
    }


def build_ranker_datasets(args, model_args):
    if args.phase == "prior":
        return None, None
    kwargs = dataset_kwargs(args, model_args)
    train_dataset = DecoyPDBBindV2(
        args.train_manifest,
        args.train_split,
        gate_report_path=args.train_gate_report,
        expected_profile=args.ranker_profile,
        limit_complexes=args.limit_train_complexes,
        **kwargs,
    )
    val_dataset = DecoyPDBBindV2(
        args.val_manifest,
        args.val_split,
        gate_report_path=args.val_gate_report,
        expected_profile=args.ranker_profile,
        limit_complexes=args.limit_val_complexes,
        **kwargs,
    )
    return train_dataset, val_dataset


def build_prior_datasets(args, model_args):
    if args.phase == "ranker":
        return None, None
    required = {
        "prior_train_manifest": args.prior_train_manifest,
        "prior_train_gate_report": args.prior_train_gate_report,
        "prior_train_split": args.prior_train_split,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "prior/native stream is required for phase prior/joint; missing " + ", ".join(missing)
        )
    train_dataset = DecoyPDBBindV2(
        args.prior_train_manifest,
        args.prior_train_split,
        gate_report_path=args.prior_train_gate_report,
        expected_profile="prior_native",
        limit_complexes=args.limit_train_complexes,
        **dataset_kwargs(args, model_args),
    )
    val_dataset = None
    if args.phase == "prior":
        val_required = {
            "prior_val_manifest": args.prior_val_manifest,
            "prior_val_gate_report": args.prior_val_gate_report,
            "prior_val_split": args.prior_val_split,
        }
        missing = [name for name, value in val_required.items() if not value]
        if missing:
            raise ValueError(
                "pure prior phase requires native validation stream; missing "
                + ", ".join(missing)
            )
        val_dataset = DecoyPDBBindV2(
            args.prior_val_manifest,
            args.prior_val_split,
            gate_report_path=args.prior_val_gate_report,
            expected_profile="prior_native",
            limit_complexes=args.limit_val_complexes,
            **dataset_kwargs(args, model_args),
        )
    return train_dataset, val_dataset


def make_loader(dataset, sampler, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def split_graph_batch(group_batch, microbatch_size: int):
    graphs = group_batch.to_data_list()
    for start in range(0, len(graphs), microbatch_size):
        yield graphs[start : start + microbatch_size]


def capture_rng(device: torch.device):
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return cpu_state, cuda_state


def restore_rng(state, device: torch.device) -> None:
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def combiner_components(output) -> torch.Tensor:
    return torch.stack(
        (
            output.pose.prior_score,
            output.pose.cross_score,
            output.pose.na_score,
            output.pose.clash_penalty,
            output.pose.strain_penalty,
        ),
        dim=-1,
    )


def assert_finite_tensor(name: str, value: torch.Tensor) -> None:
    finite = torch.isfinite(value)
    if not bool(finite.all()):
        raise FloatingPointError(
            f"non-finite {name}: shape={tuple(value.shape)} "
            f"count={int((~finite).sum())}"
        )


def assert_finite_gradients(model) -> dict[str, float]:
    unwrapped = model.module if hasattr(model, "module") else model
    bad = {}
    maximum = 0.0
    checked = 0
    for name, parameter in unwrapped.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        checked += 1
        finite = torch.isfinite(parameter.grad)
        if not bool(finite.all()):
            bad[name] = int((~finite).sum())
        elif parameter.grad.numel():
            maximum = max(maximum, float(parameter.grad.detach().abs().max()))
    if bad:
        raise FloatingPointError(f"non-finite model gradients: {bad}")
    return {"checked_tensors": float(checked), "max_abs": maximum}


def assert_finite_trainable_parameters(model) -> None:
    unwrapped = model.module if hasattr(model, "module") else model
    bad = {}
    for name, parameter in unwrapped.named_parameters():
        if not parameter.requires_grad:
            continue
        finite = torch.isfinite(parameter)
        if not bool(finite.all()):
            bad[name] = int((~finite).sum())
    if bad:
        raise FloatingPointError(f"optimizer produced non-finite parameters: {bad}")


def set_scorer_training_mode(model, args) -> dict[str, int | bool]:
    """Train heads/backbone while keeping pretrained normalization stable.

    E3NN BatchNorm running statistics were learned with much larger diffusion
    batches.  Re-estimating them from four poses is both methodologically
    wrong and numerically unstable.  During head warm-up the entire frozen
    backbone stays in eval mode; afterwards its trainable layers are enabled
    while all BatchNorm modules continue using the transferred running stats.
    """

    model.train()
    unwrapped = model.module if hasattr(model, "module") else model
    backbone = unwrapped.backbone
    warmup = int(args.current_epoch) < int(args.head_warmup_epochs)
    backbone_frozen = bool(getattr(args, "freeze_backbone", False)) or warmup
    if backbone_frozen:
        backbone.eval()
    else:
        backbone.train()
    frozen_norms = 0
    for module in backbone.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or (
            module.__class__.__name__ == "BatchNorm"
            and hasattr(module, "running_mean")
            and hasattr(module, "running_var")
        ):
            module.eval()
            frozen_norms += 1
    return {
        "backbone_eval": backbone_frozen,
        "frozen_normalization_modules": frozen_norms,
    }


@torch.no_grad()
def calibrate_combiner(model, dataset, device, *, groups: int, microbatch_size: int) -> dict:
    sampler = GroupBatchSampler(
        dataset.pose_group_uids, groups_per_batch=1, shuffle=False, seed=0
    )
    loader = make_loader(dataset, sampler, num_workers=0)
    model.eval()
    values = []
    for group_index, group_batch in enumerate(loader):
        if groups > 0 and group_index >= groups:
            break
        for graph_chunk in split_graph_batch(group_batch, microbatch_size):
            microbatch = Batch.from_data_list(graph_chunk).to(device)
            components = combiner_components(model(microbatch))
            assert_finite_tensor("combiner calibration components", components)
            values.append(components.cpu())
    if not values:
        raise ValueError("combiner calibration received no train poses")
    values = torch.cat(values, dim=0)
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    assert_finite_tensor("combiner calibration mean", mean)
    assert_finite_tensor("combiner calibration std", std)
    inactive = std < 1e-3
    mean[inactive] = 0.0
    std[inactive] = 1.0
    model.combiner.set_calibration(mean.to(device), std.to(device))
    return {"mean": mean.tolist(), "std": std.tolist(), "n_poses": int(values.shape[0])}


def _gradient_family_stats(model, prefixes: tuple[str, ...]) -> dict[str, float]:
    count = 0
    nonzero = 0
    maximum = 0.0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or not name.startswith(prefixes):
            continue
        count += 1
        if parameter.grad is not None:
            value = float(parameter.grad.detach().abs().max())
            maximum = max(maximum, value)
            nonzero += int(value > 0.0)
    return {"required_tensors": float(count), "nonzero_grad_tensors": float(nonzero), "max_abs_grad": maximum}


def assert_na_gradient_gate(model, *, require_backbone: bool) -> dict[str, dict[str, float]]:
    """Fail closed when an enabled NA residual is disconnected from the loss."""
    unwrapped = model.module if hasattr(model, "module") else model
    report = {
        "na_residual": _gradient_family_stats(unwrapped, ("na_residual.",)),
        "na_backbone": _gradient_family_stats(
            unwrapped,
            (
                "backbone.nusurf_fusion.",
                "backbone.nusurf_refine_blocks.",
                "backbone.nuc_feat_proj.",
                "backbone.nuc_feat_gate.",
            ),
        ),
    }
    required_families = ["na_residual"] + (["na_backbone"] if require_backbone else [])
    failed = {
        family: report[family]
        for family in required_families
        if report[family]["required_tensors"] <= 0
        or report[family]["nonzero_grad_tensors"] <= 0
        or report[family]["max_abs_grad"] <= 0.0
    }
    if failed:
        raise RuntimeError(f"NA gradient gate failed: {failed}")
    return report


def two_pass_group_step(
    model,
    optimizer,
    group_batch,
    accelerator,
    args,
    *,
    zero_grad: bool = True,
    optimizer_step: bool = True,
    loss_scale: float = 1.0,
    group_loss_weight: float = 1.0,
) -> dict[str, float]:
    if not math.isfinite(group_loss_weight) or group_loss_weight <= 0:
        raise ValueError("group loss weight must be finite and positive")
    graph_chunks = list(split_graph_batch(group_batch, args.pose_microbatch_size))
    if not graph_chunks:
        raise ValueError("empty pose group batch")
    recorded_rng = []
    recorded_scores = []
    recorded_logit_lt5 = []
    recorded_logit_lt2_given_lt5 = []
    cached_prior_scores = []
    cached_pose_states = []
    rmsd_values = []
    group_values = []

    set_scorer_training_mode(model, args)
    with torch.no_grad():
        for graph_chunk in graph_chunks:
            microbatch = Batch.from_data_list(graph_chunk).to(accelerator.device)
            if args.freeze_backbone:
                unwrapped = model.module if hasattr(model, "module") else model
                surface_state = unwrapped.backbone.encode_surface_static(microbatch)
                ligand_state = unwrapped.backbone.encode_ligand_intra(microbatch)
                prior = unwrapped.forward_prior(surface_state, ligand_state)
                pose_state = unwrapped.backbone.encode_pose(
                    microbatch, surface_state, ligand_state
                )
                # The frozen backbone/prior may consume RNG (for example a
                # transferred dropout layer).  Capture only the trainable-head
                # boundary so the cached second pass reproduces head dropout
                # exactly without replaying frozen computation.
                recorded_rng.append(capture_rng(accelerator.device))
                pose = unwrapped.score_pose(
                    microbatch,
                    surface_state,
                    ligand_state,
                    prior,
                    pose_state=pose_state,
                )
                output = type("CachedScorerOutput", (), {"pose": pose})()
                cached_prior_scores.append(prior.score.detach().cpu())
                cached_pose_states.append(
                    pose_state.__class__(
                        *(value.detach().cpu() for value in pose_state)
                    )
                )
            else:
                recorded_rng.append(capture_rng(accelerator.device))
                output = model(microbatch)
            recorded_scores.append(output.pose.score.detach())
            recorded_logit_lt5.append(output.pose.ordinal_logit_lt5.detach())
            recorded_logit_lt2_given_lt5.append(
                output.pose.ordinal_logit_lt2_given_lt5.detach()
            )
            rmsd_values.append(microbatch.decoy_rmsd.float().view(-1))
            group_values.append(microbatch.decoy_group_id.long().view(-1))

    scores = torch.cat(recorded_scores)
    logit_lt5 = torch.cat(recorded_logit_lt5)
    logit_lt2_given_lt5 = torch.cat(recorded_logit_lt2_given_lt5)
    rmsd = torch.cat(rmsd_values)
    group = torch.cat(group_values)
    assert_finite_tensor("first-pass rank scores", scores)
    assert_finite_tensor("first-pass <5 logits", logit_lt5)
    assert_finite_tensor("first-pass <2|<5 logits", logit_lt2_given_lt5)
    assert_finite_tensor("rank RMSD labels", rmsd)
    output_gradients, rank_logs = ranking_output_gradients(
        scores,
        logit_lt5,
        logit_lt2_given_lt5,
        rmsd,
        group,
        listwise_weight=args.listwise_weight if args.phase != "prior" else 0.0,
        pairwise_weight=args.pairwise_weight if args.phase != "prior" else 0.0,
        rmsd_temperature=args.rmsd_temperature,
        score_temperature=args.score_temperature,
        min_rmsd_gap=args.min_rmsd_gap,
        pairwise_margin=args.pairwise_margin,
        ordinal_weight=args.ordinal_weight if args.phase != "prior" else 0.0,
        listwise_mode=args.listwise_mode,
        pairwise_mode=args.pairwise_mode,
        pose_bin_weights=args.pose_bin_weights,
    )
    score_gradient, lt5_gradient, lt2_gradient = output_gradients
    assert_finite_tensor("ranking score gradients", score_gradient)
    assert_finite_tensor("ranking <5 gradients", lt5_gradient)
    assert_finite_tensor("ranking <2|<5 gradients", lt2_gradient)
    nonfinite_logs = {
        name: value for name, value in rank_logs.items() if not math.isfinite(float(value))
    }
    if nonfinite_logs:
        raise FloatingPointError(f"non-finite ranking losses: {nonfinite_logs}")

    if zero_grad:
        optimizer.zero_grad(set_to_none=True)
    offset = 0
    recompute_max_difference = 0.0
    for chunk_index, graph_chunk in enumerate(graph_chunks):
        restore_rng(recorded_rng[chunk_index], accelerator.device)
        should_sync = optimizer_step and chunk_index == len(graph_chunks) - 1
        sync_context = contextlib.nullcontext() if should_sync else accelerator.no_sync(model)
        with sync_context:
            if args.freeze_backbone:
                cached_pose = cached_pose_states[chunk_index].__class__(
                    *(
                        value.to(accelerator.device, non_blocking=True)
                        for value in cached_pose_states[chunk_index]
                    )
                )
                output = model(
                    None,
                    cached_pose_state=cached_pose,
                    cached_prior_score=cached_prior_scores[chunk_index].to(
                        accelerator.device, non_blocking=True
                    ),
                )
            else:
                microbatch = Batch.from_data_list(graph_chunk).to(accelerator.device)
                output = model(microbatch)
            assert_finite_tensor("second-pass rank scores", output.pose.score)
            count = output.pose.score.numel()
            expected = scores[offset : offset + count]
            recompute_max_difference = max(
                recompute_max_difference,
                float((output.pose.score.detach() - expected).abs().max()),
            )
            rank_surrogate = (
                output.pose.score * score_gradient[offset : offset + count]
            ).sum() + (
                output.pose.ordinal_logit_lt5 * lt5_gradient[offset : offset + count]
            ).sum() + (
                output.pose.ordinal_logit_lt2_given_lt5
                * lt2_gradient[offset : offset + count]
            ).sum()
            accelerator.backward(float(loss_scale) * group_loss_weight * rank_surrogate)
            offset += count
    if recompute_max_difference > args.two_pass_parity_atol:
        raise RuntimeError(
            "two-pass dropout/RNG parity failed: "
            f"max score difference={recompute_max_difference}, "
            f"allowed_atol={args.two_pass_parity_atol}"
        )
    na_gradient_gate = None
    gradient_finite_gate = assert_finite_gradients(model)
    if optimizer_step:
        if args.scorer_variant in {"full_v2_b", "gated_na_s2"}:
            na_gradient_gate = assert_na_gradient_gate(
                model,
                require_backbone=(
                    args.scorer_variant == "full_v2_b"
                    and not args.freeze_backbone
                    and args.current_epoch >= args.head_warmup_epochs
                ),
            )
        if args.gradient_clip > 0:
            gradient_norm = accelerator.clip_grad_norm_(
                model.parameters(), args.gradient_clip
            )
            if not math.isfinite(float(gradient_norm)):
                raise FloatingPointError(
                    f"non-finite gradient norm before optimizer step: {float(gradient_norm)}"
                )
        optimizer.step()
        assert_finite_trainable_parameters(model)
    return {
        "rank_loss": float(rank_logs["rank_loss"]),
        "listwise_loss": float(rank_logs["listwise_loss"]),
        "pairwise_loss": float(rank_logs["pairwise_loss"]),
        "ordinal_loss": float(rank_logs["ordinal_loss"]),
        "ranker_total_loss": float(rank_logs["ranker_total_loss"]),
        "weighted_ranker_total_loss": group_loss_weight * float(rank_logs["ranker_total_loss"]),
        "group_loss_weight": group_loss_weight,
        "recompute_max_difference": recompute_max_difference,
        "na_gradient_max": (
            max(stats["max_abs_grad"] for stats in na_gradient_gate.values())
            if na_gradient_gate
            else 0.0
        ),
        "gradient_max": gradient_finite_gate["max_abs"],
        "frozen_feature_reuse": float(bool(args.freeze_backbone)),
    }


def prior_stream_step(model, optimizer, prior_batch, accelerator, args) -> dict[str, float]:
    """One optimizer step using only native poses and the prior objective."""
    set_scorer_training_mode(model, args)
    optimizer.zero_grad(set_to_none=True)
    microbatch = prior_batch.to(accelerator.device)
    # Use the wrapped forward so DDP reducer hooks are installed on every
    # trainable parameter.  Only ``output.prior`` enters the objective.
    prior = model(microbatch, prior_only=True)
    nll_sum, pair_count = near_native_prior_nll_sum_count(
        prior.pi,
        prior.mu,
        prior.sigma,
        prior.candidate_distances,
        prior.pair_mask,
        microbatch.decoy_is_prior_valid.view(-1),
        contact_radius=args.prior_contact_radius,
    )
    if int(pair_count) <= 0:
        raise RuntimeError("native prior stream contained no valid contact pair")
    objective = args.prior_weight * nll_sum / pair_count.to(nll_sum.dtype)
    assert_finite_tensor("native prior objective", objective)
    accelerator.backward(objective)
    assert_finite_gradients(model)
    if args.gradient_clip > 0:
        gradient_norm = accelerator.clip_grad_norm_(model.parameters(), args.gradient_clip)
        if not math.isfinite(float(gradient_norm)):
            raise FloatingPointError(
                f"non-finite native-prior gradient norm: {float(gradient_norm)}"
            )
    optimizer.step()
    assert_finite_trainable_parameters(model)
    return {
        "prior_nll_sum": float(nll_sum.detach()),
        "prior_pairs": float(pair_count),
    }


@torch.no_grad()
def evaluate_prior(model, loader, accelerator, args) -> dict[str, float]:
    """Evaluate exact pair-weighted native MDN NLL across all ranks."""
    model.eval()
    nll_sum = torch.zeros((), device=accelerator.device)
    pair_count = torch.zeros((), device=accelerator.device)
    graph_count = torch.zeros((), device=accelerator.device)
    for prior_batch in loader:
        microbatch = prior_batch.to(accelerator.device)
        prior = model(microbatch, prior_only=True)
        batch_nll, batch_pairs = near_native_prior_nll_sum_count(
            prior.pi,
            prior.mu,
            prior.sigma,
            prior.candidate_distances,
            prior.pair_mask,
            microbatch.decoy_is_prior_valid.view(-1),
            contact_radius=args.prior_contact_radius,
        )
        nll_sum += batch_nll
        pair_count += batch_pairs
        graph_count += int(microbatch.num_graphs)
    packed = torch.stack((nll_sum, pair_count, graph_count)).unsqueeze(0)
    gathered = accelerator.gather(packed)
    totals = gathered.sum(dim=0)
    if float(totals[1]) <= 0:
        raise RuntimeError("native validation stream contained no valid contact pair")
    return {
        "prior_nll": float((totals[0] / totals[1]).cpu()),
        "prior_pairs": float(totals[1].cpu()),
        "prior_graphs": float(totals[2].cpu()),
    }


@torch.no_grad()
def evaluate(model, loader, dataset, accelerator, args) -> tuple[dict[str, float], list[dict]]:
    model.eval()
    local_records = []
    for group_batch in loader:
        for graph_chunk in split_graph_batch(group_batch, args.pose_microbatch_size):
            microbatch = Batch.from_data_list(graph_chunk).to(accelerator.device)
            output = model(microbatch)
            component_rows = zip(
                output.pose.score.detach().cpu(),
                output.pose.prior_score.detach().cpu(),
                output.pose.cross_score.detach().cpu(),
                output.pose.na_score.detach().cpu(),
                output.pose.clash_penalty.detach().cpu(),
                output.pose.strain_penalty.detach().cpu(),
                output.pose.ordinal_logit_lt5.detach().cpu(),
                output.pose.ordinal_logit_lt2_given_lt5.detach().cpu(),
                output.pose.prob_lt2.detach().cpu(),
                output.pose.prob_lt5.detach().cpu(),
                output.pose.coverage_score.detach().cpu(),
                output.pose.na_gate.detach().cpu(),
            )
            for graph, components in zip(graph_chunk, component_rows):
                (
                    score,
                    prior_score,
                    cross_score,
                    na_score,
                    clash_penalty,
                    strain_penalty,
                    logit_lt5,
                    logit_lt2_given_lt5,
                    prob_lt2,
                    prob_lt5,
                    coverage_score,
                    na_gate,
                ) = components
                local_records.append(
                    {
                        "pose_uid": str(graph.decoy_pose_uid),
                        "pose_group_uid": str(graph.decoy_pose_group_uid),
                        "score": float(score),
                        "prior_score": float(prior_score),
                        "cross_score": float(cross_score),
                        "na_score": float(na_score),
                        "clash_penalty": float(clash_penalty),
                        "strain_penalty": float(strain_penalty),
                        "ordinal_logit_lt5": float(logit_lt5),
                        "ordinal_logit_lt2_given_lt5": float(logit_lt2_given_lt5),
                        "prob_lt2": float(prob_lt2),
                        "prob_lt5": float(prob_lt5),
                        "coverage_score": float(coverage_score),
                        "na_gate": float(na_gate),
                        "rmsd": float(graph.decoy_rmsd.view(-1)[0]),
                        "candidate_rank_input": int(
                            graph.decoy_candidate_rank_input.view(-1)[0]
                        ),
                    }
                )
    records = gather_pose_records(accelerator, local_records)
    assert_complete_group_records(records, dataset.expected_group_counts)
    score = torch.tensor([record["score"] for record in records])
    rmsd = torch.tensor([record["rmsd"] for record in records])
    group = torch.tensor(
        [stable_group_integer(record["pose_group_uid"]) for record in records], dtype=torch.long
    )
    original_rank = torch.tensor(
        [record["candidate_rank_input"] for record in records], dtype=torch.long
    )
    return grouped_rerank_metrics(score, rmsd, group, original_rank=original_rank), records


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument(
        "--scorer-variant",
        choices=(
            "aligned_a",
            "full_v2_b",
            "distribution_s1",
            "gated_na_s2",
        ),
        required=True,
    )
    parser.add_argument("--g22-source-root")
    parser.add_argument("--diffusion-checkpoint", required=True)
    parser.add_argument("--protein-scorer-checkpoint")
    parser.add_argument("--scorer-restart")
    parser.add_argument("--mdn-reference-json", required=True)
    parser.add_argument("--train-manifest")
    parser.add_argument("--train-gate-report")
    parser.add_argument("--train-split")
    parser.add_argument("--ranker-profile", choices=("raw", "curated"), default="raw")
    parser.add_argument("--val-manifest")
    parser.add_argument("--val-gate-report")
    parser.add_argument("--val-split")
    parser.add_argument("--prior-train-manifest")
    parser.add_argument("--prior-train-gate-report")
    parser.add_argument("--prior-train-split")
    parser.add_argument("--prior-val-manifest")
    parser.add_argument("--prior-val-gate-report")
    parser.add_argument("--prior-val-split")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--surface-path", required=True)
    parser.add_argument("--surface-scaler-json", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--phase", choices=("prior", "ranker", "joint"), default="ranker")
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--head-lr", type=float)
    parser.add_argument("--backbone-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--pose-microbatch-size", type=int, default=1)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--dataset-workers", type=int, default=1)
    parser.add_argument("--max-surface-vertices", type=int, default=512)
    parser.add_argument("--per-complex-timeout-sec", type=int, default=180)
    parser.add_argument("--remove-hs", action="store_true")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--prior-weight", type=float, default=0.2)
    parser.add_argument("--listwise-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-weight", type=float, default=0.25)
    parser.add_argument("--ordinal-weight", type=float)
    parser.add_argument("--rmsd-temperature", type=float, default=1.0)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--min-rmsd-gap", type=float, default=0.5)
    parser.add_argument("--pairwise-margin", type=float, default=0.2)
    parser.add_argument(
        "--listwise-mode",
        choices=("rmsd_softmax", "threshold_topheavy"),
        default="rmsd_softmax",
    )
    parser.add_argument(
        "--pairwise-mode", choices=("gap", "boundary"), default="gap"
    )
    parser.add_argument("--distribution-policy-json")
    parser.add_argument("--distribution-samples-per-epoch", type=int, default=0)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--prior-contact-radius", type=float, default=7.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    # CUDA scatter reductions are atomically accumulated and can differ by a
    # few 1e-5 between otherwise identical forward passes.  This remains far
    # below the score changes caused by a genuinely different dropout mask.
    parser.add_argument("--two-pass-parity-atol", type=float, default=1e-3)
    parser.add_argument("--log-every-groups", type=int, default=25)
    parser.add_argument("--calibration-groups", type=int, default=16)
    parser.add_argument("--full-recalibration-epoch", type=int, default=2)
    parser.add_argument("--effective-groups-per-update", type=int, default=4)
    parser.add_argument("--head-warmup-epochs", type=int, default=2)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--evaluate-initial-state", action="store_true")
    parser.add_argument("--allow-nonformal-smoke", action="store_true")
    parser.add_argument("--limit-train-complexes", type=int, default=0)
    parser.add_argument("--limit-val-complexes", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.groups_per_batch != 1:
        raise ValueError("formal V2 scorer currently fixes groups_per_batch=1")
    if (args.limit_train_complexes or args.limit_val_complexes) and not args.allow_nonformal_smoke:
        raise ValueError("complex limits are smoke-only and require --allow-nonformal-smoke")
    if args.phase != "prior" and not args.allow_nonformal_smoke and args.ranker_profile != "raw":
        raise ValueError("formal A/B scorer training requires untouched raw K40 ranker groups")
    if (
        args.phase != "prior"
        and not args.allow_nonformal_smoke
        and args.max_surface_vertices != 512
    ):
        raise ValueError("formal A/B scorer comparison fixes max_surface_vertices=512")
    if args.phase == "prior" and args.scorer_variant != "aligned_a":
        raise ValueError("protein native-prior pretraining requires scorer_variant=aligned_a")
    if (
        args.phase != "prior"
        and not args.scorer_restart
        and not args.protein_scorer_checkpoint
        and not args.allow_nonformal_smoke
    ):
        raise ValueError(
            "formal ranker training requires --protein-scorer-checkpoint; "
            "direct diffusion-only initialization is no longer a formal V2 method"
        )
    if args.effective_groups_per_update < 1:
        raise ValueError("effective_groups_per_update must be positive")
    if args.backbone_lr_multiplier <= 0 or args.backbone_lr_multiplier > 1:
        raise ValueError("backbone_lr_multiplier must be in (0,1]")
    threshold_variants = {"full_v2_b", "distribution_s1", "gated_na_s2"}
    if args.ordinal_weight is None:
        args.ordinal_weight = 0.25 if args.scorer_variant in threshold_variants else 0.0
    if args.scorer_variant == "aligned_a" and args.ordinal_weight != 0:
        raise ValueError("aligned_a has no ordinal head and requires ordinal_weight=0")
    if args.scorer_variant in threshold_variants and args.ordinal_weight <= 0:
        raise ValueError(f"{args.scorer_variant} requires ordinal_weight>0")
    optimized_variants = {"distribution_s1", "gated_na_s2"}
    if args.scorer_variant in optimized_variants:
        if args.phase != "ranker":
            raise ValueError("S1/S2 optimization contract requires phase=ranker")
        if not args.distribution_policy_json:
            raise ValueError("S1/S2 require a validation-derived distribution policy")
        if args.listwise_mode != "threshold_topheavy" or args.pairwise_mode != "boundary":
            raise ValueError("S1/S2 require threshold_topheavy + boundary objectives")
        if not args.freeze_backbone:
            raise ValueError("S1/S2 require the transferred protein backbone to stay frozen")
        if not args.evaluate_initial_state:
            raise ValueError(
                "S1/S2 require --evaluate-initial-state so the untrained protein-MDN "
                "initialization remains eligible for strict-validation selection"
            )
    if args.evaluate_initial_state and args.scorer_restart:
        raise ValueError("--evaluate-initial-state is only valid for a fresh scorer run")
    if args.evaluate_initial_state and args.phase == "prior":
        raise ValueError("--evaluate-initial-state is defined only for ranker validation")
    if args.scorer_variant == "full_v2_b":
        if not args.g22_source_root:
            raise ValueError("full_v2_b requires --g22-source-root")
        env_root = os.environ.get("SURFNA_V2_G22_SOURCE_ROOT", "")
        if Path(env_root).resolve() != Path(args.g22_source_root).resolve():
            raise ValueError(
                "SURFNA_V2_G22_SOURCE_ROOT must equal --g22-source-root before Python starts"
            )
    args.head_lr = args.lr if args.head_lr is None else args.head_lr
    args.pose_bin_weights = None
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp])
    set_seed(args.seed, device_specific=False)
    ranker_required = {
        "train_manifest": args.train_manifest,
        "train_gate_report": args.train_gate_report,
        "train_split": args.train_split,
        "val_manifest": args.val_manifest,
        "val_gate_report": args.val_gate_report,
        "val_split": args.val_split,
    }
    distribution_policy = None
    if args.phase != "prior":
        missing = [name for name, value in ranker_required.items() if not value]
        if missing:
            raise ValueError("ranker stream is required; missing " + ", ".join(missing))
        train_gate = validate_gate(
            args.train_gate_report,
            args.train_manifest,
            allow_nonformal=args.allow_nonformal_smoke,
        )
        val_gate = validate_gate(
            args.val_gate_report,
            args.val_manifest,
            allow_nonformal=args.allow_nonformal_smoke,
        )
        if args.distribution_policy_json:
            distribution_policy = load_distribution_policy(
                args.distribution_policy_json,
                args.train_manifest,
                args.val_manifest,
            )
            args.pose_bin_weights = distribution_policy["pose_bin_weight_tuple"]
    else:
        train_gate = val_gate = None
    prior_gate = None
    prior_val_gate = None
    if args.phase != "ranker":
        if not args.prior_train_gate_report or not args.prior_train_manifest:
            raise ValueError("phase prior/joint requires the separate native prior manifest and gate")
        prior_gate = validate_gate(
            args.prior_train_gate_report,
            args.prior_train_manifest,
            allow_nonformal=args.allow_nonformal_smoke,
        )
        if args.phase == "prior":
            if not args.prior_val_gate_report or not args.prior_val_manifest:
                raise ValueError("pure prior phase requires a native validation manifest and gate")
            prior_val_gate = validate_gate(
                args.prior_val_gate_report,
                args.prior_val_manifest,
                allow_nonformal=args.allow_nonformal_smoke,
            )
    overlap_train_manifest = (
        args.prior_train_manifest if args.phase == "prior" else args.train_manifest
    )
    overlap_val_manifest = (
        args.prior_val_manifest if args.phase == "prior" else args.val_manifest
    )
    train_val_overlap = sorted(
        manifest_complexes(overlap_train_manifest)
        & manifest_complexes(overlap_val_manifest)
    )
    if train_val_overlap:
        raise ValueError(
            f"scorer train/validation complex overlap is forbidden: {train_val_overlap[:10]}"
        )

    model_args = load_scorer_model_args(args.model_config)
    model_args.scorer_variant = args.scorer_variant
    model_args.g22_source_root = args.g22_source_root
    model_args.mdn_reference_json = args.mdn_reference_json
    model_args.allow_unfitted_mdn_reference = False
    t_to_sigma = partial(t_to_sigma_compl, args=model_args)
    model = build_surfna_v2_scorer(model_args, accelerator.device, t_to_sigma)
    restart_checkpoint = None
    start_epoch = 0
    if args.scorer_restart:
        restart_checkpoint = torch.load(args.scorer_restart, map_location="cpu")
        state = checkpoint_state(restart_checkpoint)
        model.load_state_dict(state, strict=True)
        start_epoch = int(restart_checkpoint.get("epoch", -1)) + 1
        if start_epoch >= args.n_epochs:
            raise ValueError(
                f"restart epoch {start_epoch - 1} already reaches n_epochs={args.n_epochs}"
            )
        transfer_report = {
            "scorer_restart": str(Path(args.scorer_restart).resolve()),
            "scorer_restart_sha256": sha256_file(args.scorer_restart),
            "start_epoch": start_epoch,
        }
    else:
        diffusion_checkpoint = torch.load(args.diffusion_checkpoint, map_location="cpu")
        diffusion_transfer = load_v2_diffusion_backbone(
            model,
            diffusion_checkpoint,
            surface_feature_schema="v2_full8",
            require_dynamic_ranker=args.phase != "prior",
            require_static_full_coverage=True,
            require_na_transfer=(
                args.scorer_variant == "full_v2_b" and args.phase != "prior"
            ),
        ).to_dict()
        protein_transfer = None
        if args.protein_scorer_checkpoint:
            protein_checkpoint = torch.load(
                args.protein_scorer_checkpoint, map_location="cpu"
            )
            protein_transfer = load_protein_mdn_scorer_transfer(
                model, protein_checkpoint
            ).to_dict()
        transfer_report = {
            "diffusion_backbone": diffusion_transfer,
            "protein_mdn_scorer": protein_transfer,
        }
    phase_report = configure_phase(model, args.phase)
    if args.freeze_backbone:
        for name, parameter in model.named_parameters():
            if name.startswith("backbone."):
                parameter.requires_grad_(False)
        phase_report = {
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "backbone_frozen": True,
        }
    train_dataset, val_dataset = build_ranker_datasets(args, model_args)
    prior_dataset, prior_val_dataset = build_prior_datasets(args, model_args)

    calibration = (
        calibrate_combiner(
            model,
            train_dataset,
            accelerator.device,
            groups=args.calibration_groups,
            microbatch_size=args.pose_microbatch_size,
        )
        if train_dataset is not None
        else {"skipped": "pure_native_prior_phase"}
    )
    backbone_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone_parameters if name.startswith("backbone.") else head_parameters).append(parameter)
    parameter_groups = []
    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": args.head_lr * args.backbone_lr_multiplier,
                "group_name": "backbone",
            }
        )
    if head_parameters:
        parameter_groups.append(
            {"params": head_parameters, "lr": args.head_lr, "group_name": "heads"}
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    if restart_checkpoint is not None:
        optimizer_state = restart_checkpoint.get("optimizer")
        if not isinstance(optimizer_state, dict):
            raise ValueError("scorer restart lacks optimizer state required for exact continuation")
        optimizer.load_state_dict(optimizer_state)
    model, optimizer = accelerator.prepare(model, optimizer)

    train_sampler = val_sampler = train_loader = val_loader = None
    group_loss_weights_by_uid = {}
    if train_dataset is not None:
        full_coverage = (distribution_policy or {}).get("sampling_mode") == "full_coverage_group_loss_weighted"
        if full_coverage:
            if accelerator.num_processes != 1:
                raise ValueError("full-coverage S1/S2 policy is frozen for one GPU per run")
            if args.distribution_samples_per_epoch not in (0, len(train_dataset.expected_group_counts)):
                raise ValueError("full-coverage epoch must visit every group exactly once")
            group_loss_weights_by_uid = {
                uid: distribution_policy["group_loss_weights"][label]
                for uid, label in group_difficulty_from_dataset(train_dataset).items()
            }
            if not math.isclose(sum(group_loss_weights_by_uid.values()) / len(group_loss_weights_by_uid),
                                1.0, abs_tol=1e-8):
                raise ValueError("actual dataset group weights do not have mean one")
        if distribution_policy is not None and not full_coverage:
            train_sampler = DistributionAwareDistributedGroupBatchSampler(
                train_dataset.pose_group_uids,
                difficulty_by_group=group_difficulty_from_dataset(train_dataset),
                target_probabilities=distribution_policy[
                    "final_group_target_probabilities"
                ],
                samples_per_epoch=(
                    args.distribution_samples_per_epoch
                    if args.distribution_samples_per_epoch > 0
                    else len(train_dataset.expected_group_counts)
                ),
                groups_per_batch=1,
                shuffle=True,
                seed=args.seed,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                drop_rank_tail=True,
            )
        else:
            train_sampler = DistributedGroupBatchSampler(
                train_dataset.pose_group_uids,
                groups_per_batch=1,
                shuffle=True,
                seed=args.seed,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                drop_rank_tail=True,
            )
        val_sampler = DistributedGroupBatchSampler(
            val_dataset.pose_group_uids,
            groups_per_batch=1,
            shuffle=False,
            seed=args.seed,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            drop_rank_tail=False,
        )
        train_loader = make_loader(train_dataset, train_sampler, args.loader_workers)
        val_loader = make_loader(val_dataset, val_sampler, args.loader_workers)
        if full_coverage:
            coverage = train_sampler.audit()
            if (coverage["duplicate_local_rows"] or coverage["missing_local_rows"]
                    or coverage["groups_split_across_batches"]
                    or len(train_sampler) != len(group_loss_weights_by_uid)):
                raise ValueError(f"full-coverage sampler gate failed: {coverage}")
    prior_sampler = prior_loader = prior_val_sampler = prior_val_loader = None
    if prior_dataset is not None:
        prior_sampler = DistributedGroupBatchSampler(
            prior_dataset.pose_group_uids,
            groups_per_batch=args.effective_groups_per_update,
            shuffle=True,
            seed=args.seed + 17,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            drop_rank_tail=True,
        )
        prior_loader = make_loader(prior_dataset, prior_sampler, args.loader_workers)
    if prior_val_dataset is not None:
        prior_val_sampler = DistributedGroupBatchSampler(
            prior_val_dataset.pose_group_uids,
            groups_per_batch=args.effective_groups_per_update,
            shuffle=False,
            seed=args.seed + 19,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            drop_rank_tail=False,
        )
        prior_val_loader = make_loader(
            prior_val_dataset, prior_val_sampler, args.loader_workers
        )

    run_dir = Path(args.run_dir)
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "args": vars(args),
            "train_gate": train_gate,
            "val_gate": val_gate,
            "prior_gate": prior_gate,
            "prior_val_gate": prior_val_gate,
            "transfer_report": transfer_report,
            "phase_report": phase_report,
            "calibration": calibration,
            "distribution_policy": distribution_policy,
            "distribution_policy_sha256": (
                sha256_file(args.distribution_policy_json)
                if args.distribution_policy_json
                else None
            ),
            "train_sampler": train_sampler.audit() if train_sampler is not None else None,
            "val_sampler": val_sampler.audit() if val_sampler is not None else None,
            "prior_sampler": prior_sampler.audit() if prior_sampler is not None else None,
            "prior_val_sampler": (
                prior_val_sampler.audit() if prior_val_sampler is not None else None
            ),
            "reference_json_sha256": sha256_file(args.mdn_reference_json),
            "diffusion_checkpoint_sha256": sha256_file(args.diffusion_checkpoint),
            "protein_scorer_checkpoint_sha256": (
                sha256_file(args.protein_scorer_checkpoint)
                if args.protein_scorer_checkpoint
                else None
            ),
            "train_manifest_sha256": (
                sha256_file(args.train_manifest) if args.train_manifest else None
            ),
            "val_manifest_sha256": (
                sha256_file(args.val_manifest) if args.val_manifest else None
            ),
            "prior_train_manifest_sha256": (
                sha256_file(args.prior_train_manifest) if args.prior_train_manifest else None
            ),
            "prior_val_manifest_sha256": (
                sha256_file(args.prior_val_manifest) if args.prior_val_manifest else None
            ),
            "train_split_sha256": (
                sha256_file(args.train_split) if args.train_split else None
            ),
            "val_split_sha256": sha256_file(args.val_split) if args.val_split else None,
            "prior_train_split_sha256": (
                sha256_file(args.prior_train_split) if args.prior_train_split else None
            ),
            "prior_val_split_sha256": (
                sha256_file(args.prior_val_split) if args.prior_val_split else None
            ),
            "surface_scaler_sha256": sha256_file(args.surface_scaler_json),
            "equivariant_graph_rms_cap": float(
                accelerator.unwrap_model(model).backbone.scorer_equivariant_rms_cap
            ),
            "trainer_sha256": sha256_file(__file__),
            "backbone_source_file": str(
                Path(args.g22_source_root).resolve() / "models" / "surface_score_model_v3.py"
                if args.scorer_variant == "full_v2_b"
                else "protein/current"
            ),
            "backbone_source_sha256": (
                sha256_file(
                    str(Path(args.g22_source_root) / "models" / "surface_score_model_v3.py")
                )
                if args.scorer_variant == "full_v2_b"
                else None
            ),
            "early_stopping": {
                "triggered": False,
                "patience": args.early_stop_patience,
                "min_epochs": args.min_epochs,
                "stopped_epoch": None,
            },
            "start_epoch": start_epoch,
        }
        (run_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
    accelerator.wait_for_everyone()

    if restart_checkpoint is not None and restart_checkpoint.get("selection_key") is not None:
        best_key = tuple(float(value) for value in restart_checkpoint["selection_key"])
    else:
        best_key = (
            (-math.inf,)
            if args.phase == "prior"
            else (-math.inf, -math.inf, -math.inf, -math.inf)
        )
    epochs_without_improvement = 0
    if args.evaluate_initial_state:
        initial_started = time.monotonic()
        initial_val_metrics, _ = evaluate(
            model, val_loader, val_dataset, accelerator, args
        )
        initial_key = checkpoint_selection_key(initial_val_metrics)
        best_key = initial_key
        if accelerator.is_main_process:
            initial_report = {
                "epoch": -1,
                "stage": "initial_state_before_any_optimizer_step",
                "train": {},
                "val": initial_val_metrics,
                "selection_key": initial_key,
                "epoch_wall_sec": time.monotonic() - initial_started,
                "trainability": configure_epoch_trainability(
                    accelerator.unwrap_model(model),
                    args.phase,
                    -1,
                    args.head_warmup_epochs,
                    freeze_backbone=args.freeze_backbone,
                ),
                "calibration": calibration,
            }
            with (run_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(initial_report, sort_keys=True) + "\n")
            unwrapped = accelerator.unwrap_model(model)
            initial_state = {
                "epoch": -1,
                "model": copy.deepcopy(unwrapped.state_dict()),
                "optimizer": optimizer.state_dict(),
                "selection_key": initial_key,
                "val_metrics": initial_val_metrics,
            }
            torch.save(initial_state, run_dir / "initial_model.pt")
            torch.save(initial_state, run_dir / "best_model.pt")
            print(json.dumps(initial_report, sort_keys=True), flush=True)
        accelerator.wait_for_everyone()
    for epoch in range(start_epoch, args.n_epochs):
        epoch_started = time.monotonic()
        args.current_epoch = epoch
        trainability = configure_epoch_trainability(
            accelerator.unwrap_model(model),
            args.phase,
            epoch,
            args.head_warmup_epochs,
            freeze_backbone=args.freeze_backbone,
        )
        if (
            train_dataset is not None
            and args.full_recalibration_epoch >= 0
            and epoch == args.full_recalibration_epoch
        ):
            calibration = calibrate_combiner(
                accelerator.unwrap_model(model),
                train_dataset,
                accelerator.device,
                groups=0,
                microbatch_size=args.pose_microbatch_size,
            )
        if prior_sampler is not None:
            prior_sampler.set_epoch(epoch)
        prior_nll_sum = 0.0
        prior_pairs = 0.0
        if prior_loader is not None:
            for prior_batch in prior_loader:
                step = prior_stream_step(model, optimizer, prior_batch, accelerator, args)
                prior_nll_sum += step["prior_nll_sum"]
                prior_pairs += step["prior_pairs"]
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        totals = {}
        n_steps = 0
        accumulation_index = 0
        accumulation_target = 1
        for group_index, group_batch in enumerate(train_loader or ()):
            if accumulation_index == 0:
                accumulation_target = min(
                    args.effective_groups_per_update, len(train_loader) - group_index
                )
            accumulation_index += 1
            should_step = accumulation_index == accumulation_target
            step = two_pass_group_step(
                model,
                optimizer,
                group_batch,
                accelerator,
                args,
                zero_grad=accumulation_index == 1,
                optimizer_step=should_step,
                loss_scale=1.0 / accumulation_target,
                group_loss_weight=group_objective_weight(group_batch, group_loss_weights_by_uid),
            )
            if should_step:
                accumulation_index = 0
            for key, value in step.items():
                if key == "recompute_max_difference":
                    totals[key] = max(totals.get(key, 0.0), float(value))
                else:
                    totals[key] = totals.get(key, 0.0) + float(value)
            n_steps += 1
            if (
                accelerator.is_main_process
                and args.log_every_groups > 0
                and n_steps % args.log_every_groups == 0
            ):
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "train_groups_complete_local": n_steps,
                            "train_groups_total_local": len(train_sampler),
                            "elapsed_sec": time.monotonic() - epoch_started,
                            "recompute_max_difference": totals.get(
                                "recompute_max_difference", 0.0
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        metric_keys = sorted(totals)
        local = torch.tensor(
            [
                totals.get(key, 0.0)
                if key == "recompute_max_difference"
                else totals.get(key, 0.0) / max(n_steps, 1)
                for key in metric_keys
            ],
            device=accelerator.device,
        )
        # accelerate 0.15 interprets ``reduce(..., reduction="mean")`` as a
        # mean over tensor elements as well as ranks, collapsing this metric
        # vector to a scalar.  Gather an explicit rank axis instead so every
        # named metric remains distinct on both one and multiple GPUs.
        if metric_keys:
            gathered_values = accelerator.gather(local.unsqueeze(0))
            global_values = gathered_values.mean(dim=0)
            if "recompute_max_difference" in metric_keys:
                parity_index = metric_keys.index("recompute_max_difference")
                global_values[parity_index] = gathered_values[:, parity_index].max()
            train_metrics = dict(zip(metric_keys, global_values.cpu().tolist()))
        else:
            train_metrics = {}
        if prior_loader is not None:
            prior_local = torch.tensor(
                [[prior_nll_sum, prior_pairs]], device=accelerator.device
            )
            prior_global = accelerator.gather(prior_local).sum(dim=0)
            if float(prior_global[1]) <= 0:
                raise RuntimeError("native training stream contained no valid contact pair")
            train_metrics["prior_nll"] = float(
                (prior_global[0] / prior_global[1]).cpu()
            )
            train_metrics["prior_pairs"] = float(prior_global[1].cpu())
        if args.phase == "prior":
            val_metrics = evaluate_prior(model, prior_val_loader, accelerator, args)
            current_key = (-val_metrics["prior_nll"],)
        else:
            val_metrics, _ = evaluate(model, val_loader, val_dataset, accelerator, args)
            current_key = checkpoint_selection_key(val_metrics)
        improved = current_key > best_key
        if improved:
            best_key = current_key
        if accelerator.is_main_process:
            epoch_report = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "selection_key": current_key,
                "epoch_wall_sec": time.monotonic() - epoch_started,
                "trainability": trainability,
                "calibration": calibration,
            }
            with (run_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(epoch_report, sort_keys=True) + "\n")
            unwrapped = accelerator.unwrap_model(model)
            state = {
                "epoch": epoch,
                "model": copy.deepcopy(unwrapped.state_dict()),
                "optimizer": optimizer.state_dict(),
                "selection_key": current_key,
                "val_metrics": val_metrics,
            }
            torch.save(state, run_dir / "last_model.pt")
            if improved:
                torch.save(state, run_dir / "best_model.pt")
            print(json.dumps(epoch_report, sort_keys=True), flush=True)
        accelerator.wait_for_everyone()
        epochs_without_improvement = 0 if improved else epochs_without_improvement + 1
        stop = (
            epoch + 1 >= args.min_epochs
            and args.early_stop_patience > 0
            and epochs_without_improvement >= args.early_stop_patience
        )
        stop_tensor = torch.tensor(int(stop), device=accelerator.device)
        if accelerator.num_processes > 1:
            torch.distributed.broadcast(stop_tensor, src=0)
        if bool(stop_tensor.item()):
            if accelerator.is_main_process:
                (run_dir / "EARLY_STOPPED").write_text(f"epoch={epoch}\n")
                metadata_path = run_dir / "run_metadata.json"
                metadata = json.loads(metadata_path.read_text())
                metadata["early_stopping"].update(
                    {"triggered": True, "stopped_epoch": epoch}
                )
                metadata_path.write_text(
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n"
                )
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
