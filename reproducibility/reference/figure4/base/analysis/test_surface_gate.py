#!/usr/bin/env python3
"""Run the preregistered Surface-output gate integrity tests on one real graph."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from mechanism_common import (
    FROZEN_CODE_ROOT,
    apply_joint_chemistry_shuffle,
    atomic_json,
    checkpoint_path,
    complex_name,
    graph_invariant_hashes,
    load_model_args,
    load_validation_dataset,
    stable_state_sha256,
    stable_tensor_sha256,
)
from evaluate_fixed_pose_mechanism import verify_zero_z_integrator_equivalence


ABBA_CYCLES = 16
NULL_ENVELOPE_EPS_MULTIPLIER = 4.0
MAX_NULL_ENVELOPE_ABS = 1e-5
MIN_POSITIVE_EFFECT_TO_TAU_RATIO = 1000.0
MIN_POSITIVE_EFFECT_ABS = 1e-4
HEAD_NAMES = ("translation", "rotation", "torsion")


def load_state(path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu")
    state = raw.get("model", raw) if isinstance(raw, dict) else raw
    if not isinstance(state, dict):
        raise TypeError(f"not a model state dict: {path}")
    return {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}


def instantiate_models(model_args, device: torch.device):
    from utils.diffusion_utils import t_to_sigma as t_to_sigma_impl
    import utils.utils as model_utils

    sigma_fn = partial(t_to_sigma_impl, args=model_args)
    patched_class = model_utils.SurfaceScoreModelV3
    patched_model = model_utils.get_model(
        model_args,
        device,
        t_to_sigma=sigma_fn,
        no_parallel=True,
        model_type=model_args.model_type,
    )

    original_path = FROZEN_CODE_ROOT / "models/surface_score_model_v3.py"
    spec = importlib.util.spec_from_file_location("surfna_frozen_original_surface_score_model_v3", original_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen original model: {original_path}")
    original_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original_module)
    try:
        model_utils.SurfaceScoreModelV3 = original_module.TensorProductScoreModel
        original_model = model_utils.get_model(
            model_args,
            device,
            t_to_sigma=sigma_fn,
            no_parallel=True,
            model_type=model_args.model_type,
        )
    finally:
        model_utils.SurfaceScoreModelV3 = patched_class
    return original_model, patched_model


def prepare_batch(graph, t: float, model_args, device: torch.device) -> Batch:
    from utils.diffusion_utils import set_time

    graph = copy.deepcopy(graph)
    set_time(graph, t, t, t, 1, model_args.all_atoms, device=None)
    return Batch.from_data_list([graph]).to(device)


def flattened_outputs(outputs) -> torch.Tensor:
    nonempty = [value.reshape(-1) for value in outputs if value.numel()]
    return torch.cat(nonempty) if nonempty else torch.empty(0)


def gradient_norm(model, graph, t: float, model_args, device: torch.device, alpha: float) -> float:
    batch = prepare_batch(graph, t, model_args, device)
    batch["surface"].x = batch["surface"].x.detach().clone().requires_grad_(True)
    model.analysis_surface_gate_alpha = alpha
    outputs = model(batch)
    flat = flattened_outputs(outputs)
    coefficients = torch.linspace(0.5, 1.5, flat.numel(), device=flat.device, dtype=flat.dtype)
    objective = torch.sum(coefficients * flat)
    gradient = torch.autograd.grad(
        objective,
        batch["surface"].x,
        allow_unused=True,
        retain_graph=False,
        create_graph=False,
    )[0]
    return 0.0 if gradient is None else float(torch.linalg.vector_norm(gradient.detach()).cpu())


def output_diff(left, right) -> float:
    values = []
    for left_value, right_value in zip(left, right):
        if left_value.numel() == 0 and right_value.numel() == 0:
            continue
        values.append(float((left_value.detach() - right_value.detach()).abs().max().cpu()))
    return max(values, default=0.0)


def forward_once(model, graph, t, model_args, device, alpha=None):
    if alpha is not None:
        model.analysis_surface_gate_alpha = float(alpha)
    with torch.no_grad():
        outputs = model(prepare_batch(graph, t, model_args, device))
    return tuple(value.detach().float().reshape(-1).cpu().clone() for value in outputs)


def collect_abba(left, right, cycles: int = ABBA_CYCLES):
    """Collect order-balanced repeated forwards for two conditions."""
    left_outputs = []
    right_outputs = []
    for _ in range(cycles):
        left_outputs.append(forward_once(*left))
        right_outputs.append(forward_once(*right))
        right_outputs.append(forward_once(*right))
        left_outputs.append(forward_once(*left))
    return left_outputs, right_outputs


def head_repeat_diagnostics(left_outputs, right_outputs):
    """Measure centers and within-condition CUDA-float32 repeat envelopes per head."""
    epsilon = float(torch.finfo(torch.float32).eps)
    diagnostics = {}
    for head_index, head_name in enumerate(HEAD_NAMES):
        left = torch.stack([outputs[head_index] for outputs in left_outputs]).double()
        right = torch.stack([outputs[head_index] for outputs in right_outputs]).double()
        if left.shape[1:] != right.shape[1:] or left.numel() == 0:
            raise RuntimeError(f"invalid repeated-output shape for {head_name}: {left.shape}/{right.shape}")
        left_center = left.mean(dim=0)
        right_center = right.mean(dim=0)
        left_envelope = float((left - left_center).abs().max())
        right_envelope = float((right - right_center).abs().max())
        center_difference = (left_center - right_center).abs()
        max_difference, flat_index = center_difference.reshape(-1).max(dim=0)
        index = int(flat_index)
        scale = max(
            1.0,
            float(left.abs().max()),
            float(right.abs().max()),
        )
        tolerance = (
            left_envelope
            + right_envelope
            + NULL_ENVELOPE_EPS_MULTIPLIER * epsilon * scale
        )
        observed = float(max_difference)
        left_value = float(left_center.reshape(-1)[index])
        right_value = float(right_center.reshape(-1)[index])
        finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        envelopes_within_cap = (
            left_envelope <= MAX_NULL_ENVELOPE_ABS
            and right_envelope <= MAX_NULL_ENVELOPE_ABS
        )
        diagnostics[head_name] = {
            "repeat_count_left": int(left.shape[0]),
            "repeat_count_right": int(right.shape[0]),
            "num_output_scalars": int(left_center.numel()),
            "left_null_envelope_max_abs": left_envelope,
            "right_null_envelope_max_abs": right_envelope,
            "output_scale_max_abs": scale,
            "float32_epsilon": epsilon,
            "epsilon_multiplier": NULL_ENVELOPE_EPS_MULTIPLIER,
            "maximum_null_envelope_abs": MAX_NULL_ENVELOPE_ABS,
            "null_envelopes_within_cap": envelopes_within_cap,
            "tolerance": tolerance,
            "center_max_abs_difference": observed,
            "argmax_flat_index": index,
            "left_center_at_argmax": left_value,
            "right_center_at_argmax": right_value,
            "relative_difference_at_argmax": observed / max(abs(left_value), abs(right_value), epsilon),
            "all_repeats_finite": finite,
        }
    return diagnostics


def head_null_envelope_certificate(left_outputs, right_outputs):
    """Certify equivalence against bounded, empirically measured repeat envelopes."""
    diagnostics = head_repeat_diagnostics(left_outputs, right_outputs)
    for item in diagnostics.values():
        item["passed"] = (
            item["all_repeats_finite"]
            and item["null_envelopes_within_cap"]
            and item["center_max_abs_difference"] <= item["tolerance"]
        )
    passed = all(item["passed"] for item in diagnostics.values())
    return {"passed": passed, "heads": diagnostics}


def centered_head_effect(reference_outputs, intervention_outputs):
    diagnostics = {}
    for head_index, head_name in enumerate(HEAD_NAMES):
        reference = torch.stack([outputs[head_index] for outputs in reference_outputs]).double()
        intervention = torch.stack([outputs[head_index] for outputs in intervention_outputs]).double()
        difference = (reference.mean(0) - intervention.mean(0)).abs()
        value, flat_index = difference.reshape(-1).max(dim=0)
        diagnostics[head_name] = {
            "center_max_abs_effect": float(value),
            "argmax_flat_index": int(flat_index),
            "all_repeats_finite": bool(
                torch.isfinite(reference).all() and torch.isfinite(intervention).all()
            ),
        }
    return diagnostics


def positive_control_certificate(reference_outputs, intervention_outputs):
    """Require a large matched-head effect under the same ABBA noise accounting."""
    repeat_diagnostics = head_repeat_diagnostics(reference_outputs, intervention_outputs)
    effects = centered_head_effect(reference_outputs, intervention_outputs)
    heads = {}
    for head_name in HEAD_NAMES:
        repeat = repeat_diagnostics[head_name]
        effect = effects[head_name]
        required = max(
            MIN_POSITIVE_EFFECT_TO_TAU_RATIO * repeat["tolerance"],
            MIN_POSITIVE_EFFECT_ABS,
        )
        heads[head_name] = {
            **repeat,
            **effect,
            "required_minimum_effect": required,
            "effect_to_tau_ratio": (
                effect["center_max_abs_effect"] / max(repeat["tolerance"], 1e-30)
            ),
            "passed_positive_separation": (
                effect["all_repeats_finite"]
                and repeat["all_repeats_finite"]
                and repeat["null_envelopes_within_cap"]
                and effect["center_max_abs_effect"] >= required
            ),
        }
    passed_heads = [name for name, item in heads.items() if item["passed_positive_separation"]]
    return {
        "passed": bool(passed_heads) and all(
            item["all_repeats_finite"] and item["null_envelopes_within_cap"]
            for item in heads.values()
        ),
        "passed_heads": passed_heads,
        "heads": heads,
        "maximum_head_effect": max(
            item["center_max_abs_effect"] for item in heads.values()
        ),
    }


def surface_jacobian_certificate(model, graph, t, model_args, device, alpha, attribute):
    """Compute a VJP for every score scalar with respect to one Surface input tensor."""
    batch = prepare_batch(graph, t, model_args, device)
    source = getattr(batch["surface"], attribute).detach().clone().requires_grad_(True)
    setattr(batch["surface"], attribute, source)
    model.analysis_surface_gate_alpha = float(alpha)
    outputs = model(batch)
    scalars = [
        (head_name, index, value.reshape(-1)[index])
        for head_name, value in zip(HEAD_NAMES, outputs)
        for index in range(value.numel())
    ]
    per_head = {
        name: {"scalar_count": 0, "nonzero_vjp_count": 0, "maximum_abs_vjp": 0.0}
        for name in HEAD_NAMES
    }
    all_finite = True
    for scalar_index, (head_name, _, scalar) in enumerate(scalars):
        gradient = torch.autograd.grad(
            scalar,
            source,
            allow_unused=True,
            retain_graph=scalar_index + 1 < len(scalars),
            create_graph=False,
        )[0]
        maximum = 0.0 if gradient is None else float(gradient.detach().abs().max().cpu())
        finite = gradient is None or bool(torch.isfinite(gradient).all())
        all_finite = all_finite and finite and math.isfinite(maximum)
        record = per_head[head_name]
        record["scalar_count"] += 1
        record["nonzero_vjp_count"] += int(maximum > 0.0)
        record["maximum_abs_vjp"] = max(record["maximum_abs_vjp"], maximum)
    maximum = max(record["maximum_abs_vjp"] for record in per_head.values())
    return {
        "alpha": float(alpha),
        "surface_attribute": attribute,
        "all_finite": all_finite,
        "all_vjps_exact_zero": all(
            record["maximum_abs_vjp"] == 0.0 for record in per_head.values()
        ),
        "any_vjp_nonzero": any(
            record["maximum_abs_vjp"] > 0.0 for record in per_head.values()
        ),
        "maximum_abs_vjp": maximum,
        "heads": per_head,
    }


def stable_value_sha256(value) -> str:
    """Hash tensor containers without relying on Python object identities."""
    digest = hashlib.sha256()

    def update(item) -> None:
        if torch.is_tensor(item):
            digest.update(b"torch:")
            digest.update(stable_tensor_sha256(item).encode())
        elif isinstance(item, np.ndarray):
            digest.update(b"numpy:")
            digest.update(str(tuple(item.shape)).encode())
            digest.update(str(item.dtype).encode())
            if item.dtype.hasobject:
                update(item.tolist())
            else:
                digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict:")
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"{type(item).__name__}:{len(item)}:".encode())
            for element in item:
                update(element)
        else:
            digest.update(f"scalar:{type(item).__name__}:{item!r}".encode())

    update(value)
    return digest.hexdigest()


def expanded_graph_invariant_hashes(graph) -> dict[str, str]:
    """Snapshot all frozen graph inputs that the gated score model can consume."""
    payload = dict(graph_invariant_hashes(graph))
    node_attributes = {
        "ligand": (
            "x", "pos", "orig_pos", "edge_mask", "mask_rotate", "node_t", "batch", "ptr",
        ),
        "receptor": (
            "x", "pos", "center_pos", "atoms_pos", "nucleic_feat",
            "nucleic_anchor_pos", "nucleic_anchor_type", "nucleic_anchor_parent_index",
            "node_t", "batch", "ptr",
        ),
        "surface": ("x", "pos", "face", "node_t", "batch", "ptr"),
        "atom": ("x", "pos", "node_t", "batch", "ptr"),
    }
    for node_type, attributes in node_attributes.items():
        if node_type not in graph.node_types:
            continue
        store = graph[node_type]
        for attribute in attributes:
            value = getattr(store, attribute, None)
            if value is not None:
                payload[f"node::{node_type}::{attribute}"] = stable_value_sha256(value)

    for attribute in ("complex_t", "original_center"):
        value = getattr(graph, attribute, None)
        if value is not None:
            payload[f"graph::{attribute}"] = stable_value_sha256(value)

    for edge_type in graph.edge_types:
        store = graph[edge_type]
        edge_label = "|".join(edge_type)
        for attribute in ("edge_index", "edge_attr", "edge_weight"):
            value = getattr(store, attribute, None)
            if value is not None:
                payload[f"edge::{edge_label}::{attribute}"] = stable_value_sha256(value)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.analysis_root.resolve()
    report_path = root / "03_integrity/surface_gate_integrity_report.json"
    if report_path.exists():
        raise SystemExit(f"refusing to overwrite an existing gate report: {report_path}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("surface-gate integrity must run on CUDA")

    bank_path = root / "04_mechanism_pose_bank/mechanism_pose_bank_v1.pt"
    permutations_path = root / "04_mechanism_pose_bank/shuffle_permutation_manifest.json"
    bank = torch.load(bank_path, map_location="cpu")
    entry = next(
        item for item in bank["entries"]
        if float(item["noise_level"]) == 0.30 and item["true_torsion_score"].numel() > 0
    )
    permutations = json.loads(permutations_path.read_text())
    model_args = load_model_args("Transfer-All", 0)
    integrator_integrity = verify_zero_z_integrator_equivalence(model_args, bank["entries"])
    dataset = load_validation_dataset(model_args)
    graph = dataset[int(entry["dataset_index"])]
    if complex_name(graph) != entry["complex_id"]:
        raise RuntimeError("gate test graph does not match pose-bank identity")
    graph["ligand"].pos = entry["initial_ligand_coordinates"].clone()
    shuffled_graphs = {}
    for shuffle_seed in (20260911, 20260912, 20260913):
        shuffled_graph = copy.deepcopy(graph)
        permutation = torch.tensor(
            permutations["permutations"][entry["complex_id"]][str(shuffle_seed)]["permutation"],
            dtype=torch.long,
        )
        apply_joint_chemistry_shuffle(shuffled_graph, permutation)
        shuffled_graphs[str(shuffle_seed)] = shuffled_graph
    position_counterfactual_graph = copy.deepcopy(graph)
    position_shift = torch.tensor(
        [2.5, -1.75, 1.25], dtype=position_counterfactual_graph["surface"].pos.dtype
    )
    position_counterfactual_graph["surface"].pos = (
        position_counterfactual_graph["surface"].pos + position_shift
    )

    original_model, patched_model = instantiate_models(model_args, device)
    if next(patched_model.parameters()).dtype != torch.float32:
        raise RuntimeError("surface-gate integrity requires float32 model parameters")
    state_path = checkpoint_path(root, "Transfer-All", 0, 200)
    state = load_state(state_path)
    original_model.load_state_dict(state, strict=True)
    patched_model.load_state_dict(state, strict=True)
    original_model.eval()
    patched_model.eval()
    torch.manual_seed(20260903)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260903)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    original_keys = set(original_model.state_dict())
    patched_keys = set(patched_model.state_dict())
    analysis_parameter_names = [
        name for name, _ in patched_model.named_parameters()
        if "analysis_surface_gate" in name
    ]
    analysis_buffer_names = [
        name for name, _ in patched_model.named_buffers()
        if "analysis_surface_gate" in name
    ]

    common = (0.30, model_args, device)
    original_repeats, alpha1_true_repeats = collect_abba(
        (original_model, graph, *common, None),
        (patched_model, graph, *common, 1.0),
    )
    alpha1_equivalence = head_null_envelope_certificate(
        original_repeats, alpha1_true_repeats
    )

    alpha0_chemistry_certificates = {}
    alpha1_chemistry_positive_controls = {}
    for shuffle_seed, shuffled_graph in shuffled_graphs.items():
        alpha0_true_repeats, alpha0_shuffle_repeats = collect_abba(
            (patched_model, graph, *common, 0.0),
            (patched_model, shuffled_graph, *common, 0.0),
        )
        certificate = head_null_envelope_certificate(
            alpha0_true_repeats, alpha0_shuffle_repeats
        )
        alpha0_chemistry_certificates[shuffle_seed] = certificate
        alpha1_true_positive_repeats, alpha1_shuffle_repeats = collect_abba(
            (patched_model, graph, *common, 1.0),
            (patched_model, shuffled_graph, *common, 1.0),
        )
        alpha1_chemistry_positive_controls[shuffle_seed] = positive_control_certificate(
            alpha1_true_positive_repeats, alpha1_shuffle_repeats
        )

    alpha0_position_true_repeats, alpha0_position_shifted_repeats = collect_abba(
        (patched_model, graph, *common, 0.0),
        (patched_model, position_counterfactual_graph, *common, 0.0),
    )
    alpha0_position_certificate = head_null_envelope_certificate(
        alpha0_position_true_repeats, alpha0_position_shifted_repeats
    )
    alpha1_position_true_repeats, alpha1_position_shifted_repeats = collect_abba(
        (patched_model, graph, *common, 1.0),
        (patched_model, position_counterfactual_graph, *common, 1.0),
    )
    alpha1_position_positive_control = positive_control_certificate(
        alpha1_position_true_repeats, alpha1_position_shifted_repeats
    )
    alpha1_position_positive_control["surface_position_shift"] = position_shift.tolist()

    jacobian_certificates = {
        "alpha0_surface_x": surface_jacobian_certificate(
            patched_model, graph, *common, 0.0, "x"
        ),
        "alpha1_surface_x": surface_jacobian_certificate(
            patched_model, graph, *common, 1.0, "x"
        ),
        "alpha0_surface_pos": surface_jacobian_certificate(
            patched_model, graph, *common, 0.0, "pos"
        ),
        "alpha1_surface_pos": surface_jacobian_certificate(
            patched_model, graph, *common, 1.0, "pos"
        ),
    }

    alpha1_original_diff = output_diff(original_repeats[0], alpha1_true_repeats[0])
    alpha0_chemistry_diff = max(
        item["center_max_abs_difference"]
        for certificate in alpha0_chemistry_certificates.values()
        for item in certificate["heads"].values()
    )
    alpha1_chemistry_diff = max(
        record["maximum_head_effect"]
        for record in alpha1_chemistry_positive_controls.values()
    )

    batch_for_invariants = prepare_batch(graph, 0.30, model_args, device)
    invariants_before = expanded_graph_invariant_hashes(batch_for_invariants)
    patched_model.analysis_surface_gate_alpha = 0.5
    with torch.no_grad():
        patched_model(batch_for_invariants)
    invariants_after = expanded_graph_invariant_hashes(batch_for_invariants)
    invariant_mismatches = sorted(
        key for key in set(invariants_before) | set(invariants_after)
        if invariants_before.get(key) != invariants_after.get(key)
    )

    tests = {
        "A_alpha1_matches_original_under_per_head_null_envelope": alpha1_equivalence["passed"],
        "B_alpha0_invariant_to_three_chemistry_shuffles": all(
            record["passed"] for record in alpha0_chemistry_certificates.values()
        ),
        "B_alpha0_invariant_to_surface_position_counterfactual": (
            alpha0_position_certificate["passed"]
        ),
        "B_alpha1_chemistry_positive_controls_separated": all(
            record["passed"] for record in alpha1_chemistry_positive_controls.values()
        ),
        "B_alpha1_position_positive_control_separated": (
            alpha1_position_positive_control["passed"]
        ),
        "C_alpha0_surface_x_full_scalar_vjp_exact_zero": (
            jacobian_certificates["alpha0_surface_x"]["all_finite"]
            and jacobian_certificates["alpha0_surface_x"]["all_vjps_exact_zero"]
        ),
        "C_alpha0_surface_pos_full_scalar_vjp_exact_zero": (
            jacobian_certificates["alpha0_surface_pos"]["all_finite"]
            and jacobian_certificates["alpha0_surface_pos"]["all_vjps_exact_zero"]
        ),
        "C_alpha1_surface_x_has_nonzero_vjp": (
            jacobian_certificates["alpha1_surface_x"]["all_finite"]
            and jacobian_certificates["alpha1_surface_x"]["any_vjp_nonzero"]
        ),
        "C_alpha1_surface_pos_has_nonzero_vjp": (
            jacobian_certificates["alpha1_surface_pos"]["all_finite"]
            and jacobian_certificates["alpha1_surface_pos"]["any_vjp_nonzero"]
        ),
        "D_all_frozen_graph_inputs_unchanged": not invariant_mismatches,
        "E_zero_z_sde_matches_frozen_sampling": integrator_integrity["passed"],
        "checkpoint_state_keys_unchanged": original_keys == patched_keys,
        "gate_not_parameter_or_buffer": not analysis_parameter_names and not analysis_buffer_names,
        "all_three_shuffle_seeds_covered": (
            set(alpha0_chemistry_certificates) == {"20260911", "20260912", "20260913"}
        ),
        "backend_is_cuda_float32": (
            device.type == "cuda" and next(patched_model.parameters()).dtype == torch.float32
        ),
    }
    report = {
        "schema_version": "surfna-lb200-surface-output-gate-integrity-v3",
        "passed": all(tests.values()),
        "model": {"arm": "Transfer-All", "seed": 0, "epoch": 200},
        "complex_id": entry["complex_id"],
        "noise_level": 0.30,
        "dtype": str(next(patched_model.parameters()).dtype),
        "gate_semantics": (
            "alpha multiplies every Surface-to-ligand residual and every Surface-conditioned "
            "ligand patch-tokenizer residual; alpha=0 disconnects Surface outputs from all "
            "translation, rotation, and torsion score heads"
        ),
        "thresholds": {
            "equivalence_rule": (
                "per-head |mean(left)-mean(right)| <= E_left + E_right + "
                "4*float32_eps*max(1, output_scale), with E measured from ABBA identical repeats"
            ),
            "abba_cycles": ABBA_CYCLES,
            "repeats_per_condition": 2 * ABBA_CYCLES,
            "positive_control_repeats_per_condition": 2 * ABBA_CYCLES,
            "null_envelope_epsilon_multiplier": NULL_ENVELOPE_EPS_MULTIPLIER,
            "maximum_null_envelope_abs": MAX_NULL_ENVELOPE_ABS,
            "minimum_positive_effect_to_tau_ratio": MIN_POSITIVE_EFFECT_TO_TAU_RATIO,
            "minimum_positive_effect_absolute": MIN_POSITIVE_EFFECT_ABS,
            "alpha0_vjp_rule": "every score scalar VJP with respect to Surface.x and Surface.pos is exactly zero or disconnected",
        },
        "measurements": {
            "alpha1_vs_frozen_original_max_abs_diff": alpha1_original_diff,
            "alpha0_true_vs_shuffled_max_abs_diff": alpha0_chemistry_diff,
            "alpha1_true_vs_shuffled_max_abs_diff": alpha1_chemistry_diff,
            "original_state_sha256": stable_state_sha256(original_model.state_dict()),
            "patched_state_sha256": stable_state_sha256(patched_model.state_dict()),
            "graph_invariant_tensor_count": len(invariants_before),
            "graph_invariant_mismatches": invariant_mismatches,
        },
        "backend": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn_allowed": bool(torch.backends.cudnn.allow_tf32),
        },
        "alpha1_original_null_envelope": alpha1_equivalence,
        "alpha0_chemistry_null_envelopes": alpha0_chemistry_certificates,
        "alpha0_position_null_envelope": alpha0_position_certificate,
        "alpha1_chemistry_positive_controls": alpha1_chemistry_positive_controls,
        "alpha1_position_positive_control": alpha1_position_positive_control,
        "jacobian_certificates": jacobian_certificates,
        "graph_invariant_keys": sorted(invariants_before),
        "zero_z_integrator_equivalence": integrator_integrity,
        "tests": tests,
        "analysis_parameter_names": analysis_parameter_names,
        "analysis_buffer_names": analysis_buffer_names,
    }
    atomic_json(report_path, report)
    if not report["passed"]:
        raise RuntimeError(f"surface gate integrity failed: {tests}")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
