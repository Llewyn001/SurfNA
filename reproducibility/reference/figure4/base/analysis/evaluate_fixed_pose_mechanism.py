#!/usr/bin/env python3
"""Evaluate one arm/seed/checkpoint on the frozen noisy-pose mechanism bank."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from mechanism_common import (
    ARMS,
    DOSE_EPOCHS,
    DOSE_GATE_ALPHAS,
    EPOCHS,
    FROZEN_CODE_ROOT,
    INFERENCE_STEPS,
    MAIN_GATE_ALPHAS,
    SHUFFLE_SEEDS,
    apply_joint_chemistry_shuffle,
    arm_slug,
    atomic_json,
    atomic_torch_save,
    checkpoint_path,
    complex_name,
    equal_component_mean,
    evaluator_rmsd,
    heavy_atom_mask,
    load_score_model,
    load_validation_dataset,
    native_ligand_coordinates,
    normalized_mse,
    sha256_file,
    stable_tensor_sha256,
    tensor_cosine,
)


CSV_FIELDS = [
    "arm", "training_seed", "epoch", "complex_id", "noise_level", "pose_repeat",
    "condition", "gate_alpha", "shuffle_seed", "entry_sha256", "initial_coordinates_sha256",
    "translation_cosine", "translation_normalized_mse", "translation_direction_sign_agreement",
    "rotation_cosine", "rotation_normalized_mse", "torsion_cosine", "torsion_normalized_mse",
    "combined_alignment", "combined_alignment_components", "RMSD_before", "RMSD_after",
    "delta_RMSD_1step", "direct_RMSD_before", "direct_RMSD_after", "direct_delta_RMSD_1step",
    "rmsd_source_method_before", "rmsd_source_method_after", "num_rotatable_bonds",
    "valid", "failure_reason",
]


def chunks(items, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def direct_rmsd(graph, coordinates) -> float:
    coords = coordinates.detach().cpu().numpy() if torch.is_tensor(coordinates) else np.asarray(coordinates)
    mask = heavy_atom_mask(graph)
    native = native_ligand_coordinates(graph)
    return float(np.sqrt(((coords[mask] - native[mask]) ** 2).sum(axis=1).mean()))


def deterministic_step_size(t: float) -> float:
    from utils.diffusion_utils import get_t_schedule

    schedule = list(map(float, get_t_schedule(INFERENCE_STEPS))) + [0.0]
    matches = [index for index, value in enumerate(schedule[:-1]) if abs(value - t) < 1e-8]
    if len(matches) != 1:
        raise RuntimeError(f"noise level {t} is not a unique node of the frozen 20-step schedule")
    index = matches[0]
    dt = schedule[index] - schedule[index + 1]
    if dt <= 0:
        raise RuntimeError(f"invalid reverse-diffusion interval at t={t}: {dt}")
    return dt


def one_step_updates(model_args, entry, tr_score, rot_score, tor_score):
    """Match sampling.py's SDE drift with its stochastic z term fixed to zero."""
    dt = deterministic_step_size(float(entry["noise_level"]))
    tr_sigma = float(entry["tr_sigma"])
    rot_sigma = float(entry["rot_sigma"])
    tor_sigma = float(entry["tor_sigma"])
    tr_g = tr_sigma * torch.sqrt(torch.tensor(
        2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)
    ))
    rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(
        np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)
    ))
    tr_score_cpu = tr_score.detach().cpu()
    rot_score_cpu = rot_score.detach().cpu()
    tr_z = torch.zeros(tr_score_cpu.shape)
    rot_z = torch.zeros(rot_score_cpu.shape)
    tr_update = (tr_g ** 2 * dt * tr_score_cpu + tr_g * np.sqrt(dt) * tr_z).cpu()
    rot_update = (rot_score_cpu * dt * rot_g ** 2 + rot_g * np.sqrt(dt) * rot_z).cpu()
    tor_update = None
    if tor_score.numel():
        tor_g = tor_sigma * torch.sqrt(torch.tensor(
            2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)
        ))
        tor_score_cpu = tor_score.detach().cpu()
        tor_z = torch.zeros(tor_score_cpu.shape)
        tor_update = (
            tor_g ** 2 * dt * tor_score_cpu + tor_g * np.sqrt(dt) * tor_z
        ).numpy()
    return tr_update, rot_update, tor_update, dt


def frozen_sampling_zero_z_reference_updates(model_args, entry, tr_score, rot_score, tor_score):
    """Literal zero-z SDE branch of frozen utils/sampling.py lines 116--138."""
    dt = deterministic_step_size(float(entry["noise_level"]))
    tr_sigma = float(entry["tr_sigma"])
    rot_sigma = float(entry["rot_sigma"])
    tor_sigma = float(entry["tor_sigma"])
    tr_g = tr_sigma * torch.sqrt(torch.tensor(
        2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)
    ))
    rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(
        np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)
    ))
    tr_score_cpu = tr_score.detach().cpu()
    rot_score_cpu = rot_score.detach().cpu()
    tr_z = torch.zeros(tr_score_cpu.shape)
    rot_z = torch.zeros(rot_score_cpu.shape)
    tr_update = (tr_g ** 2 * dt * tr_score_cpu + tr_g * np.sqrt(dt) * tr_z).cpu()
    rot_update = (rot_score_cpu * dt * rot_g ** 2 + rot_g * np.sqrt(dt) * rot_z).cpu()
    tor_update = None
    if tor_score.numel():
        tor_g = tor_sigma * torch.sqrt(torch.tensor(
            2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)
        ))
        tor_score_cpu = tor_score.detach().cpu()
        tor_z = torch.zeros(tor_score_cpu.shape)
        tor_update = (
            tor_g ** 2 * dt * tor_score_cpu + tor_g * np.sqrt(dt) * tor_z
        ).numpy()
    return tr_update, rot_update, tor_update, dt


def verify_zero_z_integrator_equivalence(model_args, entries, tolerance: float = 1e-6) -> dict:
    """Fail fast unless our one-step helper matches the frozen sampling SDE formula."""
    representatives = {}
    for entry in entries:
        representatives.setdefault(float(entry["noise_level"]), entry)
    if not representatives:
        raise RuntimeError("cannot validate the one-step integrator without pose-bank entries")

    tr_score = torch.tensor([[0.25, -0.75, 1.50]], dtype=torch.float32)
    rot_score = torch.tensor([[-1.25, 0.50, 0.125]], dtype=torch.float32)
    tor_score = torch.tensor([0.625, -0.375, 1.125], dtype=torch.float32)
    records = []
    global_max = 0.0
    for noise_level, entry in sorted(representatives.items()):
        observed = one_step_updates(model_args, entry, tr_score, rot_score, tor_score)
        reference = frozen_sampling_zero_z_reference_updates(
            model_args, entry, tr_score, rot_score, tor_score
        )
        tr_diff = float((observed[0] - reference[0]).abs().max())
        rot_diff = float((observed[1] - reference[1]).abs().max())
        tor_diff = float(np.max(np.abs(observed[2] - reference[2])))
        dt_diff = abs(float(observed[3]) - float(reference[3]))
        maximum = max(tr_diff, rot_diff, tor_diff, dt_diff)
        global_max = max(global_max, maximum)
        records.append({
            "noise_level": noise_level,
            "dt": float(observed[3]),
            "translation_max_abs_diff": tr_diff,
            "rotation_max_abs_diff": rot_diff,
            "torsion_max_abs_diff": tor_diff,
            "dt_abs_diff": dt_diff,
            "passed": maximum <= tolerance,
        })

    sampling_path = FROZEN_CODE_ROOT / "utils/sampling.py"
    report = {
        "schema_version": "surfna-lb200-zero-z-sde-equivalence-v1",
        "passed": global_max <= tolerance and all(record["passed"] for record in records),
        "tolerance": tolerance,
        "maximum_abs_difference": global_max,
        "reference_source": str(sampling_path),
        "reference_source_sha256": sha256_file(sampling_path),
        "reference_formula_lines": "116-138",
        "noise_levels_checked": sorted(representatives),
        "records": records,
    }
    if not report["passed"]:
        raise RuntimeError(
            "one-step zero-z SDE implementation differs from frozen sampling.py: "
            f"max_abs_diff={global_max}"
        )
    return report


def assert_graph_torsion_topology(graph, entry) -> tuple[int, dict[str, int]]:
    """Assert pose-bank, edge-mask, and torsion-mask topology agree for one graph."""
    graph_edge_mask = graph["ligand"].edge_mask.detach().cpu().reshape(-1).bool()
    entry_edge_mask = entry["rotatable_bond_mask"].detach().cpu().reshape(-1).bool()
    if graph_edge_mask.shape != entry_edge_mask.shape or not torch.equal(graph_edge_mask, entry_edge_mask):
        raise RuntimeError(
            f"torsion edge-mask identity mismatch for {entry['complex_id']} "
            f"at noise={entry['noise_level']} repeat={entry['pose_repeat']}"
        )

    mask_rotate = graph["ligand"].mask_rotate
    while isinstance(mask_rotate, (list, tuple)):
        if len(mask_rotate) != 1:
            raise RuntimeError(
                f"ambiguous per-graph mask_rotate container for {entry['complex_id']}: "
                f"length={len(mask_rotate)}"
            )
        mask_rotate = mask_rotate[0]
    mask_rotate_array = (
        mask_rotate.detach().cpu().numpy()
        if torch.is_tensor(mask_rotate)
        else np.asarray(mask_rotate)
    )
    if mask_rotate_array.ndim != 2:
        raise RuntimeError(
            f"mask_rotate must be rank 2 for {entry['complex_id']}, "
            f"observed shape={mask_rotate_array.shape}"
        )

    counts = {
        "graph_edge_mask": int(graph_edge_mask.sum().item()),
        "graph_mask_rotate_rows": int(mask_rotate_array.shape[0]),
        "bank_rotatable_bond_mask": int(entry_edge_mask.sum().item()),
        "bank_torsion_noise": int(entry["torsion_noise"].numel()),
        "bank_true_torsion_score": int(entry["true_torsion_score"].numel()),
    }
    if len(set(counts.values())) != 1:
        raise RuntimeError(
            f"three-party torsion topology mismatch for {entry['complex_id']} "
            f"at noise={entry['noise_level']} repeat={entry['pose_repeat']}: {counts}"
        )
    return counts["bank_true_torsion_score"], counts


def batched_graph_torsion_counts(batch, expected_graphs: int) -> list[int]:
    """Recover each graph's torsion count from the batched bond topology."""
    edge_mask = batch["ligand"].edge_mask.reshape(-1).bool()
    candidates = [
        edge_type for edge_type in batch.edge_types
        if edge_type[0] == "ligand"
        and edge_type[2] == "ligand"
        and int(batch[edge_type].edge_index.shape[1]) == int(edge_mask.numel())
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "cannot identify a unique ligand bond edge store aligned to edge_mask: "
            f"candidates={candidates}, edge_mask_length={edge_mask.numel()}"
        )
    edge_index = batch[candidates[0]].edge_index
    selected_sources = batch["ligand"].batch[edge_index[0, edge_mask]]
    selected_targets = batch["ligand"].batch[edge_index[1, edge_mask]]
    if not torch.equal(selected_sources, selected_targets):
        raise RuntimeError("a rotatable ligand bond crosses graph boundaries after batching")
    return (
        torch.bincount(selected_sources, minlength=expected_graphs)
        .detach().cpu().tolist()
    )


def evaluate_prediction(model_args, graph, entry, tr_prediction, rot_prediction, tor_prediction):
    from utils.diffusion_utils import modify_conformer

    tr_target = entry["true_translation_score"]
    rot_target = entry["true_rotation_score"]
    tor_target = entry["true_torsion_score"]
    tr_cos = tensor_cosine(tr_prediction, tr_target)
    rot_cos = tensor_cosine(rot_prediction, rot_target)
    tor_cos = tensor_cosine(tor_prediction, tor_target)
    combined, components = equal_component_mean((tr_cos, rot_cos, tor_cos))
    tr_sign = float(torch.dot(tr_prediction.reshape(-1).cpu(), tr_target.reshape(-1).cpu()) > 0)

    row = {
        "translation_cosine": tr_cos,
        "translation_normalized_mse": normalized_mse(tr_prediction, tr_target),
        "translation_direction_sign_agreement": tr_sign,
        "rotation_cosine": rot_cos,
        "rotation_normalized_mse": normalized_mse(rot_prediction, rot_target),
        "torsion_cosine": tor_cos,
        "torsion_normalized_mse": normalized_mse(tor_prediction, tor_target),
        "combined_alignment": combined,
        "combined_alignment_components": components,
        "RMSD_before": float(entry["rmsd_before"]),
        "rmsd_source_method_before": entry["rmsd_source_method"],
        "direct_RMSD_before": direct_rmsd(graph, graph["ligand"].pos),
        "num_rotatable_bonds": int(tor_prediction.numel()),
        "valid": True,
        "failure_reason": "",
    }
    predictions = [tr_prediction, rot_prediction, tor_prediction]
    if not all(bool(torch.isfinite(value).all()) for value in predictions):
        raise FloatingPointError("model emitted a non-finite score")
    tr_update, rot_update, tor_update, _ = one_step_updates(
        model_args,
        entry,
        tr_prediction,
        rot_prediction,
        tor_prediction,
    )
    updated = copy.deepcopy(graph)
    if isinstance(updated["ligand"].mask_rotate, list):
        updated["ligand"].mask_rotate = updated["ligand"].mask_rotate[0]
    modify_conformer(
        updated,
        tr_update,
        rot_update.squeeze(0),
        tor_update,
    )
    if not bool(torch.isfinite(updated["ligand"].pos).all()):
        raise FloatingPointError("one-step integrator emitted non-finite coordinates")
    rmsd_after, rmsd_method_after = evaluator_rmsd(updated, updated["ligand"].pos)
    direct_after = direct_rmsd(updated, updated["ligand"].pos)
    row.update({
        "RMSD_after": rmsd_after,
        "delta_RMSD_1step": float(entry["rmsd_before"]) - rmsd_after,
        "direct_RMSD_after": direct_after,
        "direct_delta_RMSD_1step": row["direct_RMSD_before"] - direct_after,
        "rmsd_source_method_after": rmsd_method_after,
    })
    raw = {
        "translation_prediction": tr_prediction.detach().cpu().clone(),
        "rotation_prediction": rot_prediction.detach().cpu().clone(),
        "torsion_prediction": tor_prediction.detach().cpu().clone(),
        "translation_update": tr_update.detach().cpu().clone(),
        "rotation_update": rot_update.detach().cpu().clone(),
        "torsion_update": None if tor_update is None else torch.from_numpy(tor_update.copy()),
        "updated_ligand_coordinates": updated["ligand"].pos.detach().cpu().clone(),
    }
    return row, raw


def condition_specs(epoch: int):
    specs = [
        {"condition": "true_chemistry", "gate_alpha": alpha, "shuffle_seed": None}
        for alpha in MAIN_GATE_ALPHAS
    ]
    if epoch in DOSE_EPOCHS:
        for alpha in DOSE_GATE_ALPHAS:
            if alpha not in MAIN_GATE_ALPHAS:
                specs.append({"condition": "true_chemistry", "gate_alpha": alpha, "shuffle_seed": None})
    specs.extend(
        {"condition": "joint_shuffled_chemistry", "gate_alpha": 1.0, "shuffle_seed": seed}
        for seed in SHUFFLE_SEEDS
    )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--epoch", type=int, choices=EPOCHS, required=True)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.analysis_root.resolve()
    output_dir = root / "05_mechanism_raw" / arm_slug(args.arm) / f"seed{args.seed}" / f"epoch_{args.epoch:04d}"
    completion_path = output_dir / "MECHANISM_COMPLETE.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite an existing mechanism attempt: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    device = torch.device(args.device)
    bank_path = root / "04_mechanism_pose_bank/mechanism_pose_bank_v1.pt"
    bank_sha = sha256_file(bank_path)
    bank = torch.load(bank_path, map_location="cpu")
    permutation_path = root / "04_mechanism_pose_bank/shuffle_permutation_manifest.json"
    permutation_manifest = json.loads(permutation_path.read_text())
    model, model_args, ckpt, state_sha = load_score_model(
        root,
        args.arm,
        args.seed,
        args.epoch,
        device,
    )
    if not hasattr(model, "analysis_surface_gate_alpha"):
        raise RuntimeError("analysis overlay was not imported; gated model attribute is absent")
    integrator_integrity = verify_zero_z_integrator_equivalence(model_args, bank["entries"])
    dataset = load_validation_dataset(model_args)
    clean_graphs = [dataset[index] for index in range(len(dataset))]
    entries_by_noise = defaultdict(list)
    for entry in bank["entries"]:
        entries_by_noise[float(entry["noise_level"])].append(entry)

    rows = []
    raw_rows = []
    input_hashes_by_entry = {}
    failures = []
    torsion_topology_checks = 0
    from utils.diffusion_utils import set_time

    for condition in condition_specs(args.epoch):
        alpha = float(condition["gate_alpha"])
        shuffle_seed = condition["shuffle_seed"]
        model.analysis_surface_gate_alpha = alpha
        for noise_level in sorted(entries_by_noise):
            for entry_chunk in chunks(entries_by_noise[noise_level], args.batch_size):
                graphs = []
                graph_torsion_counts = []
                for entry in entry_chunk:
                    graph = copy.deepcopy(clean_graphs[int(entry["dataset_index"])])
                    if complex_name(graph) != entry["complex_id"]:
                        raise RuntimeError("validation identity drift while materializing pose bank")
                    graph["ligand"].pos = entry["initial_ligand_coordinates"].clone()
                    current_hash = stable_tensor_sha256(graph["ligand"].pos)
                    previous_hash = input_hashes_by_entry.setdefault(entry["entry_sha256"], current_hash)
                    if previous_hash != current_hash:
                        raise RuntimeError("initial noisy coordinates changed across intervention conditions")
                    torsion_count, _ = assert_graph_torsion_topology(graph, entry)
                    graph_torsion_counts.append(torsion_count)
                    torsion_topology_checks += 1
                    if shuffle_seed is not None:
                        record = permutation_manifest["permutations"][entry["complex_id"]][str(shuffle_seed)]
                        permutation = torch.tensor(record["permutation"], dtype=torch.long)
                        apply_joint_chemistry_shuffle(graph, permutation)
                    graphs.append(graph)

                batch = Batch.from_data_list(graphs).to(device)
                batched_counts = batched_graph_torsion_counts(batch, len(graphs))
                if batched_counts != graph_torsion_counts:
                    raise RuntimeError(
                        "per-graph torsion topology changed during batching: "
                        f"individual={graph_torsion_counts}, batched={batched_counts}"
                    )
                set_time(
                    batch,
                    noise_level,
                    noise_level,
                    noise_level,
                    len(graphs),
                    model_args.all_atoms,
                    device,
                )
                with torch.no_grad():
                    tr_scores, rot_scores, tor_scores = model(batch)
                tr_scores = tr_scores.detach().cpu()
                rot_scores = rot_scores.detach().cpu()
                tor_scores = tor_scores.detach().cpu()
                torsion_offset = 0
                for index, (entry, graph) in enumerate(zip(entry_chunk, graphs)):
                    torsion_count = graph_torsion_counts[index]
                    tor_prediction = tor_scores[torsion_offset:torsion_offset + torsion_count]
                    if int(tor_prediction.numel()) != torsion_count:
                        raise RuntimeError(
                            f"per-graph torsion output slice mismatch for {entry['complex_id']}: "
                            f"expected={torsion_count}, observed={tor_prediction.numel()}"
                        )
                    torsion_offset += torsion_count
                    base = {
                        "arm": args.arm,
                        "training_seed": args.seed,
                        "epoch": args.epoch,
                        "complex_id": entry["complex_id"],
                        "noise_level": noise_level,
                        "pose_repeat": entry["pose_repeat"],
                        "condition": condition["condition"],
                        "gate_alpha": alpha,
                        "shuffle_seed": "" if shuffle_seed is None else shuffle_seed,
                        "entry_sha256": entry["entry_sha256"],
                        "initial_coordinates_sha256": input_hashes_by_entry[entry["entry_sha256"]],
                    }
                    try:
                        metrics, raw = evaluate_prediction(
                            model_args,
                            graph,
                            entry,
                            tr_scores[index:index + 1],
                            rot_scores[index:index + 1],
                            tor_prediction,
                        )
                        row = {**base, **metrics}
                        raw_rows.append({**base, **raw})
                    except Exception as error:
                        reason = f"{type(error).__name__}: {error}"
                        failures.append({**base, "failure_reason": reason})
                        row = {
                            **base,
                            "translation_cosine": math.nan,
                            "translation_normalized_mse": math.nan,
                            "translation_direction_sign_agreement": math.nan,
                            "rotation_cosine": math.nan,
                            "rotation_normalized_mse": math.nan,
                            "torsion_cosine": math.nan,
                            "torsion_normalized_mse": math.nan,
                            "combined_alignment": math.nan,
                            "combined_alignment_components": 0,
                            "RMSD_before": float(entry["rmsd_before"]),
                            "RMSD_after": math.nan,
                            "delta_RMSD_1step": math.nan,
                            "direct_RMSD_before": direct_rmsd(graph, graph["ligand"].pos),
                            "direct_RMSD_after": math.nan,
                            "direct_delta_RMSD_1step": math.nan,
                            "rmsd_source_method_before": entry["rmsd_source_method"],
                            "rmsd_source_method_after": "failed",
                            "num_rotatable_bonds": torsion_count,
                            "valid": False,
                            "failure_reason": reason,
                        }
                    rows.append(row)
                if torsion_offset != int(tor_scores.numel()):
                    raise RuntimeError(
                        f"torsion output cardinality mismatch: consumed={torsion_offset}, observed={tor_scores.numel()}"
                    )

    expected_conditions = len(condition_specs(args.epoch))
    expected_rows = len(bank["entries"]) * expected_conditions
    if len(rows) != expected_rows:
        raise RuntimeError(f"mechanism row count mismatch: {len(rows)} != {expected_rows}")
    if failures:
        atomic_json(output_dir / "FAILURES.json", {
            "count": len(failures),
            "records": failures,
        })
        raise RuntimeError(f"{len(failures)} fixed-pose mechanism evaluations failed")

    csv_path = output_dir / "mechanism_denoising_long.csv"
    with tempfile.NamedTemporaryFile("w", newline="", dir=output_dir, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        temporary_csv = Path(handle.name)
    os.replace(temporary_csv, csv_path)
    raw_path = output_dir / "mechanism_scores_raw.pt"
    atomic_torch_save(raw_path, {
        "schema_version": "surfna-lb200-mechanism-scores-raw-v1",
        "arm": args.arm,
        "training_seed": args.seed,
        "epoch": args.epoch,
        "checkpoint": str(ckpt),
        "checkpoint_file_sha256": sha256_file(ckpt),
        "checkpoint_state_sha256": state_sha,
        "pose_bank_sha256": bank_sha,
        "integrator": {
            "inference_steps": INFERENCE_STEPS,
            "step_rule": "canonical_sde_drift_with_stochastic_z_fixed_to_zero",
            "dt_by_noise": {
                str(record["noise_level"]): record["dt"]
                for record in integrator_integrity["records"]
            },
            "frozen_sampling_equivalence": integrator_integrity,
        },
        "torsion_topology_integrity": {
            "passed": True,
            "checks": torsion_topology_checks,
            "parties": [
                "individual_graph_edge_mask_and_mask_rotate",
                "pose_bank_rotatable_mask_noise_and_target",
                "batched_ligand_bond_topology_and_model_output_partition",
            ],
        },
        "records": raw_rows,
    })
    completion = {
        "schema_version": "surfna-lb200-mechanism-completion-v1",
        "arm": args.arm,
        "training_seed": args.seed,
        "epoch": args.epoch,
        "checkpoint": str(ckpt),
        "checkpoint_file_sha256": sha256_file(ckpt),
        "checkpoint_state_sha256": state_sha,
        "pose_bank_sha256": bank_sha,
        "zero_z_integrator_equivalence": integrator_integrity,
        "torsion_topology_checks": torsion_topology_checks,
        "conditions": condition_specs(args.epoch),
        "row_count": len(rows),
        "valid_row_count": sum(bool(row["valid"]) for row in rows),
        "failure_count": 0,
        "csv": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "raw_scores": str(raw_path),
        "raw_scores_sha256": sha256_file(raw_path),
        "wall_seconds": time.time() - started,
    }
    atomic_json(completion_path, completion)
    print(json.dumps(completion, sort_keys=True))


if __name__ == "__main__":
    main()
