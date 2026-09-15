#!/usr/bin/env python3
"""Run the preregistered one-complex smoke gate before any full matrix launch."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path

import torch
from torch_geometric.data import Batch

from evaluate_fixed_pose_mechanism import assert_graph_torsion_topology, evaluate_prediction
from mechanism_common import (
    CHEMISTRY_CHANNELS,
    NMSE_EPS,
    apply_joint_chemistry_shuffle,
    atomic_csv,
    atomic_json,
    complex_name,
    graph_invariant_hashes,
    load_model_args,
    load_score_model,
    load_validation_dataset,
    sha256_file,
    stable_tensor_sha256,
)


def predict(model, model_args, graph, t: float, alpha: float, device: torch.device):
    from utils.diffusion_utils import set_time

    model.analysis_surface_gate_alpha = float(alpha)
    batch = Batch.from_data_list([copy.deepcopy(graph)]).to(device)
    set_time(batch, t, t, t, 1, model_args.all_atoms, device)
    with torch.no_grad():
        outputs = model(batch)
    outputs = tuple(value.detach().cpu() for value in outputs)
    if not all(bool(torch.isfinite(value).all()) for value in outputs):
        raise FloatingPointError("non-finite model output in smoke test")
    return outputs


def nonchem_invariant_hashes(graph):
    hashes = graph_invariant_hashes(graph)
    hashes.pop("surface_x", None)
    hashes["surface_nonchem_x"] = stable_tensor_sha256(graph["surface"].x[:, [3, 7]])
    return hashes


def cosine_certificate(prediction: torch.Tensor, target: torch.Tensor, value: float) -> dict:
    prediction = prediction.detach().float().reshape(-1)
    target = target.detach().float().reshape(-1)
    prediction_norm = float(torch.linalg.vector_norm(prediction).cpu())
    target_norm = float(torch.linalg.vector_norm(target).cpu())
    structurally_undefined = target.numel() == 0 or target_norm <= NMSE_EPS
    denominator = prediction_norm * target_norm
    well_defined = (
        not structurally_undefined
        and prediction.numel() == target.numel()
        and denominator > NMSE_EPS
    )
    prediction_degenerate = (
        not structurally_undefined
        and (
            prediction.numel() != target.numel()
            or prediction.numel() == 0
            or denominator <= NMSE_EPS
        )
    )
    valid = math.isfinite(float(value)) if well_defined else math.isnan(float(value))
    return {
        "prediction_numel": int(prediction.numel()),
        "target_numel": int(target.numel()),
        "prediction_norm": prediction_norm,
        "target_norm": target_norm,
        "denominator": denominator,
        "nmse_epsilon": NMSE_EPS,
        "well_defined": well_defined,
        "structurally_undefined": structurally_undefined,
        "prediction_degenerate": prediction_degenerate,
        "value_matches_definition": valid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.analysis_root.resolve()
    output = root / "03_integrity"
    report_path = output / "full_one_complex_smoke_report.json"
    csv_path = output / "full_one_complex_smoke_metrics.csv"
    if report_path.exists() or csv_path.exists():
        raise SystemExit("refusing to overwrite an existing full smoke-test result")

    device = torch.device(args.device)
    bank = torch.load(root / "04_mechanism_pose_bank/mechanism_pose_bank_v1.pt", map_location="cpu")
    entry = next(
        item for item in bank["entries"]
        if float(item["noise_level"]) == 0.30 and item["true_torsion_score"].numel() > 0
    )
    permutations = json.loads(
        (root / "04_mechanism_pose_bank/shuffle_permutation_manifest.json").read_text()
    )

    reference_args = load_model_args("Transfer-All", 0)
    dataset = load_validation_dataset(reference_args)
    base_graph = dataset[int(entry["dataset_index"])]
    if complex_name(base_graph) != entry["complex_id"]:
        raise RuntimeError("smoke-test pose-bank identity mismatch")
    base_graph["ligand"].pos = entry["initial_ligand_coordinates"].clone()
    initial_coordinates_sha256 = stable_tensor_sha256(base_graph["ligand"].pos)
    torsion_count, torsion_topology_counts = assert_graph_torsion_topology(base_graph, entry)

    shuffled_graph = copy.deepcopy(base_graph)
    permutation = torch.tensor(
        permutations["permutations"][entry["complex_id"]]["20260911"]["permutation"],
        dtype=torch.long,
    )
    apply_joint_chemistry_shuffle(shuffled_graph, permutation)

    patch_graph = copy.deepcopy(base_graph)
    surface_pos = patch_graph["surface"].pos
    patch_size = min(max(8, int(round(surface_pos.shape[0] / 32))), int(surface_pos.shape[0]))
    distances = torch.linalg.vector_norm(surface_pos - surface_pos[0], dim=1)
    patch_indices = torch.argsort(distances)[:patch_size]
    chemistry = patch_graph["surface"].x[:, list(CHEMISTRY_CHANNELS)]
    patch_graph["surface"].x[patch_indices[:, None], torch.tensor(CHEMISTRY_CHANNELS)] = chemistry.mean(0)

    variants = {
        "true_alpha0": (base_graph, 0.0),
        "true_alpha05": (base_graph, 0.5),
        "true_alpha1": (base_graph, 1.0),
        "joint_shuffle_alpha1": (shuffled_graph, 1.0),
        "one_patch_mean_neutralized_alpha1": (patch_graph, 1.0),
    }
    invariant_before = nonchem_invariant_hashes(base_graph)
    invariant_shuffle = nonchem_invariant_hashes(shuffled_graph)
    invariant_patch = nonchem_invariant_hashes(patch_graph)

    rows = []
    checkpoints = []
    for epoch in (25, 200):
        model, model_args, checkpoint, state_sha = load_score_model(
            root, "Transfer-All", 0, epoch, device
        )
        checkpoints.append({
            "epoch": epoch,
            "checkpoint": str(checkpoint),
            "checkpoint_state_sha256": state_sha,
        })
        for condition, (graph, alpha) in variants.items():
            tr_score, rot_score, tor_score = predict(
                model, model_args, graph, float(entry["noise_level"]), alpha, device
            )
            metrics, _ = evaluate_prediction(
                model_args, graph, entry, tr_score, rot_score, tor_score
            )
            cosine_certificates = {
                "translation": cosine_certificate(
                    tr_score, entry["true_translation_score"], metrics["translation_cosine"]
                ),
                "rotation": cosine_certificate(
                    rot_score, entry["true_rotation_score"], metrics["rotation_cosine"]
                ),
                "torsion": cosine_certificate(
                    tor_score, entry["true_torsion_score"], metrics["torsion_cosine"]
                ),
            }
            numeric = [
                metrics["translation_normalized_mse"],
                metrics["rotation_normalized_mse"],
                metrics["torsion_normalized_mse"],
                metrics["combined_alignment"],
                metrics["RMSD_before"],
                metrics["RMSD_after"],
                metrics["delta_RMSD_1step"],
            ]
            if not all(math.isfinite(float(value)) for value in numeric):
                raise FloatingPointError(f"non-finite smoke metric: epoch={epoch} condition={condition}")
            rows.append({
                "arm": "Transfer-All",
                "training_seed": 0,
                "epoch": epoch,
                "complex_id": entry["complex_id"],
                "noise_level": entry["noise_level"],
                "pose_repeat": entry["pose_repeat"],
                "condition": condition,
                "gate_alpha": alpha,
                "translation_cosine": metrics["translation_cosine"],
                "translation_normalized_mse": metrics["translation_normalized_mse"],
                "translation_cosine_defined": cosine_certificates["translation"]["well_defined"],
                "translation_cosine_structurally_undefined": cosine_certificates["translation"]["structurally_undefined"],
                "translation_prediction_degenerate": cosine_certificates["translation"]["prediction_degenerate"],
                "translation_cosine_value_matches_definition": cosine_certificates["translation"]["value_matches_definition"],
                "rotation_cosine": metrics["rotation_cosine"],
                "rotation_normalized_mse": metrics["rotation_normalized_mse"],
                "rotation_cosine_defined": cosine_certificates["rotation"]["well_defined"],
                "rotation_cosine_structurally_undefined": cosine_certificates["rotation"]["structurally_undefined"],
                "rotation_prediction_degenerate": cosine_certificates["rotation"]["prediction_degenerate"],
                "rotation_cosine_value_matches_definition": cosine_certificates["rotation"]["value_matches_definition"],
                "torsion_cosine": metrics["torsion_cosine"],
                "torsion_normalized_mse": metrics["torsion_normalized_mse"],
                "torsion_cosine_defined": cosine_certificates["torsion"]["well_defined"],
                "torsion_cosine_structurally_undefined": cosine_certificates["torsion"]["structurally_undefined"],
                "torsion_prediction_degenerate": cosine_certificates["torsion"]["prediction_degenerate"],
                "torsion_cosine_value_matches_definition": cosine_certificates["torsion"]["value_matches_definition"],
                "combined_alignment": metrics["combined_alignment"],
                "combined_alignment_components": metrics["combined_alignment_components"],
                "RMSD_before": metrics["RMSD_before"],
                "RMSD_after": metrics["RMSD_after"],
                "delta_RMSD_1step": metrics["delta_RMSD_1step"],
                "initial_coordinates_sha256": initial_coordinates_sha256,
            })

    atomic_csv(csv_path, list(rows[0]), rows)
    with csv_path.open(newline="") as handle:
        reloaded = list(csv.DictReader(handle))
    metrics_csv_sha256 = sha256_file(csv_path)
    tests = {
        "ema025_loaded": any(item["epoch"] == 25 for item in checkpoints),
        "ema200_loaded": any(item["epoch"] == 200 for item in checkpoints),
        "normal_forward_finite": len(rows) == 10,
        "all_cosines_match_production_definition": all(
            row[f"{head}_cosine_value_matches_definition"]
            for row in rows
            for head in ("translation", "rotation", "torsion")
        ),
        "combined_component_counts_match_defined_cosines": all(
            int(row["combined_alignment_components"])
            == sum(bool(row[f"{head}_cosine_defined"]) for head in ("translation", "rotation", "torsion"))
            for row in rows
        ),
        "at_least_two_alignment_components_per_condition": all(
            int(row["combined_alignment_components"]) >= 2 for row in rows
        ),
        "ema025_has_no_degenerate_nonempty_score_head": all(
            not row[f"{head}_prediction_degenerate"]
            for row in rows if int(row["epoch"]) == 25
            for head in ("translation", "rotation", "torsion")
        ),
        "ema200_has_no_degenerate_nonempty_score_head": all(
            not row[f"{head}_prediction_degenerate"]
            for row in rows if int(row["epoch"]) == 200
            for head in ("translation", "rotation", "torsion")
        ),
        "gate_0_05_1_covered": {0.0, 0.5, 1.0}.issubset({float(row["gate_alpha"]) for row in rows}),
        "true_and_shuffle_covered": any("joint_shuffle" in row["condition"] for row in rows),
        "one_patch_occlusion_covered": any("one_patch" in row["condition"] for row in rows),
        "fixed_pose_hash_constant": len({row["initial_coordinates_sha256"] for row in rows}) == 1,
        "one_step_rmsd_finite": all(math.isfinite(float(row["delta_RMSD_1step"])) for row in rows),
        "csv_roundtrip": len(reloaded) == len(rows),
        "shuffle_preserves_nonchem_graph_inputs": invariant_before == invariant_shuffle,
        "patch_preserves_nonchem_graph_inputs": invariant_before == invariant_patch,
        "torsion_topology_three_party_agreement": (
            torsion_count > 0 and len(set(torsion_topology_counts.values())) == 1
        ),
    }
    report = {
        "schema_version": "surfna-lb200-one-complex-smoke-v2",
        "passed": all(tests.values()),
        "test_data_used": False,
        "complex_id": entry["complex_id"],
        "entry_sha256": entry["entry_sha256"],
        "patch_seed_vertex": 0,
        "patch_vertex_count": patch_size,
        "torsion_count": torsion_count,
        "torsion_topology_counts": torsion_topology_counts,
        "checkpoints": checkpoints,
        "tests": tests,
        "epochs_tested": sorted({int(item["epoch"]) for item in checkpoints}),
        "metrics_csv": str(csv_path),
        "metrics_csv_sha256": metrics_csv_sha256,
        "metrics_row_count": len(rows),
        "undefined_cosine_count": sum(
            not row[f"{head}_cosine_defined"]
            for row in rows
            for head in ("translation", "rotation", "torsion")
        ),
        "structurally_undefined_cosine_count": sum(
            row[f"{head}_cosine_structurally_undefined"]
            for row in rows
            for head in ("translation", "rotation", "torsion")
        ),
        "degenerate_prediction_count": sum(
            row[f"{head}_prediction_degenerate"]
            for row in rows
            for head in ("translation", "rotation", "torsion")
        ),
        "degenerate_prediction_count_epoch25": sum(
            row[f"{head}_prediction_degenerate"]
            for row in rows if int(row["epoch"]) == 25
            for head in ("translation", "rotation", "torsion")
        ),
        "degenerate_prediction_count_epoch200": sum(
            row[f"{head}_prediction_degenerate"]
            for row in rows if int(row["epoch"]) == 200
            for head in ("translation", "rotation", "torsion")
        ),
    }
    atomic_json(report_path, report)
    if not report["passed"]:
        raise RuntimeError(f"full one-complex smoke gate failed: {tests}")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
