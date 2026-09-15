#!/usr/bin/env python3
"""Deterministically rebuild one SurfNA mechanism CSV from frozen v5 raw tensors.

This is a numerical-QC repair, not a new model evaluation.  It preserves the
canonical float32 model predictions, one-step updates, stored updated
coordinates, and the immutable legacy pose-bank RMSD_before.  Alignment,
after-RMSD, direct-RMSD, and delta scalars are rebuilt with float64 arithmetic;
delta_RMSD therefore has a deliberately mixed-provenance legacy baseline.  The
parent v5 artifacts are read-only and hash-locked by a frozen v6 manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mechanism_common import (
    ARMS,
    DOSE_EPOCHS,
    DOSE_GATE_ALPHAS,
    EPOCHS,
    MAIN_GATE_ALPHAS,
    SHUFFLE_SEEDS,
    arm_slug,
    atomic_json,
    equal_component_mean,
    evaluator_rmsd,
    heavy_atom_mask,
    load_model_args,
    load_validation_dataset,
    model_dir,
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

STABILITY_THRESHOLDS_ANGSTROM = (100.0, 1_000.0, 1_000_000.0, 1_000_000_000_000.0)
REPLAY_TOLERANCE = 1e-6


def expected_conditions(epoch: int) -> set[tuple[str, float, int | None]]:
    alphas = list(MAIN_GATE_ALPHAS)
    if epoch in DOSE_EPOCHS:
        alphas = list(DOSE_GATE_ALPHAS)
    result = {("true_chemistry", float(alpha), None) for alpha in alphas}
    result.update(
        ("joint_shuffled_chemistry", 1.0, int(seed)) for seed in SHUFFLE_SEEDS
    )
    return result


def normalize_shuffle(value: Any) -> int | None:
    text = str(value).strip()
    return None if not text or text.lower() == "none" else int(text)


def direct_rmsd_float64(
    graph: Any, coordinates: torch.Tensor | np.ndarray, *,
    prepared_mask: np.ndarray | None = None, prepared_native: np.ndarray | None = None,
) -> float:
    coords = (
        coordinates.detach().cpu().numpy()
        if torch.is_tensor(coordinates)
        else np.asarray(coordinates)
    )
    coords = np.asarray(coords, dtype=np.float64)
    native = np.asarray(
        native_ligand_coordinates(graph) if prepared_native is None else prepared_native,
        dtype=np.float64,
    )
    mask = heavy_atom_mask(graph) if prepared_mask is None else prepared_mask
    value = float(np.sqrt(np.mean(np.sum((coords[mask] - native[mask]) ** 2, axis=1))))
    if not math.isfinite(value) or value < 0:
        raise FloatingPointError(f"non-finite direct RMSD: {value!r}")
    return value


def load_manifest_unit(
    manifest_path: Path, arm: str, seed: int, epoch: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "surfna-lb200-mechanism-recompute-input-manifest-v2":
        raise RuntimeError(f"unexpected input manifest schema: {manifest.get('schema_version')!r}")
    if manifest.get("derivation_policy", {}).get("updated_coordinates") != "reuse_parent_float32_exactly":
        raise RuntimeError("input manifest does not freeze reuse of parent updated coordinates")
    matches = [
        unit for unit in manifest.get("units", [])
        if unit.get("arm") == arm
        and int(unit.get("training_seed", -1)) == seed
        and int(unit.get("epoch", -1)) == epoch
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one manifest unit for {arm}/seed{seed}/epoch{epoch}, found {len(matches)}")
    return manifest, matches[0]


def verify_frozen_validation_cache(
    manifest: dict[str, Any], unit: dict[str, Any], model_args: Any
) -> dict[str, Any]:
    contract_path = Path(manifest["validation_cache_contract_path"])
    if (
        not contract_path.is_file()
        or sha256_file(contract_path) != manifest["validation_cache_contract_sha256"]
    ):
        raise RuntimeError("frozen validation cache contract is missing or hash-drifted")
    contract = json.loads(contract_path.read_text())
    if contract != manifest.get("validation_cache_contract"):
        raise RuntimeError("embedded and file validation cache contracts disagree")
    model_parameters = Path(unit["model_parameters"])
    if (
        not model_parameters.is_file()
        or sha256_file(model_parameters) != unit["model_parameters_sha256"]
    ):
        raise RuntimeError("model_parameters.yml is missing or hash-drifted")
    if model_parameters.resolve() != (model_dir(unit["arm"], int(unit["training_seed"])) / "model_parameters.yml").resolve():
        raise RuntimeError("loaded model-parameter path differs from the frozen source unit")
    if Path(model_args.cache_path).resolve() != Path(contract["cache_root"]).resolve():
        raise RuntimeError("model cache root differs from the frozen validation contract")
    if Path(model_args.split_val).resolve() != Path(contract["split_path"]).resolve():
        raise RuntimeError("model validation split differs from the frozen validation contract")
    for path_key, hash_key in (
        ("split_path", "split_sha256"),
        ("heterographs_path", "heterographs_sha256"),
        ("rdkit_ligands_path", "rdkit_ligands_sha256"),
    ):
        artifact = Path(contract[path_key])
        if not artifact.is_file() or sha256_file(artifact) != contract[hash_key]:
            raise RuntimeError(f"frozen validation cache artifact drift: {artifact}")
    return contract


def verify_parent_unit(unit: dict[str, Any], arm: str, seed: int, epoch: int) -> tuple[dict, dict, Path]:
    completion_path = Path(unit["source_completion"])
    csv_path = Path(unit["source_csv"])
    raw_path = Path(unit["source_raw"])
    for path, key in (
        (completion_path, "source_completion_sha256"),
        (csv_path, "source_csv_sha256"),
        (raw_path, "source_raw_sha256"),
    ):
        if not path.is_file() or sha256_file(path) != unit[key]:
            raise RuntimeError(f"parent artifact missing or hash-drifted: {path}")
    completion = json.loads(completion_path.read_text())
    if (
        completion.get("schema_version") != "surfna-lb200-mechanism-completion-v1"
        or completion.get("arm") != arm
        or int(completion.get("training_seed", -1)) != seed
        or int(completion.get("epoch", -1)) != epoch
        or int(completion.get("failure_count", -1)) != 0
        or completion.get("csv_sha256") != unit["source_csv_sha256"]
        or completion.get("raw_scores_sha256") != unit["source_raw_sha256"]
    ):
        raise RuntimeError(f"parent completion contract mismatch: {completion_path}")
    raw = torch.load(raw_path, map_location="cpu")
    if (
        raw.get("schema_version") != "surfna-lb200-mechanism-scores-raw-v1"
        or raw.get("arm") != arm
        or int(raw.get("training_seed", -1)) != seed
        or int(raw.get("epoch", -1)) != epoch
        or raw.get("pose_bank_sha256") != unit["pose_bank_sha256"]
    ):
        raise RuntimeError(f"parent raw tensor contract mismatch: {raw_path}")
    return completion, raw, raw_path


def replay_coordinates(graph: Any, record: dict[str, Any]) -> tuple[torch.Tensor, float, bool]:
    from utils.diffusion_utils import modify_conformer

    if isinstance(graph["ligand"].mask_rotate, list):
        graph["ligand"].mask_rotate = graph["ligand"].mask_rotate[0]
    torsion = record["torsion_update"]
    torsion_numpy = None if torsion is None else torsion.detach().cpu().numpy()
    initial = graph["ligand"].pos
    try:
        modify_conformer(
            graph,
            record["translation_update"].detach().cpu(),
            record["rotation_update"].detach().cpu().squeeze(0),
            torsion_numpy,
        )
        observed = graph["ligand"].pos.detach().cpu().clone()
    finally:
        graph["ligand"].pos = initial
    stored = record["updated_ligand_coordinates"].detach().cpu()
    if observed.shape != stored.shape:
        raise RuntimeError(f"coordinate replay shape mismatch: {tuple(observed.shape)} != {tuple(stored.shape)}")
    difference = (observed.double() - stored.double()).abs()
    maximum = float(difference.max()) if difference.numel() else 0.0
    exact = bool(torch.equal(observed, stored))
    if not math.isfinite(maximum) or maximum > REPLAY_TOLERANCE:
        raise RuntimeError(f"coordinate replay mismatch: max_abs_diff={maximum}")
    return stored, maximum, exact


def require_tensor_finite(record: dict[str, Any], field: str, *, allow_none: bool = False) -> None:
    value = record.get(field)
    if value is None and allow_none:
        return
    if not torch.is_tensor(value):
        raise TypeError(f"{field} is not a tensor")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"parent raw tensor contains non-finite {field}")


def recompute(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = args.output_root.resolve()
    source_root = args.source_root.resolve()
    if output_root == source_root:
        raise RuntimeError("output root must be independent from the parent v5 root")
    output_dir = (
        output_root / "05_mechanism_raw" / arm_slug(args.arm)
        / f"seed{args.seed}" / f"epoch_{args.epoch:04d}"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite a recompute attempt: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "00_protocol/MECHANISM_RECOMPUTE_INPUT_MANIFEST.json"
    manifest, unit = load_manifest_unit(manifest_path, args.arm, args.seed, args.epoch)
    if Path(manifest["source_root"]).resolve() != source_root or Path(manifest["output_root"]).resolve() != output_root:
        raise RuntimeError("runtime roots differ from the frozen v6 input manifest")
    frozen_code = manifest.get("code", {})
    for name, record in frozen_code.items():
        path = Path(record.get("path", ""))
        expected = record.get("sha256")
        if not expected or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"frozen v6/production code digest mismatch: {name}/{path}")
    if Path(frozen_code.get("recompute", {}).get("path", "")).resolve() != Path(__file__).resolve():
        raise RuntimeError("running recompute script is not the frozen v6 script")
    completion, raw, raw_path = verify_parent_unit(unit, args.arm, args.seed, args.epoch)

    bank_path = Path(manifest["pose_bank_path"])
    if not bank_path.is_file() or sha256_file(bank_path) != manifest["pose_bank_sha256"]:
        raise RuntimeError("frozen pose bank missing or hash-drifted")
    bank = torch.load(bank_path, map_location="cpu")
    entries = bank.get("entries", [])
    entry_by_sha = {str(entry["entry_sha256"]): entry for entry in entries}
    if len(entries) != 1780 or len(entry_by_sha) != 1780:
        raise RuntimeError(f"pose-bank cardinality mismatch: {len(entries)}/{len(entry_by_sha)}")

    model_args = load_model_args(args.arm, args.seed)
    cache_contract = verify_frozen_validation_cache(manifest, unit, model_args)
    dataset = load_validation_dataset(model_args)
    if Path(dataset.full_cache_path).resolve() != Path(cache_contract["full_cache_path"]).resolve():
        raise RuntimeError("loaded PDBBind cache path differs from the frozen validation contract")
    clean_graphs = [dataset[index] for index in range(len(dataset))]
    graph_names = []
    for graph in clean_graphs:
        name = getattr(graph, "name", None)
        if isinstance(name, (list, tuple)):
            name = name[0]
        graph_names.append(str(name))
    if len(graph_names) != 89 or len(set(graph_names)) != 89:
        raise RuntimeError("validation graph identity/cardinality mismatch")
    from utils.utils import remove_all_hs
    rmsd_prepared = {}
    for dataset_index, graph in enumerate(clean_graphs):
        prepared_mask = heavy_atom_mask(graph)
        prepared_native = np.asarray(native_ligand_coordinates(graph), dtype=np.float64)
        try:
            prepared_mol = remove_all_hs(graph.mol[0])
        except Exception:
            prepared_mol = None
        rmsd_prepared[dataset_index] = (prepared_mask, prepared_native, prepared_mol)

    specs = expected_conditions(args.epoch)
    expected_rows = len(entries) * len(specs)
    records = raw.get("records", [])
    if len(records) != expected_rows or int(unit["expected_row_count"]) != expected_rows:
        raise RuntimeError(f"raw record count mismatch: {len(records)} != {expected_rows}")

    initial_hashes = {
        sha: stable_tensor_sha256(entry["initial_ligand_coordinates"])
        for sha, entry in entry_by_sha.items()
    }
    before_direct: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    observed_specs: dict[str, set[tuple[str, float, int | None]]] = defaultdict(set)
    allowable_nan_counts: Counter[str] = Counter()
    rmsd_methods: Counter[str] = Counter()
    divergence_counts = {str(int(threshold)): 0 for threshold in STABILITY_THRESHOLDS_ANGSTROM}
    replay_max_abs_diff = 0.0
    replay_exact_count = 0
    max_rmsd_after = 0.0
    max_direct_rmsd_after = 0.0

    for record_index, record in enumerate(records):
        if (
            record.get("arm") != args.arm
            or int(record.get("training_seed", -1)) != args.seed
            or int(record.get("epoch", -1)) != args.epoch
        ):
            raise RuntimeError(f"raw identity drift at record {record_index}")
        entry_sha = str(record.get("entry_sha256", ""))
        entry = entry_by_sha.get(entry_sha)
        if entry is None:
            raise RuntimeError(f"unknown pose-bank entry at record {record_index}: {entry_sha}")
        condition = str(record["condition"])
        gate_alpha = float(record["gate_alpha"])
        shuffle_seed = normalize_shuffle(record.get("shuffle_seed", ""))
        intervention = (condition, gate_alpha, shuffle_seed)
        if intervention not in specs:
            raise RuntimeError(f"unexpected intervention at record {record_index}: {intervention}")
        unique = (entry_sha, *intervention)
        if unique in seen:
            raise RuntimeError(f"duplicate raw key at record {record_index}: {unique}")
        seen.add(unique)
        observed_specs[entry_sha].add(intervention)

        if str(record.get("complex_id")) != str(entry["complex_id"]):
            raise RuntimeError(f"complex identity drift at record {record_index}")
        if float(record.get("noise_level")) != float(entry["noise_level"]):
            raise RuntimeError(f"noise identity drift at record {record_index}")
        if int(record.get("pose_repeat", -1)) != int(entry["pose_repeat"]):
            raise RuntimeError(f"pose-repeat identity drift at record {record_index}")
        if str(record.get("initial_coordinates_sha256")) != initial_hashes[entry_sha]:
            raise RuntimeError(f"initial-coordinate hash drift at record {record_index}")

        for field in (
            "translation_prediction", "rotation_prediction", "torsion_prediction",
            "translation_update", "rotation_update", "updated_ligand_coordinates",
        ):
            require_tensor_finite(record, field)
        require_tensor_finite(record, "torsion_update", allow_none=True)

        dataset_index = int(entry["dataset_index"])
        graph = clean_graphs[dataset_index]
        if graph_names[dataset_index] != str(entry["complex_id"]):
            raise RuntimeError(f"dataset index drift at record {record_index}")
        graph["ligand"].pos = entry["initial_ligand_coordinates"].clone()
        stored_updated, replay_diff, replay_exact = replay_coordinates(graph, record)
        replay_max_abs_diff = max(replay_max_abs_diff, replay_diff)
        replay_exact_count += int(replay_exact)

        tr_prediction = record["translation_prediction"]
        rot_prediction = record["rotation_prediction"]
        tor_prediction = record["torsion_prediction"]
        tr_target = entry["true_translation_score"]
        rot_target = entry["true_rotation_score"]
        tor_target = entry["true_torsion_score"]
        if int(tor_prediction.numel()) != int(tor_target.numel()):
            raise RuntimeError(f"torsion cardinality drift at record {record_index}")

        tr_cos = tensor_cosine(tr_prediction, tr_target)
        rot_cos = tensor_cosine(rot_prediction, rot_target)
        tor_cos = tensor_cosine(tor_prediction, tor_target)
        tr_nmse = normalized_mse(tr_prediction, tr_target)
        rot_nmse = normalized_mse(rot_prediction, rot_target)
        tor_nmse = normalized_mse(tor_prediction, tor_target)
        combined, components = equal_component_mean((tr_cos, rot_cos, tor_cos))
        tr_dot = float(torch.dot(tr_prediction.detach().double().reshape(-1), tr_target.detach().double().reshape(-1)))
        tr_sign = float(tr_dot > 0)

        if not all(math.isfinite(value) for value in (tr_nmse, rot_nmse, combined, tr_sign)):
            raise FloatingPointError(f"disallowed non-finite alignment metric at record {record_index}")
        for field, value, prediction, target in (
            ("translation_cosine", tr_cos, tr_prediction, tr_target),
            ("rotation_cosine", rot_cos, rot_prediction, rot_target),
            ("torsion_cosine", tor_cos, tor_prediction, tor_target),
            ("torsion_normalized_mse", tor_nmse, tor_prediction, tor_target),
        ):
            if math.isfinite(value):
                continue
            prediction_d = prediction.detach().double().reshape(-1)
            target_d = target.detach().double().reshape(-1)
            empty = prediction_d.numel() == 0 or target_d.numel() == 0
            zero_norm = (
                not empty
                and (float(torch.linalg.vector_norm(prediction_d)) * float(torch.linalg.vector_norm(target_d))) <= 1e-8
            )
            if field == "torsion_normalized_mse" and not empty:
                raise FloatingPointError(f"non-empty torsion NMSE is non-finite at record {record_index}")
            if field.endswith("_cosine") and not (empty or zero_norm):
                raise FloatingPointError(f"unexplained non-finite cosine at record {record_index}: {field}")
            allowable_nan_counts[field] += 1
        if not math.isfinite(combined) or components < 1 or components > 3:
            raise FloatingPointError(f"invalid combined alignment at record {record_index}")

        prepared_mask, prepared_native, prepared_mol = rmsd_prepared[dataset_index]
        if entry_sha not in before_direct:
            before_direct[entry_sha] = direct_rmsd_float64(
                graph, graph["ligand"].pos,
                prepared_mask=prepared_mask, prepared_native=prepared_native,
            )
        rmsd_before = float(entry["rmsd_before"])
        rmsd_after, rmsd_method = evaluator_rmsd(
            graph, stored_updated, prepared_mask=prepared_mask,
            prepared_native=prepared_native, prepared_mol=prepared_mol,
        )
        direct_after = direct_rmsd_float64(
            graph, stored_updated,
            prepared_mask=prepared_mask, prepared_native=prepared_native,
        )
        delta = rmsd_before - rmsd_after
        direct_delta = before_direct[entry_sha] - direct_after
        required_derived = (
            rmsd_before, rmsd_after, delta, before_direct[entry_sha], direct_after, direct_delta,
        )
        if not all(math.isfinite(value) for value in required_derived):
            raise FloatingPointError(f"non-finite float64 RMSD metric at record {record_index}")
        if rmsd_before < 0 or rmsd_after < 0 or before_direct[entry_sha] < 0 or direct_after < 0:
            raise FloatingPointError(f"negative RMSD metric at record {record_index}")
        rmsd_methods[rmsd_method] += 1
        max_rmsd_after = max(max_rmsd_after, rmsd_after)
        max_direct_rmsd_after = max(max_direct_rmsd_after, direct_after)
        for threshold in STABILITY_THRESHOLDS_ANGSTROM:
            divergence_counts[str(int(threshold))] += int(rmsd_after > threshold)

        rows.append({
            "arm": args.arm,
            "training_seed": args.seed,
            "epoch": args.epoch,
            "complex_id": entry["complex_id"],
            "noise_level": float(entry["noise_level"]),
            "pose_repeat": int(entry["pose_repeat"]),
            "condition": condition,
            "gate_alpha": gate_alpha,
            "shuffle_seed": "" if shuffle_seed is None else shuffle_seed,
            "entry_sha256": entry_sha,
            "initial_coordinates_sha256": initial_hashes[entry_sha],
            "translation_cosine": tr_cos,
            "translation_normalized_mse": tr_nmse,
            "translation_direction_sign_agreement": tr_sign,
            "rotation_cosine": rot_cos,
            "rotation_normalized_mse": rot_nmse,
            "torsion_cosine": tor_cos,
            "torsion_normalized_mse": tor_nmse,
            "combined_alignment": combined,
            "combined_alignment_components": components,
            "RMSD_before": rmsd_before,
            "RMSD_after": rmsd_after,
            "delta_RMSD_1step": delta,
            "direct_RMSD_before": before_direct[entry_sha],
            "direct_RMSD_after": direct_after,
            "direct_delta_RMSD_1step": direct_delta,
            "rmsd_source_method_before": entry["rmsd_source_method"],
            "rmsd_source_method_after": rmsd_method,
            "num_rotatable_bonds": int(tor_prediction.numel()),
            "valid": True,
            "failure_reason": "",
        })

    if len(seen) != expected_rows or set(observed_specs) != set(entry_by_sha):
        raise RuntimeError("raw key-space cardinality mismatch after recompute")
    if any(observed_specs[entry_sha] != specs for entry_sha in entry_by_sha):
        raise RuntimeError("incomplete intervention cross-product after recompute")
    if len(rows) != expected_rows:
        raise RuntimeError(f"recomputed row count mismatch: {len(rows)} != {expected_rows}")

    csv_path = output_dir / "mechanism_denoising_long.csv"
    with tempfile.NamedTemporaryFile("w", newline="", dir=output_dir, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, csv_path)

    completion_path = output_dir / "MECHANISM_COMPLETE.json"
    completion_v2 = {
        "schema_version": "surfna-lb200-mechanism-recompute-completion-v2",
        "status": "complete",
        "arm": args.arm,
        "training_seed": args.seed,
        "epoch": args.epoch,
        "derivation_precision": "float64-derived-metrics-with-legacy-pose-bank-RMSD_before",
        "scientific_protocol_change": False,
        "reinference_performed": False,
        "scientific_output_reintegration_performed": False,
        "coordinate_integrity_replay_performed": True,
        "replayed_coordinates_used_as_metrics_input": False,
        "stored_parent_coordinates_used_as_metrics_input": True,
        "RMSD_before_policy": "reuse_immutable_legacy_float32_pose_bank_metric",
        "delta_RMSD_1step_policy": "legacy_RMSD_before_minus_float64_RMSD_after",
        "direct_RMSD_before_policy": "recompute_from_pose_bank_coordinates_in_float64",
        "parent_root": str(source_root),
        "source_completion": str(unit["source_completion"]),
        "source_completion_sha256": unit["source_completion_sha256"],
        "source_csv": str(unit["source_csv"]),
        "source_csv_sha256": unit["source_csv_sha256"],
        "source_raw_scores": str(raw_path),
        "source_raw_scores_sha256": unit["source_raw_sha256"],
        "pose_bank": str(bank_path),
        "pose_bank_sha256": manifest["pose_bank_sha256"],
        "validation_cache_contract_sha256": manifest["validation_cache_contract_sha256"],
        "model_parameters_sha256": unit["model_parameters_sha256"],
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "recompute_code": str(Path(__file__).resolve()),
        "recompute_code_sha256": sha256_file(Path(__file__).resolve()),
        "conditions": [
            {"condition": condition, "gate_alpha": alpha, "shuffle_seed": shuffle}
            for condition, alpha, shuffle in sorted(specs, key=str)
        ],
        "row_count": len(rows),
        "valid_row_count": len(rows),
        "failure_count": 0,
        "derived_disallowed_nonfinite_count": 0,
        "allowable_nan_counts": dict(allowable_nan_counts),
        "raw_tensor_finiteness_passed": True,
        "key_space_complete": True,
        "coordinate_replay": {
            "passed": True,
            "tolerance": REPLAY_TOLERANCE,
            "max_abs_difference": replay_max_abs_diff,
            "exact_match_count": replay_exact_count,
            "checked_count": len(rows),
        },
        "numerical_stability": {
            "endpoint_status": "post_failure_qc_descriptive",
            "thresholds_angstrom": list(STABILITY_THRESHOLDS_ANGSTROM),
            "divergence_counts": divergence_counts,
            "max_RMSD_after": max_rmsd_after,
            "max_direct_RMSD_after": max_direct_rmsd_after,
            "rmsd_source_methods": dict(rmsd_methods),
        },
        "csv": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "wall_seconds": time.time() - started,
    }
    atomic_json(completion_path, completion_v2)
    return {**completion_v2, "completion": str(completion_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--epoch", type=int, choices=EPOCHS, required=True)
    args = parser.parse_args()
    result = recompute(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
