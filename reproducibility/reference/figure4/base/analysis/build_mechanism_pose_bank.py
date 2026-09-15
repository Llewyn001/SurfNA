#!/usr/bin/env python3
"""Build the frozen 89 x 4 x 5 validation noisy-pose mechanism bank."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch

from mechanism_common import (
    CHEMISTRY_CHANNELS,
    EXPECTED_COUNTS,
    L2_RELEASE_ROOT,
    NOISE_LEVELS,
    POSE_BANK_SEED,
    POSE_REPEATS,
    SHUFFLE_SEEDS,
    apply_joint_chemistry_shuffle,
    arm_slug,
    atomic_csv,
    atomic_json,
    atomic_torch_save,
    complex_name,
    evaluator_rmsd,
    graph_invariant_hashes,
    load_model_args,
    load_validation_dataset,
    make_joint_shuffle_permutation,
    seed_everything,
    sha256_file,
    stable_tensor_sha256,
)


def entry_digest(entry: dict) -> str:
    digest = hashlib.sha256()
    for key in (
        "complex_id",
        "dataset_index",
        "noise_level",
        "pose_repeat",
        "pose_bank_seed",
        "tr_sigma",
        "rot_sigma",
        "tor_sigma",
        "rmsd_before",
        "rmsd_source_method",
    ):
        digest.update(f"{key}={entry[key]}".encode())
    for key in (
        "base_ligand_coordinates",
        "initial_ligand_coordinates",
        "translation_noise",
        "rotation_noise",
        "torsion_noise",
        "true_translation_score",
        "true_rotation_score",
        "true_torsion_score",
        "rotatable_bond_mask",
    ):
        digest.update(key.encode())
        digest.update(stable_tensor_sha256(entry[key]).encode())
    return digest.hexdigest()


def sorted_rows(value: torch.Tensor) -> np.ndarray:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    if array.shape[0] == 0:
        return array
    keys = tuple(array[:, index] for index in reversed(range(array.shape[1])))
    return array[np.lexsort(keys)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis_root.resolve()
    output = root / "04_mechanism_pose_bank"
    output.mkdir(parents=True, exist_ok=True)
    bank_path = output / "mechanism_pose_bank_v1.pt"
    if bank_path.exists():
        raise SystemExit(f"refusing to overwrite an existing pose bank: {bank_path}")

    seed_everything(POSE_BANK_SEED)
    model_args = load_model_args("Transfer-All", 0)
    dataset = load_validation_dataset(model_args)
    from datasets.pdbbind import NoiseTransform
    from utils import so3
    from utils.diffusion_utils import t_to_sigma as t_to_sigma_impl

    sigma_fn = partial(t_to_sigma_impl, args=model_args)
    transform = NoiseTransform(
        t_to_sigma=sigma_fn,
        no_torsion=model_args.no_torsion,
        all_atom=model_args.all_atoms,
    )

    entries = []
    manifest_rows = []
    counts = Counter()
    rmsd_by_noise = defaultdict(list)
    fallback_entries = []
    finite_failures = []
    torsion_failures = []
    per_complex_entries = Counter()

    for dataset_index in range(len(dataset)):
        clean_graph = dataset[dataset_index]
        name = complex_name(clean_graph)
        for noise_level in NOISE_LEVELS:
            tr_sigma, rot_sigma, tor_sigma = sigma_fn(noise_level, noise_level, noise_level)
            for pose_repeat in range(POSE_REPEATS):
                graph = copy.deepcopy(clean_graph)
                if not torch.is_tensor(graph["ligand"].pos):
                    graph["ligand"].pos = copy.deepcopy(random.choice(graph["ligand"].pos))
                base_coordinates = graph["ligand"].pos.detach().cpu().clone().float()
                tr_update = torch.normal(mean=0.0, std=float(tr_sigma), size=(1, 3))
                rot_update_np = np.asarray(so3.sample_vec(eps=float(rot_sigma)), dtype=np.float32)
                n_torsions = int(graph["ligand"].edge_mask.sum().item())
                torsion_updates_np = np.random.normal(
                    loc=0.0,
                    scale=float(tor_sigma),
                    size=n_torsions,
                ).astype(np.float32)
                transform.apply_noise(
                    graph,
                    noise_level,
                    noise_level,
                    noise_level,
                    tr_update=tr_update,
                    rot_update=rot_update_np,
                    torsion_updates=torsion_updates_np,
                )
                tor_score = (
                    torch.empty(0, dtype=torch.float32)
                    if graph.tor_score is None
                    else graph.tor_score.detach().cpu().clone().float()
                )
                entry = {
                    "complex_id": name,
                    "dataset_index": dataset_index,
                    "noise_level": float(noise_level),
                    "pose_repeat": pose_repeat,
                    "pose_bank_seed": POSE_BANK_SEED,
                    "base_ligand_coordinates": base_coordinates,
                    "initial_ligand_coordinates": graph["ligand"].pos.detach().cpu().clone().float(),
                    "translation_noise": tr_update.detach().cpu().clone().float(),
                    "rotation_noise": torch.from_numpy(rot_update_np.copy()).float(),
                    "torsion_noise": torch.from_numpy(torsion_updates_np.copy()).float(),
                    "true_translation_score": graph.tr_score.detach().cpu().clone().float(),
                    "true_rotation_score": graph.rot_score.detach().cpu().clone().float(),
                    "true_torsion_score": tor_score,
                    "rotatable_bond_mask": graph["ligand"].edge_mask.detach().cpu().clone().bool(),
                    "tr_sigma": float(tr_sigma),
                    "rot_sigma": float(rot_sigma),
                    "tor_sigma": float(tor_sigma),
                }
                rmsd_before, rmsd_method = evaluator_rmsd(graph, entry["initial_ligand_coordinates"])
                entry["rmsd_before"] = float(rmsd_before)
                entry["rmsd_source_method"] = rmsd_method
                entry["entry_sha256"] = entry_digest(entry)

                tensor_fields = [
                    value for value in entry.values()
                    if torch.is_tensor(value) and torch.is_floating_point(value)
                ]
                if not all(bool(torch.isfinite(value).all()) for value in tensor_fields) or not math.isfinite(rmsd_before):
                    finite_failures.append(entry["entry_sha256"])
                if entry["torsion_noise"].numel() != n_torsions or entry["true_torsion_score"].numel() != n_torsions:
                    torsion_failures.append(entry["entry_sha256"])
                if rmsd_method != "symmetry_aware":
                    fallback_entries.append(entry["entry_sha256"])

                entries.append(entry)
                counts[str(noise_level)] += 1
                rmsd_by_noise[str(noise_level)].append(rmsd_before)
                per_complex_entries[name] += 1
                manifest_rows.append({
                    "complex_id": name,
                    "dataset_index": dataset_index,
                    "noise_level": f"{noise_level:.2f}",
                    "pose_repeat": pose_repeat,
                    "pose_bank_seed": POSE_BANK_SEED,
                    "num_ligand_atoms": int(entry["initial_ligand_coordinates"].shape[0]),
                    "num_rotatable_bonds": n_torsions,
                    "tr_sigma": entry["tr_sigma"],
                    "rot_sigma": entry["rot_sigma"],
                    "tor_sigma": entry["tor_sigma"],
                    "rmsd_before": rmsd_before,
                    "rmsd_source_method": rmsd_method,
                    "entry_sha256": entry["entry_sha256"],
                })

    expected_entries = EXPECTED_COUNTS["val"] * len(NOISE_LEVELS) * POSE_REPEATS
    if len(entries) != expected_entries:
        raise RuntimeError(f"pose-bank cardinality mismatch: {len(entries)} != {expected_entries}")

    permutation_payload = {
        "schema_version": "surfna-lb200-joint-chemistry-permutation-v1",
        "chemistry_channels": list(CHEMISTRY_CHANNELS),
        "shuffle_seeds": list(SHUFFLE_SEEDS),
        "permutations": {},
    }
    shuffle_integrity = []
    max_mean_diff = 0.0
    max_std_diff = 0.0
    max_quantile_diff = 0.0
    max_fixed_ratio = 0.0
    for dataset_index in range(len(dataset)):
        graph = dataset[dataset_index]
        name = complex_name(graph)
        features = graph["surface"].x.detach().cpu().clone()
        chemistry = features[:, list(CHEMISTRY_CHANNELS)]
        permutation_payload["permutations"][name] = {}
        for shuffle_seed in SHUFFLE_SEEDS:
            permutation, attempts = make_joint_shuffle_permutation(name, int(features.shape[0]), shuffle_seed)
            shuffled_graph = copy.deepcopy(graph)
            before_invariants = graph_invariant_hashes(shuffled_graph)
            apply_joint_chemistry_shuffle(shuffled_graph, permutation)
            after_invariants = graph_invariant_hashes(shuffled_graph)
            shuffled = shuffled_graph["surface"].x.detach().cpu()
            shuffled_chemistry = shuffled[:, list(CHEMISTRY_CHANNELS)]
            non_chemistry = [index for index in range(8) if index not in CHEMISTRY_CHANNELS]
            fixed_ratio = float((permutation == torch.arange(permutation.numel())).float().mean())
            mean_diff = float((chemistry.mean(0) - shuffled_chemistry.mean(0)).abs().max())
            std_diff = float((chemistry.std(0) - shuffled_chemistry.std(0)).abs().max())
            quantiles = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
            quantile_diff = float(
                (torch.quantile(chemistry, quantiles, dim=0) - torch.quantile(shuffled_chemistry, quantiles, dim=0))
                .abs().max()
            )
            joint_multiset_equal = np.array_equal(sorted_rows(chemistry), sorted_rows(shuffled_chemistry))
            nonchem_equal = torch.equal(features[:, non_chemistry], shuffled[:, non_chemistry])
            invariants_equal = before_invariants == after_invariants
            finite = bool(torch.isfinite(shuffled).all())
            passed = (
                fixed_ratio < 0.05 and joint_multiset_equal and nonchem_equal and invariants_equal and finite
                and mean_diff <= 1e-6 and std_diff <= 1e-6 and quantile_diff <= 1e-6
            )
            if not passed:
                raise RuntimeError(f"joint chemistry shuffle integrity failed: {name}/{shuffle_seed}")
            max_mean_diff = max(max_mean_diff, mean_diff)
            max_std_diff = max(max_std_diff, std_diff)
            max_quantile_diff = max(max_quantile_diff, quantile_diff)
            max_fixed_ratio = max(max_fixed_ratio, fixed_ratio)
            permutation_payload["permutations"][name][str(shuffle_seed)] = {
                "dataset_index": dataset_index,
                "num_vertices": int(features.shape[0]),
                "attempts": attempts,
                "fixed_point_ratio": fixed_ratio,
                "permutation": permutation.tolist(),
            }
            shuffle_integrity.append({"complex_id": name, "shuffle_seed": shuffle_seed, "passed": True})

    medians = {key: float(np.median(values)) for key, values in rmsd_by_noise.items()}
    ordered_medians = [medians[str(level)] for level in NOISE_LEVELS]
    monotonic_median_rmsd = all(
        ordered_medians[index + 1] > ordered_medians[index]
        for index in range(len(ordered_medians) - 1)
    )
    bank_payload = {
        "schema_version": "surfna-lb200-mechanism-pose-bank-v1",
        "pose_bank_seed": POSE_BANK_SEED,
        "noise_levels": list(NOISE_LEVELS),
        "poses_per_complex_noise": POSE_REPEATS,
        "validation_count": EXPECTED_COUNTS["val"],
        "entry_count": len(entries),
        "split_path": str(model_args.split_val),
        "split_sha256": sha256_file(Path(model_args.split_val)),
        "cache_path": str(dataset.full_cache_path),
        "cache_heterographs_sha256": sha256_file(Path(dataset.full_cache_path) / "heterographs.pkl"),
        "cache_rdkit_ligands_sha256": sha256_file(Path(dataset.full_cache_path) / "rdkit_ligands.pkl"),
        "entries": entries,
    }
    split_names = [
        line.strip()
        for line in Path(model_args.split_val).read_text().splitlines()
        if line.strip()
    ]
    cache_contract = {
        "schema_version": "surfna-lb200-frozen-validation-cache-v1",
        "cohort": "L2 validation89",
        "complex_count": EXPECTED_COUNTS["val"],
        "complex_names_in_order": split_names,
        "split_path": str(Path(model_args.split_val).resolve()),
        "split_sha256": sha256_file(Path(model_args.split_val)),
        "cache_root": str(Path(model_args.cache_path).resolve()),
        "full_cache_path": str(Path(dataset.full_cache_path).resolve()),
        "heterographs_path": str((Path(dataset.full_cache_path) / "heterographs.pkl").resolve()),
        "heterographs_sha256": sha256_file(Path(dataset.full_cache_path) / "heterographs.pkl"),
        "rdkit_ligands_path": str((Path(dataset.full_cache_path) / "rdkit_ligands.pkl").resolve()),
        "rdkit_ligands_sha256": sha256_file(Path(dataset.full_cache_path) / "rdkit_ligands.pkl"),
        "automatic_rebuild_allowed": False,
    }
    if len(split_names) != EXPECTED_COUNTS["val"]:
        raise RuntimeError(
            f"frozen validation split cardinality mismatch: {len(split_names)} != {EXPECTED_COUNTS['val']}"
        )
    cache_contract_path = output / "frozen_validation_cache_contract.json"
    permutation_path = output / "shuffle_permutation_manifest.json"
    chemistry_integrity_path = output / "chemistry_shuffle_integrity_report.json"
    bank_integrity_path = output / "mechanism_pose_bank_integrity_report.json"
    atomic_json(cache_contract_path, cache_contract)
    atomic_torch_save(bank_path, bank_payload)
    bank_sha = sha256_file(bank_path)
    (output / "mechanism_pose_bank_v1_sha256.txt").write_text(f"{bank_sha}  {bank_path.name}\n")
    atomic_csv(
        output / "mechanism_pose_bank_v1_manifest.csv",
        list(manifest_rows[0]),
        manifest_rows,
    )
    atomic_json(permutation_path, permutation_payload)
    atomic_json(chemistry_integrity_path, {
        "schema_version": "surfna-lb200-chemistry-shuffle-integrity-v1",
        "passed": True,
        "complexes": len(dataset),
        "permutations": len(shuffle_integrity),
        "max_fixed_point_ratio": max_fixed_ratio,
        "max_abs_channel_mean_diff": max_mean_diff,
        "max_abs_channel_std_diff": max_std_diff,
        "max_abs_channel_quantile_diff": max_quantile_diff,
        "coordinates_and_edge_indices_unchanged": True,
        "si_and_boundary_unchanged": True,
        "joint_chemistry_row_multiset_unchanged": True,
        "all_finite": True,
    })
    integrity = {
        "schema_version": "surfna-lb200-mechanism-pose-bank-integrity-v1",
        "passed": (
            not finite_failures
            and not torsion_failures
            and len(per_complex_entries) == EXPECTED_COUNTS["val"]
            and all(value == len(NOISE_LEVELS) * POSE_REPEATS for value in per_complex_entries.values())
            and all(counts[str(level)] == EXPECTED_COUNTS["val"] * POSE_REPEATS for level in NOISE_LEVELS)
            and monotonic_median_rmsd
        ),
        "bank_sha256": bank_sha,
        "entry_count": len(entries),
        "counts_by_noise": dict(counts),
        "median_rmsd_by_noise": medians,
        "median_rmsd_strictly_increases": monotonic_median_rmsd,
        "finite_failure_count": len(finite_failures),
        "torsion_cardinality_failure_count": len(torsion_failures),
        "symmetry_rmsd_fallback_count": len(fallback_entries),
        "failed_entry_sha256": sorted(set(finite_failures + torsion_failures)),
    }
    atomic_json(bank_integrity_path, integrity)
    if not integrity["passed"]:
        raise RuntimeError("mechanism pose-bank integrity gate failed; see the frozen report")
    atomic_json(output / "POSE_BANK_COMPLETE.json", {
        "schema_version": "surfna-lb200-pose-bank-completion-v1",
        "bank": str(bank_path),
        "bank_sha256": bank_sha,
        "shuffle_permutation_manifest": str(permutation_path),
        "shuffle_permutation_manifest_sha256": sha256_file(permutation_path),
        "chemistry_shuffle_integrity_report": str(chemistry_integrity_path),
        "chemistry_shuffle_integrity_report_sha256": sha256_file(chemistry_integrity_path),
        "integrity_report": str(bank_integrity_path),
        "integrity_report_sha256": sha256_file(bank_integrity_path),
        "frozen_validation_cache_contract": str(cache_contract_path),
        "frozen_validation_cache_contract_sha256": sha256_file(cache_contract_path),
    })
    print(json.dumps({"bank": str(bank_path), "sha256": bank_sha, "entries": len(entries)}, sort_keys=True))


if __name__ == "__main__":
    main()
