#!/usr/bin/env python3
"""Freeze the independent generation-intervention supplement before inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ARMS = ("Transfer-All", "Transfer-NonSurface")
SEEDS = (0, 1, 2)
EPOCHS = (25, 100, 200)
CONDITIONS = (
    ("baseline_true_gate1", 1.0, "true_chemistry", "none"),
    ("surface_off_true_gate0", 0.0, "true_chemistry", "none"),
    ("chemistry_shuffle_20260911", 1.0, "joint_shuffled_chemistry", "20260911"),
    ("chemistry_shuffle_20260912", 1.0, "joint_shuffled_chemistry", "20260912"),
    ("chemistry_shuffle_20260913", 1.0, "joint_shuffled_chemistry", "20260913"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-v5", type=Path, required=True)
    parser.add_argument("--parent-v6", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    contract_path = root / "00_protocol/FROZEN_CONTRACT.json"
    matrix_path = root / "00_protocol/formal_matrix.tsv"
    if contract_path.exists() or matrix_path.exists():
        raise SystemExit("refusing to overwrite an already frozen supplement")

    code_paths = {
        "evaluator": root / "01_code/evaluate_generation_intervention.py",
        "unit_validator": root / "01_code/validate_unit.py",
        "summary": root / "01_code/summarize_supplement.py",
        "array_launcher": root / "01_code/run_generation_array.sbatch",
        "smoke_launcher": root / "01_code/run_smoke.sbatch",
        "summary_launcher": root / "01_code/run_summary.sbatch",
        "surface_gate_overlay": root / "01_code/overlay/models/surface_score_model_v3.py",
        "protocol": root / "00_protocol/PROTOCOL.md",
    }
    input_paths = {
        "shuffle_permutation_manifest": root / "03_intervention_inputs/shuffle_permutation_manifest.json",
        "frozen_validation_cache_contract": root / "03_intervention_inputs/frozen_validation_cache_contract.json",
        "checkpoint_inventory": root / "03_intervention_inputs/nine_model_checkpoint_inventory.json",
        "receptor_inventory": root / "03_intervention_inputs/VAL89_RECEPTOR_INVENTORY.json",
    }
    for path in [*code_paths.values(), *input_paths.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)

    inventory = json.loads(input_paths["checkpoint_inventory"].read_text())
    if not inventory.get("passed") or len(inventory.get("records", [])) != 72:
        raise RuntimeError("parent checkpoint inventory is not the frozen 72-checkpoint inventory")
    records = {
        (record["arm"], int(record["seed"]), int(record["epoch"])): record
        for record in inventory["records"]
    }

    rows = []
    index = 0
    for arm in ARMS:
        for seed in SEEDS:
            for epoch in EPOCHS:
                record = records[(arm, seed, epoch)]
                checkpoint = Path(record["path"])
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                observed_checkpoint_sha = sha256_file(checkpoint)
                if observed_checkpoint_sha != record["file_sha256"]:
                    raise RuntimeError(f"checkpoint hash drift: {checkpoint}")
                model_parameters = checkpoint.parent / "model_parameters.yml"
                model_parameters_sha = sha256_file(model_parameters)
                for condition_id, gate_alpha, chemistry_condition, shuffle_seed in CONDITIONS:
                    rows.append({
                        "index": index,
                        "arm": arm,
                        "seed": seed,
                        "epoch": epoch,
                        "condition_id": condition_id,
                        "gate_alpha": gate_alpha,
                        "chemistry_condition": chemistry_condition,
                        "shuffle_seed": shuffle_seed,
                        "model_dir": str(checkpoint.parent),
                        "checkpoint": checkpoint.name,
                        "checkpoint_sha256": observed_checkpoint_sha,
                        "model_parameters_sha256": model_parameters_sha,
                    })
                    index += 1
    if len(rows) != 90:
        raise RuntimeError(f"formal matrix cardinality drift: {len(rows)}")

    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    with matrix_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    permutation = json.loads(input_paths["shuffle_permutation_manifest"].read_text())
    if tuple(permutation.get("chemistry_channels", [])) != (0, 1, 2, 4, 5, 6):
        raise RuntimeError("unexpected chemistry channels in frozen permutation manifest")
    if len(permutation.get("permutations", {})) != 89:
        raise RuntimeError("frozen permutation manifest does not cover val89")

    cache_contract = json.loads(input_paths["frozen_validation_cache_contract"].read_text())
    split_path = Path(cache_contract["split_path"])
    split_names = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    if len(split_names) != 89 or len(set(split_names)) != 89:
        raise RuntimeError("frozen validation split is not 89 unique complexes")
    smoke_subset = root / "03_intervention_inputs/smoke_subset.txt"
    smoke_subset.write_text(split_names[0] + "\n")

    v6_matrix = args.parent_v6 / "03_integrity/MECHANISM_RECOMPUTE_MATRIX_COMPLETE.json"
    v6_summary = args.parent_v6 / "07_statistics/DYNAMIC_SUMMARY_COMPLETE.json"
    for path in (v6_matrix, v6_summary):
        if not path.is_file():
            raise FileNotFoundError(path)

    contract = {
        "schema_version": "surfna-generation-intervention-supplement-v1",
        "status": "FROZEN_BEFORE_SMOKE_OR_FORMAL_RESULTS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "parent_v5_read_only": str(args.parent_v5.resolve()),
        "parent_v6_read_only": str(args.parent_v6.resolve()),
        "parent_v6_integrity": {
            "mechanism_matrix_path": str(v6_matrix.resolve()),
            "mechanism_matrix_sha256": sha256_file(v6_matrix),
            "summary_path": str(v6_summary.resolve()),
            "summary_sha256": sha256_file(v6_summary),
        },
        "cohort": "HomologyClean L2 validation89",
        "validation_complexes": 89,
        "samples_per_complex": 10,
        "inference_steps": 20,
        "sampling_seed": 20260826,
        "arms": list(ARMS),
        "training_seeds": list(SEEDS),
        "ema_epochs": list(EPOCHS),
        "condition_ids": [item[0] for item in CONDITIONS],
        "formal_unit_count": 90,
        "chemistry_channels": [0, 1, 2, 4, 5, 6],
        "shuffle_seeds": [20260911, 20260912, 20260913],
        "shuffle_permutation_manifest": str(input_paths["shuffle_permutation_manifest"].resolve()),
        "shuffle_permutation_manifest_sha256": sha256_file(input_paths["shuffle_permutation_manifest"]),
        "frozen_validation_cache_contract": str(input_paths["frozen_validation_cache_contract"].resolve()),
        "frozen_validation_cache_contract_sha256": sha256_file(input_paths["frozen_validation_cache_contract"]),
        "validation_split": str(split_path.resolve()),
        "validation_split_sha256": sha256_file(split_path),
        "test128_policy": "not rerun; existing endpoint only; no intervention selection",
        "primary_endpoint": "pose_density_rmsd_lt2",
        "secondary_endpoints": ["best_of_k10_coverage_rmsd_lt2", "best_of_k10_rmsd"],
        "matrix_path": str(matrix_path.resolve()),
        "matrix_sha256": sha256_file(matrix_path),
        "smoke_subset": str(smoke_subset.resolve()),
        "smoke_subset_sha256": sha256_file(smoke_subset),
        "code_sha256": {name: sha256_file(path) for name, path in code_paths.items()},
        "input_sha256": {name: sha256_file(path) for name, path in input_paths.items()},
    }
    atomic_json(contract_path, contract)
    print(json.dumps({"frozen": True, "formal_units": len(rows), "contract": str(contract_path)}))


if __name__ == "__main__":
    main()
