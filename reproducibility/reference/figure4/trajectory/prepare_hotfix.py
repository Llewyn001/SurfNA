#!/usr/bin/env python3
"""Freeze the validation-only L2 receptor-path hotfix before GPU results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


CODE_FILES = (
    "HOTFIX_PROTOCOL.md",
    "evaluate_trajectory_accelerate_hotfix.py",
    "prepare_hotfix.py",
    "run_summary_hotfix.sbatch",
    "run_validation_hotfix_smoke.sbatch",
    "run_validation_trajectory_array_hotfix.sbatch",
    "validate_hotfix.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def atomic_write(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"stale temporary hotfix artifact: {temporary}")
    temporary.write_text(value)
    os.replace(temporary, path)


def pdb_atom_count(path: Path) -> int:
    count = 0
    with path.open(errors="strict") as handle:
        for line in handle:
            if not line.startswith("ATOM  "):
                continue
            coords = tuple(float(line[start:end]) for start, end in ((30, 38), (38, 46), (46, 54)))
            if not all(math.isfinite(value) for value in coords):
                raise RuntimeError(f"non-finite PDB coordinate: {path}")
            count += 1
    if count == 0:
        raise RuntimeError(f"receptor has no coordinate records: {path}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--hotfix-root", type=Path, required=True)
    parser.add_argument("--l2-root", type=Path, required=True)
    args = parser.parse_args()

    analysis_root = args.analysis_root.resolve()
    hotfix_root = args.hotfix_root.resolve()
    l2_root = args.l2_root.resolve()
    manifest_path = hotfix_root / "HOTFIX_MANIFEST.json"
    sidecar_path = hotfix_root / "HOTFIX_MANIFEST.sha256"
    inventory_path = hotfix_root / "VAL89_RECEPTOR_INVENTORY.json"
    smoke_subset_path = hotfix_root / "smoke_subset.txt"
    for path in (manifest_path, sidecar_path, inventory_path, smoke_subset_path):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite frozen hotfix artifact: {path}")

    split_path = l2_root / "splits/val.txt"
    data_root = l2_root / "data/trainval"
    names = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    if len(names) != 89 or len(set(names)) != 89:
        raise RuntimeError(f"expected 89 unique validation identities, found {len(names)}/{len(set(names))}")

    receptor_records = []
    ligand_records = []
    for name in names:
        complex_dir = data_root / name
        if not complex_dir.is_dir() or complex_dir.is_symlink():
            raise RuntimeError(f"invalid complex directory: {complex_dir}")
        receptor = complex_dir / f"{name}_protein_processed.pdb"
        ligand = complex_dir / f"{name}_ligand.sdf"
        for label, path in (("receptor", receptor), ("ligand", ligand)):
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"missing or symlinked {label}: {path}")
        receptor_records.append(
            {
                "complex_name": name,
                "path": str(receptor.resolve()),
                "relative_path": str(receptor.resolve().relative_to(l2_root)),
                "sha256": sha256(receptor),
                "size_bytes": receptor.stat().st_size,
                "atom_record_count": pdb_atom_count(receptor),
            }
        )
        ligand_records.append(
            {
                "complex_name": name,
                "path": str(ligand.resolve()),
                "relative_path": str(ligand.resolve().relative_to(l2_root)),
                "sha256": sha256(ligand),
                "size_bytes": ligand.stat().st_size,
            }
        )

    inventory = {
        "schema_version": "surfna-lb200-val89-raw-input-inventory-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "l2_root": str(l2_root),
        "data_root": str(data_root.resolve()),
        "split_path": str(split_path.resolve()),
        "split_sha256": sha256(split_path),
        "complex_count": len(names),
        "canonical_receptor_suffix": "_protein_processed.pdb",
        "receptors": receptor_records,
        "ligands": ligand_records,
        "passed": True,
    }
    inventory_text = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    smoke_subset_text = names[0] + "\n"

    code_records = []
    for relative in CODE_FILES:
        path = hotfix_root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or symlinked hotfix code: {path}")
        code_records.append(
            {
                "relative_path": relative,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )

    original_evaluator = analysis_root / "01_code/analysis/evaluate_trajectory_accelerate.py"
    manifest = {
        "schema_version": "surfna-lb200-validation-hotfix-v1",
        "status": "FROZEN_BEFORE_HOTFIX_SMOKE_OR_FORMAL_RESULTS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_root": str(analysis_root),
        "hotfix_root": str(hotfix_root),
        "scientific_scope": "validation receptor filename resolution only",
        "formal_matrix": {
            "arms": ["Scratch-Full8", "Transfer-All", "Transfer-NonSurface"],
            "seeds": [0, 1, 2],
            "epochs": [25, 50, 75, 100, 125, 150, 175, 200],
            "unit_count": 72,
            "validation_complexes": 89,
            "poses_per_complex": 10,
            "inference_steps": 20,
            "sampling_seed": 20260826,
        },
        "failed_attempt": {
            "validation_job": 243122,
            "summary_job": 243123,
            "failed_units": 54,
            "cancelled_unstarted_units": 18,
            "valid_scientific_outputs": 0,
            "reason": "evaluator omitted canonical _protein_processed.pdb receptor filename",
        },
        "original_evaluator": {
            "path": str(original_evaluator.resolve()),
            "sha256": sha256(original_evaluator),
        },
        "receptor_inventory": {
            "path": str(inventory_path),
            "sha256": sha256_text(inventory_text),
            "complex_count": 89,
        },
        "smoke_subset": {
            "path": str(smoke_subset_path),
            "sha256": sha256_text(smoke_subset_text),
            "complex_name": names[0],
        },
        "code_records": code_records,
        "passed": True,
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_sha = sha256_text(manifest_text)
    atomic_write(inventory_path, inventory_text)
    atomic_write(smoke_subset_path, smoke_subset_text)
    atomic_write(manifest_path, manifest_text)
    atomic_write(sidecar_path, f"{manifest_sha}  {manifest_path.name}\n")
    print(json.dumps({"passed": True, "manifest_sha256": manifest_sha, "receptors": 89}, sort_keys=True))


if __name__ == "__main__":
    main()
