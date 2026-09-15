#!/usr/bin/env python3
"""Validate one frozen generation-intervention unit and write its receipt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


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


def load_contract(root: Path):
    path = root / "00_protocol/FROZEN_CONTRACT.json"
    contract = json.loads(path.read_text())
    if contract.get("schema_version") != "surfna-generation-intervention-supplement-v1":
        raise RuntimeError("unexpected contract schema")
    if contract.get("formal_unit_count") != 90:
        raise RuntimeError("formal unit count is not frozen at 90")
    for name, expected in contract["code_sha256"].items():
        mapping = {
            "evaluator": root / "01_code/evaluate_generation_intervention.py",
            "unit_validator": root / "01_code/validate_unit.py",
            "summary": root / "01_code/summarize_supplement.py",
            "array_launcher": root / "01_code/run_generation_array.sbatch",
            "smoke_launcher": root / "01_code/run_smoke.sbatch",
            "summary_launcher": root / "01_code/run_summary.sbatch",
            "surface_gate_overlay": root / "01_code/overlay/models/surface_score_model_v3.py",
            "protocol": root / "00_protocol/PROTOCOL.md",
        }
        if sha256_file(mapping[name]) != expected:
            raise RuntimeError(f"frozen code hash drift: {name}")
    if sha256_file(Path(contract["matrix_path"])) != contract["matrix_sha256"]:
        raise RuntimeError("formal matrix hash drift")
    if sha256_file(Path(contract["shuffle_permutation_manifest"])) != contract["shuffle_permutation_manifest_sha256"]:
        raise RuntimeError("shuffle permutation hash drift")
    if sha256_file(Path(contract["validation_split"])) != contract["validation_split_sha256"]:
        raise RuntimeError("validation split hash drift")
    return contract


def matrix_row(root: Path, index: int):
    with (root / "00_protocol/formal_matrix.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 90 or [int(row["index"]) for row in rows] != list(range(90)):
        raise RuntimeError("formal matrix index/cardinality drift")
    return rows[index]


def output_dir(root: Path, row) -> Path:
    slug = row["arm"].lower().replace("-", "_")
    return root / "05_generation_raw" / slug / f"seed{row['seed']}" / f"epoch_{int(row['epoch']):04d}" / row["condition_id"]


def validate_preflight(root: Path, index: int):
    contract = load_contract(root)
    row = matrix_row(root, index)
    checkpoint = Path(row["model_dir"]) / row["checkpoint"]
    if sha256_file(checkpoint) != row["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint hash drift: {checkpoint}")
    if sha256_file(Path(row["model_dir"]) / "model_parameters.yml") != row["model_parameters_sha256"]:
        raise RuntimeError("model_parameters.yml hash drift")
    target = output_dir(root, row)
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(f"refusing to overwrite an existing unit attempt: {target}")
    return contract, row, target


def validate_manifest(root: Path, row, target: Path, expected_complexes: int, receipt_name: str):
    contract = load_contract(root)
    manifest_path = target / "validation_pose_manifest.tsv"
    log_path = target / "evaluate.log"
    if not manifest_path.is_file() or not log_path.is_file():
        raise FileNotFoundError("unit manifest or evaluator log is missing")
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected_rows = expected_complexes * 10
    if len(rows) != expected_rows:
        raise RuntimeError(f"manifest row count mismatch: {len(rows)} != {expected_rows}")
    grouped = {}
    for item in rows:
        grouped.setdefault(item["complex_name"], []).append(item)
        if not math.isfinite(float(item["rmsd"])):
            raise RuntimeError("non-finite RMSD in generated-pose manifest")
        if float(item["analysis_surface_gate_alpha"]) != float(row["gate_alpha"]):
            raise RuntimeError("gate metadata mismatch")
        if item["chemistry_condition"] != row["chemistry_condition"]:
            raise RuntimeError("chemistry-condition metadata mismatch")
        expected_shuffle_seed = "" if row["shuffle_seed"] == "none" else row["shuffle_seed"]
        if item["chemistry_shuffle_seed"] != expected_shuffle_seed:
            raise RuntimeError("chemistry-shuffle seed metadata mismatch")
    if len(grouped) != expected_complexes:
        raise RuntimeError("complex cardinality mismatch")
    for name, group in grouped.items():
        if sorted(int(item["sample_idx"]) for item in group) != list(range(10)):
            raise RuntimeError(f"sample index mismatch for {name}")
    log_text = log_path.read_text(errors="replace")
    if "Generation intervention:" not in log_text:
        raise RuntimeError("evaluator did not report the frozen generation intervention")
    receptor_hits = log_text.count("_protein_processed.pdb")
    if receptor_hits < expected_complexes:
        raise RuntimeError(f"canonical receptor evidence is incomplete: {receptor_hits} < {expected_complexes}")
    if row["chemistry_condition"] == "joint_shuffled_chemistry" and "Applied frozen joint chemistry shuffle" not in log_text:
        raise RuntimeError("joint chemistry shuffle was not reported")

    all_rmsd = [float(item["rmsd"]) for item in rows]
    best = [min(float(item["rmsd"]) for item in group) for group in grouped.values()]
    receipt = {
        "schema_version": "surfna-generation-intervention-unit-v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "index": int(row["index"]),
        "arm": row["arm"],
        "seed": int(row["seed"]),
        "epoch": int(row["epoch"]),
        "condition_id": row["condition_id"],
        "gate_alpha": float(row["gate_alpha"]),
        "chemistry_condition": row["chemistry_condition"],
        "shuffle_seed": None if row["shuffle_seed"] == "none" else int(row["shuffle_seed"]),
        "complex_count": len(grouped),
        "pose_count": len(rows),
        "failure_count": 0,
        "pose_density_rmsd_lt2": sum(value < 2.0 for value in all_rmsd) / len(all_rmsd),
        "best_of_k10_coverage_rmsd_lt2": sum(value < 2.0 for value in best) / len(best),
        "manifest_sha256": sha256_file(manifest_path),
        "contract_sha256": sha256_file(root / "00_protocol/FROZEN_CONTRACT.json"),
        "checkpoint_sha256": row["checkpoint_sha256"],
        "passed": True,
    }
    atomic_json(target / receipt_name, receipt)
    print(json.dumps(receipt, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--target", type=Path)
    parser.add_argument("--expected-complexes", type=int, default=89)
    parser.add_argument("--receipt-name", default="UNIT_COMPLETE.json")
    args = parser.parse_args()
    root = args.root.resolve()
    if not 0 <= args.index < 90:
        raise ValueError("unit index must be in [0, 89]")
    if args.preflight:
        _, row, target = validate_preflight(root, args.index)
        print(json.dumps({"preflight": True, "index": args.index, "target": str(target), "row": row}, sort_keys=True))
    else:
        row = matrix_row(root, args.index)
        target = args.target.resolve() if args.target else output_dir(root, row)
        validate_manifest(root, row, target, args.expected_complexes, args.receipt_name)


if __name__ == "__main__":
    main()
