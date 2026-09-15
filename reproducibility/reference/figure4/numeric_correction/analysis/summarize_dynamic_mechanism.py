#!/usr/bin/env python3
"""Fail-closed aggregation and paired inference for the SurfNA LB200 mechanism study.

This program consumes the complete 3 arms x 3 seeds x 8 formal epochs matrix.  It
never selects checkpoints, never reads test poses to tune an analysis, and
never draws figures.  Its inferential unit is a complex; pose, noisy-pose, and
shuffle repeats are aggregated before paired hierarchical bootstrapping.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


ARMS = ("Scratch-Full8", "Transfer-All", "Transfer-NonSurface")
ARM_SLUGS = {
    "Scratch-Full8": "scratch_full8",
    "Transfer-All": "transfer_all",
    "Transfer-NonSurface": "transfer_nonsurface",
}
SEEDS = (0, 1, 2)
EPOCHS = (25, 50, 75, 100, 125, 150, 175, 200)
NOISE_LEVELS = (0.10, 0.30, 0.60, 0.90)
POSE_REPEATS = tuple(range(5))
SHUFFLE_SEEDS = (20260911, 20260912, 20260913)
DOSE_EPOCHS = (50, 100, 200)
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260903
STABILITY_THRESHOLDS_ANGSTROM = (100.0, 1_000.0, 1_000_000.0, 1_000_000_000_000.0)

EXPECTED_TEST200_HASHES = {
    "test_pose_level_metrics.csv": "b300b340f8607f883eb8cfa1a66fba0597c8ad6270a7b1413f69245368ba18a3",
    "test_complex_level_metrics.csv": "f5eb1efdc6c26f9c5417c822fe7c58f4784c9d8e03cd210a70f814608c702d79",
    "test_seed_level_summary.csv": "004b009b914efa1a7bea0261f5dfcc9246ed661b8a756770dad7469bbf0db3e0",
}
TEST200_DESTINATIONS = {
    "test_pose_level_metrics.csv": "test200_pose_level.csv",
    "test_complex_level_metrics.csv": "test200_complex_level.csv",
    "test_seed_level_summary.csv": "test200_summary.csv",
}

TASK_METRICS = (
    "pose_density_lt2",
    "best_of_k10_lt2",
    "best_rmsd",
    "valid_pose_rate",
)
MECHANISM_METRICS = (
    "combined_alignment",
    "translation_cosine",
    "rotation_cosine",
    "torsion_cosine",
    "delta_RMSD_1step",
    "median_delta_RMSD_1step",
    "one_step_improvement_fraction",
)
MECHANISM_NUMERIC_FIELDS = (
    "translation_cosine",
    "translation_normalized_mse",
    "translation_direction_sign_agreement",
    "rotation_cosine",
    "rotation_normalized_mse",
    "torsion_cosine",
    "torsion_normalized_mse",
    "combined_alignment",
    "RMSD_before",
    "RMSD_after",
    "delta_RMSD_1step",
    "direct_RMSD_before",
    "direct_RMSD_after",
    "direct_delta_RMSD_1step",
)
MECHANISM_LONG_FIELDS = [
    "arm", "training_seed", "epoch", "complex_id", "noise_level", "pose_repeat",
    "condition", "gate_alpha", "shuffle_seed", "entry_sha256", "initial_coordinates_sha256",
    "translation_cosine", "translation_normalized_mse", "translation_direction_sign_agreement",
    "rotation_cosine", "rotation_normalized_mse", "torsion_cosine", "torsion_normalized_mse",
    "combined_alignment", "combined_alignment_components", "RMSD_before", "RMSD_after",
    "delta_RMSD_1step", "direct_RMSD_before", "direct_RMSD_after", "direct_delta_RMSD_1step",
    "rmsd_source_method_before", "rmsd_source_method_after", "num_rotatable_bonds",
    "valid", "failure_reason",
]


class SummaryValidationError(RuntimeError):
    """Raised when any frozen input unit is missing, duplicated, or invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_csv(path: Path, rows: Iterable[dict], fields: list[str] | tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: object, *, field: str) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise SummaryValidationError(f"invalid Boolean in {field}: {value!r}")


def parse_finite(value: object, *, field: str, nonnegative: bool = False) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise SummaryValidationError(f"invalid numeric value in {field}: {value!r}") from error
    if not math.isfinite(parsed) or (nonnegative and parsed < 0):
        raise SummaryValidationError(f"non-finite/out-of-range value in {field}: {value!r}")
    return parsed


def optional_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def finite_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else math.nan


def finite_median(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(finite)) if finite else math.nan


def normalized_aulc(points: dict[int, float]) -> float:
    if set(points) != set(EPOCHS) or any(not math.isfinite(float(points[epoch])) for epoch in EPOCHS):
        return math.nan
    area = 0.0
    for left, right in zip(EPOCHS[:-1], EPOCHS[1:]):
        area += (right - left) * (float(points[left]) + float(points[right])) / 2.0
    return area / 175.0


def expected_conditions(epoch: int) -> set[tuple[str, float, int | None]]:
    alphas = [0.0, 0.5, 1.0]
    if epoch in DOSE_EPOCHS:
        alphas.extend([0.25, 0.75])
    conditions = {("true_chemistry", alpha, None) for alpha in alphas}
    conditions.update(("joint_shuffled_chemistry", 1.0, seed) for seed in SHUFFLE_SEEDS)
    return conditions


def load_v6_recompute_contract(root: Path) -> tuple[dict, dict, dict[tuple, dict], str]:
    manifest_path = root / "00_protocol/MECHANISM_RECOMPUTE_INPUT_MANIFEST.json"
    sidecar_path = manifest_path.with_suffix(".sha256")
    matrix_path = root / "03_integrity/MECHANISM_RECOMPUTE_MATRIX_COMPLETE.json"
    contract_path = root / "00_protocol/FROZEN_V6_NUMERIC_QC_CONTRACT.json"
    contract_sidecar = contract_path.with_suffix(".sha256")
    for path in (manifest_path, sidecar_path, matrix_path, contract_path, contract_sidecar):
        if not path.is_file():
            raise SummaryValidationError(f"missing v6 numerical-QC contract artifact: {path}")
    manifest_sha = sha256_file(manifest_path)
    if sidecar_path.read_text().strip() != manifest_sha:
        raise SummaryValidationError("v6 input manifest sidecar digest mismatch")
    manifest = json.loads(manifest_path.read_text())
    contract = json.loads(contract_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    if (
        manifest.get("schema_version") != "surfna-lb200-mechanism-recompute-input-manifest-v2"
        or len(manifest.get("units", [])) != 72
        or len(manifest.get("validation_units", [])) != 72
        or contract.get("schema_version") != "surfna-lb200-v6-numeric-qc-contract-v1"
        or contract.get("input_manifest_sha256") != manifest_sha
        or matrix.get("schema_version") != "surfna-lb200-mechanism-recompute-matrix-complete-v2"
        or matrix.get("status") != "complete"
        or int(matrix.get("valid_units", -1)) != 72
        or int(matrix.get("expected_units", -1)) != 72
        or int(matrix.get("derived_disallowed_nonfinite_count", -1)) != 0
        or matrix.get("input_manifest_sha256") != manifest_sha
    ):
        raise SummaryValidationError("v6 numerical-QC contract or matrix-completion gate failed")
    if contract_sidecar.read_text().strip() != sha256_file(contract_path):
        raise SummaryValidationError("v6 numeric-QC contract sidecar digest mismatch")
    for local_path, manifest_path_key, manifest_hash_key in (
        (
            root / "04_mechanism_pose_bank/mechanism_pose_bank_v1.pt",
            "pose_bank_path", "pose_bank_sha256",
        ),
        (
            root / "04_mechanism_pose_bank/mechanism_pose_bank_v1_manifest.csv",
            "pose_bank_manifest_path", "pose_bank_manifest_sha256",
        ),
        (
            root / "04_mechanism_pose_bank/POSE_BANK_COMPLETE.json",
            "pose_bank_completion_path", "pose_bank_completion_sha256",
        ),
        (
            root / "04_mechanism_pose_bank/frozen_validation_cache_contract.json",
            "validation_cache_contract_path", "validation_cache_contract_sha256",
        ),
    ):
        source_path = Path(manifest.get(manifest_path_key, ""))
        expected_hash = manifest.get(manifest_hash_key)
        if (
            not expected_hash or not source_path.is_file() or not local_path.is_file()
            or sha256_file(source_path) != expected_hash
            or sha256_file(local_path) != expected_hash
        ):
            raise SummaryValidationError(f"frozen copied/source artifact drift: {local_path}")
    hotfix_manifest = root / "validation_hotfix_v1/HOTFIX_MANIFEST.json"
    hotfix_sidecar = root / "validation_hotfix_v1/HOTFIX_MANIFEST.sha256"
    hotfix_digest = manifest.get("validation_hotfix_manifest_sha256")
    if (
        not hotfix_manifest.is_file()
        or not hotfix_sidecar.is_file()
        or not hotfix_digest
        or sha256_file(hotfix_manifest) != hotfix_digest
        or sha256_file(hotfix_sidecar) != manifest.get("validation_hotfix_sidecar_sha256")
    ):
        raise SummaryValidationError("copied validation hotfix manifest digest mismatch")
    for code_name, record in manifest.get("code", {}).items():
        code_path = Path(record.get("path", ""))
        if not code_path.is_file() or sha256_file(code_path) != record.get("sha256"):
            raise SummaryValidationError(f"frozen v6 code digest mismatch: {code_name}/{code_path}")
    if manifest.get("code", {}).get("summary", {}).get("sha256") != sha256_file(Path(__file__).resolve()):
        raise SummaryValidationError("running summary script differs from frozen v6 manifest")
    units = {}
    for unit in manifest["units"]:
        key = (unit["arm"], int(unit["training_seed"]), int(unit["epoch"]))
        if key in units:
            raise SummaryValidationError(f"duplicate v6 input-manifest unit: {key}")
        units[key] = unit
    if len(units) != 72:
        raise SummaryValidationError(f"v6 input-manifest unit cardinality mismatch: {len(units)}")
    matrix_units = {}
    for record in matrix.get("units", []):
        key = (record.get("arm"), int(record.get("training_seed", -1)), int(record.get("epoch", -1)))
        if key in matrix_units:
            raise SummaryValidationError(f"duplicate matrix-completion unit: {key}")
        matrix_units[key] = record
    if set(matrix_units) != set(units):
        raise SummaryValidationError("matrix-completion unit inventory differs from frozen input manifest")
    for (arm, seed, epoch), record in matrix_units.items():
        directory = (
            root / "05_mechanism_raw" / ARM_SLUGS[arm]
            / f"seed{seed}" / f"epoch_{epoch:04d}"
        )
        csv_path = directory / "mechanism_denoising_long.csv"
        receipt_path = directory / "MECHANISM_COMPLETE.json"
        if (
            Path(record.get("csv", "")).resolve() != csv_path.resolve()
            or Path(record.get("receipt", "")).resolve() != receipt_path.resolve()
            or not csv_path.is_file() or not receipt_path.is_file()
            or sha256_file(csv_path) != record.get("csv_sha256")
            or sha256_file(receipt_path) != record.get("receipt_sha256")
        ):
            raise SummaryValidationError(f"matrix-completion output binding failed: {arm}/seed{seed}/epoch{epoch}")
    return manifest, matrix, units, manifest_sha


def unit_paths(root: Path) -> tuple[dict[tuple, Path], dict[tuple, Path]]:
    validation = {}
    mechanism = {}
    for arm in ARMS:
        slug = ARM_SLUGS[arm]
        for seed in SEEDS:
            for epoch in EPOCHS:
                validation[(arm, seed, epoch)] = (
                    root / "06_validation_trajectory/raw" / slug / f"seed{seed}"
                    / f"epoch_{epoch:04d}" / "validation_pose_manifest.tsv"
                )
                mechanism[(arm, seed, epoch)] = (
                    root / "05_mechanism_raw" / slug / f"seed{seed}"
                    / f"epoch_{epoch:04d}" / "mechanism_denoising_long.csv"
                )
    return validation, mechanism


def precheck_unit_inventory(root: Path, completeness: list[dict]) -> tuple[dict, dict]:
    validation, mechanism = unit_paths(root)
    expected_validation = set(validation.values())
    expected_mechanism = set(mechanism.values())
    validation_root = root / "06_validation_trajectory/raw"
    mechanism_root = root / "05_mechanism_raw"
    discovered_validation = set(validation_root.rglob("validation_pose_manifest.tsv")) if validation_root.exists() else set()
    discovered_mechanism = set(mechanism_root.rglob("mechanism_denoising_long.csv")) if mechanism_root.exists() else set()
    issues = []
    for kind, expected, discovered in (
        ("validation", expected_validation, discovered_validation),
        ("mechanism", expected_mechanism, discovered_mechanism),
    ):
        missing = sorted(expected - discovered)
        unexpected = sorted(discovered - expected)
        if missing:
            issues.append(f"{kind}: missing {len(missing)} units: {missing[:5]}")
        if unexpected:
            issues.append(f"{kind}: unexpected/duplicate-layout {len(unexpected)} units: {unexpected[:5]}")
    for kind, mapping in (("validation", validation), ("mechanism", mechanism)):
        for (arm, seed, epoch), path in mapping.items():
            completeness.append({
                "unit_type": kind,
                "arm": arm,
                "training_seed": seed,
                "epoch": epoch,
                "path": str(path),
                "exists": path.is_file(),
                "status": "pending" if path.is_file() else "missing",
                "row_count": "",
                "expected_row_count": "",
                "sha256": "",
                "reason": "" if path.is_file() else "canonical input file missing",
            })
    if issues:
        raise SummaryValidationError("; ".join(issues))
    return validation, mechanism


def update_completeness(
    completeness: list[dict], kind: str, arm: str, seed: int, epoch: int, **values: object
) -> None:
    matches = [
        row for row in completeness
        if row["unit_type"] == kind and row["arm"] == arm
        and int(row["training_seed"]) == seed and int(row["epoch"]) == epoch
    ]
    if len(matches) != 1:
        raise SummaryValidationError(
            f"internal completeness-key collision: {kind}/{arm}/seed{seed}/epoch{epoch}"
        )
    matches[0].update(values)


def load_frozen_validation_contract(root: Path) -> tuple[list[str], dict]:
    path = root / "04_mechanism_pose_bank/frozen_validation_cache_contract.json"
    if not path.is_file():
        raise SummaryValidationError(f"missing frozen validation cache contract: {path}")
    payload = json.loads(path.read_text())
    names = [str(value) for value in payload.get("complex_names_in_order", [])]
    if len(names) != 89 or len(set(names)) != 89 or int(payload.get("complex_count", -1)) != 89:
        raise SummaryValidationError("frozen validation cache contract is not an exact 89-complex cohort")
    for path_key, hash_key in (
        ("split_path", "split_sha256"),
        ("heterographs_path", "heterographs_sha256"),
        ("rdkit_ligands_path", "rdkit_ligands_sha256"),
    ):
        artifact = Path(payload[path_key])
        if not artifact.is_file() or sha256_file(artifact) != payload[hash_key]:
            raise SummaryValidationError(f"frozen validation artifact drift: {artifact}")
    return names, payload


def load_pose_bank_manifest(root: Path, names: list[str]) -> tuple[dict[str, dict], dict]:
    directory = root / "04_mechanism_pose_bank"
    completion_path = directory / "POSE_BANK_COMPLETE.json"
    manifest_path = directory / "mechanism_pose_bank_v1_manifest.csv"
    if not completion_path.is_file() or not manifest_path.is_file():
        raise SummaryValidationError("pose-bank completion or manifest is missing")
    completion = json.loads(completion_path.read_text())
    required_hash_pairs = (
        ("bank", "bank_sha256"),
        ("shuffle_permutation_manifest", "shuffle_permutation_manifest_sha256"),
        ("chemistry_shuffle_integrity_report", "chemistry_shuffle_integrity_report_sha256"),
        ("integrity_report", "integrity_report_sha256"),
        ("frozen_validation_cache_contract", "frozen_validation_cache_contract_sha256"),
    )
    for path_key, hash_key in required_hash_pairs:
        artifact = Path(completion.get(path_key, ""))
        if not artifact.is_file() or sha256_file(artifact) != completion.get(hash_key):
            raise SummaryValidationError(f"POSE_BANK_COMPLETE hash contract failed for {path_key}: {artifact}")

    entries: dict[str, dict] = {}
    by_cell: dict[tuple[str, float], list[int]] = defaultdict(list)
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "complex_id", "dataset_index", "noise_level", "pose_repeat", "num_rotatable_bonds",
            "rmsd_before", "rmsd_source_method", "entry_sha256",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise SummaryValidationError(f"malformed pose-bank manifest fields: {reader.fieldnames}")
        for row in reader:
            entry_sha = str(row["entry_sha256"])
            name = str(row["complex_id"])
            noise = parse_finite(row["noise_level"], field="pose-bank noise_level")
            repeat = int(row["pose_repeat"])
            if entry_sha in entries or name not in names or noise not in NOISE_LEVELS or repeat not in POSE_REPEATS:
                raise SummaryValidationError(f"duplicate/invalid pose-bank entry: {name}/{noise}/{repeat}/{entry_sha}")
            entries[entry_sha] = {
                "complex_id": name,
                "dataset_index": int(row["dataset_index"]),
                "noise_level": noise,
                "pose_repeat": repeat,
                "num_rotatable_bonds": int(row["num_rotatable_bonds"]),
                "rmsd_before": parse_finite(row["rmsd_before"], field="pose-bank rmsd_before", nonnegative=True),
                "rmsd_source_method": str(row["rmsd_source_method"]),
            }
            by_cell[(name, noise)].append(repeat)
    if len(entries) != 1780 or set(name for name, _ in by_cell) != set(names):
        raise SummaryValidationError(f"pose-bank manifest cardinality mismatch: entries={len(entries)}")
    if any(sorted(repeats) != list(POSE_REPEATS) for repeats in by_cell.values()):
        raise SummaryValidationError("pose-bank manifest has missing or duplicate pose repeats")
    if len(by_cell) != 89 * len(NOISE_LEVELS):
        raise SummaryValidationError("pose-bank manifest has an incomplete complex x noise matrix")
    return entries, completion


def parse_validation_units(
    paths: dict[tuple, Path], names: list[str], cache_contract: dict,
    long_path: Path, completeness: list[dict], expected_hotfix_digest: str,
    expected_validation_units: dict[tuple, dict],
) -> dict[tuple[str, int, int, str], dict[str, float]]:
    complex_metrics: dict[tuple[str, int, int, str], dict[str, float]] = {}
    sampling_seed_by_complex: dict[str, int] = {}
    fields = [
        "arm", "training_seed", "epoch", "complex_id", "pose_id", "RMSD", "is_lt2",
        "is_valid", "rmsd_source_method", "complex_sampling_seed", "source_manifest_sha256",
    ]
    with long_path.open("w", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=fields)
        writer.writeheader()
        for (arm, seed, epoch), path in sorted(paths.items()):
            source_sha = sha256_file(path)
            run_dir = path.parent
            try:
                frozen_unit = expected_validation_units.get((arm, seed, epoch))
                if frozen_unit is None:
                    raise SummaryValidationError(
                        f"missing frozen validation unit: {arm}/seed{seed}/epoch{epoch}"
                    )
                if source_sha != frozen_unit.get("source_manifest_sha256"):
                    raise SummaryValidationError(f"validation manifest hash drift: {path}")
                if not (run_dir / "RUN_EXIT_OK.txt").is_file():
                    raise SummaryValidationError(f"missing validation RUN_EXIT_OK: {run_dir}")
                if (
                    sha256_file(run_dir / "RUN_EXIT_OK.txt")
                    != frozen_unit.get("source_run_exit_ok_sha256")
                    or sha256_file(run_dir / "RUN_EXIT.txt")
                    != frozen_unit.get("source_run_exit_sha256")
                ):
                    raise SummaryValidationError(f"validation receipt hash drift: {run_dir}")
                hotfix_sidecar = run_dir / "HOTFIX_MANIFEST.sha256"
                if (
                    not hotfix_sidecar.is_file()
                    or hotfix_sidecar.read_text().split()[0] != expected_hotfix_digest
                    or sha256_file(hotfix_sidecar) != frozen_unit.get("hotfix_sidecar_sha256")
                ):
                    raise SummaryValidationError(f"validation hotfix digest mismatch: {run_dir}")
                exit_text = (run_dir / "RUN_EXIT.txt").read_text().strip()
                if exit_text != "exit_code=0":
                    raise SummaryValidationError(f"nonzero/malformed validation exit receipt: {run_dir}/{exit_text}")
                grouped: dict[str, dict[int, float]] = defaultdict(dict)
                methods: dict[str, set[str]] = defaultdict(set)
                with path.open(newline="") as handle:
                    reader = csv.DictReader(handle, delimiter="\t")
                    required = {
                        "complex_name", "sample_idx", "rmsd", "rmsd_source_method",
                        "split_path", "complex_sampling_seed",
                    }
                    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                        raise SummaryValidationError(f"{path}: malformed fields={reader.fieldnames}")
                    rows = list(reader)
                if len(rows) != 890:
                    raise SummaryValidationError(f"{path}: expected 890 rows, observed {len(rows)}")
                for row in rows:
                    name = str(row["complex_name"])
                    pose_id = int(row["sample_idx"])
                    rmsd = parse_finite(row["rmsd"], field=f"{path}:rmsd", nonnegative=True)
                    method = str(row["rmsd_source_method"]).strip()
                    complex_sampling_seed = int(row["complex_sampling_seed"])
                    if name not in names or pose_id not in range(10) or not method:
                        raise SummaryValidationError(f"{path}: invalid identity/pose/method: {name}/{pose_id}/{method!r}")
                    if os.path.realpath(row["split_path"]) != os.path.realpath(cache_contract["split_path"]):
                        raise SummaryValidationError(f"{path}: split path drift for {name}")
                    if pose_id in grouped[name]:
                        raise SummaryValidationError(f"{path}: duplicate pose row {name}/{pose_id}")
                    previous_seed = sampling_seed_by_complex.setdefault(name, complex_sampling_seed)
                    if previous_seed != complex_sampling_seed:
                        raise SummaryValidationError(f"validation sampling seed drift for {name}")
                    grouped[name][pose_id] = rmsd
                    methods[name].add(method)
                    writer.writerow({
                        "arm": arm,
                        "training_seed": seed,
                        "epoch": epoch,
                        "complex_id": name,
                        "pose_id": pose_id,
                        "RMSD": rmsd,
                        "is_lt2": rmsd < 2.0,
                        "is_valid": True,
                        "rmsd_source_method": method,
                        "complex_sampling_seed": complex_sampling_seed,
                        "source_manifest_sha256": source_sha,
                    })
                if set(grouped) != set(names) or any(sorted(values) != list(range(10)) for values in grouped.values()):
                    raise SummaryValidationError(f"{path}: not an exact validation89 x K10 manifest")
                for name in names:
                    rmsds = [grouped[name][pose] for pose in range(10)]
                    complex_metrics[(arm, seed, epoch, name)] = {
                        "pose_density_lt2": sum(value < 2.0 for value in rmsds) / 10.0,
                        "best_of_k10_lt2": float(min(rmsds) < 2.0),
                        "best_rmsd": min(rmsds),
                        "median_rmsd": float(statistics.median(rmsds)),
                        "valid_pose_rate": 1.0,
                        "rmsd_source_method_count": len(methods[name]),
                    }
                update_completeness(
                    completeness, "validation", arm, seed, epoch,
                    status="valid", row_count=890, expected_row_count=890, sha256=source_sha,
                )
            except Exception as error:
                update_completeness(
                    completeness, "validation", arm, seed, epoch,
                    status="invalid", sha256=source_sha, reason=f"{type(error).__name__}: {error}",
                )
                raise
    if len(complex_metrics) != 72 * 89:
        raise SummaryValidationError(f"validation complex matrix incomplete: {len(complex_metrics)}")
    return complex_metrics


def write_task_summaries(
    stage: Path, complex_metrics: dict, names: list[str], promotions: list[tuple[Path, Path]], root: Path,
) -> tuple[dict, list[dict], list[dict]]:
    complex_rows = []
    seed_rows = []
    aulc_values: dict[tuple[str, int, str, str], float] = {}
    for arm in ARMS:
        for seed in SEEDS:
            for epoch in EPOCHS:
                one_epoch = [complex_metrics[(arm, seed, epoch, name)] for name in names]
                for name, values in zip(names, one_epoch):
                    complex_rows.append({
                        "arm": arm, "training_seed": seed, "epoch": epoch, "complex_id": name,
                        **values,
                    })
                seed_rows.append({
                    "arm": arm,
                    "training_seed": seed,
                    "epoch": epoch,
                    "n_complexes": len(names),
                    "pose_density_lt2": finite_mean(row["pose_density_lt2"] for row in one_epoch),
                    "best_of_k10_coverage_lt2": finite_mean(row["best_of_k10_lt2"] for row in one_epoch),
                    "median_best_rmsd": finite_median(row["best_rmsd"] for row in one_epoch),
                    "mean_best_rmsd": finite_mean(row["best_rmsd"] for row in one_epoch),
                    "valid_pose_rate": finite_mean(row["valid_pose_rate"] for row in one_epoch),
                })
            for name in names:
                for metric in TASK_METRICS:
                    points = {
                        epoch: complex_metrics[(arm, seed, epoch, name)][metric]
                        for epoch in EPOCHS
                    }
                    aulc_values[(arm, seed, name, metric)] = normalized_aulc(points)

    aulc_rows = []
    for (arm, seed, name, metric), value in sorted(aulc_values.items()):
        aulc_rows.append({
            "arm": arm, "training_seed": seed, "metric": metric,
            "aggregation_level": "complex", "complex_id": name, "AULC_25_200": value,
        })
    for arm in ARMS:
        for seed in SEEDS:
            for metric in TASK_METRICS:
                values = [aulc_values[(arm, seed, name, metric)] for name in names]
                aulc_rows.append({
                    "arm": arm, "training_seed": seed, "metric": metric,
                    "aggregation_level": "seed_mean", "complex_id": "",
                    "AULC_25_200": finite_mean(values),
                })

    outputs = (
        ("validation_complex", complex_rows, [
            "arm", "training_seed", "epoch", "complex_id", "pose_density_lt2",
            "best_of_k10_lt2", "best_rmsd", "median_rmsd", "valid_pose_rate",
            "rmsd_source_method_count",
        ], root / "06_validation_trajectory/nine_model_validation_complex_trajectory.csv"),
        ("validation_seed", seed_rows, [
            "arm", "training_seed", "epoch", "n_complexes", "pose_density_lt2",
            "best_of_k10_coverage_lt2", "median_best_rmsd", "mean_best_rmsd", "valid_pose_rate",
        ], root / "06_validation_trajectory/nine_model_validation_seed_trajectory.csv"),
        ("validation_aulc", aulc_rows, [
            "arm", "training_seed", "metric", "aggregation_level", "complex_id", "AULC_25_200",
        ], root / "06_validation_trajectory/nine_model_validation_AULC.csv"),
    )
    for filename, rows, fields, destination in outputs:
        source = stage / f"{filename}.csv"
        write_csv(source, rows, fields)
        promotions.append((source, destination))
    return aulc_values, seed_rows, aulc_rows


def condition_key(row: dict, epoch: int) -> tuple[str, float, int | None]:
    condition = str(row["condition"])
    alpha = float(row["gate_alpha"])
    shuffle_text = str(row.get("shuffle_seed", "")).strip()
    shuffle = None if not shuffle_text else int(shuffle_text)
    key = (condition, alpha, shuffle)
    if key not in expected_conditions(epoch):
        raise SummaryValidationError(f"unexpected intervention condition at epoch {epoch}: {key}")
    return key


def parse_mechanism_units(
    paths: dict[tuple, Path], bank_entries: dict[str, dict], names: list[str],
    long_path: Path, complex_path: Path, completeness: list[dict],
    recompute_units: dict[tuple, dict], recompute_manifest_sha: str,
) -> dict[tuple, dict[str, float]]:
    values: dict[tuple, dict[str, float]] = {}
    coordinate_hashes: dict[str, str] = {}
    complex_fields = [
        "arm", "training_seed", "epoch", "complex_id", "noise_level", "condition",
        "gate_alpha", "shuffle_seed", "n_pose_repeats", "combined_alignment",
        "translation_cosine", "rotation_cosine", "torsion_cosine", "delta_RMSD_1step",
        "median_delta_RMSD_1step",
        "one_step_improvement_fraction", "valid_combined_alignment", "valid_translation_cosine",
        "valid_rotation_cosine", "valid_torsion_cosine", "valid_delta_RMSD_1step",
        "valid_median_delta_RMSD_1step",
    ]
    with long_path.open("w", newline="") as long_handle, complex_path.open("w", newline="") as complex_handle:
        long_writer = csv.DictWriter(long_handle, fieldnames=MECHANISM_LONG_FIELDS)
        complex_writer = csv.DictWriter(complex_handle, fieldnames=complex_fields)
        long_writer.writeheader()
        complex_writer.writeheader()
        for (arm, seed, epoch), path in sorted(paths.items()):
            source_sha = sha256_file(path)
            completion_path = path.parent / "MECHANISM_COMPLETE.json"
            try:
                if not completion_path.is_file():
                    raise SummaryValidationError(f"missing mechanism completion: {completion_path}")
                completion = json.loads(completion_path.read_text())
                unit = recompute_units.get((arm, seed, epoch))
                if unit is None:
                    raise SummaryValidationError(f"missing frozen v6 source unit: {arm}/seed{seed}/epoch{epoch}")
                if (
                    completion.get("schema_version") != "surfna-lb200-mechanism-recompute-completion-v2"
                    or completion.get("status") != "complete"
                    or completion.get("arm") != arm
                    or int(completion.get("training_seed", -1)) != seed
                    or int(completion.get("epoch", -1)) != epoch
                    or int(completion.get("failure_count", -1)) != 0
                    or completion.get("csv_sha256") != source_sha
                    or completion.get("source_raw_scores_sha256") != unit["source_raw_sha256"]
                    or completion.get("source_completion_sha256") != unit["source_completion_sha256"]
                    or completion.get("source_csv_sha256") != unit["source_csv_sha256"]
                    or completion.get("input_manifest_sha256") != recompute_manifest_sha
                    or completion.get("derivation_precision") != "float64-derived-metrics-with-legacy-pose-bank-RMSD_before"
                    or completion.get("scientific_protocol_change") is not False
                    or completion.get("reinference_performed") is not False
                    or completion.get("scientific_output_reintegration_performed") is not False
                    or completion.get("coordinate_integrity_replay_performed") is not True
                    or completion.get("replayed_coordinates_used_as_metrics_input") is not False
                    or completion.get("stored_parent_coordinates_used_as_metrics_input") is not True
                    or completion.get("RMSD_before_policy") != "reuse_immutable_legacy_float32_pose_bank_metric"
                    or completion.get("delta_RMSD_1step_policy") != "legacy_RMSD_before_minus_float64_RMSD_after"
                    or completion.get("direct_RMSD_before_policy") != "recompute_from_pose_bank_coordinates_in_float64"
                    or completion.get("validation_cache_contract_sha256") != unit["validation_cache_contract_sha256"]
                    or completion.get("model_parameters_sha256") != unit["model_parameters_sha256"]
                    or int(completion.get("derived_disallowed_nonfinite_count", -1)) != 0
                ):
                    raise SummaryValidationError(f"mechanism completion contract mismatch: {completion_path}")
                replay = completion.get("coordinate_replay", {})
                if (
                    replay.get("passed") is not True
                    or int(replay.get("checked_count", -1)) != int(unit["expected_row_count"])
                    or float(replay.get("tolerance", math.nan)) != 1e-6
                    or not math.isfinite(float(replay.get("max_abs_difference", math.nan)))
                    or float(replay.get("max_abs_difference")) > 1e-6
                ):
                    raise SummaryValidationError(f"coordinate replay integrity not passed: {completion_path}")

                specs = expected_conditions(epoch)
                expected_rows = len(bank_entries) * len(specs)
                observed_specs: dict[str, set[tuple[str, float, int | None]]] = defaultdict(set)
                seen_rows: set[tuple] = set()
                grouped: dict[tuple, dict] = {}
                row_count = 0
                with path.open(newline="") as handle:
                    reader = csv.DictReader(handle)
                    if reader.fieldnames is None or not set(MECHANISM_LONG_FIELDS).issubset(reader.fieldnames):
                        raise SummaryValidationError(f"{path}: malformed fields={reader.fieldnames}")
                    for row in reader:
                        row_count += 1
                        if row["arm"] != arm or int(row["training_seed"]) != seed or int(row["epoch"]) != epoch:
                            raise SummaryValidationError(f"{path}: row arm/seed/epoch identity drift")
                        if not parse_bool(row["valid"], field=f"{path}:valid") or str(row["failure_reason"]).strip():
                            raise SummaryValidationError(f"{path}: invalid mechanism row encountered")
                        entry_sha = str(row["entry_sha256"])
                        entry = bank_entries.get(entry_sha)
                        if entry is None:
                            raise SummaryValidationError(f"{path}: unknown pose-bank entry {entry_sha}")
                        name = str(row["complex_id"])
                        noise = float(row["noise_level"])
                        repeat = int(row["pose_repeat"])
                        if (
                            name != entry["complex_id"] or noise != entry["noise_level"]
                            or repeat != entry["pose_repeat"] or name not in names
                        ):
                            raise SummaryValidationError(f"{path}: pose-bank identity drift for {entry_sha}")
                        if int(row["num_rotatable_bonds"]) != entry["num_rotatable_bonds"]:
                            raise SummaryValidationError(f"{path}: torsion count drift for {entry_sha}")
                        if abs(float(row["RMSD_before"]) - entry["rmsd_before"]) > 1e-7:
                            raise SummaryValidationError(f"{path}: RMSD_before drift for {entry_sha}")
                        intervention = condition_key(row, epoch)
                        unique = (entry_sha, *intervention)
                        if unique in seen_rows or intervention in observed_specs[entry_sha]:
                            raise SummaryValidationError(f"{path}: duplicate mechanism row {unique}")
                        seen_rows.add(unique)
                        observed_specs[entry_sha].add(intervention)
                        coordinate_hash = str(row["initial_coordinates_sha256"])
                        if len(coordinate_hash) != 64:
                            raise SummaryValidationError(f"{path}: malformed input-coordinate hash")
                        previous_hash = coordinate_hashes.setdefault(entry_sha, coordinate_hash)
                        if previous_hash != coordinate_hash:
                            raise SummaryValidationError(f"input coordinates changed across mechanism units: {entry_sha}")
                        numeric = {
                            field: parse_finite(row[field], field=f"{path}:{field}")
                            for field in (
                                "translation_normalized_mse", "translation_direction_sign_agreement",
                                "rotation_normalized_mse", "combined_alignment", "RMSD_before",
                                "RMSD_after", "delta_RMSD_1step", "direct_RMSD_before",
                                "direct_RMSD_after", "direct_delta_RMSD_1step",
                            )
                        }
                        for field in (
                            "translation_normalized_mse", "rotation_normalized_mse", "RMSD_before",
                            "RMSD_after", "direct_RMSD_before", "direct_RMSD_after",
                        ):
                            if numeric[field] < 0:
                                raise SummaryValidationError(f"{path}: negative {field}")
                        if numeric["translation_direction_sign_agreement"] not in (0.0, 1.0):
                            raise SummaryValidationError(f"{path}: invalid translation sign agreement")
                        if not math.isclose(
                            numeric["delta_RMSD_1step"],
                            numeric["RMSD_before"] - numeric["RMSD_after"],
                            rel_tol=1e-12, abs_tol=1e-10,
                        ):
                            raise SummaryValidationError(f"{path}: delta_RMSD arithmetic mismatch")
                        if not math.isclose(
                            numeric["direct_delta_RMSD_1step"],
                            numeric["direct_RMSD_before"] - numeric["direct_RMSD_after"],
                            rel_tol=1e-12, abs_tol=1e-10,
                        ):
                            raise SummaryValidationError(f"{path}: direct delta_RMSD arithmetic mismatch")
                        cosines = []
                        for field in ("translation_cosine", "rotation_cosine", "torsion_cosine"):
                            value = optional_float(row[field])
                            if math.isfinite(value):
                                if value < -1.0 - 1e-12 or value > 1.0 + 1e-12:
                                    raise SummaryValidationError(f"{path}: {field} outside [-1,1]")
                                cosines.append(value)
                        components = int(row["combined_alignment_components"])
                        if (
                            components != len(cosines) or not cosines
                            or not math.isclose(
                                numeric["combined_alignment"], sum(cosines) / len(cosines),
                                rel_tol=1e-12, abs_tol=1e-10,
                            )
                        ):
                            raise SummaryValidationError(f"{path}: combined alignment arithmetic mismatch")
                        metrics = {
                            metric: optional_float(row[metric])
                            for metric in MECHANISM_METRICS
                            if metric not in ("one_step_improvement_fraction", "median_delta_RMSD_1step")
                        }
                        metrics["median_delta_RMSD_1step"] = numeric["delta_RMSD_1step"]
                        metrics["one_step_improvement_fraction"] = float(
                            float(row["delta_RMSD_1step"]) > 0
                        )
                        group_key = (name, noise, *intervention)
                        accumulator = grouped.setdefault(group_key, {
                            "repeats": set(),
                            "sums": defaultdict(float),
                            "counts": defaultdict(int),
                            "values": defaultdict(list),
                        })
                        if repeat in accumulator["repeats"]:
                            raise SummaryValidationError(f"{path}: duplicate pose repeat in {group_key}/{repeat}")
                        accumulator["repeats"].add(repeat)
                        for metric, value in metrics.items():
                            if math.isfinite(value):
                                accumulator["sums"][metric] += value
                                accumulator["counts"][metric] += 1
                                accumulator["values"][metric].append(value)
                        long_writer.writerow(row)
                if row_count != expected_rows or len(seen_rows) != expected_rows:
                    raise SummaryValidationError(
                        f"{path}: expected {expected_rows} unique rows, observed {row_count}/{len(seen_rows)}"
                    )
                if set(observed_specs) != set(bank_entries) or any(value != specs for value in observed_specs.values()):
                    raise SummaryValidationError(f"{path}: incomplete intervention cross-product")
                if int(completion.get("row_count", -1)) != row_count or int(completion.get("valid_row_count", -1)) != row_count:
                    raise SummaryValidationError(f"{path}: completion row counts disagree with CSV")
                expected_groups = 89 * len(NOISE_LEVELS) * len(specs)
                if len(grouped) != expected_groups:
                    raise SummaryValidationError(f"{path}: mechanism complex group count {len(grouped)} != {expected_groups}")
                for (name, noise, condition, alpha, shuffle), accumulator in sorted(grouped.items()):
                    if accumulator["repeats"] != set(POSE_REPEATS):
                        raise SummaryValidationError(f"{path}: incomplete pose repeats in {name}/{noise}/{condition}")
                    result = {}
                    for metric in MECHANISM_METRICS:
                        count = accumulator["counts"].get(metric, 0)
                        result[metric] = (
                            float(statistics.median(accumulator["values"][metric]))
                            if metric == "median_delta_RMSD_1step" and count
                            else accumulator["sums"][metric] / count if count else math.nan
                        )
                    key = (arm, seed, epoch, name, noise, condition, alpha, shuffle)
                    if key in values:
                        raise SummaryValidationError(f"duplicate mechanism complex cell: {key}")
                    values[key] = result
                    complex_writer.writerow({
                        "arm": arm,
                        "training_seed": seed,
                        "epoch": epoch,
                        "complex_id": name,
                        "noise_level": noise,
                        "condition": condition,
                        "gate_alpha": alpha,
                        "shuffle_seed": "" if shuffle is None else shuffle,
                        "n_pose_repeats": len(accumulator["repeats"]),
                        **result,
                        **{
                            f"valid_{metric}": accumulator["counts"].get(metric, 0)
                            for metric in MECHANISM_METRICS if metric != "one_step_improvement_fraction"
                        },
                    })
                update_completeness(
                    completeness, "mechanism", arm, seed, epoch,
                    status="valid", row_count=row_count, expected_row_count=expected_rows, sha256=source_sha,
                )
            except Exception as error:
                update_completeness(
                    completeness, "mechanism", arm, seed, epoch,
                    status="invalid", sha256=source_sha, reason=f"{type(error).__name__}: {error}",
                )
                raise
    expected_cells = (
        len(ARMS) * len(SEEDS) * len(names) * len(NOISE_LEVELS)
        * sum(len(expected_conditions(epoch)) for epoch in EPOCHS)
    )
    if len(values) != expected_cells:
        raise SummaryValidationError(
            f"mechanism complex-cell cardinality mismatch: {len(values)} != {expected_cells}"
        )
    return values


def write_stability_outputs(
    mechanism_long_path: Path, stage: Path, root: Path, promotions: list[tuple[Path, Path]],
) -> dict:
    """Report post-failure numerical stability without dropping or clipping any row."""
    grouped: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with mechanism_long_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rmsd = parse_finite(row["RMSD_after"], field="v6 stability:RMSD_after", nonnegative=True)
            arm = str(row["arm"])
            seed = int(row["training_seed"])
            epoch = int(row["epoch"])
            noise = float(row["noise_level"])
            condition = str(row["condition"])
            alpha = float(row["gate_alpha"])
            shuffle = str(row["shuffle_seed"]).strip()
            keys = (
                ("condition", arm, seed, epoch, noise, condition, alpha, shuffle),
                ("unit", arm, seed, epoch, "all", "all", "all", "all"),
                ("arm", arm, "all", "all", "all", "all", "all", "all"),
                ("global", "all", "all", "all", "all", "all", "all", "all"),
            )
            for key in keys:
                accumulator = grouped[key]
                accumulator["n_rows"] += 1
                accumulator["max_RMSD_after"] = max(accumulator["max_RMSD_after"], rmsd)
                for threshold in STABILITY_THRESHOLDS_ANGSTROM:
                    accumulator[f"gt_{int(threshold)}A"] += int(rmsd > threshold)
    fields = [
        "aggregation_level", "arm", "training_seed", "epoch", "noise_level",
        "condition", "gate_alpha", "shuffle_seed", "n_rows", "max_RMSD_after",
    ]
    for threshold in STABILITY_THRESHOLDS_ANGSTROM:
        fields.extend([f"count_RMSD_gt_{int(threshold)}A", f"rate_RMSD_gt_{int(threshold)}A"])
    rows = []
    for key, accumulator in sorted(grouped.items(), key=lambda item: str(item[0])):
        level, arm, seed, epoch, noise, condition, alpha, shuffle = key
        n_rows = int(accumulator["n_rows"])
        row = {
            "aggregation_level": level,
            "arm": arm,
            "training_seed": seed,
            "epoch": epoch,
            "noise_level": noise,
            "condition": condition,
            "gate_alpha": alpha,
            "shuffle_seed": shuffle,
            "n_rows": n_rows,
            "max_RMSD_after": accumulator["max_RMSD_after"],
        }
        for threshold in STABILITY_THRESHOLDS_ANGSTROM:
            count = int(accumulator[f"gt_{int(threshold)}A"])
            row[f"count_RMSD_gt_{int(threshold)}A"] = count
            row[f"rate_RMSD_gt_{int(threshold)}A"] = count / n_rows
        rows.append(row)
    csv_path = stage / "one_step_stability_endpoints.csv"
    write_csv(csv_path, rows, fields)
    promotions.append((csv_path, root / "07_statistics/one_step_stability_endpoints.csv"))
    arm_rows = {
        row["arm"]: row for row in rows if row["aggregation_level"] == "arm"
    }
    global_rows = [row for row in rows if row["aggregation_level"] == "global"]
    if set(arm_rows) != set(ARMS) or len(global_rows) != 1:
        raise SummaryValidationError("stability endpoint aggregation is incomplete")
    payload = {
        "schema_version": "surfna-lb200-one-step-stability-endpoints-v1",
        "status": "post_failure_qc_descriptive_not_original_confirmatory_endpoint",
        "primary_threshold_angstrom": 100.0,
        "sensitivity_thresholds_angstrom": list(STABILITY_THRESHOLDS_ANGSTROM[1:]),
        "denominator": "all valid fixed-pose one-step rows",
        "global": global_rows[0],
        "by_arm": arm_rows,
    }
    json_path = stage / "one_step_stability_summary.json"
    atomic_json(json_path, payload)
    promotions.append((json_path, root / "07_statistics/one_step_stability_summary.json"))
    return payload


def mechanism_seed_trajectories(values: dict, names: list[str]) -> list[dict]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    pooled: dict[tuple, list[float]] = defaultdict(list)
    for (arm, seed, epoch, name, noise, condition, alpha, shuffle), metrics in values.items():
        for metric, value in metrics.items():
            if math.isfinite(value):
                grouped[(arm, seed, epoch, noise, condition, alpha, shuffle, metric)].append(value)
                pooled[(arm, seed, epoch, name, condition, alpha, shuffle, metric)].append(value)
    for (arm, seed, epoch, name, condition, alpha, shuffle, metric), noise_values in pooled.items():
        value = finite_mean(noise_values)
        if math.isfinite(value):
            grouped[(arm, seed, epoch, "all", condition, alpha, shuffle, metric)].append(value)
    rows = []
    for key, observations in sorted(grouped.items(), key=lambda item: str(item[0])):
        arm, seed, epoch, noise, condition, alpha, shuffle, metric = key
        rows.append({
            "arm": arm,
            "training_seed": seed,
            "epoch": epoch,
            "noise_level": noise,
            "condition": condition,
            "gate_alpha": alpha,
            "shuffle_seed": "" if shuffle is None else shuffle,
            "metric": metric,
            "value": finite_mean(observations),
            "n_complex_values": len(observations),
        })
    return rows


def mechanism_condition_aulc(seed_rows: list[dict]) -> list[dict]:
    series: dict[tuple, dict[int, float]] = defaultdict(dict)
    counts: dict[tuple, dict[int, int]] = defaultdict(dict)
    for row in seed_rows:
        key = (
            row["arm"], int(row["training_seed"]), str(row["noise_level"]), row["condition"],
            float(row["gate_alpha"]), str(row["shuffle_seed"]), row["metric"],
        )
        series[key][int(row["epoch"])] = float(row["value"])
        counts[key][int(row["epoch"])] = int(row["n_complex_values"])
    rows = []
    for key, points in sorted(series.items(), key=lambda item: str(item[0])):
        value = normalized_aulc(points)
        if not math.isfinite(value):
            continue
        arm, seed, noise, condition, alpha, shuffle, metric = key
        rows.append({
            "arm": arm,
            "training_seed": seed,
            "noise_level": noise,
            "condition": condition,
            "gate_alpha": alpha,
            "shuffle_seed": shuffle,
            "metric": metric,
            "AULC_25_200": value,
            "epoch_nodes": len(points),
            "minimum_complex_values_per_epoch": min(counts[key].values()),
        })
    return rows


def mechanism_effects(values: dict, names: list[str]) -> tuple[dict, dict, dict, dict]:
    surface: dict[tuple, float] = {}
    chemistry: dict[tuple, float] = {}
    delta_surface: dict[tuple, float] = {}
    mediated_chemistry: dict[tuple, float] = {}
    for arm in ARMS:
        for seed in SEEDS:
            for epoch in EPOCHS:
                for name in names:
                    for noise in NOISE_LEVELS:
                        for metric in MECHANISM_METRICS:
                            true_on = values[(arm, seed, epoch, name, noise, "true_chemistry", 1.0, None)][metric]
                            true_off = values[(arm, seed, epoch, name, noise, "true_chemistry", 0.0, None)][metric]
                            shuffled = [
                                values[(
                                    arm, seed, epoch, name, noise,
                                    "joint_shuffled_chemistry", 1.0, shuffle_seed,
                                )][metric]
                                for shuffle_seed in SHUFFLE_SEEDS
                            ]
                            if math.isfinite(true_on) and math.isfinite(true_off):
                                surface[(arm, seed, epoch, name, noise, metric)] = true_on - true_off
                            shuffle_mean = finite_mean(shuffled)
                            if math.isfinite(true_on) and math.isfinite(shuffle_mean):
                                chemistry[(arm, seed, epoch, name, noise, metric)] = true_on - shuffle_mean
    for seed in SEEDS:
        for epoch in EPOCHS:
            for name in names:
                for noise in NOISE_LEVELS:
                    for metric in MECHANISM_METRICS:
                        ta_key = ("Transfer-All", seed, epoch, name, noise, metric)
                        tn_key = ("Transfer-NonSurface", seed, epoch, name, noise, metric)
                        if ta_key in surface and tn_key in surface:
                            delta_surface[(seed, epoch, name, noise, metric)] = surface[ta_key] - surface[tn_key]
                        if ta_key in chemistry and tn_key in chemistry:
                            mediated_chemistry[(seed, epoch, name, noise, metric)] = (
                                chemistry[ta_key] - chemistry[tn_key]
                            )
    return surface, chemistry, delta_surface, mediated_chemistry


def pool_effect_over_noise(mapping: dict, prefix: tuple, metric: str) -> float:
    return finite_mean(mapping.get((*prefix, noise, metric), math.nan) for noise in NOISE_LEVELS)


def effect_aulcs(
    surface: dict, chemistry: dict, delta_surface: dict, mediated_chemistry: dict,
    names: list[str],
) -> dict[tuple, float]:
    output: dict[tuple, float] = {}
    for family, mapping in (("G_surface", surface), ("G_chem", chemistry)):
        for arm in ARMS:
            for seed in SEEDS:
                for name in names:
                    for metric in MECHANISM_METRICS:
                        points = {
                            epoch: pool_effect_over_noise(mapping, (arm, seed, epoch, name), metric)
                            for epoch in EPOCHS
                        }
                        value = normalized_aulc(points)
                        if math.isfinite(value):
                            output[(family, arm, seed, name, metric)] = value
    for family, mapping in (("DeltaG_surface", delta_surface), ("M_chem", mediated_chemistry)):
        for seed in SEEDS:
            for name in names:
                for metric in MECHANISM_METRICS:
                    points = {
                        epoch: pool_effect_over_noise(mapping, (seed, epoch, name), metric)
                        for epoch in EPOCHS
                    }
                    value = normalized_aulc(points)
                    if math.isfinite(value):
                        output[(family, "TA-minus-TN", seed, name, metric)] = value
    return output


def write_mechanism_summaries(
    stage: Path, root: Path, promotions: list[tuple[Path, Path]], values: dict, names: list[str],
) -> tuple[list[dict], dict, dict, dict, dict, dict]:
    seed_rows = mechanism_seed_trajectories(values, names)
    aulc_rows = mechanism_condition_aulc(seed_rows)
    seed_path = stage / "mechanism_seed.csv"
    aulc_path = stage / "mechanism_aulc.csv"
    write_csv(seed_path, seed_rows, [
        "arm", "training_seed", "epoch", "noise_level", "condition", "gate_alpha",
        "shuffle_seed", "metric", "value", "n_complex_values",
    ])
    write_csv(aulc_path, aulc_rows, [
        "arm", "training_seed", "noise_level", "condition", "gate_alpha", "shuffle_seed",
        "metric", "AULC_25_200", "epoch_nodes", "minimum_complex_values_per_epoch",
    ])
    promotions.extend([
        (seed_path, root / "07_statistics/nine_model_mechanism_seed_trajectory.csv"),
        (aulc_path, root / "07_statistics/nine_model_mechanism_AULC.csv"),
    ])

    surface, chemistry, delta_surface, mediated_chemistry = mechanism_effects(values, names)
    effect_aulc = effect_aulcs(surface, chemistry, delta_surface, mediated_chemistry, names)
    effect_complex_rows = []
    paired_complex_rows = []
    for family, mapping in (("G_surface", surface), ("G_chem", chemistry)):
        for (arm, seed, epoch, name, noise, metric), value in sorted(mapping.items()):
            effect_complex_rows.append({
                "family": family, "arm": arm, "training_seed": seed, "epoch": epoch,
                "complex_id": name, "noise_level": noise, "metric": metric, "value": value,
            })
    for family, mapping in (("DeltaG_surface", delta_surface), ("M_chem", mediated_chemistry)):
        for (seed, epoch, name, noise, metric), value in sorted(mapping.items()):
            paired_complex_rows.append({
                "family": family, "contrast": "Transfer-All-minus-Transfer-NonSurface",
                "training_seed": seed, "epoch": epoch, "complex_id": name,
                "noise_level": noise, "metric": metric, "value": value,
            })
    effect_aulc_rows = [
        {
            "family": family, "arm_or_contrast": arm, "training_seed": seed,
            "complex_id": name, "metric": metric, "AULC_25_200": value,
        }
        for (family, arm, seed, name, metric), value in sorted(effect_aulc.items())
    ]
    effect_complex_path = stage / "mechanism_effect_complex.csv"
    paired_complex_path = stage / "paired_mechanism_effect_complex.csv"
    effect_aulc_path = stage / "mechanism_effect_aulc.csv"
    write_csv(effect_complex_path, effect_complex_rows, [
        "family", "arm", "training_seed", "epoch", "complex_id", "noise_level", "metric", "value",
    ])
    write_csv(paired_complex_path, paired_complex_rows, [
        "family", "contrast", "training_seed", "epoch", "complex_id", "noise_level", "metric", "value",
    ])
    write_csv(effect_aulc_path, effect_aulc_rows, [
        "family", "arm_or_contrast", "training_seed", "complex_id", "metric", "AULC_25_200",
    ])
    promotions.extend([
        (effect_complex_path, root / "07_statistics/nine_model_mechanism_effect_complex.csv"),
        (paired_complex_path, root / "07_statistics/paired_mechanism_effect_complex.csv"),
        (effect_aulc_path, root / "07_statistics/mechanism_effect_AULC.csv"),
    ])
    return seed_rows, surface, chemistry, delta_surface, mediated_chemistry, effect_aulc


def write_gate_dose_response(
    stage: Path, root: Path, promotions: list[tuple[Path, Path]], values: dict, names: list[str],
) -> list[dict]:
    """Summarize the preregistered five-alpha cross-sectional checks.

    EMA0 is unavailable by the frozen protocol, so only the actually observed
    50/100/200 checkpoints are emitted.  These are descriptive and are never
    used for checkpoint selection or as a hard success gate.
    """
    alphas = (0.0, 0.25, 0.5, 0.75, 1.0)
    trajectory_rows = []
    summary_rows = []
    for arm in ARMS:
        for seed in SEEDS:
            for epoch in DOSE_EPOCHS:
                for metric in MECHANISM_METRICS:
                    for noise in (*NOISE_LEVELS, "all"):
                        by_alpha: dict[float, list[float]] = {alpha: [] for alpha in alphas}
                        complete_complex_profiles = []
                        for name in names:
                            profile = []
                            for alpha in alphas:
                                if noise == "all":
                                    observation = finite_mean(
                                        values[(
                                            arm, seed, epoch, name, level,
                                            "true_chemistry", alpha, None,
                                        )][metric]
                                        for level in NOISE_LEVELS
                                    )
                                else:
                                    observation = values[(
                                        arm, seed, epoch, name, noise,
                                        "true_chemistry", alpha, None,
                                    )][metric]
                                profile.append(observation)
                                if math.isfinite(observation):
                                    by_alpha[alpha].append(observation)
                            if all(math.isfinite(value) for value in profile):
                                complete_complex_profiles.append(profile)
                        means = {alpha: finite_mean(by_alpha[alpha]) for alpha in alphas}
                        for alpha in alphas:
                            trajectory_rows.append({
                                "arm": arm,
                                "training_seed": seed,
                                "epoch": epoch,
                                "noise_level": noise,
                                "metric": metric,
                                "gate_alpha": alpha,
                                "mean_value": means[alpha],
                                "n_complexes": len(by_alpha[alpha]),
                            })
                        if not all(math.isfinite(means[alpha]) for alpha in alphas):
                            raise SummaryValidationError(
                                f"dose response has no finite seed mean: {arm}/seed{seed}/epoch{epoch}/{noise}/{metric}"
                            )
                        differences = [means[right] - means[left] for left, right in zip(alphas, alphas[1:])]
                        mean_vector = np.asarray([means[alpha] for alpha in alphas], dtype=np.float64)
                        alpha_value_correlation: float | str = ""
                        if float(np.std(mean_vector)) > 0:
                            alpha_value_correlation = float(np.corrcoef(
                                np.asarray(alphas, dtype=np.float64), mean_vector,
                            )[0, 1])
                        monotonic_complexes = sum(
                            all(profile[index + 1] >= profile[index] for index in range(4))
                            for profile in complete_complex_profiles
                        )
                        summary_rows.append({
                            "arm": arm,
                            "training_seed": seed,
                            "epoch": epoch,
                            "noise_level": noise,
                            "metric": metric,
                            "alpha_0": means[0.0],
                            "alpha_0_25": means[0.25],
                            "alpha_0_5": means[0.5],
                            "alpha_0_75": means[0.75],
                            "alpha_1": means[1.0],
                            "delta_alpha1_minus_alpha0": means[1.0] - means[0.0],
                            "consecutive_nondecreasing_steps": sum(value >= 0 for value in differences),
                            "monotonic_seed_mean": all(value >= 0 for value in differences),
                            "pearson_alpha_value": alpha_value_correlation,
                            "n_complete_complexes": len(complete_complex_profiles),
                            "complex_monotonic_fraction": (
                                monotonic_complexes / len(complete_complex_profiles)
                                if complete_complex_profiles else ""
                            ),
                            "endpoint_status": "descriptive_not_hard_gate",
                        })
    if len(trajectory_rows) != len(ARMS) * len(SEEDS) * len(DOSE_EPOCHS) * len(MECHANISM_METRICS) * 5 * 5:
        raise SummaryValidationError("dose-response trajectory cardinality mismatch")
    trajectory_path = stage / "surface_gate_dose_response_trajectory.csv"
    summary_path = stage / "surface_gate_dose_response_summary.csv"
    write_csv(trajectory_path, trajectory_rows, [
        "arm", "training_seed", "epoch", "noise_level", "metric", "gate_alpha",
        "mean_value", "n_complexes",
    ])
    write_csv(summary_path, summary_rows, [
        "arm", "training_seed", "epoch", "noise_level", "metric", "alpha_0",
        "alpha_0_25", "alpha_0_5", "alpha_0_75", "alpha_1",
        "delta_alpha1_minus_alpha0", "consecutive_nondecreasing_steps",
        "monotonic_seed_mean", "pearson_alpha_value", "n_complete_complexes",
        "complex_monotonic_fraction", "endpoint_status",
    ])
    promotions.extend([
        (trajectory_path, root / "07_statistics/surface_gate_dose_response_trajectory.csv"),
        (summary_path, root / "07_statistics/surface_gate_dose_response_summary.csv"),
    ])
    return summary_rows


def aggregate_effect_trajectory(mapping: dict, family: str, names: list[str], paired: bool) -> list[dict]:
    rows = []
    arms = (None,) if paired else ARMS
    for arm in arms:
        for seed in SEEDS:
            for epoch in EPOCHS:
                for metric in MECHANISM_METRICS:
                    for noise in (*NOISE_LEVELS, "all"):
                        observations = []
                        for name in names:
                            if noise == "all":
                                prefix = (seed, epoch, name) if paired else (arm, seed, epoch, name)
                                value = pool_effect_over_noise(mapping, prefix, metric)
                            else:
                                key = (
                                    (seed, epoch, name, noise, metric)
                                    if paired else (arm, seed, epoch, name, noise, metric)
                                )
                                value = mapping.get(key, math.nan)
                            if math.isfinite(value):
                                observations.append(value)
                        rows.append({
                            "family": family,
                            "arm_or_contrast": (
                                "Transfer-All-minus-Transfer-NonSurface" if paired else arm
                            ),
                            "training_seed": seed,
                            "epoch": epoch,
                            "noise_level": noise,
                            "metric": metric,
                            "value": finite_mean(observations),
                            "n_complexes": len(observations),
                        })
    return rows


_BOOTSTRAP_PLAN_CACHE: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}


def bootstrap_plan(replicates: int, n_complexes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    key = (replicates, n_complexes, seed)
    if key not in _BOOTSTRAP_PLAN_CACHE:
        rng = np.random.default_rng(seed + 7919 * n_complexes)
        seed_draws = rng.integers(0, len(SEEDS), size=(replicates, len(SEEDS)), dtype=np.int8)
        complex_draws = rng.integers(
            0, n_complexes,
            size=(len(SEEDS), len(SEEDS), replicates, n_complexes),
            dtype=np.int16,
        )
        _BOOTSTRAP_PLAN_CACHE[key] = seed_draws, complex_draws
    return _BOOTSTRAP_PLAN_CACHE[key]


def percentile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def hierarchical_bootstrap(
    mapping: dict[tuple[int, str], float], names: list[str], *, family: str, contrast: str,
    metric: str, endpoint: str, epoch: int | None, noise: float | str | None,
    effect_definition: str, higher_is_better: bool, replicates: int, seed: int,
) -> tuple[dict, list[dict], list[dict]]:
    valid_names = [
        name for name in names
        if all(math.isfinite(float(mapping.get((training_seed, name), math.nan))) for training_seed in SEEDS)
    ]
    if not valid_names:
        raise SummaryValidationError(f"no jointly valid complexes for bootstrap {family}/{metric}/{endpoint}")
    matrix = np.asarray([
        [float(mapping[(training_seed, name)]) for name in valid_names]
        for training_seed in SEEDS
    ], dtype=np.float64)
    seed_effects = matrix.mean(axis=1)
    point = float(seed_effects.mean())
    seed_draws, complex_draws = bootstrap_plan(replicates, len(valid_names), seed)
    replicate_seed_means = np.empty((replicates, len(SEEDS)), dtype=np.float64)
    for slot in range(len(SEEDS)):
        for source_seed_index in range(len(SEEDS)):
            selected = seed_draws[:, slot] == source_seed_index
            indices = complex_draws[slot, source_seed_index, selected]
            replicate_seed_means[selected, slot] = matrix[source_seed_index, indices].mean(axis=1)
    bootstrap_values = replicate_seed_means.mean(axis=1)
    raw_low = percentile(bootstrap_values, 0.025)
    raw_high = percentile(bootstrap_values, 0.975)
    multiplier = 1.0 if higher_is_better else -1.0
    favorable_point = multiplier * point
    favorable_low = raw_low if higher_is_better else -raw_high
    favorable_high = raw_high if higher_is_better else -raw_low
    favorable_seed_effects = multiplier * seed_effects
    result = {
        "family": family,
        "contrast": contrast,
        "metric": metric,
        "endpoint": endpoint,
        "epoch": "" if epoch is None else epoch,
        "noise_level": "" if noise is None else noise,
        "effect_definition": effect_definition,
        "higher_is_better": higher_is_better,
        "raw_estimate": point,
        "raw_ci_low": raw_low,
        "raw_ci_high": raw_high,
        "favorable_estimate": favorable_point,
        "favorable_ci_low": favorable_low,
        "favorable_ci_high": favorable_high,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "n_training_seeds": len(SEEDS),
        "n_paired_complexes": len(valid_names),
        "favorable_seed_directions": int(np.sum(favorable_seed_effects > 0)),
        "zero_seed_directions": int(np.sum(favorable_seed_effects == 0)),
    }
    seed_rows = [
        {
            "family": family, "contrast": contrast, "metric": metric, "endpoint": endpoint,
            "epoch": "" if epoch is None else epoch, "noise_level": "" if noise is None else noise,
            "training_seed": training_seed, "raw_effect": float(seed_effects[index]),
            "favorable_effect": float(favorable_seed_effects[index]),
            "n_paired_complexes": len(valid_names),
        }
        for index, training_seed in enumerate(SEEDS)
    ]
    complex_rows = [
        {
            "family": family, "contrast": contrast, "metric": metric, "endpoint": endpoint,
            "epoch": "" if epoch is None else epoch, "noise_level": "" if noise is None else noise,
            "training_seed": training_seed, "complex_id": name,
            "raw_effect": float(matrix[seed_index, complex_index]),
            "favorable_effect": float(multiplier * matrix[seed_index, complex_index]),
        }
        for seed_index, training_seed in enumerate(SEEDS)
        for complex_index, name in enumerate(valid_names)
    ]
    return result, seed_rows, complex_rows


def load_and_copy_test200(
    test_root: Path, stage: Path, root: Path, promotions: list[tuple[Path, Path]], completeness: list[dict],
) -> tuple[dict, list[dict], list[dict]]:
    ready_path = test_root / "FROZEN_PRE_EVALUATION_READY.json"
    if not ready_path.is_file():
        raise SummaryValidationError(f"missing frozen test200 READY contract: {ready_path}")
    source_hashes = {"FROZEN_PRE_EVALUATION_READY.json": sha256_file(ready_path)}
    source_rows: dict[str, list[dict]] = {}
    for source_name, expected_hash in EXPECTED_TEST200_HASHES.items():
        source = test_root / "statistics" / source_name
        observed_hash = sha256_file(source) if source.is_file() else ""
        completeness.append({
            "unit_type": "test200_frozen_csv", "arm": "all", "training_seed": "all", "epoch": 200,
            "path": str(source), "exists": source.is_file(),
            "status": "valid" if observed_hash == expected_hash else "invalid",
            "row_count": "", "expected_row_count": "", "sha256": observed_hash,
            "reason": "" if observed_hash == expected_hash else f"expected_sha256={expected_hash}",
        })
        if observed_hash != expected_hash:
            raise SummaryValidationError(
                f"frozen test200 CSV hash mismatch: {source_name} {observed_hash} != {expected_hash}"
            )
        with source.open(newline="") as handle:
            source_rows[source_name] = list(csv.DictReader(handle))
        source_hashes[source_name] = observed_hash
        staged_copy = stage / TEST200_DESTINATIONS[source_name]
        shutil.copy2(source, staged_copy)
        if sha256_file(staged_copy) != observed_hash:
            raise SummaryValidationError(f"test200 byte-copy hash mismatch: {source_name}")
        promotions.append((
            staged_copy,
            root / "06_validation_trajectory" / TEST200_DESTINATIONS[source_name],
        ))

    pose_rows = source_rows["test_pose_level_metrics.csv"]
    complex_rows = source_rows["test_complex_level_metrics.csv"]
    summary_rows = source_rows["test_seed_level_summary.csv"]
    if len(pose_rows) != 9 * 1280 or len(complex_rows) != 9 * 128 or len(summary_rows) != 9 * 5:
        raise SummaryValidationError(
            f"frozen test200 cardinality mismatch: pose={len(pose_rows)}, complex={len(complex_rows)}, "
            f"summary={len(summary_rows)}"
        )
    pose_keys = set()
    complexes_by_run: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in pose_rows:
        arm = row["arm"]
        seed = int(row["training_seed"])
        name = row["complex_name"]
        sample = int(row["sample_idx"])
        rmsd = parse_finite(row["rmsd"], field="test200 rmsd", nonnegative=True)
        key = (arm, seed, name, sample)
        if arm not in ARMS or seed not in SEEDS or sample not in range(10) or key in pose_keys:
            raise SummaryValidationError(f"invalid/duplicate frozen test200 pose key: {key}")
        if parse_bool(row["rmsd_lt2"], field="test200 rmsd_lt2") != (rmsd < 2.0):
            raise SummaryValidationError(f"frozen test200 threshold-label mismatch: {key}")
        pose_keys.add(key)
        complexes_by_run[(arm, seed)].add(name)
    if any(len(values) != 128 for values in complexes_by_run.values()) or len(complexes_by_run) != 9:
        raise SummaryValidationError("frozen test200 pose CSV does not contain nine exact 128-complex runs")

    complex_lookup = {}
    for row in complex_rows:
        key = (row["arm"], int(row["training_seed"]), row["complex_name"])
        if key in complex_lookup or key[0] not in ARMS or key[1] not in SEEDS:
            raise SummaryValidationError(f"invalid/duplicate frozen test200 complex key: {key}")
        complex_lookup[key] = {
            "pose_density_lt2": parse_finite(row["pose_density_lt2"], field="test200 pose density"),
            "best_of_k10_lt2": float(parse_bool(row["best_of_k10_lt2"], field="test200 coverage")),
            "best_rmsd": parse_finite(row["best_rmsd"], field="test200 best RMSD", nonnegative=True),
        }
    for arm in ARMS:
        for seed in SEEDS:
            if {key[2] for key in complex_lookup if key[:2] == (arm, seed)} != complexes_by_run[(arm, seed)]:
                raise SummaryValidationError(f"test200 pose/complex identity mismatch: {arm}/seed{seed}")
    normalized_complex_rows = [
        {
            "arm": arm,
            "training_seed": seed,
            "complex_name": name,
            **metrics,
        }
        for (arm, seed, name), metrics in sorted(complex_lookup.items())
    ]
    return source_hashes, normalized_complex_rows, summary_rows


def add_bootstrap(
    outputs: tuple[list[dict], list[dict], list[dict]], mapping: dict, names: list[str], **kwargs,
) -> None:
    result, seed_rows, complex_rows = hierarchical_bootstrap(mapping, names, **kwargs)
    outputs[0].append(result)
    outputs[1].extend(seed_rows)
    outputs[2].extend(complex_rows)


def build_bootstrap_outputs(
    task_complex: dict, task_aulc: dict, names: list[str], surface: dict, chemistry: dict,
    delta_surface: dict, mediated_chemistry: dict, effect_aulc: dict,
    test_complex_rows: list[dict], replicates: int, seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    results: list[dict] = []
    seed_rows: list[dict] = []
    complex_rows: list[dict] = []
    outputs = (results, seed_rows, complex_rows)

    for metric in TASK_METRICS:
        higher = metric != "best_rmsd"
        for epoch in (25, 200):
            mapping = {
                (training_seed, name): (
                    task_complex[("Transfer-All", training_seed, epoch, name)][metric]
                    - task_complex[("Transfer-NonSurface", training_seed, epoch, name)][metric]
                )
                for training_seed in SEEDS for name in names
            }
            add_bootstrap(
                outputs, mapping, names, family="validation_task",
                contrast="Transfer-All-minus-Transfer-NonSurface", metric=metric,
                endpoint=f"epoch_{epoch}", epoch=epoch, noise=None,
                effect_definition="Transfer-All minus Transfer-NonSurface",
                higher_is_better=higher, replicates=replicates, seed=seed,
            )
        mapping = {
            (training_seed, name): (
                task_aulc[("Transfer-All", training_seed, name, metric)]
                - task_aulc[("Transfer-NonSurface", training_seed, name, metric)]
            )
            for training_seed in SEEDS for name in names
        }
        add_bootstrap(
            outputs, mapping, names, family="validation_task",
            contrast="Transfer-All-minus-Transfer-NonSurface", metric=metric,
            endpoint="AULC_25_200", epoch=None, noise=None,
            effect_definition="AULC(Transfer-All) minus AULC(Transfer-NonSurface)",
            higher_is_better=higher, replicates=replicates, seed=seed,
        )

    mechanism_families = (
        ("G_surface", surface, "Transfer-All", "A(alpha=1)-A(alpha=0) in Transfer-All"),
        ("G_surface", surface, "Transfer-NonSurface", "A(alpha=1)-A(alpha=0) in Transfer-NonSurface"),
        ("DeltaG_surface", delta_surface, None, "G_surface(Transfer-All)-G_surface(Transfer-NonSurface)"),
        ("G_chem", chemistry, "Transfer-All", "A(true)-mean[A(shuffle)] in Transfer-All"),
        ("G_chem", chemistry, "Transfer-NonSurface", "A(true)-mean[A(shuffle)] in Transfer-NonSurface"),
        ("M_chem", mediated_chemistry, None, "G_chem(Transfer-All)-G_chem(Transfer-NonSurface)"),
    )
    for family, mapping_source, arm, definition in mechanism_families:
        contrast = arm or "Transfer-All-minus-Transfer-NonSurface"
        for metric in MECHANISM_METRICS:
            for epoch in (25, 200):
                mapping = {}
                for training_seed in SEEDS:
                    for name in names:
                        prefix = (
                            (training_seed, epoch, name)
                            if arm is None else (arm, training_seed, epoch, name)
                        )
                        mapping[(training_seed, name)] = pool_effect_over_noise(
                            mapping_source, prefix, metric
                        )
                add_bootstrap(
                    outputs, mapping, names, family=family, contrast=contrast, metric=metric,
                    endpoint=f"epoch_{epoch}", epoch=epoch, noise="all",
                    effect_definition=definition, higher_is_better=True,
                    replicates=replicates, seed=seed,
                )
            aulc_arm = "TA-minus-TN" if arm is None else arm
            mapping = {
                (training_seed, name): effect_aulc.get(
                    (family, aulc_arm, training_seed, name, metric), math.nan
                )
                for training_seed in SEEDS for name in names
            }
            add_bootstrap(
                outputs, mapping, names, family=family, contrast=contrast, metric=metric,
                endpoint="AULC_25_200", epoch=None, noise="all",
                effect_definition=f"AULC of {definition}", higher_is_better=True,
                replicates=replicates, seed=seed,
            )

    for family, mapping_source, definition in (
        ("DeltaG_surface_epoch_x_noise", delta_surface, "paired Surface-path interaction"),
        ("M_chem_epoch_x_noise", mediated_chemistry, "paired shuffled-chemistry sensitivity interaction"),
    ):
        for metric in ("combined_alignment", "delta_RMSD_1step"):
            for epoch in EPOCHS:
                for noise in NOISE_LEVELS:
                    mapping = {
                        (training_seed, name): mapping_source.get(
                            (training_seed, epoch, name, noise, metric), math.nan
                        )
                        for training_seed in SEEDS for name in names
                    }
                    add_bootstrap(
                        outputs, mapping, names, family=family,
                        contrast="Transfer-All-minus-Transfer-NonSurface", metric=metric,
                        endpoint="epoch_x_noise", epoch=epoch, noise=noise,
                        effect_definition=definition, higher_is_better=True,
                        replicates=replicates, seed=seed,
                    )

    test_lookup = {
        (row["arm"], int(row["training_seed"]), row["complex_name"]): row
        for row in test_complex_rows
    }
    test_names = sorted({row["complex_name"] for row in test_complex_rows})
    for metric in ("pose_density_lt2", "best_of_k10_lt2", "best_rmsd"):
        higher = metric != "best_rmsd"
        mapping = {
            (training_seed, name): (
                float(test_lookup[("Transfer-All", training_seed, name)][metric])
                - float(test_lookup[("Transfer-NonSurface", training_seed, name)][metric])
            )
            for training_seed in SEEDS for name in test_names
        }
        add_bootstrap(
            outputs, mapping, test_names, family="test200_task",
            contrast="Transfer-All-minus-Transfer-NonSurface", metric=metric,
            endpoint="fixed_EMA_epoch_200", epoch=200, noise=None,
            effect_definition="frozen test128 Transfer-All minus Transfer-NonSurface",
            higher_is_better=higher, replicates=replicates, seed=seed,
        )
    return outputs


def descriptive_rows(
    task_complex: dict, task_aulc: dict, names: list[str], surface: dict, chemistry: dict,
    effect_aulc: dict, test_summary_rows: list[dict],
) -> list[dict]:
    rows = []
    for metric in TASK_METRICS:
        for endpoint, epoch in (("epoch_25", 25), ("epoch_200", 200)):
            seed_values = []
            for seed in SEEDS:
                value = finite_mean(task_complex[("Scratch-Full8", seed, epoch, name)][metric] for name in names)
                seed_values.append(value)
                rows.append({
                    "domain": "validation", "family": "task", "metric": metric,
                    "endpoint": endpoint, "training_seed": seed, "value": value,
                    "standard_deviation": "", "n_complexes": len(names),
                })
            rows.append({
                "domain": "validation", "family": "task", "metric": metric,
                "endpoint": endpoint, "training_seed": "all_seed_mean", "value": finite_mean(seed_values),
                "standard_deviation": statistics.stdev(seed_values), "n_complexes": len(names),
            })
        seed_values = []
        for seed in SEEDS:
            value = finite_mean(task_aulc[("Scratch-Full8", seed, name, metric)] for name in names)
            seed_values.append(value)
            rows.append({
                "domain": "validation", "family": "task", "metric": metric,
                "endpoint": "AULC_25_200", "training_seed": seed, "value": value,
                "standard_deviation": "", "n_complexes": len(names),
            })
        rows.append({
            "domain": "validation", "family": "task", "metric": metric,
            "endpoint": "AULC_25_200", "training_seed": "all_seed_mean", "value": finite_mean(seed_values),
            "standard_deviation": statistics.stdev(seed_values), "n_complexes": len(names),
        })

    for family, mapping in (("G_surface", surface), ("G_chem", chemistry)):
        for metric in MECHANISM_METRICS:
            for endpoint, epoch in (("epoch_25", 25), ("epoch_200", 200)):
                seed_values = []
                for seed in SEEDS:
                    value = finite_mean(
                        pool_effect_over_noise(mapping, ("Scratch-Full8", seed, epoch, name), metric)
                        for name in names
                    )
                    seed_values.append(value)
                    rows.append({
                        "domain": "fixed_pose_mechanism", "family": family, "metric": metric,
                        "endpoint": endpoint, "training_seed": seed, "value": value,
                        "standard_deviation": "", "n_complexes": len(names),
                    })
                rows.append({
                    "domain": "fixed_pose_mechanism", "family": family, "metric": metric,
                    "endpoint": endpoint, "training_seed": "all_seed_mean", "value": finite_mean(seed_values),
                    "standard_deviation": statistics.stdev(seed_values), "n_complexes": len(names),
                })
            seed_values = []
            for seed in SEEDS:
                values = [
                    effect_aulc.get((family, "Scratch-Full8", seed, name, metric), math.nan)
                    for name in names
                ]
                value = finite_mean(values)
                seed_values.append(value)
                rows.append({
                    "domain": "fixed_pose_mechanism", "family": family, "metric": metric,
                    "endpoint": "AULC_25_200", "training_seed": seed, "value": value,
                    "standard_deviation": "", "n_complexes": sum(math.isfinite(v) for v in values),
                })
            rows.append({
                "domain": "fixed_pose_mechanism", "family": family, "metric": metric,
                "endpoint": "AULC_25_200", "training_seed": "all_seed_mean", "value": finite_mean(seed_values),
                "standard_deviation": statistics.stdev(seed_values), "n_complexes": len(names),
            })

    for row in test_summary_rows:
        if row["arm"] == "Scratch-Full8" and row["subset"] == "All128":
            for metric in (
                "complex_level_pose_density_lt2_pct", "best_of_k10_coverage_lt2_pct",
                "median_best_rmsd", "mean_best_rmsd",
            ):
                rows.append({
                    "domain": "frozen_test128", "family": "task", "metric": metric,
                    "endpoint": "fixed_EMA_epoch_200", "training_seed": int(row["training_seed"]),
                    "value": float(row[metric]), "standard_deviation": "",
                    "n_complexes": int(row["n_complexes"]),
                })
    return rows


def inference_status(row: dict) -> str:
    if float(row["favorable_ci_low"]) > 0 and int(row["favorable_seed_directions"]) == 3:
        return "supported"
    if float(row["favorable_ci_high"]) < 0:
        return "not supported"
    return "inconclusive"


def find_bootstrap(rows: list[dict], family: str, metric: str, endpoint: str, contrast: str | None = None) -> dict:
    matches = [
        row for row in rows
        if row["family"] == family and row["metric"] == metric and row["endpoint"] == endpoint
        and (contrast is None or row["contrast"] == contrast)
    ]
    if len(matches) != 1:
        raise SummaryValidationError(
            f"expected one bootstrap result for {family}/{metric}/{endpoint}/{contrast}, found {len(matches)}"
        )
    return matches[0]


def fmt_effect(row: dict) -> str:
    return (
        f"{float(row['favorable_estimate']):.4f} "
        f"[{float(row['favorable_ci_low']):.4f}, {float(row['favorable_ci_high']):.4f}]"
    )


def build_report(
    path: Path, root: Path, completeness: list[dict], bootstrap_rows: list[dict],
    validation_seed_rows: list[dict], test_summary_rows: list[dict], source_hashes: dict,
    pose_bank_completion: dict, stability: dict, dose_response_rows: list[dict],
    recompute_manifest_sha: str,
    wall_seconds: float,
) -> None:
    task_200 = find_bootstrap(bootstrap_rows, "validation_task", "pose_density_lt2", "epoch_200")
    task_aulc = find_bootstrap(bootstrap_rows, "validation_task", "pose_density_lt2", "AULC_25_200")
    test_200 = find_bootstrap(bootstrap_rows, "test200_task", "pose_density_lt2", "fixed_EMA_epoch_200")
    surface_25 = find_bootstrap(
        bootstrap_rows, "DeltaG_surface", "combined_alignment", "epoch_25",
        "Transfer-All-minus-Transfer-NonSurface",
    )
    surface_aulc = find_bootstrap(
        bootstrap_rows, "DeltaG_surface", "combined_alignment", "AULC_25_200",
        "Transfer-All-minus-Transfer-NonSurface",
    )
    surface_200 = find_bootstrap(
        bootstrap_rows, "DeltaG_surface", "combined_alignment", "epoch_200",
        "Transfer-All-minus-Transfer-NonSurface",
    )
    chemistry_25 = find_bootstrap(
        bootstrap_rows, "M_chem", "combined_alignment", "epoch_25",
        "Transfer-All-minus-Transfer-NonSurface",
    )
    chemistry_aulc = find_bootstrap(
        bootstrap_rows, "M_chem", "combined_alignment", "AULC_25_200",
        "Transfer-All-minus-Transfer-NonSurface",
    )
    chemistry_200 = find_bootstrap(
        bootstrap_rows, "M_chem", "combined_alignment", "epoch_200",
        "Transfer-All-minus-Transfer-NonSurface",
    )
    status_h1 = inference_status(test_200)
    status_h2 = inference_status(task_aulc)
    status_h3 = inference_status(surface_aulc)
    status_h4 = inference_status(chemistry_aulc)

    task_table = []
    for arm in ARMS:
        for epoch in (25, 200):
            rows = [
                row for row in validation_seed_rows
                if row["arm"] == arm and int(row["epoch"]) == epoch
            ]
            task_table.append(
                f"| {arm} | {epoch} | "
                f"{finite_mean(row['pose_density_lt2'] for row in rows):.4f} | "
                f"{finite_mean(row['best_of_k10_coverage_lt2'] for row in rows):.4f} | "
                f"{finite_mean(row['median_best_rmsd'] for row in rows):.4f} |"
            )
    test_table = []
    for row in test_summary_rows:
        if row["subset"] == "All128":
            test_table.append(
                f"| {row['arm']} | {row['training_seed']} | "
                f"{float(row['complex_level_pose_density_lt2_pct']):.3f}% | "
                f"{float(row['best_of_k10_coverage_lt2_pct']):.3f}% | "
                f"{float(row['median_best_rmsd']):.3f} |"
            )
    invalid_units = [row for row in completeness if row["status"] != "valid"]
    stability_table = []
    for arm in ARMS:
        row = stability["by_arm"][arm]
        stability_table.append(
            f"| {arm} | {int(row['n_rows'])} | "
            f"{float(row['rate_RMSD_gt_100A']):.6f} | "
            f"{float(row['rate_RMSD_gt_1000A']):.6f} | "
            f"{float(row['rate_RMSD_gt_1000000A']):.6f} | "
            f"{float(row['rate_RMSD_gt_1000000000000A']):.6f} | "
            f"{float(row['max_RMSD_after']):.6e} |"
        )
    dose_table = []
    for arm in ARMS:
        for epoch in DOSE_EPOCHS:
            selected = [
                row for row in dose_response_rows
                if row["arm"] == arm and int(row["epoch"]) == epoch
                and row["noise_level"] == "all"
                and row["metric"] == "combined_alignment"
            ]
            if len(selected) != 3:
                raise SummaryValidationError(f"missing three-seed dose summary: {arm}/epoch{epoch}")
            dose_table.append(
                f"| {arm} | {epoch} | "
                f"{finite_mean(float(row['delta_alpha1_minus_alpha0']) for row in selected):.4f} | "
                f"{sum(parse_bool(row['monotonic_seed_mean'], field='dose monotonic seed mean') for row in selected)}/3 | "
                f"{finite_mean(float(row['complex_monotonic_fraction']) for row in selected):.4f} |"
            )
    report = f"""# SurfNA Low-Budget Dynamic Surface-Transfer Mechanism Report

Generated: {datetime.now(timezone.utc).isoformat()}

## Reproducibility and completeness

- Matrix: 3 arms x 3 training seeds x 8 fixed epochs (25--200) = 72 validation units and 72 fixed-pose mechanism units.
- Epoch 0 was retained only as a failed diagnostic initialization audit and is excluded from formal statistics.
- Validation cohort: 89 complexes, K=10, fixed inference protocol.
- Mechanism bank: 89 x 4 noise levels x 5 pose repeats = 1,780 immutable entries.
- Primary inferential unit: complex. Poses are averaged within complex; chemistry shuffles are averaged within the same pose/complex before contrasts.
- Inference: paired hierarchical bootstrap over training seed and then paired complexes; 10,000 replicates, seed 20260903, percentile 95% CI.
- Test role: frozen external EMA-200 endpoint only; no checkpoint selection or analysis adaptation used test128.
- Valid source units: {sum(row['status'] == 'valid' for row in completeness)}/{len(completeness)}; invalid units: {len(invalid_units)}.
- Summary wall time: {wall_seconds:.1f} seconds.
- Pose-bank SHA-256: `{pose_bank_completion['bank_sha256']}`.
- Frozen test200 source hashes: `{json.dumps(source_hashes, sort_keys=True)}`.
- v6 recompute input manifest SHA-256: `{recompute_manifest_sha}`.

## Numerical-QC repair and one-step stability

The parent v5 model predictions, zero-z one-step updates, stored updated coordinates, and immutable legacy float32 pose-bank `RMSD_before` values were reused byte-for-byte. Alignment, after-RMSD, direct-RMSD, and delta scalars were regenerated with float64 arithmetic for all 72 units; `delta_RMSD_1step` therefore deliberately combines the frozen legacy baseline with a float64 after-RMSD. There was no model re-inference, scientific-output re-integration, row deletion, clipping, or trust-region intervention. A deterministic integration replay was performed only as an integrity check and its coordinates were never used to replace the stored parent coordinates. Coordinate replay, cache/model-parameter/parent hashes, key-space completeness, arithmetic identities, and derived-finiteness gates passed before this report was generated.

The divergence endpoint was added after the v5 summary failure and is therefore **post-failure descriptive QC**, not an original confirmatory endpoint. Its denominator is every valid fixed-pose one-step row; lower is more stable.

| Arm | Rows | RMSD>100 Å | RMSD>1,000 Å | RMSD>10^6 Å | RMSD>10^12 Å | Maximum RMSD |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(stability_table)}

These rows are retained rather than clipped. Consequently, extreme Scratch-Full8 updates remain part of the numerical-stability result and may dominate arithmetic means involving one-step RMSD; the bounded score-alignment endpoints should be interpreted separately.

## Nine-model validation task trajectory

Values below are means over the three complete seeds. Fractions use a 0–1 scale.

| Arm | Epoch | Pose density <2 Å | Best-of-K10 coverage <2 Å | Median best RMSD |
|---|---:|---:|---:|---:|
{chr(10).join(task_table)}

Paired Transfer-All minus Transfer-NonSurface effects (reported in the favorable direction):

| Endpoint | Pose-density effect [95% CI] | Direction-consistent seeds |
|---|---:|---:|
| Validation epoch 200 | {fmt_effect(task_200)} | {task_200['favorable_seed_directions']}/3 |
| Validation AULC 25–200 | {fmt_effect(task_aulc)} | {task_aulc['favorable_seed_directions']}/3 |
| Frozen test128 EMA200 | {fmt_effect(test_200)} | {test_200['favorable_seed_directions']}/3 |

## Frozen test128 EMA-200 endpoint

The three pre-existing CSVs were reused byte-for-byte and pinned by their pre-frozen SHA-256 values.

| Arm | Seed | Pose density <2 Å | Best-of-K10 coverage <2 Å | Median best RMSD |
|---|---:|---:|---:|---:|
{chr(10).join(test_table)}

## Surface-gate mechanism

`G_surface = A(alpha=1) - A(alpha=0)` and `DeltaG_surface = G_surface(Transfer-All) - G_surface(Transfer-NonSurface)`.

| Endpoint | DeltaG_surface combined-alignment [95% CI] | Seeds in favorable direction |
|---|---:|---:|
| Epoch 25 | {fmt_effect(surface_25)} | {surface_25['favorable_seed_directions']}/3 |
| AULC 25–200 | {fmt_effect(surface_aulc)} | {surface_aulc['favorable_seed_directions']}/3 |
| Epoch 200 | {fmt_effect(surface_200)} | {surface_200['favorable_seed_directions']}/3 |

Full `G_surface` results for both transfer arms, Scratch-Full8, all seven metrics, all epochs, and each noise level are retained in the machine-readable trajectory and bootstrap tables.

### Five-level Surface-gate dose response

The preregistered alpha scan (0, 0.25, 0.5, 0.75, 1) is summarized cross-sectionally at the three available checkpoints. EMA0 remains unavailable because its functional output failed the frozen initialization audit; no epoch-0 value was imputed. Monotonicity is descriptive and is not a hard acceptance gate.

| Arm | Epoch | Mean A(1)-A(0), combined alignment | Monotonic seed means | Mean fraction of monotonic complexes |
|---|---:|---:|---:|---:|
{chr(10).join(dose_table)}

## Chemistry counterfactual sensitivity

`G_chem = A(true chemistry) - mean[A(three joint shuffles)]` and `M_chem = G_chem(Transfer-All) - G_chem(Transfer-NonSurface)`. This difference-in-differences measures transfer-specific reliance on correctly localized chemistry fields; it is not a formal causal mediation decomposition.

| Endpoint | M_chem combined-alignment [95% CI] | Seeds in favorable direction |
|---|---:|---:|
| Epoch 25 | {fmt_effect(chemistry_25)} | {chemistry_25['favorable_seed_directions']}/3 |
| AULC 25–200 | {fmt_effect(chemistry_aulc)} | {chemistry_aulc['favorable_seed_directions']}/3 |
| Epoch 200 | {fmt_effect(chemistry_200)} | {chemistry_200['favorable_seed_directions']}/3 |

The joint chemistry vector is the intervention unit. Three fixed shuffles are averaged inside each complex before any seed-level or population-level inference.

## Epoch x noise localization

All 8 formal epochs x 4 noise levels are reported for both paired interactions and for combined alignment plus one-step RMSD improvement. These are descriptive localization results; no single heat-map cell is used for checkpoint selection or as the sole basis for a mechanistic claim.

## Scratch-Full8 descriptive reference

Scratch-Full8 has all three seeds throughout the validation trajectory, mechanism conditions, AULC, and frozen test endpoint. It is reported descriptively and is not substituted for the prespecified paired Transfer-All versus Transfer-NonSurface contrast.

## Statistical conclusions

- H1 fixed-budget task effect on frozen test128: **{status_h1}**.
- H2 validation adaptation AULC effect: **{status_h2}**.
- H3 transfer-specific Surface-path contribution: **{status_h3}**.
- H4 transfer-specific shuffled-chemistry sensitivity/feature-reliance interaction: **{status_h4}**.
- H5 noise-stage localization: **descriptive result**; see the epoch-by-noise bootstrap rows.

## Strongest defensible claim

The strongest claim must be bounded by the statuses above. Evidence consistent with a transfer-specific physicochemical mechanism requires the paired task trajectory, `DeltaG_surface`, and `M_chem` estimates to be directionally consistent with hierarchical-bootstrap intervals excluding zero. If task transfer is supported but shuffled-chemistry sensitivity is inconclusive, the defensible claim is limited to a transferable Surface pathway and cannot be attributed specifically to the shuffled physicochemical channels. The chemistry intervention supports feature-reliance attribution but does not by itself establish formal causal mediation. These analyses do not establish prospective virtual-screening performance, affinity prediction, or causality outside the frozen L2 validation/test cohorts.

## Output notes

- No figures were generated by this summary job.
- Every pose/repeat remains in the long tables; no unfavorable complex was removed.
- Raw effect, favorable-direction effect, all seed effects, complex effects, valid complex counts, and 95% intervals are available in `07_statistics`.
- Post-failure stability counts and rates are available in `07_statistics/one_step_stability_endpoints.csv` and `one_step_stability_summary.json`.
- Five-level gate-dose results are available in `07_statistics/surface_gate_dose_response_trajectory.csv` and `surface_gate_dose_response_summary.csv`; EMA0 is explicitly unavailable.
"""
    path.write_text(report)


def write_failure_attempt(root: Path, completeness: list[dict], error: Exception, started: float) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempt = root / "07_statistics/summary_attempts" / f"failed_{timestamp}_{os.getpid()}"
    attempt.mkdir(parents=True, exist_ok=False)
    write_csv(attempt / "failure_summary.csv", [{
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "wall_seconds": time.time() - started,
    }], ["status", "error_type", "error", "wall_seconds"])
    atomic_json(attempt / "completeness_report.json", {
        "schema_version": "surfna-lb200-dynamic-summary-completeness-v1",
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "units": completeness,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })


def run(args: argparse.Namespace, completeness: list[dict]) -> dict:
    root = args.analysis_root.resolve()
    test_root = args.test_root.resolve()
    started = time.time()
    recompute_manifest, recompute_matrix, recompute_units, recompute_manifest_sha = (
        load_v6_recompute_contract(root)
    )
    validation_paths, mechanism_paths = precheck_unit_inventory(root, completeness)
    names, cache_contract = load_frozen_validation_contract(root)
    bank_entries, pose_bank_completion = load_pose_bank_manifest(root, names)
    expected_validation_units = {}
    for unit in recompute_manifest["validation_units"]:
        key = (unit["arm"], int(unit["training_seed"]), int(unit["epoch"]))
        if key in expected_validation_units:
            raise SummaryValidationError(f"duplicate frozen validation unit: {key}")
        expected_validation_units[key] = unit
    if len(expected_validation_units) != 72:
        raise SummaryValidationError("frozen validation input matrix is not 72 units")

    final_destinations = [
        root / "06_validation_trajectory/nine_model_validation_long.csv",
        root / "06_validation_trajectory/nine_model_validation_complex_trajectory.csv",
        root / "06_validation_trajectory/nine_model_validation_seed_trajectory.csv",
        root / "06_validation_trajectory/nine_model_validation_AULC.csv",
        root / "06_validation_trajectory/test200_pose_level.csv",
        root / "06_validation_trajectory/test200_complex_level.csv",
        root / "06_validation_trajectory/test200_summary.csv",
        root / "07_statistics/nine_model_mechanism_long.csv",
        root / "07_statistics/nine_model_mechanism_complex_trajectory.csv",
        root / "07_statistics/nine_model_mechanism_seed_trajectory.csv",
        root / "07_statistics/nine_model_mechanism_AULC.csv",
        root / "07_statistics/nine_model_mechanism_effect_complex.csv",
        root / "07_statistics/paired_mechanism_effect_complex.csv",
        root / "07_statistics/mechanism_effect_AULC.csv",
        root / "07_statistics/G_surface_trajectory.csv",
        root / "07_statistics/G_chem_trajectory.csv",
        root / "07_statistics/DeltaG_surface_trajectory.csv",
        root / "07_statistics/M_chem_trajectory.csv",
        root / "07_statistics/hierarchical_bootstrap_effects.csv",
        root / "07_statistics/seed_level_effects.csv",
        root / "07_statistics/complex_level_effects.csv",
        root / "07_statistics/scratch_descriptive.csv",
        root / "07_statistics/one_step_stability_endpoints.csv",
        root / "07_statistics/one_step_stability_summary.json",
        root / "07_statistics/surface_gate_dose_response_trajectory.csv",
        root / "07_statistics/surface_gate_dose_response_summary.csv",
        root / "07_statistics/test200_frozen_sources.json",
        root / "07_statistics/failure_summary.csv",
        root / "07_statistics/completeness_report.json",
        root / "FINAL_DYNAMIC_MECHANISM_REPORT.md",
        root / "07_statistics/DYNAMIC_SUMMARY_COMPLETE.json",
    ]
    collisions = [path for path in final_destinations if path.exists()]
    if collisions:
        raise SummaryValidationError(f"refusing to overwrite existing summary outputs: {collisions[:5]}")

    statistics_root = root / "07_statistics"
    statistics_root.mkdir(parents=True, exist_ok=True)
    promotions: list[tuple[Path, Path]] = []
    with tempfile.TemporaryDirectory(prefix=".dynamic_summary_stage_", dir=statistics_root) as temporary:
        stage = Path(temporary)
        validation_long = stage / "validation_long.csv"
        task_complex = parse_validation_units(
            validation_paths, names, cache_contract, validation_long, completeness,
            recompute_manifest["validation_hotfix_manifest_sha256"],
            expected_validation_units,
        )
        promotions.append((validation_long, root / "06_validation_trajectory/nine_model_validation_long.csv"))
        task_aulc, validation_seed_rows, _ = write_task_summaries(
            stage, task_complex, names, promotions, root
        )

        mechanism_long = stage / "mechanism_long.csv"
        mechanism_complex = stage / "mechanism_complex.csv"
        mechanism_values = parse_mechanism_units(
            mechanism_paths, bank_entries, names, mechanism_long, mechanism_complex, completeness,
            recompute_units, recompute_manifest_sha,
        )
        promotions.extend([
            (mechanism_long, root / "07_statistics/nine_model_mechanism_long.csv"),
            (mechanism_complex, root / "07_statistics/nine_model_mechanism_complex_trajectory.csv"),
        ])
        stability = write_stability_outputs(mechanism_long, stage, root, promotions)
        (
            mechanism_seed_rows, surface, chemistry, delta_surface,
            mediated_chemistry, mechanism_effect_aulc,
        ) = write_mechanism_summaries(stage, root, promotions, mechanism_values, names)
        dose_response_rows = write_gate_dose_response(
            stage, root, promotions, mechanism_values, names
        )

        trajectory_fields = [
            "family", "arm_or_contrast", "training_seed", "epoch", "noise_level",
            "metric", "value", "n_complexes",
        ]
        for family, mapping, paired, filename in (
            ("G_surface", surface, False, "G_surface_trajectory.csv"),
            ("G_chem", chemistry, False, "G_chem_trajectory.csv"),
            ("DeltaG_surface", delta_surface, True, "DeltaG_surface_trajectory.csv"),
            ("M_chem", mediated_chemistry, True, "M_chem_trajectory.csv"),
        ):
            rows = aggregate_effect_trajectory(mapping, family, names, paired)
            source = stage / filename
            write_csv(source, rows, trajectory_fields)
            promotions.append((source, root / "07_statistics" / filename))

        test_hashes, test_complex_rows, test_summary_rows = load_and_copy_test200(
            test_root, stage, root, promotions, completeness
        )
        bootstrap_rows, bootstrap_seed_rows, bootstrap_complex_rows = build_bootstrap_outputs(
            task_complex, task_aulc, names, surface, chemistry, delta_surface,
            mediated_chemistry, mechanism_effect_aulc, test_complex_rows,
            args.bootstrap_replicates, args.bootstrap_seed,
        )
        bootstrap_path = stage / "hierarchical_bootstrap_effects.csv"
        seed_effect_path = stage / "seed_level_effects.csv"
        complex_effect_path = stage / "complex_level_effects.csv"
        write_csv(bootstrap_path, bootstrap_rows, [
            "family", "contrast", "metric", "endpoint", "epoch", "noise_level",
            "effect_definition", "higher_is_better", "raw_estimate", "raw_ci_low", "raw_ci_high",
            "favorable_estimate", "favorable_ci_low", "favorable_ci_high", "bootstrap_replicates",
            "bootstrap_seed", "n_training_seeds", "n_paired_complexes",
            "favorable_seed_directions", "zero_seed_directions",
        ])
        write_csv(seed_effect_path, bootstrap_seed_rows, [
            "family", "contrast", "metric", "endpoint", "epoch", "noise_level",
            "training_seed", "raw_effect", "favorable_effect", "n_paired_complexes",
        ])
        write_csv(complex_effect_path, bootstrap_complex_rows, [
            "family", "contrast", "metric", "endpoint", "epoch", "noise_level",
            "training_seed", "complex_id", "raw_effect", "favorable_effect",
        ])
        promotions.extend([
            (bootstrap_path, root / "07_statistics/hierarchical_bootstrap_effects.csv"),
            (seed_effect_path, root / "07_statistics/seed_level_effects.csv"),
            (complex_effect_path, root / "07_statistics/complex_level_effects.csv"),
        ])

        scratch_rows = descriptive_rows(
            task_complex, task_aulc, names, surface, chemistry,
            mechanism_effect_aulc, test_summary_rows,
        )
        scratch_path = stage / "scratch_descriptive.csv"
        write_csv(scratch_path, scratch_rows, [
            "domain", "family", "metric", "endpoint", "training_seed", "value",
            "standard_deviation", "n_complexes",
        ])
        promotions.append((scratch_path, root / "07_statistics/scratch_descriptive.csv"))

        test_source_path = stage / "test200_frozen_sources.json"
        atomic_json(test_source_path, {
            "schema_version": "surfna-lb200-frozen-test200-reuse-v1",
            "test_root": str(test_root),
            "source_hashes": test_hashes,
            "copied_byte_for_byte": True,
            "used_for_checkpoint_selection": False,
        })
        promotions.append((test_source_path, root / "07_statistics/test200_frozen_sources.json"))

        failure_path = stage / "failure_summary.csv"
        write_csv(failure_path, [], ["unit_type", "arm", "training_seed", "epoch", "reason"])
        promotions.append((failure_path, root / "07_statistics/failure_summary.csv"))
        if any(row["status"] != "valid" for row in completeness):
            raise SummaryValidationError("one or more source units did not finish strict validation")
        completeness_path = stage / "completeness_report.json"
        atomic_json(completeness_path, {
            "schema_version": "surfna-lb200-dynamic-summary-completeness-v2-numeric-qc",
            "status": "complete",
            "expected_validation_units": 72,
            "valid_validation_units": sum(
                row["unit_type"] == "validation" and row["status"] == "valid" for row in completeness
            ),
            "expected_mechanism_units": 72,
            "valid_mechanism_units": sum(
                row["unit_type"] == "mechanism" and row["status"] == "valid" for row in completeness
            ),
            "recompute_manifest_sha256": recompute_manifest_sha,
            "recompute_matrix_completion_sha256": sha256_file(
                root / "03_integrity/MECHANISM_RECOMPUTE_MATRIX_COMPLETE.json"
            ),
            "derived_disallowed_nonfinite_count": 0,
            "frozen_test_csv_units": sum(row["unit_type"] == "test200_frozen_csv" for row in completeness),
            "units": completeness,
        })
        promotions.append((completeness_path, root / "07_statistics/completeness_report.json"))

        report_path = stage / "FINAL_DYNAMIC_MECHANISM_REPORT.md"
        build_report(
            report_path, root, completeness, bootstrap_rows, validation_seed_rows,
            test_summary_rows, test_hashes, pose_bank_completion, stability,
            dose_response_rows, recompute_manifest_sha, time.time() - started,
        )
        promotions.append((report_path, root / "FINAL_DYNAMIC_MECHANISM_REPORT.md"))

        if len({destination for _, destination in promotions}) != len(promotions):
            raise SummaryValidationError("internal duplicate output destination")
        collisions = [destination for _, destination in promotions if destination.exists()]
        if collisions:
            raise SummaryValidationError(f"refusing to overwrite outputs created concurrently: {collisions[:5]}")
        output_inventory = [
            {
                "relative_path": str(destination.relative_to(root)),
                "sha256": sha256_file(source),
                "size_bytes": source.stat().st_size,
            }
            for source, destination in promotions
        ]
        completion_payload = {
            "schema_version": "surfna-lb200-dynamic-summary-completion-v2-numeric-qc",
            "status": "complete",
            "validation_units": 72,
            "mechanism_units": 72,
            "models": 9,
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "recompute_manifest_sha256": recompute_manifest_sha,
            "recompute_matrix_completion_sha256": sha256_file(
                root / "03_integrity/MECHANISM_RECOMPUTE_MATRIX_COMPLETE.json"
            ),
            "derived_disallowed_nonfinite_count": 0,
            "stability_endpoint_status": stability["status"],
            "test200_source_hashes": test_hashes,
            "outputs": output_inventory,
            "wall_seconds": time.time() - started,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        completion_source = stage / "DYNAMIC_SUMMARY_COMPLETE.json"
        atomic_json(completion_source, completion_payload)

        for source, destination in promotions:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
        completion_destination = root / "07_statistics/DYNAMIC_SUMMARY_COMPLETE.json"
        os.replace(completion_source, completion_destination)
        return {**completion_payload, "completion": str(completion_destination)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.bootstrap_replicates != BOOTSTRAP_REPLICATES:
        raise SystemExit(f"bootstrap replicates are frozen at {BOOTSTRAP_REPLICATES}")
    if args.bootstrap_seed != BOOTSTRAP_SEED:
        raise SystemExit(f"bootstrap seed is frozen at {BOOTSTRAP_SEED}")
    started = time.time()
    completeness: list[dict] = []
    try:
        result = run(args, completeness)
    except Exception as error:
        try:
            root = args.analysis_root.resolve()
            _, _ = unit_paths(root)
            # Recover the latest canonical unit inventory when failure happened before run() returned it.
            if not completeness:
                try:
                    precheck_unit_inventory(root, completeness)
                except Exception:
                    pass
            write_failure_attempt(root, completeness, error, started)
        except Exception as reporting_error:
            print(f"failed to write failure audit: {type(reporting_error).__name__}: {reporting_error}")
        raise
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
