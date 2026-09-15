#!/usr/bin/env python3
"""Fail-closed validation for the 72-unit v6 float64 recompute matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ARMS = ("Scratch-Full8", "Transfer-All", "Transfer-NonSurface")
ARM_SLUGS = {
    "Scratch-Full8": "scratch_full8",
    "Transfer-All": "transfer_all",
    "Transfer-NonSurface": "transfer_nonsurface",
}
SEEDS = (0, 1, 2)
EPOCHS = (25, 50, 75, 100, 125, 150, 175, 200)
MAIN_GATE_ALPHAS = (0.0, 0.5, 1.0)
DOSE_GATE_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DOSE_EPOCHS = (50, 100, 200)
SHUFFLE_SEEDS = (20260911, 20260912, 20260913)
REPLAY_TOLERANCE = 1e-6
REQUIRED_FINITE = (
    "translation_normalized_mse",
    "translation_direction_sign_agreement",
    "rotation_normalized_mse",
    "combined_alignment",
    "RMSD_before",
    "RMSD_after",
    "delta_RMSD_1step",
    "direct_RMSD_before",
    "direct_RMSD_after",
    "direct_delta_RMSD_1step",
)
THRESHOLDS = (100.0, 1_000.0, 1_000_000.0, 1_000_000_000_000.0)


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


def finite(value: object, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid numeric {field}: {value!r}") from error
    if not math.isfinite(parsed):
        raise RuntimeError(f"non-finite {field}: {value!r}")
    return parsed


def expected_conditions(epoch: int) -> set[tuple[str, float, int | None]]:
    alphas = DOSE_GATE_ALPHAS if epoch in DOSE_EPOCHS else MAIN_GATE_ALPHAS
    result = {("true_chemistry", alpha, None) for alpha in alphas}
    result.update(("joint_shuffled_chemistry", 1.0, seed) for seed in SHUFFLE_SEEDS)
    return result


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    manifest_path = root / "00_protocol/MECHANISM_RECOMPUTE_INPUT_MANIFEST.json"
    sidecar = manifest_path.with_suffix(".sha256")
    manifest = json.loads(manifest_path.read_text())
    if sidecar.read_text().strip() != sha256_file(manifest_path):
        raise SystemExit("v6 input manifest digest mismatch")
    if Path(manifest.get("output_root", "")).resolve() != root:
        raise SystemExit("runtime output root differs from frozen v6 input manifest")
    for path_key, hash_key in (
        ("pose_bank_path", "pose_bank_sha256"),
        ("pose_bank_manifest_path", "pose_bank_manifest_sha256"),
        ("pose_bank_completion_path", "pose_bank_completion_sha256"),
        ("validation_cache_contract_path", "validation_cache_contract_sha256"),
    ):
        artifact = Path(manifest.get(path_key, ""))
        if not artifact.is_file() or sha256_file(artifact) != manifest.get(hash_key):
            raise SystemExit(f"frozen input artifact drift: {path_key}/{artifact}")
    cache_contract = json.loads(Path(manifest["validation_cache_contract_path"]).read_text())
    if cache_contract != manifest.get("validation_cache_contract"):
        raise SystemExit("embedded and file validation cache contracts disagree")
    for path_key, hash_key in (
        ("split_path", "split_sha256"),
        ("heterographs_path", "heterographs_sha256"),
        ("rdkit_ligands_path", "rdkit_ligands_sha256"),
    ):
        artifact = Path(cache_contract[path_key])
        if not artifact.is_file() or sha256_file(artifact) != cache_contract[hash_key]:
            raise SystemExit(f"frozen validation cache artifact drift: {artifact}")
    for code_name, record in manifest.get("code", {}).items():
        code_path = Path(record.get("path", ""))
        if not code_path.is_file() or sha256_file(code_path) != record.get("sha256"):
            raise SystemExit(f"frozen v6 code digest mismatch: {code_name}/{code_path}")
    own_hash = manifest.get("code", {}).get("matrix_validator", {}).get("sha256")
    if own_hash != sha256_file(Path(__file__).resolve()):
        raise SystemExit("running matrix validator differs from frozen v6 manifest")
    units = manifest.get("units", [])
    if len(units) != 72:
        raise SystemExit(f"expected 72 frozen units, found {len(units)}")

    expected_paths = set()
    global_counts = Counter()
    arm_counts: dict[str, Counter] = defaultdict(Counter)
    unit_summaries = []
    source_raw_hashes = set()
    for unit in units:
        arm = unit["arm"]
        seed = int(unit["training_seed"])
        epoch = int(unit["epoch"])
        directory = (
            root / "05_mechanism_raw" / ARM_SLUGS[arm]
            / f"seed{seed}" / f"epoch_{epoch:04d}"
        )
        csv_path = directory / "mechanism_denoising_long.csv"
        receipt_path = directory / "MECHANISM_COMPLETE.json"
        expected_paths.add(csv_path)
        if not csv_path.is_file() or not receipt_path.is_file():
            raise SystemExit(f"missing v2 unit output: {directory}")
        receipt = json.loads(receipt_path.read_text())
        source_raw_hashes.add(receipt.get("source_raw_scores_sha256"))
        model_parameters = Path(unit.get("model_parameters", ""))
        if (
            not model_parameters.is_file()
            or sha256_file(model_parameters) != unit.get("model_parameters_sha256")
        ):
            raise SystemExit(f"frozen model parameters drift: {model_parameters}")
        if (
            receipt.get("schema_version") != "surfna-lb200-mechanism-recompute-completion-v2"
            or receipt.get("status") != "complete"
            or receipt.get("arm") != arm
            or int(receipt.get("training_seed", -1)) != seed
            or int(receipt.get("epoch", -1)) != epoch
            or receipt.get("source_raw_scores_sha256") != unit["source_raw_sha256"]
            or receipt.get("source_completion_sha256") != unit["source_completion_sha256"]
            or receipt.get("source_csv_sha256") != unit["source_csv_sha256"]
            or receipt.get("pose_bank_sha256") != manifest["pose_bank_sha256"]
            or receipt.get("input_manifest_sha256") != sha256_file(manifest_path)
            or receipt.get("derivation_precision") != "float64-derived-metrics-with-legacy-pose-bank-RMSD_before"
            or receipt.get("scientific_protocol_change") is not False
            or receipt.get("reinference_performed") is not False
            or receipt.get("scientific_output_reintegration_performed") is not False
            or receipt.get("coordinate_integrity_replay_performed") is not True
            or receipt.get("replayed_coordinates_used_as_metrics_input") is not False
            or receipt.get("stored_parent_coordinates_used_as_metrics_input") is not True
            or receipt.get("RMSD_before_policy") != "reuse_immutable_legacy_float32_pose_bank_metric"
            or receipt.get("delta_RMSD_1step_policy") != "legacy_RMSD_before_minus_float64_RMSD_after"
            or receipt.get("direct_RMSD_before_policy") != "recompute_from_pose_bank_coordinates_in_float64"
            or receipt.get("validation_cache_contract_sha256") != manifest["validation_cache_contract_sha256"]
            or receipt.get("model_parameters_sha256") != unit["model_parameters_sha256"]
            or int(receipt.get("failure_count", -1)) != 0
            or int(receipt.get("derived_disallowed_nonfinite_count", -1)) != 0
            or receipt.get("csv_sha256") != sha256_file(csv_path)
        ):
            raise SystemExit(f"v2 receipt contract mismatch: {receipt_path}")
        replay = receipt.get("coordinate_replay", {})
        if (
            replay.get("passed") is not True
            or int(replay.get("checked_count", -1)) != int(unit["expected_row_count"])
            or float(replay.get("tolerance", math.nan)) != REPLAY_TOLERANCE
            or not math.isfinite(float(replay.get("max_abs_difference", math.nan)))
            or float(replay.get("max_abs_difference")) > REPLAY_TOLERANCE
        ):
            raise SystemExit(f"coordinate replay gate failed: {receipt_path}")

        seen = set()
        observed_specs: dict[str, set[tuple[str, float, int | None]]] = defaultdict(set)
        unit_counts = Counter()
        optional_nan_counts = Counter()
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "arm", "training_seed", "epoch", "entry_sha256", "condition",
                "gate_alpha", "shuffle_seed", "valid", "failure_reason",
                "translation_cosine", "rotation_cosine", "torsion_cosine",
                "torsion_normalized_mse", "combined_alignment_components",
                *REQUIRED_FINITE,
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise SystemExit(f"malformed v2 CSV: {csv_path}")
            for row_index, row in enumerate(reader, start=2):
                if (
                    row["arm"] != arm
                    or int(row["training_seed"]) != seed
                    or int(row["epoch"]) != epoch
                    or str(row["valid"]).lower() != "true"
                    or str(row["failure_reason"]).strip()
                ):
                    raise SystemExit(f"invalid row identity/status: {csv_path}:{row_index}")
                shuffle = str(row["shuffle_seed"]).strip()
                key = (
                    row["entry_sha256"], row["condition"], float(row["gate_alpha"]),
                    None if not shuffle else int(shuffle),
                )
                if key in seen:
                    raise SystemExit(f"duplicate v2 row: {csv_path}:{row_index}")
                seen.add(key)
                intervention = (row["condition"], float(row["gate_alpha"]), None if not shuffle else int(shuffle))
                if intervention not in expected_conditions(epoch):
                    raise SystemExit(f"unexpected intervention: {csv_path}:{row_index}:{intervention}")
                observed_specs[row["entry_sha256"]].add(intervention)
                parsed = {
                    field: finite(row[field], f"{csv_path}:{row_index}:{field}")
                    for field in REQUIRED_FINITE
                }
                components = int(row["combined_alignment_components"])
                if components < 1 or components > 3:
                    raise SystemExit(f"invalid alignment component count: {csv_path}:{row_index}")
                cosines = []
                for field in ("translation_cosine", "rotation_cosine", "torsion_cosine", "torsion_normalized_mse"):
                    try:
                        value = float(row[field])
                    except ValueError as error:
                        raise SystemExit(f"invalid optional metric: {csv_path}:{row_index}:{field}") from error
                    if math.isinf(value):
                        raise SystemExit(f"infinite optional metric: {csv_path}:{row_index}:{field}")
                    if math.isnan(value):
                        optional_nan_counts[field] += 1
                    elif field.endswith("_cosine"):
                        if value < -1.0 - 1e-12 or value > 1.0 + 1e-12:
                            raise SystemExit(f"cosine outside [-1,1]: {csv_path}:{row_index}:{field}")
                        cosines.append(value)
                    elif value < 0:
                        raise SystemExit(f"negative optional NMSE: {csv_path}:{row_index}:{field}")
                if len(cosines) != components or not close(parsed["combined_alignment"], sum(cosines) / len(cosines)):
                    raise SystemExit(f"combined alignment arithmetic mismatch: {csv_path}:{row_index}")
                for field in (
                    "translation_normalized_mse", "rotation_normalized_mse", "RMSD_before",
                    "RMSD_after", "direct_RMSD_before", "direct_RMSD_after",
                ):
                    if parsed[field] < 0:
                        raise SystemExit(f"negative metric: {csv_path}:{row_index}:{field}")
                if parsed["translation_direction_sign_agreement"] not in (0.0, 1.0):
                    raise SystemExit(f"invalid sign-agreement value: {csv_path}:{row_index}")
                if not close(
                    parsed["delta_RMSD_1step"], parsed["RMSD_before"] - parsed["RMSD_after"]
                ):
                    raise SystemExit(f"delta RMSD arithmetic mismatch: {csv_path}:{row_index}")
                if not close(
                    parsed["direct_delta_RMSD_1step"],
                    parsed["direct_RMSD_before"] - parsed["direct_RMSD_after"],
                ):
                    raise SystemExit(f"direct delta RMSD arithmetic mismatch: {csv_path}:{row_index}")
                rmsd = parsed["RMSD_after"]
                unit_counts["rows"] += 1
                arm_counts[arm]["rows"] += 1
                global_counts["rows"] += 1
                for threshold in THRESHOLDS:
                    key_name = f"gt_{int(threshold)}A"
                    event = int(rmsd > threshold)
                    unit_counts[key_name] += event
                    arm_counts[arm][key_name] += event
                    global_counts[key_name] += event
        expected_rows = int(unit["expected_row_count"])
        if len(seen) != expected_rows or unit_counts["rows"] != expected_rows:
            raise SystemExit(f"row/key-space mismatch: {csv_path} {unit_counts['rows']}/{len(seen)} != {expected_rows}")
        specs = expected_conditions(epoch)
        if len(observed_specs) != 1780 or any(value != specs for value in observed_specs.values()):
            raise SystemExit(f"incomplete intervention cross-product: {csv_path}")
        if int(receipt.get("row_count", -1)) != expected_rows or int(receipt.get("valid_row_count", -1)) != expected_rows:
            raise SystemExit(f"receipt row count mismatch: {receipt_path}")
        receipt_counts = receipt.get("numerical_stability", {}).get("divergence_counts", {})
        expected_nan_counts = {
            key: int(value) for key, value in receipt.get("allowable_nan_counts", {}).items()
        }
        observed_nan_counts = {
            key: int(optional_nan_counts[key])
            for key in ("translation_cosine", "rotation_cosine", "torsion_cosine", "torsion_normalized_mse")
            if optional_nan_counts[key]
        }
        if expected_nan_counts != observed_nan_counts:
            raise SystemExit(f"allowable-NaN count mismatch: {receipt_path}")
        for threshold in THRESHOLDS:
            key_name = f"gt_{int(threshold)}A"
            if int(receipt_counts.get(str(int(threshold)), -1)) != unit_counts[key_name]:
                raise SystemExit(f"stability count mismatch: {receipt_path}/{threshold}")
        unit_summaries.append({
            "arm": arm,
            "training_seed": seed,
            "epoch": epoch,
            "rows": unit_counts["rows"],
            "divergence_counts": {
                str(int(threshold)): unit_counts[f"gt_{int(threshold)}A"] for threshold in THRESHOLDS
            },
            "csv": str(csv_path),
            "csv_sha256": sha256_file(csv_path),
            "receipt": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
        })

    discovered = set((root / "05_mechanism_raw").rglob("mechanism_denoising_long.csv"))
    if discovered != expected_paths:
        raise SystemExit(
            f"unexpected/missing recomputed CSV inventory: expected={len(expected_paths)} discovered={len(discovered)}"
        )
    if len(source_raw_hashes) != 72 or None in source_raw_hashes:
        raise SystemExit(f"source raw hash uniqueness failed: {len(source_raw_hashes)}")

    def summarize(counter: Counter) -> dict:
        rows = int(counter["rows"])
        return {
            "rows": rows,
            "divergence_counts": {
                str(int(threshold)): int(counter[f"gt_{int(threshold)}A"])
                for threshold in THRESHOLDS
            },
            "divergence_rates": {
                str(int(threshold)): counter[f"gt_{int(threshold)}A"] / rows
                for threshold in THRESHOLDS
            },
        }

    output = root / "03_integrity/MECHANISM_RECOMPUTE_MATRIX_COMPLETE.json"
    payload = {
        "schema_version": "surfna-lb200-mechanism-recompute-matrix-complete-v2",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "valid_units": 72,
        "expected_units": 72,
        "source_raw_hashes_unique": 72,
        "derived_disallowed_nonfinite_count": 0,
        "stability_endpoint_status": "post_failure_qc_descriptive_not_original_confirmatory_endpoint",
        "global": summarize(global_counts),
        "by_arm": {arm: summarize(arm_counts[arm]) for arm in ARMS},
        "units": unit_summaries,
    }
    if output.exists():
        raise SystemExit(f"refusing to overwrite matrix completion: {output}")
    atomic_json(output, payload)
    print(json.dumps({
        "completion": str(output),
        "completion_sha256": sha256_file(output),
        "valid_units": 72,
        "global": payload["global"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
