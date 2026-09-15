#!/usr/bin/env python3
"""Freeze the independent v6 numerical-QC derivation contract and input manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
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
DOSE_EPOCHS = {50, 100, 200}
THRESHOLDS = (100.0, 1_000.0, 1_000_000.0, 1_000_000_000_000.0)
EXPECTED_HOTFIX_MANIFEST_SHA256 = (
    "c4bd99646c7fef132c4c14fc84db37b09dcce8c4516e983734e5b27d069c6275"
)
FROZEN_PRODUCTION_CODE = Path(
    "/public/home/luoyuxuan/SurfNA_V2/development/"
    "l2_generator_scratch_ddp2_20260901_v5_4090d/frozen/code/src"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: object) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    test_root = args.test_root.resolve()
    if source == output:
        raise SystemExit("source and output roots must differ")

    protocol = output / "00_protocol"
    manifest_path = protocol / "MECHANISM_RECOMPUTE_INPUT_MANIFEST.json"
    contract_path = protocol / "FROZEN_V6_NUMERIC_QC_CONTRACT.json"
    for path in (manifest_path, contract_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite frozen v6 artifact: {path}")

    pose_bank_path = source / "04_mechanism_pose_bank/mechanism_pose_bank_v1.pt"
    pose_bank_sha = sha256_file(pose_bank_path)
    pose_bank_manifest_path = source / "04_mechanism_pose_bank/mechanism_pose_bank_v1_manifest.csv"
    pose_bank_manifest_sha = sha256_file(pose_bank_manifest_path)
    pose_bank_completion_path = source / "04_mechanism_pose_bank/POSE_BANK_COMPLETE.json"
    pose_bank_completion_sha = sha256_file(pose_bank_completion_path)
    validation_cache_contract_path = (
        source / "04_mechanism_pose_bank/frozen_validation_cache_contract.json"
    )
    validation_cache_contract_sha = sha256_file(validation_cache_contract_path)
    validation_cache_contract = json.loads(validation_cache_contract_path.read_text())
    if (
        validation_cache_contract.get("schema_version")
        != "surfna-lb200-frozen-validation-cache-v1"
        or int(validation_cache_contract.get("complex_count", -1)) != 89
        or len(validation_cache_contract.get("complex_names_in_order", [])) != 89
        or validation_cache_contract.get("automatic_rebuild_allowed") is not False
    ):
        raise SystemExit("malformed frozen validation cache contract")
    for path_key, hash_key in (
        ("split_path", "split_sha256"),
        ("heterographs_path", "heterographs_sha256"),
        ("rdkit_ligands_path", "rdkit_ligands_sha256"),
    ):
        cache_artifact = Path(validation_cache_contract[path_key])
        if (
            not cache_artifact.is_file()
            or sha256_file(cache_artifact) != validation_cache_contract[hash_key]
        ):
            raise SystemExit(f"frozen validation cache artifact drift: {cache_artifact}")
    for source_artifact, copied_artifact in (
        (pose_bank_path, output / "04_mechanism_pose_bank/mechanism_pose_bank_v1.pt"),
        (pose_bank_manifest_path, output / "04_mechanism_pose_bank/mechanism_pose_bank_v1_manifest.csv"),
        (pose_bank_completion_path, output / "04_mechanism_pose_bank/POSE_BANK_COMPLETE.json"),
        (
            validation_cache_contract_path,
            output / "04_mechanism_pose_bank/frozen_validation_cache_contract.json",
        ),
    ):
        if (
            not copied_artifact.is_file()
            or sha256_file(copied_artifact) != sha256_file(source_artifact)
        ):
            raise SystemExit(f"v6 copied frozen artifact differs from parent: {copied_artifact}")
    validation_hotfix_sidecar = source / "validation_hotfix_v1/HOTFIX_MANIFEST.sha256"
    validation_hotfix_sidecar_text = validation_hotfix_sidecar.read_text()
    validation_hotfix_tokens = validation_hotfix_sidecar_text.split()
    if (
        len(validation_hotfix_tokens) != 2
        or validation_hotfix_tokens[1] != "HOTFIX_MANIFEST.json"
        or len(validation_hotfix_tokens[0]) != 64
    ):
        raise SystemExit("malformed validation hotfix SHA-256 sidecar")
    validation_hotfix_digest = validation_hotfix_tokens[0]
    if validation_hotfix_digest != EXPECTED_HOTFIX_MANIFEST_SHA256:
        raise SystemExit("validation hotfix manifest differs from the audited frozen digest")
    if sha256_file(source / "validation_hotfix_v1/HOTFIX_MANIFEST.json") != validation_hotfix_digest:
        raise SystemExit("validation hotfix manifest digest does not match its sidecar")
    for filename in ("HOTFIX_MANIFEST.json", "HOTFIX_MANIFEST.sha256"):
        parent_hotfix = source / "validation_hotfix_v1" / filename
        copied_hotfix = output / "validation_hotfix_v1" / filename
        if not copied_hotfix.is_file() or sha256_file(copied_hotfix) != sha256_file(parent_hotfix):
            raise SystemExit(f"v6 copied validation hotfix artifact differs: {copied_hotfix}")
    source_failure = source / "logs/surfna_dyn_sum_hf1_244099.err"

    code_paths = {
        "freeze": output / "01_code/analysis/freeze_v6_numeric_qc.py",
        "mechanism_common": output / "01_code/analysis/mechanism_common.py",
        "recompute": output / "01_code/analysis/recompute_mechanism_metrics_v2.py",
        "summary": output / "01_code/analysis/summarize_dynamic_mechanism.py",
        "matrix_validator": output / "01_code/analysis/validate_recompute_matrix_v2.py",
        "recompute_sbatch": output / "runtime/run_recompute_v2.sbatch",
        "matrix_validator_sbatch": output / "runtime/run_validate_recompute_matrix_v2.sbatch",
        "summary_sbatch": output / "runtime/run_summary_v6.sbatch",
        "production_pdbbind": FROZEN_PRODUCTION_CODE / "datasets/pdbbind.py",
        "production_diffusion_utils": FROZEN_PRODUCTION_CODE / "utils/diffusion_utils.py",
        "production_utils": FROZEN_PRODUCTION_CODE / "utils/utils.py",
        "production_torsion": FROZEN_PRODUCTION_CODE / "utils/torsion.py",
        "production_geometry": FROZEN_PRODUCTION_CODE / "utils/geometry.py",
    }
    code_hashes = {}
    for name, path in code_paths.items():
        if not path.is_file():
            raise SystemExit(f"missing v6 code before freeze: {path}")
        code_hashes[name] = {"path": str(path), "sha256": sha256_file(path)}

    units = []
    validation_units = []
    index = 0
    for arm in ARMS:
        slug = ARM_SLUGS[arm]
        for seed in SEEDS:
            for epoch in EPOCHS:
                unit_dir = source / "05_mechanism_raw" / slug / f"seed{seed}" / f"epoch_{epoch:04d}"
                completion_path = unit_dir / "MECHANISM_COMPLETE.json"
                csv_path = unit_dir / "mechanism_denoising_long.csv"
                raw_path = unit_dir / "mechanism_scores_raw.pt"
                completion = json.loads(completion_path.read_text())
                csv_sha = sha256_file(csv_path)
                raw_sha = sha256_file(raw_path)
                expected_rows = 1780 * (8 if epoch in DOSE_EPOCHS else 6)
                if (
                    completion.get("schema_version") != "surfna-lb200-mechanism-completion-v1"
                    or completion.get("arm") != arm
                    or int(completion.get("training_seed", -1)) != seed
                    or int(completion.get("epoch", -1)) != epoch
                    or int(completion.get("row_count", -1)) != expected_rows
                    or int(completion.get("valid_row_count", -1)) != expected_rows
                    or int(completion.get("failure_count", -1)) != 0
                    or completion.get("csv_sha256") != csv_sha
                    or completion.get("raw_scores_sha256") != raw_sha
                    or completion.get("pose_bank_sha256") != pose_bank_sha
                ):
                    raise SystemExit(f"parent unit contract mismatch: {unit_dir}")
                model_parameters_path = Path(completion["checkpoint"]).parent / "model_parameters.yml"
                if not model_parameters_path.is_file():
                    raise SystemExit(f"missing parent model parameters: {model_parameters_path}")
                units.append({
                    "index": index,
                    "arm": arm,
                    "training_seed": seed,
                    "epoch": epoch,
                    "expected_row_count": expected_rows,
                    "pose_bank_sha256": pose_bank_sha,
                    "source_completion": str(completion_path),
                    "source_completion_sha256": sha256_file(completion_path),
                    "source_csv": str(csv_path),
                    "source_csv_sha256": csv_sha,
                    "source_raw": str(raw_path),
                    "source_raw_sha256": raw_sha,
                    "checkpoint": completion.get("checkpoint"),
                    "checkpoint_file_sha256": completion.get("checkpoint_file_sha256"),
                    "checkpoint_state_sha256": completion.get("checkpoint_state_sha256"),
                    "model_parameters": str(model_parameters_path),
                    "model_parameters_sha256": sha256_file(model_parameters_path),
                    "validation_cache_contract_sha256": validation_cache_contract_sha,
                })
                validation_dir = (
                    source / "06_validation_trajectory/raw" / slug
                    / f"seed{seed}" / f"epoch_{epoch:04d}"
                )
                validation_manifest = validation_dir / "validation_pose_manifest.tsv"
                run_exit = validation_dir / "RUN_EXIT.txt"
                run_exit_ok = validation_dir / "RUN_EXIT_OK.txt"
                hotfix_sidecar = validation_dir / "HOTFIX_MANIFEST.sha256"
                with validation_manifest.open(newline="") as handle:
                    validation_rows = sum(1 for _ in csv.DictReader(handle, delimiter="\t"))
                if (
                    validation_rows != 890
                    or run_exit.read_text().strip() != "exit_code=0"
                    or not run_exit_ok.is_file()
                    or hotfix_sidecar.read_text().split()[0] != validation_hotfix_digest
                ):
                    raise SystemExit(f"validation hotfix unit contract mismatch: {validation_dir}")
                validation_units.append({
                    "index": index,
                    "arm": arm,
                    "training_seed": seed,
                    "epoch": epoch,
                    "expected_row_count": 890,
                    "source_manifest": str(validation_manifest),
                    "source_manifest_sha256": sha256_file(validation_manifest),
                    "source_run_exit": str(run_exit),
                    "source_run_exit_sha256": sha256_file(run_exit),
                    "source_run_exit_ok": str(run_exit_ok),
                    "source_run_exit_ok_sha256": sha256_file(run_exit_ok),
                    "hotfix_manifest_sha256": validation_hotfix_digest,
                    "hotfix_sidecar_sha256": sha256_file(hotfix_sidecar),
                })
                index += 1
    if len(units) != 72:
        raise SystemExit(f"expected 72 source units, found {len(units)}")
    if len(validation_units) != 72:
        raise SystemExit(f"expected 72 validation units, found {len(validation_units)}")

    test_files = {}
    for filename in (
        "test_pose_level_metrics.csv",
        "test_complex_level_metrics.csv",
        "test_seed_level_summary.csv",
    ):
        path = test_root / "statistics" / filename
        test_files[filename] = {"path": str(path), "sha256": sha256_file(path)}

    created = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "surfna-lb200-mechanism-recompute-input-manifest-v2",
        "created_utc": created,
        "source_root": str(source),
        "output_root": str(output),
        "pose_bank_path": str(pose_bank_path),
        "pose_bank_sha256": pose_bank_sha,
        "pose_bank_manifest_path": str(pose_bank_manifest_path),
        "pose_bank_manifest_sha256": pose_bank_manifest_sha,
        "pose_bank_completion_path": str(pose_bank_completion_path),
        "pose_bank_completion_sha256": pose_bank_completion_sha,
        "validation_cache_contract_path": str(validation_cache_contract_path),
        "validation_cache_contract_sha256": validation_cache_contract_sha,
        "validation_cache_contract": validation_cache_contract,
        "validation_hotfix_manifest_sha256": validation_hotfix_digest,
        "validation_hotfix_sidecar_sha256": sha256_file(validation_hotfix_sidecar),
        "test_root": str(test_root),
        "test_files": test_files,
        "code": code_hashes,
        "production_code_root": str(FROZEN_PRODUCTION_CODE),
        "derivation_policy": {
            "model_predictions": "reuse_parent_float32_exactly",
            "one_step_updates": "reuse_parent_float32_exactly",
            "updated_coordinates": "reuse_parent_float32_exactly",
            "coordinate_replay": "verify_parent_updates_reproduce_parent_coordinates_with_tolerance_1e-6",
            "derived_metrics": "recompute_alignment_and_after/direct_RMSD_metrics_for_all_72_units_in_float64",
            "reinference": False,
            "scientific_output_reintegration": False,
            "coordinate_integrity_replay": True,
            "replayed_coordinates_used_as_metrics_input": False,
            "stored_parent_coordinates_used_as_metrics_input": True,
            "RMSD_before": "reuse_immutable_legacy_float32_pose_bank_metric",
            "delta_RMSD_1step": "mixed_provenance_legacy_RMSD_before_minus_float64_RMSD_after",
            "direct_RMSD_before": "recompute_from_pose_bank_coordinates_in_float64",
            "row_deletion": False,
            "clipping_or_trust_region": False,
        },
        "stability_endpoint": {
            "status": "post_failure_qc_descriptive_not_original_confirmatory_endpoint",
            "primary_threshold_angstrom": 100.0,
            "sensitivity_thresholds_angstrom": list(THRESHOLDS[1:]),
            "denominator": "all valid fixed-pose one-step rows",
            "direction": "lower divergence rate is more stable",
        },
        "units": units,
        "validation_units": validation_units,
    }
    atomic_json(manifest_path, manifest)
    atomic_text(manifest_path.with_suffix(".sha256"), sha256_file(manifest_path) + "\n")

    contract = {
        "schema_version": "surfna-lb200-v6-numeric-qc-contract-v1",
        "created_utc": created,
        "parent_v5_root": str(source),
        "parent_v5_summary_job": 244099,
        "parent_v5_summary_failure_log": str(source_failure),
        "parent_v5_preservation": "read_only_evidence; no overwrite; no result mixing",
        "scientific_estimand": "unchanged canonical zero-z one-step SDE outputs",
        "repair_scope": "float64 recomputation of alignment, after/direct RMSD, and delta metrics while retaining the immutable legacy float32 RMSD_before; fail-closed arithmetic QC; deterministic integration replay is integrity-only and never replaces parent coordinates",
        "formal_matrix": "3 arms x 3 paired seeds x 8 epochs = 72 units",
        "validation_reuse": "byte-identical successful validation_hotfix_v1 72x890 outputs",
        "stability_endpoint": manifest["stability_endpoint"],
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "prohibited": [
            "dropping or replacing the overflow row",
            "post-hoc clipping of coordinates or metrics",
            "mixing parent v1 CSVs with v2 recomputed CSVs",
            "new model inference, scientific-output re-integration, or stochastic resampling",
            "validation/test checkpoint selection",
        ],
        "success_gate": [
            "72/72 v2 receipts and CSVs",
            "zero disallowed non-finite derived values",
            "72/72 complete intervention key spaces",
            "72/72 parent-hash and coordinate-replay checks",
            "72/72 reused validation manifests remain valid",
            "summary completion and final report include stability endpoints",
        ],
    }
    atomic_json(contract_path, contract)
    atomic_text(contract_path.with_suffix(".sha256"), sha256_file(contract_path) + "\n")
    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "units": len(units),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
