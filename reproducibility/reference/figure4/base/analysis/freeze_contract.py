#!/usr/bin/env python3
"""Freeze the nine-model SurfNA LB200 dynamic mechanism experiment contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
import yaml

from mechanism_common import (
    ARMS,
    CHEMISTRY_CHANNELS,
    DOSE_EPOCHS,
    DOSE_GATE_ALPHAS,
    EPOCHS,
    EXPECTED_COUNTS,
    EXPECTED_SCALER_SHA256,
    EXPECTED_SOURCE_SHA256,
    EXPERIMENT_ROOT,
    FROZEN_CODE_ROOT,
    INFERENCE_SEED,
    INFERENCE_STEPS,
    L2_RELEASE_ROOT,
    MAIN_GATE_ALPHAS,
    NOISE_LEVELS,
    POSE_BANK_SEED,
    POSE_REPEATS,
    SEEDS,
    SHUFFLE_SEEDS,
    arm_slug,
    atomic_json,
    checkpoint_path,
    complex_name,
    load_model_args,
    load_validation_dataset,
    make_joint_shuffle_permutation,
    model_dir,
    sha256_file,
)


EXECUTION_TREE_RELATIVE_PATHS = (
    "01_code/analysis",
    "01_code/overlay",
    "runtime",
)
EXECUTION_SUFFIXES = {".py", ".sh", ".bash", ".zsh", ".sbatch"}
REQUIRED_EXECUTION_FILES = {
    "01_code/analysis/build_mechanism_pose_bank.py",
    "01_code/analysis/evaluate_fixed_pose_mechanism.py",
    "01_code/analysis/evaluate_trajectory_accelerate.py",
    "01_code/analysis/freeze_contract.py",
    "01_code/analysis/mechanism_common.py",
    "01_code/analysis/reconstruct_epoch0.py",
    "01_code/analysis/smoke_test_full.py",
    "01_code/analysis/summarize_dynamic_mechanism.py",
    "01_code/analysis/test_surface_gate.py",
    "01_code/analysis/validate_preflight.py",
    "01_code/overlay/models/surface_score_model_v3.py",
    "runtime/prepare_posebank_and_gate.sbatch",
    "runtime/run_fixed_pose_mechanism_array.sbatch",
    "runtime/run_summary.sbatch",
    "runtime/run_validation_trajectory_array.sbatch",
}

# These hashes identify the exact frozen G2.2 implementation audited before this
# analysis was designed.  The overlay is inventoried separately below; it must
# never turn a different production model into the experiment's silent base.
EXPECTED_FROZEN_KEY_SOURCE_SHA256 = {
    "models/surface_score_model_v3.py": (
        "9154b9549f4761b683e6e79b9158ba872c6df2a2c040075d8303fd2893854486"
    ),
    "models/surfna_v2_modules.py": (
        "90d2b9441fec4d58e942285611da81384d0697be76d2cbbc72dacdb9c392fe46"
    ),
    "datasets/pdbbind.py": (
        "2ebfbbedafcaf311e8a84ff423d6e87d6abf61cd3a326c61dbcc8ae760c5bbe4"
    ),
    "utils/sampling.py": (
        "5b4630505aaa2d53356565f39f1173fb87a39fd958d1c34540580c1d5564e190"
    ),
    "utils/diffusion_utils.py": (
        "836fd8d2317afdfa7dabceffeadc2b61f01d9387af137ad548739ae27b228f6b"
    ),
    "utils/training.py": (
        "1db5d18848f0d247f3de9a5daf910ec8a1a354120bebcc3100ef948983cacc2e"
    ),
    "utils/parsing.py": (
        "fcf15ba5bde091b1d09ead30e66bbe62656003c29466a2cd40987955af7fbbe8"
    ),
    "utils/utils.py": (
        "64d372b55543db056d349c45cff31cbf84505e483f5e476421401abd566bbf7e"
    ),
    "evaluate_accelarate.py": (
        "390550ed216eafa82a78dd7de0bd62cb884334fb49418a5afa968341f3ecada6"
    ),
}

EXPECTED_AMENDMENT_SHA256 = (
    "87d69ed39f4b495ddbf9c092adb94c22d82c63880ef1d0f499244cf30a24dff6"
)


def is_execution_file(path: Path) -> bool:
    """Return whether *path* can affect an analysis/runtime execution."""
    if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
        return False
    return path.suffix.lower() in EXECUTION_SUFFIXES or bool(path.stat().st_mode & 0o111)


def execution_code_inventory(root: Path) -> dict:
    records = []
    for relative_tree in EXECUTION_TREE_RELATIVE_PATHS:
        tree = root / relative_tree
        if not tree.is_dir():
            raise RuntimeError(f"missing execution-code tree: {tree}")
        for path in sorted(candidate for candidate in tree.rglob("*") if is_execution_file(candidate)):
            records.append({
                "tree": relative_tree,
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            })
    observed = {record["relative_path"] for record in records}
    missing = sorted(REQUIRED_EXECUTION_FILES - observed)
    if missing:
        raise RuntimeError(f"required analysis execution files are missing: {missing}")
    if len(observed) != len(records):
        raise RuntimeError("duplicate execution-code paths in frozen inventory")
    return {
        "schema_version": "surfna-lb200-execution-code-inventory-v1",
        "passed": True,
        "analysis_root": str(root),
        "trees": list(EXECUTION_TREE_RELATIVE_PATHS),
        "required_files": sorted(REQUIRED_EXECUTION_FILES),
        "record_count": len(records),
        "records": records,
    }


def frozen_model_source_inventory() -> dict:
    records = []
    for relative_path, expected_sha256 in sorted(EXPECTED_FROZEN_KEY_SOURCE_SHA256.items()):
        path = FROZEN_CODE_ROOT / relative_path
        if not path.is_file():
            raise RuntimeError(f"missing frozen G2.2 source file: {path}")
        observed_sha256 = sha256_file(path)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"frozen G2.2 source hash mismatch for {relative_path}: "
                f"observed={observed_sha256}, expected={expected_sha256}"
            )
        records.append({
            "relative_path": relative_path,
            "path": str(path),
            "sha256": observed_sha256,
            "expected_sha256": expected_sha256,
            "size_bytes": path.stat().st_size,
        })
    return {
        "schema_version": "surfna-lb200-frozen-model-source-inventory-v1",
        "passed": True,
        "frozen_code_root": str(FROZEN_CODE_ROOT),
        "record_count": len(records),
        "records": records,
    }


def expected_shuffle_permutation_manifest() -> dict:
    """Freeze the deterministic intervention assignment before model analysis."""
    model_args = load_model_args("Transfer-All", 0)
    dataset = load_validation_dataset(model_args)
    payload = {
        "schema_version": "surfna-lb200-joint-chemistry-permutation-v1",
        "chemistry_channels": list(CHEMISTRY_CHANNELS),
        "shuffle_seeds": list(SHUFFLE_SEEDS),
        "permutations": {},
    }
    for dataset_index in range(len(dataset)):
        graph = dataset[dataset_index]
        name = complex_name(graph)
        if name in payload["permutations"]:
            raise RuntimeError(f"duplicate validation complex while freezing permutations: {name}")
        num_vertices = int(graph["surface"].x.shape[0])
        payload["permutations"][name] = {}
        for shuffle_seed in SHUFFLE_SEEDS:
            permutation, attempts = make_joint_shuffle_permutation(
                name,
                num_vertices,
                shuffle_seed,
            )
            fixed_ratio = float(
                (permutation == torch.arange(permutation.numel())).float().mean()
            )
            payload["permutations"][name][str(shuffle_seed)] = {
                "dataset_index": dataset_index,
                "num_vertices": num_vertices,
                "attempts": attempts,
                "fixed_point_ratio": fixed_ratio,
                "permutation": permutation.tolist(),
            }
    if len(payload["permutations"]) != EXPECTED_COUNTS["val"]:
        raise RuntimeError("expected permutation manifest does not cover validation89")
    return payload


def atomic_frozen_contract(path: Path, payload: dict) -> str:
    """Publish a contract only after its matching SHA sidecar is durable."""
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    sidecar = path.with_suffix(".sha256")
    temporary_contract = path.with_suffix(path.suffix + ".tmp")
    temporary_sidecar = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary_contract.write_text(serialized)
    temporary_sidecar.write_text(f"{digest}  {path.name}\n")
    # Sidecar-first publication means the existence of FROZEN_CONTRACT.json
    # always implies that the validator can verify it.  A crash after the first
    # replace but before the second remains safely retryable.
    os.replace(temporary_sidecar, sidecar)
    os.replace(temporary_contract, path)
    return digest


def save_yaml(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=True))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis_root.resolve()
    frozen_marker = root / "00_protocol/FROZEN_CONTRACT.json"
    if frozen_marker.exists():
        raise SystemExit(f"refusing to overwrite an existing frozen contract: {frozen_marker}")
    root.mkdir(parents=True, exist_ok=True)
    for directory in (
        "00_protocol", "00_audit", "01_code", "02_checkpoints", "03_integrity",
        "04_mechanism_pose_bank", "05_mechanism_raw", "06_validation_trajectory",
        "07_statistics", "08_figures", "09_report", "runtime", "logs",
    ):
        (root / directory).mkdir(exist_ok=True)

    if sha256_file(args.source_plan) != EXPECTED_AMENDMENT_SHA256:
        raise RuntimeError("source plan is not the frozen user-authorized 25-200 amendment")
    plan_target = root / "00_protocol/SurfNA_25_200epoch_dynamic_surface_transfer_mechanism_amendment.md"
    shutil.copy2(args.source_plan, plan_target)
    manifest_source = EXPERIMENT_ROOT / "01_initialization/parameter_transfer_manifest_final.json"
    manifest_target = root / "00_protocol/parameter_transfer_manifest.json"
    shutil.copy2(manifest_source, manifest_target)

    init0 = json.loads((EXPERIMENT_ROOT / "01_initialization/init_transfer_all_seed0.json").read_text())
    source_checkpoint = Path(init0["source_checkpoint"]["path"])
    scaler = L2_RELEASE_ROOT / "scalers/surface_v2_l2_train_only.json"
    portable = json.loads((L2_RELEASE_ROOT / "PORTABLE_RUNTIME.json").read_text())
    if portable["counts"] != EXPECTED_COUNTS:
        raise RuntimeError(f"L2 release counts drift: {portable['counts']}")
    if sha256_file(source_checkpoint) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("protein source checkpoint hash mismatch")
    if sha256_file(scaler) != EXPECTED_SCALER_SHA256:
        raise RuntimeError("train-only L2 scaler hash mismatch")

    execution_inventory_payload = execution_code_inventory(root)
    execution_inventory_path = root / "00_audit/execution_code_inventory.json"
    atomic_json(execution_inventory_path, execution_inventory_payload)
    source_inventory_payload = frozen_model_source_inventory()
    source_inventory_path = root / "00_audit/frozen_model_source_inventory.json"
    atomic_json(source_inventory_path, source_inventory_payload)

    inventory = []
    scientific_args = None
    allowed_arg_differences = {"log_dir", "n_epochs", "project", "run_name", "seed"}
    for arm in ARMS:
        for seed in SEEDS:
            model_args = vars(load_model_args(arm, seed))
            canonical = {key: value for key, value in model_args.items() if key not in allowed_arg_differences}
            if scientific_args is None:
                scientific_args = canonical
            elif canonical != scientific_args:
                differing = sorted(
                    key for key in set(canonical) | set(scientific_args)
                    if canonical.get(key) != scientific_args.get(key)
                )
                raise RuntimeError(f"scientific config drift for {arm}/seed{seed}: {differing}")
            for epoch in EPOCHS:
                path = checkpoint_path(root, arm, seed, epoch)
                if not path.is_file():
                    raise RuntimeError(f"missing checkpoint: {path}")
                raw = torch.load(path, map_location="cpu")
                state = raw.get("model", raw) if isinstance(raw, dict) else raw
                if not isinstance(state, dict) or len(state) != 463:
                    raise RuntimeError(f"unexpected checkpoint structure: {path}")
                inventory.append({
                    "arm": arm,
                    "seed": seed,
                    "epoch": epoch,
                    "path": str(path),
                    "file_sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "state_tensor_count": len(state),
                    "formal_use": True,
                })

    checkpoint_protocol = {
        "schema_version": "surfna-lb200-checkpoint-schedule-v2",
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "epochs": list(EPOCHS),
        "model_count": len(ARMS) * len(SEEDS),
        "formal_checkpoint_count": len(ARMS) * len(SEEDS) * len(EPOCHS),
        "epoch0_role": "diagnostic_preflight_only_excluded_from_formal_statistics",
        "selection": "fixed_epoch_nodes_only_no_validation_or_test_based_selection",
        "legacy_trajectory_rule": "runs trained beyond epoch 200 contribute EMA checkpoints at or below 200 only",
        "primary_comparison": "Transfer-All versus Transfer-NonSurface, paired within seed",
        "scratch_role": (
            "three-seed reference arm included in every nine-model descriptive output; "
            "not substituted for the primary Surface-specific paired causal contrast"
        ),
        "scratch_seed0_formal_source": str(model_dir("Scratch-Full8", 0)),
    }
    gate_protocol = {
        "schema_version": "surfna-lb200-surface-output-gate-v2",
        "parameter": "analysis_surface_gate_alpha",
        "training_parameter": False,
        "checkpoint_key": False,
        "default": 1.0,
        "main_values": list(MAIN_GATE_ALPHAS),
        "dose_values": list(DOSE_GATE_ALPHAS),
        "dose_epochs": list(DOSE_EPOCHS),
        "scope": [
            "initial Surface-conditioned ligand patch-tokenizer residual",
            "every Surface-to-ligand message residual after any transfer adapter",
            "every Surface-conditioned ligand patch-tokenizer realignment residual",
        ],
        "endpoint_behavior": {
            "alpha_1": "explicit untouched original branch",
            "alpha_0": "explicit original-ligand/zero-message branch with zero Surface gradient",
        },
        "integrity_protocol": {
            "backend": "CUDA float32 with TF32 disabled",
            "comparison_order": "ABBA",
            "abba_cycles": 16,
            "repeats_per_condition": 32,
            "equivalence_rule": (
                "per-head |mean(left)-mean(right)| <= E_left + E_right + "
                "4*float32_eps*max(1, output_scale)"
            ),
            "null_envelope_epsilon_multiplier": 4.0,
            "maximum_null_envelope_abs": 1e-5,
            "positive_control_repeats_per_condition": 32,
            "minimum_positive_effect_to_tau_ratio": 1000.0,
            "minimum_positive_effect_absolute": 1e-4,
            "alpha0_vjp_rule": (
                "every score-scalar VJP with respect to Surface.x and Surface.pos "
                "must be exactly zero or disconnected"
            ),
            "alpha1_vjp_rule": (
                "at least one score-scalar VJP must be nonzero for each of Surface.x "
                "and Surface.pos"
            ),
            "positive_controls": (
                "three joint chemistry shuffles and one Surface-position counterfactual; "
                "at least one matched score head must exceed its own ABBA-derived tau"
            ),
        },
    }
    chemistry_protocol = {
        "schema_version": "surfna-lb200-joint-chemistry-intervention-v1",
        "surface_schema": ["hbond", "hphob", "charge", "si", "donor", "acceptor", "apolar", "boundary"],
        "jointly_permuted_channels": list(CHEMISTRY_CHANNELS),
        "jointly_permuted_names": ["hbond", "hphob", "charge", "donor", "acceptor", "apolar"],
        "preserved_channels": [3, 7],
        "preserved_names": ["si", "boundary"],
        "shuffle_seeds": list(SHUFFLE_SEEDS),
        "permutation_scope": "within_complex_shared_across_arms_training_seeds_epochs_noise_and_poses",
        "fixed_point_ratio_max_exclusive": 0.05,
        "gate_alpha": 1.0,
        "primary_intervention": "vertex-level joint spatial reassignment",
        "fallback": "geodesic-block joint shuffle only if both models nearly completely fail",
    }
    bank_protocol = {
        "schema_version": "surfna-lb200-mechanism-pose-bank-protocol-v1",
        "cohort": "L2 validation89 only",
        "test_data_used": False,
        "noise_levels": list(NOISE_LEVELS),
        "poses_per_complex_noise": POSE_REPEATS,
        "entry_count": EXPECTED_COUNTS["val"] * len(NOISE_LEVELS) * POSE_REPEATS,
        "seed": POSE_BANK_SEED,
        "perturbation": "exact frozen datasets.pdbbind.NoiseTransform.apply_noise with explicit tr/rot/tor updates",
        "one_step_integrator": {
            "source": "frozen utils/sampling.py reverse SDE branch",
            "inference_steps": INFERENCE_STEPS,
            "dt": 1.0 / INFERENCE_STEPS,
            "stochastic_z": 0.0,
            "drift_multiplier": 1.0,
            "ode_half_drift": False,
        },
        "rmsd_primary": "formal evaluator symmetry-aware RMSD with explicit coordinate fallback labels",
        "rmsd_sensitivity": "direct heavy-atom coordinate RMSD",
    }
    statistics_protocol = {
        "schema_version": "surfna-lb200-statistics-protocol-v1",
        "formal_epoch_schedule": list(EPOCHS),
        "epoch0_role": "diagnostic_only_excluded_after_preflight_failure",
        "primary_unit": "complex",
        "primary_comparison": "Transfer-All minus Transfer-NonSurface paired within seed and complex",
        "scratch": "secondary descriptive reference",
        "bootstrap": {
            "type": "paired hierarchical bootstrap over training seed then complex",
            "replicates": 10000,
            "random_seed": 20260903,
            "confidence_interval": 0.95,
        },
        "aulc": "trapezoidal integration over fixed epochs 25 through 200 divided by 175",
        "torsion_na_rule": "exclude unavailable torsion component rather than encode as zero",
        "test_role": "fixed external endpoint at EMA200 only; never used for protocol/model/checkpoint selection",
    }
    figure_protocol = {
        "schema_version": "surfna-lb200-figure-protocol-v1",
        "figures": {
            "figure1": "nine-model validation trajectory plus primary paired TA-vs-TN contrast",
            "figure2": "Surface gate contribution over epoch and noise",
            "figure3": "true-versus-joint-shuffled chemistry contribution over epoch and noise",
            "figure4": "optional patch-level examples selected by frozen quantitative rule",
        },
        "formats": ["pdf", "svg", "png"],
    }
    common_training = {
        "release": str(L2_RELEASE_ROOT),
        "counts": EXPECTED_COUNTS,
        "epochs_budget": 200,
        "world_size": 2,
        "batch_per_gpu": 8,
        "global_batch": 16,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "ema_rate": 0.999,
        "validation_every_epochs": 25,
        "validation_k": 10,
        "validation_inference_steps": INFERENCE_STEPS,
        "validation_inference_seed": INFERENCE_SEED,
        "surface_schema": "v2_full8",
        "surface_scaler": str(scaler),
        "surface_scaler_sha256": sha256_file(scaler),
        "protein_source_checkpoint": str(source_checkpoint),
        "protein_source_checkpoint_sha256": sha256_file(source_checkpoint),
        "full_model_finetuning": True,
        "fresh_optimizer_scheduler": True,
    }
    contract_markdown = f"""# Frozen 200-Epoch Dynamic Surface Transfer Mechanism Contract

This contract implements the user-authorized amendment to the supplied mechanism plan without
changing the nine completed training trajectories. It was frozen before any formal epoch-25--200
gate, noisy-pose, or chemistry-counterfactual result was observed. Epoch 0 is retained only as a
diagnostic initialization audit and is excluded from every formal endpoint and statistic because
the reconstructed Transfer-All initialization produced non-finite/degenerate score outputs.

## Matrix

- Models: `Scratch-Full8`, `Transfer-All`, `Transfer-NonSurface` × seeds `0,1,2` = **9 models**.
- Epoch nodes: `{', '.join(map(str, EPOCHS))}`.
- Formal learning-efficiency endpoint: `AULC_25_200`, normalized by 175 epochs.
- Primary causal contrast: paired `Transfer-All − Transfer-NonSurface` within seed.
- Scratch: complete three-seed reference arm in all descriptive outputs; not a replacement for the
  Surface-specific primary contrast.
- Data: L2 v2 `801 train / 89 validation / fixed test128`.

## Frozen mechanism interventions

- Surface output gate: alpha `{MAIN_GATE_ALPHAS}` at all epochs; dose `{DOSE_GATE_ALPHAS}` at
  epochs `{DOSE_EPOCHS}`.
- Joint chemistry shuffle: channels `{CHEMISTRY_CHANNELS}`, seeds `{SHUFFLE_SEEDS}`; Surface
  coordinates, topology, shape index, and boundary stay unchanged.
- Fixed pose bank: validation89 × four noise levels `{NOISE_LEVELS}` × five poses = 1,780 entries,
  seed `{POSE_BANK_SEED}`.
- One-step readout: one interval of the canonical 20-step reverse-SDE integrator with stochastic
  noise fixed to zero (full SDE drift, not the 0.5 probability-flow ODE drift).
- Task trajectory: validation K10/20 steps at every fixed epoch for all nine models, inference seed
  `{INFERENCE_SEED}`.
- Formal test: reuse the already frozen nine-model EMA200 test128/K10 evaluation; test remains
  excluded from protocol design, checkpoint selection, and mechanism-bank construction.

## Statistics

Complex is the primary unit. Transfer-All versus Transfer-NonSurface uses paired hierarchical
bootstrap over seeds and complexes (10,000 replicates). Scratch comparisons are descriptive unless
their provenance permits the same paired interpretation. No epoch or checkpoint is selected from
validation or test performance.

## Immutable provenance

- Source plan SHA-256: `{sha256_file(plan_target)}`
- L2 scaler SHA-256: `{sha256_file(scaler)}`
- Protein source checkpoint SHA-256: `{sha256_file(source_checkpoint)}`
- Parameter-transfer manifest SHA-256: `{sha256_file(manifest_target)}`
"""
    (root / "00_protocol/FROZEN_200EPOCH_DYNAMIC_MECHANISM_CONTRACT.md").write_text(contract_markdown)
    save_yaml(root / "00_protocol/common_training_config.yml", common_training)
    atomic_json(root / "00_protocol/checkpoint_schedule.json", checkpoint_protocol)
    atomic_json(root / "00_protocol/surface_gate_protocol.json", gate_protocol)
    atomic_json(root / "00_protocol/chemistry_intervention_protocol.json", chemistry_protocol)
    atomic_json(root / "00_protocol/mechanism_pose_bank_protocol.json", bank_protocol)
    atomic_json(root / "00_protocol/statistics_protocol.json", statistics_protocol)
    atomic_json(root / "00_protocol/figure_protocol.json", figure_protocol)
    expected_permutation_path = (
        root / "00_protocol/expected_shuffle_permutation_manifest.json"
    )
    atomic_json(expected_permutation_path, expected_shuffle_permutation_manifest())
    protocol_relative_paths = (
        "00_protocol/SurfNA_25_200epoch_dynamic_surface_transfer_mechanism_amendment.md",
        "00_protocol/parameter_transfer_manifest.json",
        "00_protocol/FROZEN_200EPOCH_DYNAMIC_MECHANISM_CONTRACT.md",
        "00_protocol/common_training_config.yml",
        "00_protocol/checkpoint_schedule.json",
        "00_protocol/surface_gate_protocol.json",
        "00_protocol/chemistry_intervention_protocol.json",
        "00_protocol/mechanism_pose_bank_protocol.json",
        "00_protocol/statistics_protocol.json",
        "00_protocol/figure_protocol.json",
        "00_protocol/expected_shuffle_permutation_manifest.json",
    )
    protocol_artifact_sha256 = {
        relative_path: sha256_file(root / relative_path)
        for relative_path in protocol_relative_paths
    }
    checkpoint_inventory_path = root / "00_audit/nine_model_checkpoint_inventory.json"
    atomic_json(checkpoint_inventory_path, {
        "schema_version": "surfna-lb200-nine-model-checkpoint-inventory-v1",
        "passed": True,
        "existing_checkpoint_count": len(inventory),
        "expected_existing_checkpoint_count": 72,
        "records": inventory,
    })
    epoch0_audit_relative = "02_checkpoints/epoch0_reconstruction_audit.json"
    frozen_contract = {
        "schema_version": "surfna-lb200-dynamic-mechanism-frozen-contract-v1",
        "status": "FROZEN_BEFORE_FORMAL_EPOCH25_200_MECHANISM_RESULTS",
        "passed": True,
        "hash_algorithm": "sha256",
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "epochs": list(EPOCHS),
        "model_count": 9,
        "source_plan": str(plan_target),
        "source_plan_sha256": sha256_file(plan_target),
        "contract_markdown": str(root / "00_protocol/FROZEN_200EPOCH_DYNAMIC_MECHANISM_CONTRACT.md"),
        "parameter_transfer_manifest": str(manifest_target),
        "parameter_transfer_manifest_sha256": sha256_file(manifest_target),
        "protocol_artifact_sha256": protocol_artifact_sha256,
        "execution_code_inventory": str(execution_inventory_path),
        "execution_code_inventory_relative": execution_inventory_path.relative_to(root).as_posix(),
        "execution_code_inventory_sha256": sha256_file(execution_inventory_path),
        "frozen_model_source_inventory": str(source_inventory_path),
        "frozen_model_source_inventory_relative": source_inventory_path.relative_to(root).as_posix(),
        "frozen_model_source_inventory_sha256": sha256_file(source_inventory_path),
        "frozen_code_root": str(FROZEN_CODE_ROOT),
        "existing_checkpoint_inventory": str(checkpoint_inventory_path),
        "existing_checkpoint_inventory_relative": checkpoint_inventory_path.relative_to(root).as_posix(),
        "existing_checkpoint_inventory_sha256": sha256_file(checkpoint_inventory_path),
        "existing_checkpoint_count": len(inventory),
        "epoch0_formal_use": False,
        "epoch0_reconstruction_audit": {
            "relative_path": epoch0_audit_relative,
            "schema_version": "surfna-lb200-epoch0-reconstruction-audit-v1",
            "passed_required": True,
            "expected_model_count": len(ARMS) * len(SEEDS),
            "expected_arm_seed_pairs": [
                {"arm": arm, "seed": seed}
                for arm in ARMS
                for seed in SEEDS
            ],
        },
        "preflight_required_passed_markers": [
            "00_audit/execution_code_inventory.json",
            "00_audit/frozen_model_source_inventory.json",
            "00_audit/nine_model_checkpoint_inventory.json",
            epoch0_audit_relative,
            "03_integrity/surface_gate_integrity_report.json",
            "03_integrity/full_one_complex_smoke_report.json",
            "04_mechanism_pose_bank/mechanism_pose_bank_integrity_report.json",
            "04_mechanism_pose_bank/chemistry_shuffle_integrity_report.json",
        ],
        "derived_artifact_contract": {
            "pose_bank_completion": "04_mechanism_pose_bank/POSE_BANK_COMPLETE.json",
            "pose_bank": "04_mechanism_pose_bank/mechanism_pose_bank_v1.pt",
            "pose_bank_sha256_sidecar": (
                "04_mechanism_pose_bank/mechanism_pose_bank_v1_sha256.txt"
            ),
            "shuffle_permutation_manifest": (
                "04_mechanism_pose_bank/shuffle_permutation_manifest.json"
            ),
            "expected_shuffle_permutation_manifest": (
                "00_protocol/expected_shuffle_permutation_manifest.json"
            ),
            "expected_shuffle_permutation_manifest_sha256": sha256_file(
                expected_permutation_path
            ),
            "frozen_validation_cache_contract": (
                "04_mechanism_pose_bank/frozen_validation_cache_contract.json"
            ),
            "permutation_validation": (
                "exact deterministic reconstruction from complex id, vertex count, and frozen "
                "shuffle seed, plus passed chemistry integrity marker"
            ),
        },
        "formal_results_observed_before_freeze": False,
        "diagnostic_epoch0_results_observed_before_freeze": True,
        "diagnostic_epoch0_disposition": (
            "Transfer-All EMA000 failed finite-output preflight; retained as audit evidence and "
            "excluded from the formal epoch25-200 matrix by explicit user authorization"
        ),
    }
    frozen_contract_sha256 = atomic_frozen_contract(frozen_marker, frozen_contract)
    if sha256_file(frozen_marker) != frozen_contract_sha256:
        raise RuntimeError("frozen contract failed its immediate SHA-256 round-trip")
    print(json.dumps({"frozen": True, "root": str(root), "models": 9, "checkpoints": len(inventory)}, sort_keys=True))


if __name__ == "__main__":
    main()
