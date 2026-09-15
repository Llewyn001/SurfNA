#!/usr/bin/env python3
"""Fail-closed preflight for one frozen SurfNA LB200 analysis shard.

The validator intentionally uses only the standard library plus NumPy.  It
validates the code and contract before any model module or checkpoint is
imported, and it never creates, repairs, or rewrites an experiment artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np


ARMS = ("Scratch-Full8", "Transfer-All", "Transfer-NonSurface")
SEEDS = (0, 1, 2)
EPOCHS = (25, 50, 75, 100, 125, 150, 175, 200)
EXECUTION_TREES = ("01_code/analysis", "01_code/overlay", "runtime")
EXECUTION_SUFFIXES = {".py", ".sh", ".bash", ".zsh", ".sbatch"}
EXPECTED_CHECKPOINT_COUNT = 72
EXPECTED_MODEL_COUNT = 9
EXPECTED_STATE_TENSOR_COUNT = 463
EXPECTED_VALIDATION_COUNT = 89
EXPECTED_POSE_BANK_COUNT = 1780
EXPECTED_CHEMISTRY_CHANNELS = (0, 1, 2, 4, 5, 6)
EXPECTED_SHUFFLE_SEEDS = (20260911, 20260912, 20260913)
EXPECTED_GATE_ABBA_CYCLES = 16
EXPECTED_GATE_REPEATS_PER_CONDITION = 32
EXPECTED_GATE_EPS_MULTIPLIER = 4.0
EXPECTED_GATE_MAX_NULL_ENVELOPE_ABS = 1e-5
EXPECTED_GATE_POSITIVE_RATIO = 1000.0
EXPECTED_GATE_POSITIVE_ABS = 1e-4
EXPECTED_PROTEIN_SOURCE_SHA256 = (
    "c90a95381926534dcdddc7699ba3d5cc400afd23f9b07cbc7b614c4b9987239f"
)
EXPECTED_IGNORED_SOURCE_KEYS = {
    "ligand_patch_tokenizer.ligand_update.weight",
    "ligand_patch_tokenizer.surface_value.weight",
    "ligand_patch_tokenizer.token_norm.bias",
    "ligand_patch_tokenizer.token_norm.weight",
}
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")

FROZEN_CODE_ROOT = Path(
    "/public/home/luoyuxuan/SurfNA_V2/development/"
    "l2_generator_scratch_ddp2_20260901_v5_4090d/frozen/code/src"
)
SOURCE_EXPERIMENT_ROOT = Path(
    "/public/home/luoyuxuan/SurfNA_V2/development/"
    "surfna_physchem_transfer_experiment_20260902"
)
EXTERNAL_SCRATCH0_MODEL_DIR = Path(
    "/public/home/luoyuxuan/SurfNA_V2/development/"
    "l2_generator_scratch_ddp2_20260901_v5_4090d/runs/"
    "surfna_homologyclean_l2_fullsnapshot_20260831_v2_g22_"
    "scratch_seed0_e500_ddp2_b8"
)
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

EXPECTED_PROTOCOL_ARTIFACTS = {
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
}


class PreflightError(RuntimeError):
    """Raised for any contract or artifact mismatch."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PreflightError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {label}: {path}")
    require(not path.is_symlink(), f"symlink is forbidden for {label}: {path}")
    try:
        payload = json.loads(path.read_text(), object_pairs_hook=reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"cannot parse {label} {path}: {error}") from error
    require(isinstance(payload, dict), f"{label} must be a JSON object: {path}")
    return payload


def require_passed(payload: dict[str, Any], label: str) -> None:
    require(payload.get("passed") is True, f"{label} does not contain an exact passed=true marker")


def require_sha256(value: Any, label: str) -> str:
    require(isinstance(value, str) and HEX_SHA256.fullmatch(value) is not None, f"invalid SHA-256 for {label}")
    return value


def key_set_sha256(keys: set[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key.encode())
        digest.update(b"\0")
    return digest.hexdigest()


class FileHasher:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, int], str] = {}

    def __call__(self, path: Path, label: str, *, reject_symlink: bool = True) -> str:
        require(path.is_file(), f"missing {label}: {path}")
        if reject_symlink:
            require(not path.is_symlink(), f"symlink is forbidden for {label}: {path}")
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        if key not in self._cache:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            self._cache[key] = digest.hexdigest()
        return self._cache[key]


def safe_root_path(root: Path, relative_path: Any, label: str) -> Path:
    require(isinstance(relative_path, str) and relative_path, f"invalid relative path for {label}")
    relative = Path(relative_path)
    require(not relative.is_absolute(), f"absolute path is forbidden for {label}: {relative_path}")
    candidate = root / relative
    try:
        candidate.resolve().relative_to(root)
    except ValueError as error:
        raise PreflightError(f"{label} escapes analysis root: {relative_path}") from error
    return candidate


def is_execution_file(path: Path) -> bool:
    if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
        return False
    return path.suffix.lower() in EXECUTION_SUFFIXES or bool(path.stat().st_mode & 0o111)


def validate_contract(root: Path, hasher: FileHasher) -> tuple[dict[str, Any], str]:
    contract_path = root / "00_protocol/FROZEN_CONTRACT.json"
    sidecar_path = root / "00_protocol/FROZEN_CONTRACT.sha256"
    contract = load_json(contract_path, "frozen contract")
    contract_sha256 = hasher(contract_path, "frozen contract")
    require(sidecar_path.is_file(), f"missing frozen-contract SHA sidecar: {sidecar_path}")
    require(not sidecar_path.is_symlink(), f"contract SHA sidecar may not be a symlink: {sidecar_path}")
    tokens = sidecar_path.read_text().strip().split()
    require(len(tokens) == 2, "frozen-contract SHA sidecar must contain exactly digest and filename")
    require(tokens[0] == contract_sha256, "frozen-contract SHA sidecar mismatch")
    require(tokens[1] == contract_path.name, "frozen-contract SHA sidecar filename mismatch")
    require(contract.get("schema_version") == "surfna-lb200-dynamic-mechanism-frozen-contract-v1", "unexpected frozen-contract schema")
    require(
        contract.get("status") == "FROZEN_BEFORE_FORMAL_EPOCH25_200_MECHANISM_RESULTS",
        "frozen-contract status mismatch",
    )
    require_passed(contract, "frozen contract")
    require(contract.get("hash_algorithm") == "sha256", "frozen contract does not require SHA-256")
    require(tuple(contract.get("arms", ())) == ARMS, "contract arm matrix drift")
    require(tuple(contract.get("seeds", ())) == SEEDS, "contract seed matrix drift")
    require(tuple(contract.get("epochs", ())) == EPOCHS, "contract epoch schedule drift")
    require(contract.get("model_count") == EXPECTED_MODEL_COUNT, "contract model count drift")
    require(contract.get("existing_checkpoint_count") == EXPECTED_CHECKPOINT_COUNT, "contract checkpoint count drift")
    require(contract.get("epoch0_formal_use") is False, "epoch 0 must be excluded from formal use")
    require(
        contract.get("formal_results_observed_before_freeze") is False,
        "contract was not frozen before formal epoch25-200 mechanism results",
    )
    require(
        contract.get("diagnostic_epoch0_results_observed_before_freeze") is True,
        "epoch0 diagnostic disclosure is missing",
    )

    protocol_hashes = contract.get("protocol_artifact_sha256")
    require(isinstance(protocol_hashes, dict), "contract lacks protocol artifact hashes")
    require(set(protocol_hashes) == EXPECTED_PROTOCOL_ARTIFACTS, "protocol artifact set drift")
    for relative_path, expected_sha256 in sorted(protocol_hashes.items()):
        path = safe_root_path(root, relative_path, "protocol artifact")
        expected_sha256 = require_sha256(expected_sha256, f"protocol artifact {relative_path}")
        require(hasher(path, f"protocol artifact {relative_path}") == expected_sha256, f"protocol artifact hash mismatch: {relative_path}")
    return contract, contract_sha256


def load_linked_inventory(
    root: Path,
    contract: dict[str, Any],
    relative_key: str,
    hash_key: str,
    expected_relative: str,
    label: str,
    hasher: FileHasher,
) -> tuple[Path, dict[str, Any]]:
    require(contract.get(relative_key) == expected_relative, f"{label} path differs from the frozen location")
    path = safe_root_path(root, expected_relative, label)
    expected_sha256 = require_sha256(contract.get(hash_key), label)
    require(hasher(path, label) == expected_sha256, f"{label} file hash mismatch")
    payload = load_json(path, label)
    require_passed(payload, label)
    return path, payload


def validate_execution_code(
    root: Path,
    contract: dict[str, Any],
    hasher: FileHasher,
) -> str:
    path, inventory = load_linked_inventory(
        root,
        contract,
        "execution_code_inventory_relative",
        "execution_code_inventory_sha256",
        "00_audit/execution_code_inventory.json",
        "execution-code inventory",
        hasher,
    )
    require(inventory.get("schema_version") == "surfna-lb200-execution-code-inventory-v1", "execution-code inventory schema mismatch")
    require(Path(str(inventory.get("analysis_root", ""))).resolve() == root, "execution-code inventory analysis root mismatch")
    require(tuple(inventory.get("trees", ())) == EXECUTION_TREES, "execution tree list drift")
    records = inventory.get("records")
    require(isinstance(records, list), "execution-code records must be a list")
    require(inventory.get("record_count") == len(records), "execution-code record count mismatch")

    frozen_records: dict[str, dict[str, Any]] = {}
    for record in records:
        require(isinstance(record, dict), "malformed execution-code record")
        relative_path = record.get("relative_path")
        require(isinstance(relative_path, str) and relative_path not in frozen_records, f"duplicate/invalid execution path: {relative_path}")
        frozen_records[relative_path] = record
    required_files = inventory.get("required_files")
    require(isinstance(required_files, list), "execution inventory required_files must be a list")
    require(set(required_files).issubset(frozen_records), "required execution file is absent from inventory")
    require("01_code/analysis/validate_preflight.py" in required_files, "preflight validator itself was not frozen")

    discovered: dict[str, Path] = {}
    for relative_tree in EXECUTION_TREES:
        tree = safe_root_path(root, relative_tree, "execution tree")
        require(tree.is_dir(), f"missing execution tree: {tree}")
        for candidate in sorted(item for item in tree.rglob("*") if is_execution_file(item)):
            require(not candidate.is_symlink(), f"execution file may not be a symlink: {candidate}")
            relative_path = candidate.relative_to(root).as_posix()
            require(relative_path not in discovered, f"duplicate discovered execution path: {relative_path}")
            discovered[relative_path] = candidate
    require(set(discovered) == set(frozen_records), "execution file set differs from frozen inventory")
    for relative_path, candidate in discovered.items():
        record = frozen_records[relative_path]
        require(record.get("tree") in EXECUTION_TREES, f"invalid execution tree label: {relative_path}")
        require(relative_path == record["tree"] or relative_path.startswith(record["tree"] + "/"), f"execution record/tree mismatch: {relative_path}")
        expected_sha256 = require_sha256(record.get("sha256"), relative_path)
        require(candidate.stat().st_size == record.get("size_bytes"), f"execution file size mismatch: {relative_path}")
        require(hasher(candidate, relative_path) == expected_sha256, f"execution file hash mismatch: {relative_path}")
    return hasher(path, "execution-code inventory")


def validate_frozen_model_sources(
    root: Path,
    contract: dict[str, Any],
    hasher: FileHasher,
) -> str:
    path, inventory = load_linked_inventory(
        root,
        contract,
        "frozen_model_source_inventory_relative",
        "frozen_model_source_inventory_sha256",
        "00_audit/frozen_model_source_inventory.json",
        "frozen-model source inventory",
        hasher,
    )
    require(inventory.get("schema_version") == "surfna-lb200-frozen-model-source-inventory-v1", "frozen-model source inventory schema mismatch")
    require(Path(str(inventory.get("frozen_code_root", ""))).resolve() == FROZEN_CODE_ROOT, "frozen G2.2 code root mismatch")
    require(Path(str(contract.get("frozen_code_root", ""))).resolve() == FROZEN_CODE_ROOT, "contract frozen-code root mismatch")
    records = inventory.get("records")
    require(isinstance(records, list), "frozen-model source records must be a list")
    require(inventory.get("record_count") == len(records), "frozen-model source record count mismatch")
    by_relative: dict[str, dict[str, Any]] = {}
    for record in records:
        require(isinstance(record, dict), "malformed frozen-model source record")
        relative_path = record.get("relative_path")
        require(isinstance(relative_path, str) and relative_path not in by_relative, f"duplicate/invalid frozen source path: {relative_path}")
        by_relative[relative_path] = record
    require(set(by_relative) == set(EXPECTED_FROZEN_KEY_SOURCE_SHA256), "frozen key-source set drift")
    for relative_path, expected_sha256 in EXPECTED_FROZEN_KEY_SOURCE_SHA256.items():
        record = by_relative[relative_path]
        require(record.get("sha256") == expected_sha256, f"source inventory hash drift: {relative_path}")
        require(record.get("expected_sha256") == expected_sha256, f"source expected-hash drift: {relative_path}")
        source_path = FROZEN_CODE_ROOT / relative_path
        require(Path(str(record.get("path", ""))).resolve() == source_path, f"source path drift: {relative_path}")
        require(source_path.stat().st_size == record.get("size_bytes"), f"frozen source size mismatch: {relative_path}")
        require(hasher(source_path, f"frozen source {relative_path}") == expected_sha256, f"frozen source hash mismatch: {relative_path}")
    return hasher(path, "frozen-model source inventory")


def validate_checkpoint_inventory(
    root: Path,
    contract: dict[str, Any],
    arm: str,
    seed: int,
    epoch: int,
    hasher: FileHasher,
) -> tuple[Path | None, str | None, str]:
    path, inventory = load_linked_inventory(
        root,
        contract,
        "existing_checkpoint_inventory_relative",
        "existing_checkpoint_inventory_sha256",
        "00_audit/nine_model_checkpoint_inventory.json",
        "72-checkpoint inventory",
        hasher,
    )
    require(inventory.get("schema_version") == "surfna-lb200-nine-model-checkpoint-inventory-v1", "checkpoint inventory schema mismatch")
    records = inventory.get("records")
    require(isinstance(records, list), "checkpoint inventory records must be a list")
    require(inventory.get("existing_checkpoint_count") == EXPECTED_CHECKPOINT_COUNT, "checkpoint inventory count mismatch")
    require(inventory.get("expected_existing_checkpoint_count") == EXPECTED_CHECKPOINT_COUNT, "checkpoint expected count mismatch")
    require(len(records) == EXPECTED_CHECKPOINT_COUNT, "checkpoint record cardinality mismatch")
    expected_keys = {(a, s, e) for a in ARMS for s in SEEDS for e in EPOCHS}
    by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    observed_paths: set[str] = set()
    for record in records:
        require(isinstance(record, dict), "malformed checkpoint record")
        key = (record.get("arm"), record.get("seed"), record.get("epoch"))
        require(key in expected_keys and key not in by_key, f"invalid/duplicate checkpoint key: {key}")
        checkpoint_path = record.get("path")
        require(isinstance(checkpoint_path, str) and checkpoint_path not in observed_paths, f"duplicate/invalid checkpoint path: {checkpoint_path}")
        observed_paths.add(checkpoint_path)
        require(Path(checkpoint_path).name == f"ema_epoch_{key[2]:04d}.pt", f"checkpoint filename/epoch mismatch: {checkpoint_path}")
        require_sha256(record.get("file_sha256"), f"checkpoint {key}")
        require(isinstance(record.get("size_bytes"), int) and record["size_bytes"] > 0, f"checkpoint size missing: {key}")
        require(record.get("state_tensor_count") == EXPECTED_STATE_TENSOR_COUNT, f"checkpoint tensor count mismatch: {key}")
        require(record.get("formal_use") is True, f"checkpoint not marked for formal use: {key}")
        by_key[key] = record
    require(set(by_key) == expected_keys, "72-checkpoint arm/seed/epoch matrix is incomplete")
    inventory_sha256 = hasher(path, "72-checkpoint inventory")
    selected = by_key[(arm, seed, epoch)]
    selected_path = Path(selected["path"])
    require(selected_path.is_absolute(), "existing checkpoint path must be absolute")
    require(not selected_path.is_symlink(), f"selected checkpoint may not be a symlink: {selected_path}")
    require(selected_path.stat().st_size == selected["size_bytes"], "selected checkpoint size mismatch")
    selected_sha256 = hasher(selected_path, "selected checkpoint")
    require(selected_sha256 == selected["file_sha256"], "selected checkpoint hash mismatch")
    return selected_path, selected_sha256, inventory_sha256


def validate_epoch0_audit(
    root: Path,
    contract: dict[str, Any],
    arm: str,
    seed: int,
    epoch: int,
    hasher: FileHasher,
) -> tuple[Path | None, str | None, str]:
    reference = contract.get("epoch0_reconstruction_audit")
    require(isinstance(reference, dict), "contract lacks epoch0 audit reference")
    require(reference.get("relative_path") == "02_checkpoints/epoch0_reconstruction_audit.json", "epoch0 audit path drift")
    audit_path = safe_root_path(root, reference["relative_path"], "epoch0 reconstruction audit")
    audit = load_json(audit_path, "epoch0 reconstruction audit")
    require(audit.get("schema_version") == reference.get("schema_version") == "surfna-lb200-epoch0-reconstruction-audit-v1", "epoch0 audit schema mismatch")
    require(reference.get("passed_required") is True, "contract does not require epoch0 passed marker")
    require_passed(audit, "epoch0 reconstruction audit")
    require(reference.get("expected_model_count") == EXPECTED_MODEL_COUNT, "epoch0 reference model count drift")
    require(audit.get("model_count") == EXPECTED_MODEL_COUNT, "epoch0 audit model count mismatch")
    expected_pairs = {(a, s) for a in ARMS for s in SEEDS}
    reference_pairs = {
        (record.get("arm"), record.get("seed"))
        for record in reference.get("expected_arm_seed_pairs", [])
        if isinstance(record, dict)
    }
    require(reference_pairs == expected_pairs, "epoch0 reference arm/seed matrix drift")
    records = audit.get("records")
    require(isinstance(records, list) and len(records) == EXPECTED_MODEL_COUNT, "epoch0 audit record count mismatch")
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    copied_manifest_sha256 = require_sha256(contract.get("parameter_transfer_manifest_sha256"), "parameter-transfer manifest")
    copied_manifest = safe_root_path(root, "00_protocol/parameter_transfer_manifest.json", "parameter-transfer manifest")
    require(hasher(copied_manifest, "parameter-transfer manifest") == copied_manifest_sha256, "copied parameter-transfer manifest hash mismatch")
    copied_manifest_payload = load_json(copied_manifest, "parameter-transfer manifest")
    require(
        set(copied_manifest_payload.get("ignored_source_state_keys", ()))
        == EXPECTED_IGNORED_SOURCE_KEYS,
        "parameter-transfer ignored source-key allowlist drift",
    )
    require(
        copied_manifest_payload.get("shape_mismatch_source_keys") == [],
        "parameter-transfer manifest contains source/target shape mismatches",
    )
    selected_counts = {"Scratch-Full8": 0, "Transfer-All": 383, "Transfer-NonSurface": 168}

    def expected_parameters_source(arm_name: str, training_seed: int) -> Path:
        if arm_name == "Scratch-Full8" and training_seed == 0:
            model_root = EXTERNAL_SCRATCH0_MODEL_DIR
        elif arm_name == "Scratch-Full8" and training_seed == 1:
            model_root = SOURCE_EXPERIMENT_ROOT / "03_training/scratch_full8_seed1_e500_ddp2_b8"
        elif arm_name == "Scratch-Full8":
            model_root = SOURCE_EXPERIMENT_ROOT / f"03_training/scratch_full8_seed{training_seed}_e200_ddp2_b8"
        elif arm_name == "Transfer-All":
            model_root = SOURCE_EXPERIMENT_ROOT / f"03_training/transfer_all_seed{training_seed}_e200_ddp2_b8"
        elif training_seed == 0:
            model_root = SOURCE_EXPERIMENT_ROOT / "03_training/transfer_nonsurface_seed0_e500_ddp2_b8"
        else:
            model_root = SOURCE_EXPERIMENT_ROOT / f"03_training/transfer_nonsurface_seed{training_seed}_e200_ddp2_b8"
        return model_root / "model_parameters.yml"

    for record in records:
        require(isinstance(record, dict), "malformed epoch0 audit record")
        key = (record.get("arm"), record.get("seed"))
        require(key in expected_pairs and key not in by_key, f"invalid/duplicate epoch0 audit key: {key}")
        by_key[key] = record
        expected_output = root / "02_checkpoints" / key[0].lower().replace("-", "_") / f"seed{key[1]}" / "ema_epoch_0000.pt"
        require(Path(str(record.get("output", ""))).resolve() == expected_output, f"epoch0 output path mismatch: {key}")
        require(expected_output.is_file() and not expected_output.is_symlink(), f"epoch0 checkpoint missing or symlinked: {key}")
        require_sha256(record.get("file_sha256"), f"epoch0 checkpoint {key}")
        state_sha256 = require_sha256(record.get("state_sha256"), f"epoch0 state {key}")
        require(record.get("receipt_state_sha256") == state_sha256, f"epoch0 receipt/state digest mismatch: {key}")
        require(record.get("verification") == "exact_tensor_digest_match", f"epoch0 verification marker mismatch: {key}")
        require(record.get("tensor_count") == EXPECTED_STATE_TENSOR_COUNT, f"epoch0 tensor count mismatch: {key}")
        require(record.get("selected_source_tensor_count") == selected_counts[key[0]], f"epoch0 transfer count mismatch: {key}")
        require(record.get("source_checkpoint_sha256") == EXPECTED_PROTEIN_SOURCE_SHA256, f"epoch0 protein-source digest mismatch: {key}")
        require(record.get("source_state_tensor_count") == 387, f"epoch0 protein-source tensor count mismatch: {key}")
        require(
            set(record.get("ignored_source_state_keys", ())) == EXPECTED_IGNORED_SOURCE_KEYS,
            f"epoch0 ignored source-key set mismatch: {key}",
        )
        require(
            record.get("ignored_source_state_key_set_sha256")
            == key_set_sha256(EXPECTED_IGNORED_SOURCE_KEYS),
            f"epoch0 ignored source-key digest mismatch: {key}",
        )
        require(record.get("shape_mismatch_source_keys") == [], f"epoch0 shape-mismatch set is not empty: {key}")
        source_checkpoint_path = Path(str(record.get("source_checkpoint", "")))
        require(
            hasher(source_checkpoint_path, f"epoch0 protein source {key}")
            == EXPECTED_PROTEIN_SOURCE_SHA256,
            f"epoch0 protein-source file mismatch: {key}",
        )
        require(record.get("manifest_sha256") == copied_manifest_sha256, f"epoch0 manifest digest mismatch: {key}")
        manifest_path = Path(str(record.get("manifest", "")))
        require(hasher(manifest_path, f"epoch0 source manifest {key}") == copied_manifest_sha256, f"epoch0 source manifest file mismatch: {key}")
        receipt_path = Path(str(record.get("initialization_receipt", "")))
        receipt = load_json(receipt_path, f"epoch0 initialization receipt {key}")
        require(receipt.get("initial_full_state_sha256") == state_sha256, f"epoch0 initialization receipt mismatch: {key}")
        parameters_source = Path(str(record.get("model_parameters_source", "")))
        expected_source = expected_parameters_source(key[0], key[1])
        require(parameters_source.resolve() == expected_source.resolve(), f"epoch0 model-parameters source path mismatch: {key}")
        source_parameters_sha256 = require_sha256(
            record.get("model_parameters_source_sha256"), f"epoch0 model parameters source {key}"
        )
        require(
            hasher(parameters_source, f"epoch0 model parameters source {key}")
            == source_parameters_sha256,
            f"epoch0 model-parameters source hash mismatch: {key}",
        )
        parameters_copy = Path(str(record.get("model_parameters_copy", "")))
        expected_copy = expected_output.parent / "model_parameters.yml"
        require(parameters_copy.resolve() == expected_copy, f"epoch0 model-parameters copy path mismatch: {key}")
        copy_parameters_sha256 = require_sha256(
            record.get("model_parameters_copy_sha256"), f"epoch0 model parameters copy {key}"
        )
        require(copy_parameters_sha256 == source_parameters_sha256, f"epoch0 model-parameters source/copy hash mismatch: {key}")
        require(
            hasher(parameters_copy, f"epoch0 model parameters copy {key}")
            == copy_parameters_sha256,
            f"epoch0 model-parameters copied file hash mismatch: {key}",
        )
        require_sha256(record.get("base_init_sha256"), f"epoch0 base init {key}")
        require_sha256(record.get("source_checkpoint_sha256"), f"epoch0 protein source {key}")
    require(set(by_key) == expected_pairs, "epoch0 audit arm/seed matrix is incomplete")
    audit_sha256 = hasher(audit_path, "epoch0 reconstruction audit")
    if epoch != 0:
        return None, None, audit_sha256
    selected = by_key[(arm, seed)]
    selected_path = Path(selected["output"])
    selected_sha256 = hasher(selected_path, "selected epoch0 checkpoint")
    require(selected_sha256 == selected["file_sha256"], "selected epoch0 checkpoint hash mismatch")
    return selected_path, selected_sha256, audit_sha256


def read_hash_sidecar(path: Path, expected_name: str) -> str:
    require(path.is_file() and not path.is_symlink(), f"missing or symlinked SHA sidecar: {path}")
    tokens = path.read_text().strip().split()
    require(len(tokens) == 2, f"malformed SHA sidecar: {path}")
    digest = require_sha256(tokens[0], str(path))
    require(tokens[1] == expected_name, f"SHA sidecar filename mismatch: {path}")
    return digest


def deterministic_permutation(complex_id: str, n_vertices: int, seed: int) -> tuple[list[int], int]:
    require(n_vertices > 1, f"surface has too few vertices: {complex_id}/{n_vertices}")
    for attempt in range(10001):
        token = f"surfna-joint-shuffle-v1|{complex_id}|{seed}|{attempt}".encode()
        derived = int.from_bytes(hashlib.sha256(token).digest()[:8], "little", signed=False)
        permutation = np.random.default_rng(derived).permutation(n_vertices)
        if float(np.mean(permutation == np.arange(n_vertices))) < 0.05:
            return permutation.tolist(), attempt + 1
    raise PreflightError(f"cannot reconstruct frozen permutation: {complex_id}/{seed}")


def validate_permutations(
    path: Path,
    cache_contract: dict[str, Any],
    hasher: FileHasher,
) -> str:
    manifest = load_json(path, "shuffle permutation manifest")
    require(manifest.get("schema_version") == "surfna-lb200-joint-chemistry-permutation-v1", "permutation manifest schema mismatch")
    require(tuple(manifest.get("chemistry_channels", ())) == EXPECTED_CHEMISTRY_CHANNELS, "permuted chemistry channel set drift")
    require(tuple(manifest.get("shuffle_seeds", ())) == EXPECTED_SHUFFLE_SEEDS, "shuffle seed set drift")
    complex_names = cache_contract.get("complex_names_in_order")
    require(isinstance(complex_names, list) and len(complex_names) == EXPECTED_VALIDATION_COUNT, "cache contract complex list mismatch")
    require(len(set(complex_names)) == len(complex_names), "duplicate complex names in cache contract")
    permutations = manifest.get("permutations")
    require(isinstance(permutations, dict) and set(permutations) == set(complex_names), "permutation complex set differs from validation89")
    checked = 0
    for dataset_index, complex_id in enumerate(complex_names):
        by_seed = permutations[complex_id]
        require(isinstance(by_seed, dict) and set(by_seed) == {str(seed) for seed in EXPECTED_SHUFFLE_SEEDS}, f"permutation seed set mismatch: {complex_id}")
        for shuffle_seed in EXPECTED_SHUFFLE_SEEDS:
            record = by_seed[str(shuffle_seed)]
            require(isinstance(record, dict), f"malformed permutation record: {complex_id}/{shuffle_seed}")
            require(record.get("dataset_index") == dataset_index, f"permutation dataset index mismatch: {complex_id}/{shuffle_seed}")
            n_vertices = record.get("num_vertices")
            permutation = record.get("permutation")
            require(isinstance(n_vertices, int) and n_vertices > 1, f"invalid surface vertex count: {complex_id}/{shuffle_seed}")
            require(isinstance(permutation, list) and len(permutation) == n_vertices, f"permutation length mismatch: {complex_id}/{shuffle_seed}")
            require(all(isinstance(value, int) and not isinstance(value, bool) for value in permutation), f"non-integer permutation: {complex_id}/{shuffle_seed}")
            require(sorted(permutation) == list(range(n_vertices)), f"non-bijective permutation: {complex_id}/{shuffle_seed}")
            expected, attempts = deterministic_permutation(complex_id, n_vertices, shuffle_seed)
            require(permutation == expected, f"deterministic permutation mismatch: {complex_id}/{shuffle_seed}")
            require(record.get("attempts") == attempts, f"permutation attempt count mismatch: {complex_id}/{shuffle_seed}")
            fixed_ratio = sum(index == value for index, value in enumerate(permutation)) / n_vertices
            require(fixed_ratio < 0.05, f"permutation fixed-point threshold failed: {complex_id}/{shuffle_seed}")
            require(abs(float(record.get("fixed_point_ratio", -1.0)) - fixed_ratio) <= 1e-6, f"permutation fixed-point ratio mismatch: {complex_id}/{shuffle_seed}")
            checked += 1
    require(checked == EXPECTED_VALIDATION_COUNT * len(EXPECTED_SHUFFLE_SEEDS), "permutation cardinality mismatch")
    return hasher(path, "shuffle permutation manifest")


def validate_pose_bank_and_markers(
    root: Path,
    contract: dict[str, Any],
    hasher: FileHasher,
) -> tuple[str, str, str]:
    derived = contract.get("derived_artifact_contract")
    require(isinstance(derived, dict), "contract lacks derived-artifact contract")
    expected_paths = {
        "pose_bank_completion": "04_mechanism_pose_bank/POSE_BANK_COMPLETE.json",
        "pose_bank": "04_mechanism_pose_bank/mechanism_pose_bank_v1.pt",
        "pose_bank_sha256_sidecar": "04_mechanism_pose_bank/mechanism_pose_bank_v1_sha256.txt",
        "shuffle_permutation_manifest": "04_mechanism_pose_bank/shuffle_permutation_manifest.json",
        "expected_shuffle_permutation_manifest": "00_protocol/expected_shuffle_permutation_manifest.json",
        "frozen_validation_cache_contract": "04_mechanism_pose_bank/frozen_validation_cache_contract.json",
    }
    for key, relative_path in expected_paths.items():
        require(derived.get(key) == relative_path, f"derived artifact path drift: {key}")

    completion_path = safe_root_path(root, expected_paths["pose_bank_completion"], "pose-bank completion")
    bank_path = safe_root_path(root, expected_paths["pose_bank"], "mechanism pose bank")
    sidecar_path = safe_root_path(root, expected_paths["pose_bank_sha256_sidecar"], "pose-bank SHA sidecar")
    permutation_path = safe_root_path(root, expected_paths["shuffle_permutation_manifest"], "permutation manifest")
    expected_permutation_path = safe_root_path(
        root,
        expected_paths["expected_shuffle_permutation_manifest"],
        "expected permutation manifest",
    )
    cache_contract_path = safe_root_path(root, expected_paths["frozen_validation_cache_contract"], "validation cache contract")

    completion = load_json(completion_path, "pose-bank completion")
    require(completion.get("schema_version") == "surfna-lb200-pose-bank-completion-v1", "pose-bank completion schema mismatch")
    require(Path(str(completion.get("bank", ""))).resolve() == bank_path, "pose-bank completion points to a different bank")
    bank_sha256 = hasher(bank_path, "mechanism pose bank")
    require(completion.get("bank_sha256") == bank_sha256, "pose-bank completion hash mismatch")
    require(read_hash_sidecar(sidecar_path, bank_path.name) == bank_sha256, "pose-bank SHA sidecar mismatch")

    integrity_path = root / "04_mechanism_pose_bank/mechanism_pose_bank_integrity_report.json"
    require(Path(str(completion.get("integrity_report", ""))).resolve() == integrity_path, "pose-bank integrity-report path mismatch")
    integrity = load_json(integrity_path, "pose-bank integrity report")
    require_passed(integrity, "pose-bank integrity report")
    require(integrity.get("schema_version") == "surfna-lb200-mechanism-pose-bank-integrity-v1", "pose-bank integrity schema mismatch")
    require(integrity.get("bank_sha256") == bank_sha256, "pose-bank integrity hash mismatch")
    require(integrity.get("entry_count") == EXPECTED_POSE_BANK_COUNT, "pose-bank entry count mismatch")
    require(integrity.get("finite_failure_count") == 0, "pose-bank finite failures present")
    require(integrity.get("torsion_cardinality_failure_count") == 0, "pose-bank torsion cardinality failures present")

    require(Path(str(completion.get("frozen_validation_cache_contract", ""))).resolve() == cache_contract_path, "completion cache-contract path mismatch")
    cache_contract_sha256 = hasher(cache_contract_path, "validation cache contract")
    require(completion.get("frozen_validation_cache_contract_sha256") == cache_contract_sha256, "validation cache-contract hash mismatch")
    cache_contract = load_json(cache_contract_path, "validation cache contract")
    require(cache_contract.get("schema_version") == "surfna-lb200-frozen-validation-cache-v1", "validation cache-contract schema mismatch")
    require(cache_contract.get("automatic_rebuild_allowed") is False, "validation cache permits automatic rebuilding")
    require(cache_contract.get("complex_count") == EXPECTED_VALIDATION_COUNT, "validation cache complex count mismatch")
    for path_key, hash_key in (
        ("split_path", "split_sha256"),
        ("heterographs_path", "heterographs_sha256"),
        ("rdkit_ligands_path", "rdkit_ligands_sha256"),
    ):
        artifact_path = Path(str(cache_contract.get(path_key, "")))
        expected_sha256 = require_sha256(cache_contract.get(hash_key), f"validation cache {path_key}")
        require(hasher(artifact_path, f"validation cache {path_key}") == expected_sha256, f"validation cache artifact hash mismatch: {path_key}")

    chemistry_integrity_path = root / "04_mechanism_pose_bank/chemistry_shuffle_integrity_report.json"
    chemistry_integrity = load_json(chemistry_integrity_path, "chemistry-shuffle integrity report")
    require_passed(chemistry_integrity, "chemistry-shuffle integrity report")
    require(chemistry_integrity.get("schema_version") == "surfna-lb200-chemistry-shuffle-integrity-v1", "chemistry-shuffle integrity schema mismatch")
    require(chemistry_integrity.get("complexes") == EXPECTED_VALIDATION_COUNT, "chemistry-shuffle complex count mismatch")
    require(chemistry_integrity.get("permutations") == EXPECTED_VALIDATION_COUNT * len(EXPECTED_SHUFFLE_SEEDS), "chemistry-shuffle permutation count mismatch")
    for field in (
        "coordinates_and_edge_indices_unchanged",
        "si_and_boundary_unchanged",
        "joint_chemistry_row_multiset_unchanged",
        "all_finite",
    ):
        require(chemistry_integrity.get(field) is True, f"chemistry-shuffle integrity flag failed: {field}")

    gate_path = root / "03_integrity/surface_gate_integrity_report.json"
    gate_protocol = load_json(
        root / "00_protocol/surface_gate_protocol.json",
        "surface-gate frozen protocol",
    )
    require(
        gate_protocol.get("schema_version") == "surfna-lb200-surface-output-gate-v2",
        "surface-gate frozen protocol schema mismatch",
    )
    frozen_gate_integrity = gate_protocol.get("integrity_protocol")
    require(isinstance(frozen_gate_integrity, dict), "surface-gate frozen integrity protocol is missing")
    require(frozen_gate_integrity.get("backend") == "CUDA float32 with TF32 disabled", "surface-gate frozen backend drift")
    require(frozen_gate_integrity.get("comparison_order") == "ABBA", "surface-gate frozen comparison order drift")
    require(frozen_gate_integrity.get("abba_cycles") == EXPECTED_GATE_ABBA_CYCLES, "surface-gate frozen ABBA cycle drift")
    require(frozen_gate_integrity.get("repeats_per_condition") == EXPECTED_GATE_REPEATS_PER_CONDITION, "surface-gate frozen repeat-count drift")
    require(frozen_gate_integrity.get("null_envelope_epsilon_multiplier") == EXPECTED_GATE_EPS_MULTIPLIER, "surface-gate frozen epsilon multiplier drift")
    require(frozen_gate_integrity.get("maximum_null_envelope_abs") == EXPECTED_GATE_MAX_NULL_ENVELOPE_ABS, "surface-gate frozen envelope cap drift")
    require(frozen_gate_integrity.get("positive_control_repeats_per_condition") == EXPECTED_GATE_REPEATS_PER_CONDITION, "surface-gate frozen positive-control repeats drift")
    require(frozen_gate_integrity.get("minimum_positive_effect_to_tau_ratio") == EXPECTED_GATE_POSITIVE_RATIO, "surface-gate frozen positive-control ratio drift")
    require(frozen_gate_integrity.get("minimum_positive_effect_absolute") == EXPECTED_GATE_POSITIVE_ABS, "surface-gate frozen positive-control floor drift")
    gate = load_json(gate_path, "surface-gate integrity report")
    require_passed(gate, "surface-gate integrity report")
    require(gate.get("schema_version") == "surfna-lb200-surface-output-gate-integrity-v3", "surface-gate integrity schema mismatch")
    gate_tests = gate.get("tests")
    require(isinstance(gate_tests, dict) and gate_tests and all(value is True for value in gate_tests.values()), "one or more surface-gate integrity tests failed")
    gate_thresholds = gate.get("thresholds")
    require(isinstance(gate_thresholds, dict), "surface-gate threshold contract is missing")
    require(gate_thresholds.get("abba_cycles") == EXPECTED_GATE_ABBA_CYCLES, "surface-gate ABBA cycle count drift")
    require(gate_thresholds.get("repeats_per_condition") == EXPECTED_GATE_REPEATS_PER_CONDITION, "surface-gate repeat count drift")
    require(gate_thresholds.get("positive_control_repeats_per_condition") == EXPECTED_GATE_REPEATS_PER_CONDITION, "surface-gate positive-control repeat count drift")
    require(gate_thresholds.get("null_envelope_epsilon_multiplier") == EXPECTED_GATE_EPS_MULTIPLIER, "surface-gate epsilon multiplier drift")
    require(gate_thresholds.get("maximum_null_envelope_abs") == EXPECTED_GATE_MAX_NULL_ENVELOPE_ABS, "surface-gate null-envelope cap drift")
    require(gate_thresholds.get("minimum_positive_effect_to_tau_ratio") == EXPECTED_GATE_POSITIVE_RATIO, "surface-gate positive-control ratio drift")
    require(gate_thresholds.get("minimum_positive_effect_absolute") == EXPECTED_GATE_POSITIVE_ABS, "surface-gate positive-control floor drift")
    for key in (
        "abba_cycles", "repeats_per_condition", "positive_control_repeats_per_condition",
        "null_envelope_epsilon_multiplier", "maximum_null_envelope_abs",
        "minimum_positive_effect_to_tau_ratio", "minimum_positive_effect_absolute",
    ):
        require(gate_thresholds.get(key) == frozen_gate_integrity.get(key), f"surface-gate report/protocol mismatch: {key}")
    backend = gate.get("backend")
    require(isinstance(backend, dict), "surface-gate backend record is missing")
    require(str(backend.get("device", "")).startswith("cuda"), "surface-gate certificate was not produced on CUDA")
    require(gate.get("dtype") == "torch.float32", "surface-gate certificate did not use float32")
    require(backend.get("tf32_matmul_allowed") is False, "surface-gate certificate allowed TF32 matmul")
    require(backend.get("tf32_cudnn_allowed") is False, "surface-gate certificate allowed TF32 cuDNN")
    def require_bounded_null_certificate(certificate: Any, label: str) -> None:
        require(isinstance(certificate, dict) and certificate.get("passed") is True, f"{label} failed")
        heads = certificate.get("heads")
        require(isinstance(heads, dict) and set(heads) == {"translation", "rotation", "torsion"}, f"{label} head set drift")
        for head_name, record in heads.items():
            require(isinstance(record, dict), f"{label}/{head_name} record malformed")
            require(record.get("repeat_count_left") == EXPECTED_GATE_REPEATS_PER_CONDITION, f"{label}/{head_name} left repeat count drift")
            require(record.get("repeat_count_right") == EXPECTED_GATE_REPEATS_PER_CONDITION, f"{label}/{head_name} right repeat count drift")
            require(record.get("maximum_null_envelope_abs") == EXPECTED_GATE_MAX_NULL_ENVELOPE_ABS, f"{label}/{head_name} envelope cap drift")
            require(record.get("null_envelopes_within_cap") is True, f"{label}/{head_name} exceeded the frozen envelope cap")
            require(record.get("all_repeats_finite") is True and record.get("passed") is True, f"{label}/{head_name} failed")

    require_bounded_null_certificate(
        gate.get("alpha1_original_null_envelope"),
        "alpha=1 original-equivalence null-envelope certificate",
    )
    chemistry_nulls = gate.get("alpha0_chemistry_null_envelopes")
    require(
        isinstance(chemistry_nulls, dict)
        and set(chemistry_nulls) == {"20260911", "20260912", "20260913"},
        "alpha=0 chemistry null-envelope certificate set drift",
    )
    for shuffle_seed, certificate in chemistry_nulls.items():
        require_bounded_null_certificate(certificate, f"alpha=0 chemistry null envelope {shuffle_seed}")
    require_bounded_null_certificate(
        gate.get("alpha0_position_null_envelope"),
        "alpha=0 position null-envelope certificate",
    )
    chemistry_positive = gate.get("alpha1_chemistry_positive_controls")
    require(
        isinstance(chemistry_positive, dict)
        and set(chemistry_positive) == {"20260911", "20260912", "20260913"},
        "alpha=1 chemistry positive-control set drift",
    )
    for shuffle_seed, certificate in chemistry_positive.items():
        require(isinstance(certificate, dict) and certificate.get("passed") is True, f"alpha=1 chemistry positive control failed: {shuffle_seed}")
        heads = certificate.get("heads")
        require(isinstance(heads, dict) and set(heads) == {"translation", "rotation", "torsion"}, f"alpha=1 chemistry positive-control head set drift: {shuffle_seed}")
        require(bool(certificate.get("passed_heads")), f"alpha=1 chemistry positive control has no separated head: {shuffle_seed}")
        require(all(record.get("null_envelopes_within_cap") is True for record in heads.values()), f"alpha=1 chemistry positive-control envelope cap failed: {shuffle_seed}")
    position_positive = gate.get("alpha1_position_positive_control")
    require(isinstance(position_positive, dict) and position_positive.get("passed") is True, "alpha=1 position positive control failed")
    require(bool(position_positive.get("passed_heads")), "alpha=1 position positive control has no separated head")
    require(
        all(record.get("null_envelopes_within_cap") is True for record in position_positive.get("heads", {}).values()),
        "alpha=1 position positive-control envelope cap failed",
    )
    jacobians = gate.get("jacobian_certificates")
    require(isinstance(jacobians, dict), "surface-gate Jacobian certificates are missing")
    for label in ("alpha0_surface_x", "alpha0_surface_pos"):
        require(
            jacobians.get(label, {}).get("all_finite") is True
            and jacobians[label].get("all_vjps_exact_zero") is True,
            f"surface-gate exact disconnection certificate failed: {label}",
        )
    for label in ("alpha1_surface_x", "alpha1_surface_pos"):
        require(
            jacobians.get(label, {}).get("all_finite") is True
            and jacobians[label].get("any_vjp_nonzero") is True,
            f"surface-gate open-path Jacobian certificate failed: {label}",
        )

    smoke_path = root / "03_integrity/full_one_complex_smoke_report.json"
    smoke = load_json(smoke_path, "full one-complex smoke report")
    require_passed(smoke, "full one-complex smoke report")
    require(
        smoke.get("schema_version") == "surfna-lb200-one-complex-smoke-v2",
        "full one-complex smoke schema mismatch",
    )
    require(smoke.get("test_data_used") is False, "smoke test touched fixed test data")
    smoke_tests = smoke.get("tests")
    require(
        isinstance(smoke_tests, dict)
        and smoke_tests
        and all(value is True for value in smoke_tests.values()),
        "one or more full smoke subtests failed",
    )
    checkpoints = smoke.get("checkpoints")
    require(isinstance(checkpoints, list) and len(checkpoints) == 2, "smoke checkpoint list mismatch")
    require(
        {record.get("epoch") for record in checkpoints if isinstance(record, dict)} == {25, 200},
        "smoke test did not cover EMA epochs 25 and 200 exactly",
    )
    require(smoke.get("epochs_tested") == [25, 200], "smoke epoch summary mismatch")
    smoke_csv_path = Path(str(smoke.get("metrics_csv", "")))
    expected_smoke_csv_path = root / "03_integrity/full_one_complex_smoke_metrics.csv"
    require(smoke_csv_path.resolve() == expected_smoke_csv_path, "smoke metrics CSV path mismatch")
    require(smoke.get("metrics_row_count") == 10, "smoke metrics CSV row-count declaration mismatch")
    smoke_csv_sha256 = require_sha256(smoke.get("metrics_csv_sha256"), "smoke metrics CSV")
    require(
        hasher(smoke_csv_path, "smoke metrics CSV") == smoke_csv_sha256,
        "smoke metrics CSV hash mismatch",
    )
    with smoke_csv_path.open("r", encoding="utf-8") as handle:
        smoke_csv_row_count = sum(1 for _ in handle) - 1
    require(smoke_csv_row_count == 10, "smoke metrics CSV does not contain exactly 10 data rows")

    required_markers = contract.get("preflight_required_passed_markers")
    require(isinstance(required_markers, list), "contract lacks required passed-marker list")
    for relative_path in required_markers:
        marker_path = safe_root_path(root, relative_path, "required passed marker")
        require_passed(load_json(marker_path, f"required passed marker {relative_path}"), relative_path)

    expected_permutation_sha256 = require_sha256(
        derived.get("expected_shuffle_permutation_manifest_sha256"),
        "expected permutation manifest",
    )
    require(
        hasher(expected_permutation_path, "expected permutation manifest")
        == expected_permutation_sha256,
        "expected permutation-manifest hash mismatch",
    )
    permutation_sha256 = validate_permutations(permutation_path, cache_contract, hasher)
    require(
        permutation_sha256 == expected_permutation_sha256,
        "generated permutation manifest differs from the pre-analysis frozen assignment",
    )
    return bank_sha256, permutation_sha256, cache_contract_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--epoch", type=int, choices=EPOCHS, required=True)
    args = parser.parse_args()
    root = args.analysis_root.resolve()
    require(root.is_dir(), f"analysis root does not exist: {root}")
    hasher = FileHasher()

    contract, contract_sha256 = validate_contract(root, hasher)
    execution_inventory_sha256 = validate_execution_code(root, contract, hasher)
    frozen_source_inventory_sha256 = validate_frozen_model_sources(root, contract, hasher)
    checkpoint_path, checkpoint_sha256, checkpoint_inventory_sha256 = validate_checkpoint_inventory(
        root, contract, args.arm, args.seed, args.epoch, hasher
    )
    epoch0_path, epoch0_sha256, epoch0_audit_sha256 = validate_epoch0_audit(
        root, contract, args.arm, args.seed, args.epoch, hasher
    )
    require(checkpoint_path is not None and checkpoint_sha256 is not None, "selected checkpoint was not resolved")
    pose_bank_sha256, permutation_sha256, cache_contract_sha256 = validate_pose_bank_and_markers(
        root, contract, hasher
    )

    result = {
        "schema_version": "surfna-lb200-analysis-preflight-v1",
        "passed": True,
        "analysis_root": str(root),
        "arm": args.arm,
        "seed": args.seed,
        "epoch": args.epoch,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_inventory_sha256": checkpoint_inventory_sha256,
        "epoch0_audit_sha256": epoch0_audit_sha256,
        "frozen_contract_sha256": contract_sha256,
        "execution_code_inventory_sha256": execution_inventory_sha256,
        "frozen_model_source_inventory_sha256": frozen_source_inventory_sha256,
        "pose_bank_sha256": pose_bank_sha256,
        "permutation_manifest_sha256": permutation_sha256,
        "frozen_validation_cache_contract_sha256": cache_contract_sha256,
    }
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (PreflightError, OSError, ValueError, TypeError) as error:
        raise SystemExit(f"PRECHECK_FAILED: {error}") from error
