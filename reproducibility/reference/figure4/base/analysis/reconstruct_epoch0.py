#!/usr/bin/env python3
"""Reconstruct and hash-verify the nine frozen epoch-0 EMA states."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

from mechanism_common import (
    ARMS,
    EXPERIMENT_ROOT,
    EXPECTED_SOURCE_SHA256,
    SEEDS,
    arm_slug,
    atomic_json,
    atomic_torch_save,
    model_dir,
    sha256_file,
    stable_state_sha256,
)


EXPECTED_MODEL_STATE_COUNT = 463
EXPECTED_COMPATIBLE_COUNT = 383
EXPECTED_NONSURFACE_COMPATIBLE_COUNT = 168
EXPECTED_SURFACE_COMPATIBLE_COUNT = 215
EXPECTED_NA_SPECIFIC_COUNT = 80
EXPECTED_IGNORED_SOURCE_KEYS = {
    "ligand_patch_tokenizer.ligand_update.weight",
    "ligand_patch_tokenizer.surface_value.weight",
    "ligand_patch_tokenizer.token_norm.bias",
    "ligand_patch_tokenizer.token_norm.weight",
}


def receipt_path(arm: str, seed: int) -> Path:
    slug = arm_slug(arm)
    suffix = "e200_ddp2_b8"
    if arm == "Scratch-Full8" and seed in (0, 1):
        suffix = "e500_ddp2_b8"
    if arm == "Transfer-NonSurface" and seed == 0:
        suffix = "e500_ddp2_b8"
    return EXPERIMENT_ROOT / "03_training" / f"{slug}_seed{seed}_{suffix}.INITIALIZATION.json"


def normalize_source(raw):
    state = raw.get("model", raw) if isinstance(raw, dict) else raw
    return {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}


def require_exact_key_set(label: str, observed, expected) -> set[str]:
    observed_list = list(observed)
    observed_set = set(observed_list)
    expected_set = set(expected)
    if len(observed_list) != len(observed_set):
        raise RuntimeError(f"{label} contains duplicate keys")
    if observed_set != expected_set:
        missing = sorted(expected_set - observed_set)
        extra = sorted(observed_set - expected_set)
        raise RuntimeError(
            f"{label} key-set mismatch: missing={missing[:10]} extra={extra[:10]} "
            f"observed={len(observed_set)} expected={len(expected_set)}"
        )
    return observed_set


def tensor_mismatch_keys(left, right, keys) -> list[str]:
    mismatches = []
    for key in sorted(keys):
        if key not in left or key not in right:
            mismatches.append(key)
            continue
        if not torch.equal(left[key].detach().cpu(), right[key].detach().cpu()):
            mismatches.append(key)
    return mismatches


def subset_state_sha256(state, keys) -> str:
    subset = state.__class__()
    for key in sorted(keys):
        subset[key] = state[key]
    return stable_state_sha256(subset)


def key_set_sha256(keys) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis_root.resolve()
    audit_path = root / "02_checkpoints/epoch0_reconstruction_audit.json"
    if audit_path.exists():
        raise SystemExit(f"refusing to overwrite epoch-0 audit: {audit_path}")

    manifest_path = EXPERIMENT_ROOT / "01_initialization/parameter_transfer_manifest_final.json"
    manifest_sha256 = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "final_after_real_graph_smoke":
        raise RuntimeError("parameter-transfer manifest is not final after real-graph smoke")
    manifest_records = manifest.get("records")
    if not isinstance(manifest_records, list):
        raise RuntimeError("parameter-transfer manifest records are absent or malformed")
    record_by_key = {record["key"]: record for record in manifest_records}
    if len(record_by_key) != len(manifest_records):
        raise RuntimeError("parameter-transfer manifest contains duplicate parameter keys")
    if len(record_by_key) != EXPECTED_MODEL_STATE_COUNT:
        raise RuntimeError(
            f"target-state manifest drift: observed={len(record_by_key)} "
            f"expected={EXPECTED_MODEL_STATE_COUNT}"
        )
    malformed_compatibility = [
        record.get("key") for record in manifest_records
        if type(record.get("source_compatible")) is not bool
    ]
    if malformed_compatibility:
        raise RuntimeError(
            "manifest source_compatible must be boolean for every tensor: "
            f"{malformed_compatibility[:10]}"
        )
    invalid_groups = {
        record.get("transfer_group") for record in manifest_records
        if record.get("transfer_group") not in {"SurfacePath", "NonSurfacePath"}
    }
    if invalid_groups:
        raise RuntimeError(
            "unexpected transfer groups in manifest: "
            f"{sorted(repr(group) for group in invalid_groups)}"
        )

    manifest_compatible_keys = {
        key for key, record in record_by_key.items() if bool(record.get("source_compatible"))
    }
    manifest_naspecific_keys = set(record_by_key) - manifest_compatible_keys
    manifest_surface_keys = {
        key for key, record in record_by_key.items() if record["transfer_group"] == "SurfacePath"
    }
    manifest_nonsurface_keys = set(record_by_key) - manifest_surface_keys
    expected_transfer_nonsurface_keys = {
        key for key in manifest_compatible_keys
        if record_by_key[key]["transfer_group"] == "NonSurfacePath"
    }
    expected_transfer_surface_keys = manifest_compatible_keys - expected_transfer_nonsurface_keys
    expected_counts = {
        "compatible": (len(manifest_compatible_keys), EXPECTED_COMPATIBLE_COUNT),
        "Transfer-NonSurface": (
            len(expected_transfer_nonsurface_keys), EXPECTED_NONSURFACE_COMPATIBLE_COUNT
        ),
        "compatible SurfacePath": (
            len(expected_transfer_surface_keys), EXPECTED_SURFACE_COMPATIBLE_COUNT
        ),
        "NASpecific": (len(manifest_naspecific_keys), EXPECTED_NA_SPECIFIC_COUNT),
    }
    for label, (observed, expected) in expected_counts.items():
        if observed != expected:
            raise RuntimeError(f"{label} count drift: observed={observed} expected={expected}")
    expected_manifest_compatibility = {
        "loaded": EXPECTED_COMPATIBLE_COUNT,
        "missing": EXPECTED_NA_SPECIFIC_COUNT,
        "unexpected": 0,
        "skipped_shape": 0,
    }
    if manifest.get("source_compatibility") != expected_manifest_compatibility:
        raise RuntimeError(
            "manifest source compatibility drift: "
            f"observed={manifest.get('source_compatibility')} "
            f"expected={expected_manifest_compatibility}"
        )
    ignored_source_keys = require_exact_key_set(
        "manifest ignored source keys",
        manifest.get("ignored_source_state_keys", []),
        EXPECTED_IGNORED_SOURCE_KEYS,
    )
    shape_mismatch_source_keys = require_exact_key_set(
        "manifest shape-mismatch source keys",
        manifest.get("shape_mismatch_source_keys", []),
        set(),
    )

    staged_records = []
    paired_seed_audits = []
    source_state_cache = {}
    for seed in SEEDS:
        seed_inputs = {}
        for arm in ARMS:
            init_path = EXPERIMENT_ROOT / "01_initialization" / f"init_{arm_slug(arm)}_seed{seed}.json"
            config = json.loads(init_path.read_text())
            if config.get("arm") != arm or config.get("seed") != seed:
                raise RuntimeError(
                    f"initialization config identity mismatch: {init_path} "
                    f"contains arm={config.get('arm')} seed={config.get('seed')}"
                )
            base_path = Path(config["base_init"]["path"]).resolve(strict=True)
            source_path = Path(config["source_checkpoint"]["path"]).resolve(strict=True)
            configured_manifest = Path(config["manifest"]["path"]).resolve(strict=True)
            if configured_manifest != manifest_path.resolve(strict=True):
                raise RuntimeError(f"non-canonical parameter manifest in {init_path}: {configured_manifest}")
            if config["manifest"].get("sha256") != manifest_sha256:
                raise RuntimeError(f"declared parameter manifest hash mismatch: {init_path}")
            base_file_sha256 = sha256_file(base_path)
            if base_file_sha256 != config["base_init"]["sha256"]:
                raise RuntimeError(f"base init hash mismatch: {base_path}")
            source_file_sha256 = sha256_file(source_path)
            if config["source_checkpoint"].get("sha256") != EXPECTED_SOURCE_SHA256:
                raise RuntimeError(f"declared protein source hash mismatch: {init_path}")
            if source_file_sha256 != EXPECTED_SOURCE_SHA256:
                raise RuntimeError(f"protein source hash mismatch: {source_path}")

            base_raw = torch.load(base_path, map_location="cpu")
            if not isinstance(base_raw, dict) or not isinstance(base_raw.get("model"), dict):
                raise RuntimeError(f"paired base checkpoint has no model state: {base_path}")
            base_state = base_raw["model"]
            require_exact_key_set(f"base/manifest {arm}/seed{seed}", base_state, record_by_key)
            if source_path not in source_state_cache:
                source_state_cache[source_path] = normalize_source(
                    torch.load(source_path, map_location="cpu")
                )
            source_state = source_state_cache[source_path]
            unexpected_source_keys = set(source_state) - set(base_state)
            require_exact_key_set(
                f"source-only ignored keys/manifest {arm}/seed{seed}",
                unexpected_source_keys,
                ignored_source_keys,
            )
            shape_mismatch = {
                key for key, value in source_state.items()
                if key in base_state and tuple(value.shape) != tuple(base_state[key].shape)
            }
            require_exact_key_set(
                f"source/target shape mismatch/manifest {arm}/seed{seed}",
                shape_mismatch,
                shape_mismatch_source_keys,
            )
            compatible = {
                key: value for key, value in source_state.items()
                if key in base_state and tuple(value.shape) == tuple(base_state[key].shape)
            }
            require_exact_key_set(
                f"source-compatible/manifest {arm}/seed{seed}", compatible, manifest_compatible_keys
            )
            if not isinstance(config.get("transfer_keys"), list):
                raise RuntimeError(f"transfer_keys must be a list: {init_path}")
            selected = list(config["transfer_keys"])
            if arm == "Scratch-Full8":
                expected_selected = set()
                semantic_definition = "empty_allowlist"
            elif arm == "Transfer-All":
                expected_selected = manifest_compatible_keys
                semantic_definition = "all_383_source_compatible_keys"
            else:
                expected_selected = expected_transfer_nonsurface_keys
                semantic_definition = "all_and_only_168_manifest_NonSurfacePath_compatible_keys"
            selected_set = require_exact_key_set(
                f"{arm}/seed{seed} transfer allowlist", selected, expected_selected
            )
            state = base_state.__class__()
            for key, value in base_state.items():
                state[key] = (
                    compatible[key] if key in selected_set else value
                ).detach().cpu().clone()
            for key in base_state:
                expected = compatible[key] if key in selected_set else base_state[key]
                if not torch.equal(state[key], expected.detach().cpu()):
                    raise RuntimeError(f"epoch-0 tensor mismatch: {arm}/seed{seed}/{key}")
            digest = stable_state_sha256(state)
            current_receipt_path = receipt_path(arm, seed)
            receipt = json.loads(current_receipt_path.read_text())
            if receipt.get("arm") != arm or receipt.get("seed") != seed:
                raise RuntimeError(f"initialization receipt identity mismatch: {current_receipt_path}")
            receipt_digest = receipt["initial_full_state_sha256"]
            if digest != receipt_digest:
                raise RuntimeError(
                    f"epoch-0 state digest mismatch for {arm}/seed{seed}: "
                    f"observed={digest}, receipt={receipt_digest}"
                )
            output = root / "02_checkpoints" / arm_slug(arm) / f"seed{seed}" / "ema_epoch_0000.pt"
            parameters_copy = output.parent / "model_parameters.yml"
            parameters_source = model_dir(arm, seed) / "model_parameters.yml"
            if not parameters_source.is_file():
                raise RuntimeError(f"missing model parameters for {arm}/seed{seed}: {parameters_source}")
            parameters_source_sha256 = sha256_file(parameters_source)
            for planned_output in (output, parameters_copy):
                if planned_output.exists():
                    raise RuntimeError(f"refusing to overwrite reconstructed artifact: {planned_output}")
            record = {
                "arm": arm,
                "seed": seed,
                "output": str(output),
                "state_sha256": digest,
                "tensor_count": len(state),
                "selected_source_tensor_count": len(selected_set),
                "selected_source_key_set_sha256": key_set_sha256(selected_set),
                "selection_semantics": semantic_definition,
                "exact_expected_allowlist": True,
                "base_init": str(base_path),
                "base_init_sha256": base_file_sha256,
                "base_state_sha256": stable_state_sha256(base_state),
                "source_checkpoint": str(source_path),
                "source_checkpoint_sha256": source_file_sha256,
                "source_state_tensor_count": len(source_state),
                "ignored_source_state_keys": sorted(unexpected_source_keys),
                "ignored_source_state_key_set_sha256": key_set_sha256(unexpected_source_keys),
                "shape_mismatch_source_keys": sorted(shape_mismatch),
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha256,
                "initialization_receipt": str(current_receipt_path),
                "receipt_state_sha256": receipt_digest,
                "verification": "exact_tensor_digest_match",
                "formal_use": False,
                "analysis_role": "diagnostic_initialization_audit_only",
                "model_parameters_source": str(parameters_source),
                "model_parameters_source_sha256": parameters_source_sha256,
                "model_parameters_copy": str(parameters_copy),
                "model_parameters_copy_sha256": parameters_source_sha256,
                "provenance_note": (
                    "The formal Scratch seed0 EMA25-200 trajectory is the external completed run; "
                    "its epoch0 state is reconstructed from the deterministic seed0 target state. "
                    "The matching paired-run initialization receipt and independent workflow replay "
                    "both reproduce this digest."
                    if arm == "Scratch-Full8" and seed == 0 else ""
                ),
            }
            seed_inputs[arm] = {
                "config": config,
                "config_path": init_path,
                "base_path": base_path,
                "base_file_sha256": base_file_sha256,
                "base_state": base_state,
                "base_state_sha256": stable_state_sha256(base_state),
                "selected_keys": selected_set,
                "state": state,
                "record": record,
                "output": output,
                "parameters_copy": parameters_copy,
                "parameters_source": parameters_source,
            }

        reference_arm = ARMS[0]
        reference_base = seed_inputs[reference_arm]["base_state"]
        base_paths = {str(seed_inputs[arm]["base_path"]) for arm in ARMS}
        base_file_hashes = {seed_inputs[arm]["base_file_sha256"] for arm in ARMS}
        base_state_hashes = {seed_inputs[arm]["base_state_sha256"] for arm in ARMS}
        base_mismatches = {
            arm: tensor_mismatch_keys(reference_base, seed_inputs[arm]["base_state"], reference_base)
            for arm in ARMS
        }
        if len(base_paths) != 1 or len(base_file_hashes) != 1 or len(base_state_hashes) != 1:
            raise RuntimeError(
                f"same-seed three-arm base identity mismatch for seed{seed}: "
                f"paths={sorted(base_paths)} file_hashes={sorted(base_file_hashes)} "
                f"state_hashes={sorted(base_state_hashes)}"
            )
        if any(base_mismatches.values()):
            raise RuntimeError(
                f"same-seed three-arm base tensor mismatch for seed{seed}: {base_mismatches}"
            )

        scratch_state = seed_inputs["Scratch-Full8"]["state"]
        scratch_base_mismatches = tensor_mismatch_keys(
            scratch_state, reference_base, reference_base
        )
        if scratch_base_mismatches:
            raise RuntimeError(
                f"Scratch-Full8 is not the unmodified paired base for seed{seed}: "
                f"{scratch_base_mismatches[:10]}"
            )

        transfer_all_state = seed_inputs["Transfer-All"]["state"]
        transfer_nonsurface_state = seed_inputs["Transfer-NonSurface"]["state"]
        outside_surface_mismatches = tensor_mismatch_keys(
            transfer_all_state, transfer_nonsurface_state, manifest_nonsurface_keys
        )
        if outside_surface_mismatches:
            raise RuntimeError(
                f"Transfer-All/Transfer-NonSurface differ outside SurfacePath for seed{seed}: "
                f"{outside_surface_mismatches[:10]}"
            )

        naspecific_mismatches = {}
        for arm in ARMS:
            mismatches = tensor_mismatch_keys(
                seed_inputs[arm]["state"], reference_base, manifest_naspecific_keys
            )
            naspecific_mismatches[arm] = mismatches
            if mismatches:
                raise RuntimeError(
                    f"NASpecific tensors do not remain at paired base for {arm}/seed{seed}: "
                    f"{mismatches[:10]}"
                )

        paired_seed_audits.append({
            "seed": seed,
            "passed": True,
            "base_pairing": {
                "passed": True,
                "same_path": True,
                "same_file_sha256": True,
                "same_state_sha256": True,
                "exact_tensorwise_equal": True,
                "path": next(iter(base_paths)),
                "file_sha256": next(iter(base_file_hashes)),
                "state_sha256": next(iter(base_state_hashes)),
                "per_arm": {
                    arm: {
                        "config": str(seed_inputs[arm]["config_path"]),
                        "base_path": str(seed_inputs[arm]["base_path"]),
                        "base_file_sha256": seed_inputs[arm]["base_file_sha256"],
                        "base_state_sha256": seed_inputs[arm]["base_state_sha256"],
                        "mismatch_keys": base_mismatches[arm],
                    }
                    for arm in ARMS
                },
            },
            "allowlist_semantics": {
                arm: {
                    "passed": True,
                    "definition": seed_inputs[arm]["record"]["selection_semantics"],
                    "selected_tensor_count": len(seed_inputs[arm]["selected_keys"]),
                    "selected_key_set_sha256": key_set_sha256(seed_inputs[arm]["selected_keys"]),
                }
                for arm in ARMS
            },
            "Transfer-All_vs_Transfer-NonSurface": {
                "passed": True,
                "comparison_scope": "all target-state tensors outside manifest SurfacePath",
                "compared_tensor_count": len(manifest_nonsurface_keys),
                "mismatch_keys": outside_surface_mismatches,
                "Transfer-All_subset_sha256": subset_state_sha256(
                    transfer_all_state, manifest_nonsurface_keys
                ),
                "Transfer-NonSurface_subset_sha256": subset_state_sha256(
                    transfer_nonsurface_state, manifest_nonsurface_keys
                ),
            },
            "NASpecific_remains_base": {
                "passed": True,
                "definition": "target-state tensors absent from the compatible protein source state",
                "tensor_count": len(manifest_naspecific_keys),
                "base_subset_sha256": subset_state_sha256(
                    reference_base, manifest_naspecific_keys
                ),
                "mismatch_keys_by_arm": naspecific_mismatches,
            },
            "Scratch_equals_base": {
                "passed": True,
                "compared_tensor_count": len(reference_base),
                "mismatch_keys": scratch_base_mismatches,
            },
        })
        staged_records.extend(seed_inputs[arm] for arm in ARMS)

    # All scientific pairing assertions above are fail-closed and run before
    # the first reconstructed checkpoint is written.
    records = []
    for staged in staged_records:
        output = staged["output"]
        atomic_torch_save(output, staged["state"])
        shutil.copy2(staged["parameters_source"], staged["parameters_copy"])
        if sha256_file(staged["parameters_copy"]) != staged["record"]["model_parameters_copy_sha256"]:
            raise RuntimeError(
                f"copied model parameters failed SHA-256 round-trip: {staged['parameters_copy']}"
            )
        loaded = torch.load(output, map_location="cpu")
        if stable_state_sha256(loaded) != staged["record"]["state_sha256"]:
            raise RuntimeError(f"saved epoch-0 checkpoint failed round-trip: {output}")
        staged["record"]["file_sha256"] = sha256_file(output)
        records.append(staged["record"])

    atomic_json(audit_path, {
        "schema_version": "surfna-lb200-epoch0-reconstruction-audit-v1",
        "passed": True,
        "formal_use": False,
        "analysis_role": "diagnostic_initialization_audit_only",
        "definition": (
            "EMA is initialized by cloning all trainable parameters immediately after the paired "
            "model initialization and before the first optimizer step; therefore the epoch-0 EMA "
            "state equals the verified complete initial model state."
        ),
        "model_count": len(records),
        "paired_audit": {
            "schema_version": "surfna-lb200-paired-initialization-audit-v1",
            "passed": True,
            "fail_closed_before_checkpoint_write": True,
            "required_semantics": {
                "same_seed_three_arms_share_exact_base": True,
                "Transfer-All_equals_all_383_compatible_keys": True,
                "Transfer-NonSurface_equals_all_and_only_manifest_NonSurfacePath_168": True,
                "Scratch_transfer_allowlist_empty": True,
                "Transfer-All_and_Transfer-NonSurface_equal_outside_SurfacePath": True,
                "NASpecific_remains_at_paired_base": True,
                "source_only_keys_equal_frozen_ignored_allowlist": True,
                "source_target_shape_mismatch_set_empty": True,
            },
            "manifest_partition": {
                "target_state_tensor_count": len(record_by_key),
                "source_compatible_tensor_count": len(manifest_compatible_keys),
                "compatible_SurfacePath_tensor_count": len(expected_transfer_surface_keys),
                "compatible_NonSurfacePath_tensor_count": len(expected_transfer_nonsurface_keys),
                "all_SurfacePath_target_tensor_count": len(manifest_surface_keys),
                "all_non_SurfacePath_target_tensor_count": len(manifest_nonsurface_keys),
                "NASpecific_tensor_count": len(manifest_naspecific_keys),
                "ignored_source_tensor_count": len(ignored_source_keys),
                "ignored_source_state_keys": sorted(ignored_source_keys),
                "shape_mismatch_source_keys": sorted(shape_mismatch_source_keys),
                "key_sets": {
                    "Transfer-All": sorted(manifest_compatible_keys),
                    "Transfer-NonSurface": sorted(expected_transfer_nonsurface_keys),
                    "Scratch-Full8": [],
                    "SurfacePath_compatible": sorted(expected_transfer_surface_keys),
                    "SurfacePath_all_target": sorted(manifest_surface_keys),
                    "non_SurfacePath_all_target": sorted(manifest_nonsurface_keys),
                    "NASpecific": sorted(manifest_naspecific_keys),
                },
                "key_set_sha256": {
                    "Transfer-All": key_set_sha256(manifest_compatible_keys),
                    "Transfer-NonSurface": key_set_sha256(expected_transfer_nonsurface_keys),
                    "Scratch-Full8": key_set_sha256(set()),
                    "SurfacePath_compatible": key_set_sha256(expected_transfer_surface_keys),
                    "SurfacePath_all_target": key_set_sha256(manifest_surface_keys),
                    "non_SurfacePath_all_target": key_set_sha256(manifest_nonsurface_keys),
                    "NASpecific": key_set_sha256(manifest_naspecific_keys),
                },
            },
            "seeds": paired_seed_audits,
        },
        "records": records,
    })
    print(json.dumps({
        "passed": True,
        "model_count": len(records),
        "paired_seed_count": len(paired_seed_audits),
        "audit": str(audit_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
