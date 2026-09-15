"""Fail-closed contracts for the L2 native-MDN B arm; no legacy V3 data paths.

The numerical recipe is adapted from native_prior_v3_20260828/run_arm.py.
This module deliberately uses only the standard library, including for gates.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

DIFFUSION_SHA256 = "c90a95381926534dcdddc7699ba3d5cc400afd23f9b07cbc7b614c4b9987239f"
MDN_SHA256 = "8f997c35fa3c659f56a21a6b0996fb0faefd2a7f685359cd993231a03e04cc3a"
SCHEMA = "surfna-l2-native-mdn-contract-v2"
CHEMISTRY_COMPARISON_POLICY = {
    "revision": "rdkit_stereochemistry_comparison_v2",
    "normalize_copies_only": "Chem.AssignStereochemistry(cleanIt=True, force=True)",
    "compare": ["indexed_atomic_number", "formal_charge", "isotope", "normalized_chiral_tag",
                "indexed_CIP", "canonical_isomeric_SMILES", "indexed_bond_type", "bond_stereo_EZ"],
    "bond_drawing_direction": "audit_only_not_a_chemical_identity_constraint",
    "native_coordinates": "preserve_and_compare_in_original_atom_order",
    "graph_features_modified": False,
    "source_molecules_modified": False,
    "numeric_MDN_recipe_changed": False,
}
COORDINATE_POLICY = {
    "ligand.orig_pos": "global_native; restore ligand.pos=orig_pos-original_center",
    "receptor.pos": "already_centered; unchanged",
    "surface.pos": "already_centered; unchanged",
    "receptor.center_pos": "global; subtract original_center once",
    "receptor.nucleic_anchor_pos": "already_centered; unchanged",
    "surface.x": "L2_train_only_scaler_already_applied; unchanged",
    "evidence": "L2 raw-cache CPU audit 801 train/89 val; get_complex centers anchors, not center_pos",
}
RECIPE = {
    "arm": "protein_mdn_transfer", "scorer_variant": "aligned_a",
    "surface_feature_schema": "v2_full8", "max_surface_vertices": 512,
    "n_gaussians": 20, "scorer_hidden_dim": 192, "mdn_dropout": 0.1,
    "mdn_score_topk_per_atom": 8, "equivariant_graph_rms_cap": 100.0,
    "use_nusurf_fusion": False, "use_nucleic_feat_fusion": False,
    "objective": "native_contact_complex_balanced_NLL", "contact_cutoff_A": 7.0,
    "optimizer": "AdamW", "lr": 2e-4, "weight_decay": 1e-5,
    "batch_size": 4, "pair_chunk_size": 4096, "gradient_clip": 1.0,
    "max_epochs": 20, "min_epochs": 3, "patience": 4,
    "improvement_epsilon": 1e-6, "seed": 20260828,
    "selection": "native_val_complex_balanced_NLL_min",
    "initial_checkpoint_eligible": True, "frozen_backbone": True,
    "fresh_optimizer": True, "scheduler": None,
    "smoke_train_complexes": 8, "smoke_steps": 80, "smoke_lr": 5e-4,
    "reference_bins": 200, "reference_min_A": 0.0, "reference_max_A": 20.0,
    "reference_pseudocount": 0.5, "reference_floor": 1e-8,
    "reference_fit": "train_native_all_pairs_0_to_20A_complex_balanced",
    "test_opened": False, "native_k8_evaluation": False,
}
TABLE_NAMES = (
    ".p.npy", ".score.npy", ".so3_cdf_vals2.npy", ".so3_exp_score_norms2.npy",
    ".so3_omegas_array2.npy", ".so3_score_norms2.npy",
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def exclusive_json(path, value):
    """One-attempt receipts must never be replaced, even after a failed attempt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path, value):
    """Mutable artifacts of this one attempt only, never frozen inputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("xb") as handle:
        handle.write(json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def event(stage, **kwargs):
    print(json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                          stage=stage, **kwargs), allow_nan=False), flush=True)


def key(name):
    return hashlib.sha256(name.encode()).hexdigest()[:24]


def within(path, directory):
    path, directory = Path(path).resolve(), Path(directory).resolve()
    return path == directory or directory in path.parents


def release_path(release, relative):
    if Path(relative).is_absolute():
        raise RuntimeError(f"Release path must be relative: {relative}")
    path = (Path(release) / relative).resolve()
    if not within(path, release):
        raise RuntimeError(f"Path escapes frozen release: {relative}")
    return path


def member_names(path, expected=None):
    names = Path(path).read_text().splitlines()
    if not names or any(not n or n.strip() != n or "/" in n or "\\" in n or n in {".", ".."}
                        for n in names):
        raise RuntimeError(f"Invalid split member name: {path}")
    if len(set(names)) != len(names) or (expected is not None and len(names) != expected):
        raise RuntimeError(f"Duplicate members or split count mismatch: {path}")
    return names


def checked_file(path, expected=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise RuntimeError(f"Required input is not a file: {path}")
    actual = sha(path)
    if expected is not None and actual != expected:
        raise RuntimeError(f"Pinned input mismatch: {path}: {actual} != {expected}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def verify_record(record):
    if not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
        raise RuntimeError("Invalid SHA-256 in contract")
    checked_file(record["path"], record["sha256"])


def validate_scaler(scaler, train_names, train_sha):
    policy = scaler["fit_policy"]
    if policy.get("fit_split") != "training_only" or policy.get("fit_population") != "new_L2_NA_train_only":
        raise RuntimeError("Scaler was not fitted on new L2 NA training data only")
    for flag in ("protein_surfaces_used", "validation_surfaces_used", "test_surfaces_used"):
        if policy.get(flag) is not False:
            raise RuntimeError(f"Forbidden or unknown scaler population: {flag}")
    source = scaler["source"]["na"]
    if source["ply_count"] != len(train_names) or source["train_manifest_sha256"] != train_sha:
        raise RuntimeError("Scaler training split/count mismatch")
    members = [entry["complex_id"] for entry in source["train_surface_files"]]
    if len(members) != len(set(members)) or set(members) != set(train_names):
        raise RuntimeError("Scaler fitted-member set differs from the L2 train split")


def validate_parent(root, config):
    expected = config.get("root_contract_sha256")
    path = Path(root) / "contract.json"
    if expected is None:
        if path.exists():
            raise RuntimeError("Root contract was added after native freeze; use a new root")
        return
    checked_file(path, expected)
    data = read_json(path)["dataset"]
    ours = config["dataset"]
    if Path(data["release_root"]).resolve() != Path(ours["release_root"]):
        raise RuntimeError("Root/native dataset roots disagree")
    for split in ("train", "val"):
        entry = data[split]
        if entry["sha256"] != ours[split]["sha256"] or entry["count"] != ours[split]["count"]:
            raise RuntimeError(f"Root/native {split} disagrees")
        if Path(entry["path"]).resolve() != Path(ours[split]["path"]):
            raise RuntimeError(f"Root/native {split} path disagrees")
    if data["surface_scaler"]["sha256"] != ours["surface_scaler"]["sha256"]:
        raise RuntimeError("Root/native scaler disagrees")


def read_config(root, verify=True):
    root = Path(root).resolve()
    path = root / "native_contract.json"
    config = read_json(path)
    gate = read_json(root / "NATIVE_CONTRACT_READY.json")
    if gate.get("status") != "PASS" or gate["contract_sha256"] != sha(path):
        raise RuntimeError("Native contract has no matching ready receipt")
    if config["schema_version"] != SCHEMA or config["recipe"] != RECIPE:
        raise RuntimeError("Native recipe/schema changed; no implicit method variation allowed")
    if config.get("coordinate_policy") != COORDINATE_POLICY:
        raise RuntimeError("L2 native coordinate policy changed or is missing")
    if config.get("chemistry_comparison_policy") != CHEMISTRY_COMPARISON_POLICY:
        raise RuntimeError("Native chemistry-comparison policy changed or is missing")
    if Path(config["root"]).resolve() != root:
        raise RuntimeError("Working root moved without a new contract")
    if config["protein_diffusion_checkpoint"]["sha256"] != DIFFUSION_SHA256:
        raise RuntimeError("Native B must initialize from the frozen protein diffusion checkpoint")
    if config["protein_mdn_checkpoint"]["sha256"] != MDN_SHA256:
        raise RuntimeError("Native B must initialize from the frozen protein MDN, not a V3 NA head")
    if config.get("test_opened") is not False:
        raise RuntimeError("Native contract test exclusion missing")
    validate_parent(root, config)
    for forbidden in config["read_only_roots"]:
        if within(root, forbidden):
            raise RuntimeError("Working root overlaps an immutable input tree")
    if verify:
        for record in config["input_files"]:
            verify_record(record)
        source = Path(config["source_snapshot"]["root"])
        actual = {str(p.relative_to(source)) for p in source.rglob("*.py") if p.is_file()}
        if actual != set(config["source_snapshot"]["python_files"]):
            raise RuntimeError("Architecture source file set changed after freeze")
    train = member_names(config["dataset"]["train"]["path"], config["dataset"]["train"]["count"])
    val = member_names(config["dataset"]["val"]["path"], config["dataset"]["val"]["count"])
    if set(train) & set(val):
        raise RuntimeError("Train/val identity overlap")
    validate_scaler(read_json(config["dataset"]["surface_scaler"]["path"]), train,
                    config["dataset"]["train"]["sha256"])
    return config


def load_graph_index(root, config):
    native = Path(root) / "native"
    gate = read_json(native / "PREPARED_READY.json")
    if gate.get("status") != "PASS" or gate.get("contract_sha256") != sha(Path(root) / "native_contract.json"):
        raise RuntimeError("Native preparation gate not satisfied")
    for filename, field in (("graph_index.json", "graph_index_sha256"),
                            ("native_train_reference.json", "reference_sha256")):
        checked_file(native / filename, gate[field])
    index = read_json(native / "graph_index.json")
    rows = index["entries"]
    counts = {s: config["dataset"][s]["count"] for s in ("train", "val")}
    if index["split_counts"] != counts or gate["split_counts"] != counts:
        raise RuntimeError("Prepared split counts drift")
    if len(rows) != len({r["name"] for r in rows}) or any(r["split"] not in counts for r in rows):
        raise RuntimeError("Duplicate or forbidden graph-index members")
    for split in counts:
        expected = set(member_names(config["dataset"][split]["path"], counts[split]))
        if {r["name"] for r in rows if r["split"] == split} != expected:
            raise RuntimeError(f"Prepared {split} identity set mismatch")
    for row in rows:
        if not within(row["path"], native / "prepared"):
            raise RuntimeError("Prepared graph points outside this root")
    reference = read_json(native / "native_train_reference.json")
    if reference["fit_split"] != "train" or reference["n_complexes"] != counts["train"]:
        raise RuntimeError("Reference density used non-training data")
    if reference["source_sha256"] != sha(native / "graph_index.json"):
        raise RuntimeError("Reference density no longer matches prepared graph provenance")
    return rows
