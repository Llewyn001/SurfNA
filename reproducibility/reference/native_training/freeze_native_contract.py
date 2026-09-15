"""Pin L2 train/val cached graphs and the existing protein-native B recipe.

Standard-library-only CPU gate. It never opens test structures or checkpoints
with pickle and never imports torch, rebuilds graphs, or launches a subprocess.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from native_contract import (CHEMISTRY_COMPARISON_POLICY, COORDINATE_POLICY, DIFFUSION_SHA256, MDN_SHA256, RECIPE, SCHEMA,
    TABLE_NAMES, checked_file, event, exclusive_json, member_names, read_json,
    release_path, sha, validate_parent, validate_scaler, within)


def freeze(args):
    root, release, source = (Path(x).resolve() for x in (args.root, args.release_root, args.code_root))
    readonly = [release, source, Path(args.precomputed_dir).resolve(),
                Path(args.protein_model_config).resolve().parent,
                Path(args.protein_diffusion_checkpoint).resolve().parent,
                Path(args.protein_mdn_checkpoint).resolve().parent]
    if any(within(root, p) for p in readonly):
        raise RuntimeError("New root must be outside frozen datasets/source/weights")
    if (root / "native_contract.json").exists() or (root / "NATIVE_CONTRACT_READY.json").exists():
        raise RuntimeError("Native contract already exists; refusing overwrite/retry")
    pins = {}

    def pin(path, expected=None):
        result = checked_file(path, expected)
        previous = pins.setdefault(result["path"], result)
        if previous != result:
            raise RuntimeError("One path was pinned inconsistently")
        return result

    runtime_pin = pin(release / "PORTABLE_RUNTIME.json")
    runtime = read_json(runtime_pin["path"])
    files = {entry["path"]: entry for entry in runtime["files"]}
    if len(files) != len(runtime["files"]):
        raise RuntimeError("Duplicate release manifest paths")

    def release_pin(relative):
        if relative not in files:
            raise RuntimeError(f"Input absent from frozen release file manifest: {relative}")
        return pin(release_path(release, relative), files[relative]["sha256"])

    audit_pin = release_pin("audit/GRAPH_LOAD_AUDIT.json")
    audit = read_json(audit_pin["path"])
    if audit.get("status") != "verified" or audit.get("raw_checks_before_sanitation") is not True:
        raise RuntimeError("Frozen raw graph audit did not pass")
    dataset = {"release_root": str(release), "portable_runtime": runtime_pin,
               "graph_load_audit": audit_pin,
               "data_dir": str(release_path(release, runtime["layout"]["data_dir"])),
               "surface_dir": str(release_path(release, runtime["layout"]["surface_dir"])),
               "surface_scaler": release_pin(runtime["layout"]["scaler"])}
    sets, graph_sources, members = {}, {}, []
    for split in ("train", "val"):
        split_meta = runtime["splits"][split]
        split_pin = release_pin(split_meta["path"])
        if split_pin["sha256"] != split_meta["sha256"]:
            raise RuntimeError("Runtime split SHA mismatch")
        names = member_names(split_pin["path"], split_meta["count"])
        if runtime["counts"][split] != len(names):
            raise RuntimeError("Release count disagrees with split")
        sets[split] = set(names)
        dataset[split] = dict(split_pin, count=len(names))
        table_meta = runtime["membership"]["tables"][split]
        table_pin = release_pin(table_meta["path"])
        if table_pin["sha256"] != table_meta["sha256"]:
            raise RuntimeError("Membership table SHA mismatch")
        dataset[split]["membership"] = table_pin
        columns = runtime["membership"]["columns"]
        id_field = columns["id"]
        pdb_field = columns.get("pdb", "parent_pdb")
        group_field = columns.get("split_group", columns.get("split_group_id", "split_group_id"))
        with Path(table_pin["path"]).open() as handle:
            table = list(csv.DictReader(handle, delimiter="\t"))
        by_name = {row[id_field]: row for row in table}
        if len(by_name) != len(table) or set(by_name) != sets[split]:
            raise RuntimeError(f"Membership table / split identity mismatch: {split}")
        if any(not row.get(pdb_field) or not row.get(group_field) for row in table):
            raise RuntimeError(f"Missing nonempty parent PDB or split-group identity: {split}")
        split_audit = audit["splits"][split]
        for field in ("actual", "expected", "inference_list_actual"):
            if split_audit[field] != len(names):
                raise RuntimeError(f"Graph audit count mismatch {split}/{field}")
        for field in ("raw_graph_invalid", "loader_invalid", "ligand_matching_fallback_count"):
            if split_audit[field] != 0:
                raise RuntimeError(f"Frozen graph audit not clean {split}/{field}")
        if split_audit.get("automatic_rebuild_disabled") is not True:
            raise RuntimeError("Graph cache rebuild exclusion missing")
        cache = split_audit["cache_directory"]
        graph_sources[split] = {}
        for filename in ("heterographs.pkl", "rdkit_ligands.pkl"):
            relative = str(Path(cache) / filename)
            record = release_pin(relative)
            if record["sha256"] != split_audit["files_sha256"][relative]:
                raise RuntimeError("Graph cache SHA differs from raw audit")
            graph_sources[split][filename.removesuffix(".pkl")] = record
        for name in sorted(names):
            relative = str(Path(runtime["layout"]["data_dir"]) / name / f"{name}_ligand.sdf")
            members.append({"name": name, "split": split, "native_sdf": release_pin(relative),
                            "parent_pdb": by_name[name][pdb_field],
                            "split_group": by_name[name][group_field]})
    if sets["train"] & sets["val"]:
        raise RuntimeError("Train/val overlap")
    for field in ("parent_pdb", "split_group"):
        groups = [{m[field] for m in members if m["split"] == s and m[field]} for s in ("train", "val")]
        if groups[0] & groups[1]:
            raise RuntimeError(f"Train/val {field} overlap in frozen membership")
    validate_scaler(read_json(dataset["surface_scaler"]["path"]), sets["train"], dataset["train"]["sha256"])
    source_manifest = pin(source.parent / "source_snapshot_manifest.json")
    legacy_manifest = read_json(source_manifest["path"])
    python_files = sorted(str(p.relative_to(source)) for p in source.rglob("*.py") if p.is_file())
    if not python_files or set(python_files) != {p for p in legacy_manifest if p.endswith(".py")}:
        raise RuntimeError("Original native architecture file set differs from its snapshot manifest")
    for relative in python_files:
        pin(release_path(source, relative), legacy_manifest[relative]["sha256"])
    required = {"models/surface_score_model_v3.py", "models/surface_interaction_backbone_v2.py",
                "models/surfna_v2_scorer_components.py", "models/surfna_v2_transfer.py",
                "utils/scorer_factory_v2.py", "utils/scorer_config_v2.py"}
    if not required.issubset(python_files):
        raise RuntimeError("Source is not the frozen native-MDN architecture snapshot")
    adapter_root = Path(__file__).resolve().parent
    adapter_files = sorted(adapter_root.glob("*.py"))
    for path in adapter_files:
        pin(path)
    tables = {name: pin(Path(args.precomputed_dir) / name) for name in TABLE_NAMES}
    config = {"schema_version": SCHEMA, "root": str(root), "recipe": RECIPE, "coordinate_policy": COORDINATE_POLICY,
              "chemistry_comparison_policy": CHEMISTRY_COMPARISON_POLICY,
              "root_contract_sha256": sha(root / "contract.json") if (root / "contract.json").exists() else None,
              "dataset": dataset, "graph_sources": graph_sources, "members": members,
              "source_snapshot": {"root": str(source), "manifest": source_manifest, "python_files": python_files},
              "adapter_root": str(adapter_root), "adapter_files": [str(p) for p in adapter_files],
              "protein_model_config": pin(args.protein_model_config),
              "protein_diffusion_checkpoint": pin(args.protein_diffusion_checkpoint, DIFFUSION_SHA256),
              "protein_mdn_checkpoint": pin(args.protein_mdn_checkpoint, MDN_SHA256),
              "precomputed_dir": str(Path(args.precomputed_dir).resolve()), "precomputed_files": tables,
              "read_only_roots": [str(p) for p in readonly], "test_opened": False,
              "provenance": "native_prior_v3_20260828 B numerical recipe; new L2 data/scaler/reference/head",
              "no_legacy_na_head_or_scaler": True, "training_executed": False}
    config["input_files"] = sorted(pins.values(), key=lambda x: x["path"])
    validate_parent(root, config)
    exclusive_json(root / "native_contract.json", config)
    exclusive_json(root / "NATIVE_CONTRACT_READY.json", {
        "status": "PASS", "contract_sha256": sha(root / "native_contract.json"),
        "root_contract_sha256": config["root_contract_sha256"],
        "split_counts": {s: len(sets[s]) for s in sets}, "input_files": len(pins),
        "test_opened": False, "training_executed": False})
    event("native_contract_frozen", root=str(root), counts={s: len(sets[s]) for s in sets}, test_opened=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "release-root", "code-root", "protein-model-config",
                 "protein-diffusion-checkpoint", "protein-mdn-checkpoint", "precomputed-dir"):
        parser.add_argument("--" + name, required=True)
    freeze(parser.parse_args())
