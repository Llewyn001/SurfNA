#!/usr/bin/env python3
"""Fail closed on any drift in the frozen validation hotfix or val89 inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), f"missing or symlinked JSON: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def validate_pdb(path: Path, expected_count: int) -> None:
    count = 0
    with path.open(errors="strict") as handle:
        for line in handle:
            if not line.startswith("ATOM  "):
                continue
            coords = tuple(float(line[start:end]) for start, end in ((30, 38), (38, 46), (46, 54)))
            require(all(math.isfinite(value) for value in coords), f"non-finite receptor coordinate: {path}")
            count += 1
    require(count == expected_count and count > 0, f"receptor coordinate-count drift: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--hotfix-root", type=Path, required=True)
    parser.add_argument("--require-smoke", action="store_true")
    args = parser.parse_args()
    analysis_root = args.analysis_root.resolve()
    hotfix_root = args.hotfix_root.resolve()

    manifest_path = hotfix_root / "HOTFIX_MANIFEST.json"
    sidecar_path = hotfix_root / "HOTFIX_MANIFEST.sha256"
    tokens = sidecar_path.read_text().split()
    require(len(tokens) == 2 and tokens[1] == manifest_path.name, "malformed hotfix manifest sidecar")
    require(sha256(manifest_path) == tokens[0], "hotfix manifest digest mismatch")
    manifest = load_json(manifest_path)
    require(manifest.get("schema_version") == "surfna-lb200-validation-hotfix-v1", "hotfix schema mismatch")
    require(manifest.get("status") == "FROZEN_BEFORE_HOTFIX_SMOKE_OR_FORMAL_RESULTS", "hotfix freeze status mismatch")
    require(manifest.get("passed") is True, "hotfix manifest is not passed")
    require(Path(manifest["analysis_root"]).resolve() == analysis_root, "analysis root mismatch")
    require(Path(manifest["hotfix_root"]).resolve() == hotfix_root, "hotfix root mismatch")
    require(manifest.get("scientific_scope") == "validation receptor filename resolution only", "hotfix scope drift")

    records = manifest.get("code_records")
    require(isinstance(records, list) and len(records) == 7, "hotfix code inventory count mismatch")
    for record in records:
        path = hotfix_root / record["relative_path"]
        require(path.is_file() and not path.is_symlink(), f"missing or symlinked hotfix code: {path}")
        require(path.stat().st_size == record["size_bytes"], f"hotfix code size drift: {path}")
        require(sha256(path) == record["sha256"], f"hotfix code digest drift: {path}")

    original = manifest["original_evaluator"]
    original_path = Path(original["path"])
    require(sha256(original_path) == original["sha256"], "original frozen evaluator drift")

    inventory_ref = manifest["receptor_inventory"]
    inventory_path = Path(inventory_ref["path"])
    require(sha256(inventory_path) == inventory_ref["sha256"], "raw-input inventory drift")
    inventory = load_json(inventory_path)
    require(inventory.get("schema_version") == "surfna-lb200-val89-raw-input-inventory-v1", "raw-input schema mismatch")
    require(inventory.get("passed") is True, "raw-input inventory is not passed")
    require(inventory.get("complex_count") == 89, "raw-input complex count mismatch")
    require(inventory.get("canonical_receptor_suffix") == "_protein_processed.pdb", "canonical receptor suffix drift")
    receptors = inventory.get("receptors")
    ligands = inventory.get("ligands")
    require(isinstance(receptors, list) and len(receptors) == 89, "receptor inventory cardinality mismatch")
    require(isinstance(ligands, list) and len(ligands) == 89, "ligand inventory cardinality mismatch")
    names = []
    for record in receptors:
        path = Path(record["path"])
        require(path.name == f"{record['complex_name']}_protein_processed.pdb", f"non-canonical receptor path: {path}")
        require(path.is_file() and not path.is_symlink(), f"missing or symlinked receptor: {path}")
        require(path.stat().st_size == record["size_bytes"], f"receptor size drift: {path}")
        require(sha256(path) == record["sha256"], f"receptor digest drift: {path}")
        validate_pdb(path, record["atom_record_count"])
        names.append(record["complex_name"])
    for record in ligands:
        path = Path(record["path"])
        require(path.name == f"{record['complex_name']}_ligand.sdf", f"non-canonical ligand path: {path}")
        require(path.is_file() and not path.is_symlink(), f"missing or symlinked ligand: {path}")
        require(path.stat().st_size == record["size_bytes"], f"ligand size drift: {path}")
        require(sha256(path) == record["sha256"], f"ligand digest drift: {path}")
    split_path = Path(inventory["split_path"])
    split_names = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    require(sha256(split_path) == inventory["split_sha256"], "validation split digest drift")
    require(names == split_names and len(set(names)) == 89, "validation identity/order drift")
    smoke_ref = manifest.get("smoke_subset")
    require(isinstance(smoke_ref, dict), "smoke subset reference is missing")
    smoke_subset = Path(smoke_ref.get("path", ""))
    require(smoke_subset.resolve() == hotfix_root / "smoke_subset.txt", "smoke subset path drift")
    require(smoke_subset.is_file() and not smoke_subset.is_symlink(), "missing or symlinked smoke subset")
    require(sha256(smoke_subset) == smoke_ref.get("sha256"), "smoke subset digest drift")
    smoke_names = [line.strip() for line in smoke_subset.read_text().splitlines() if line.strip()]
    require(
        smoke_names == [smoke_ref.get("complex_name")] == names[:1],
        "smoke subset identity/order drift",
    )
    if args.require_smoke:
        smoke_path = analysis_root / "03_integrity/validation_hotfix_v1_smoke/SMOKE_COMPLETE.json"
        smoke = load_json(smoke_path)
        require(smoke.get("schema_version") == "surfna-lb200-validation-hotfix-smoke-v1", "smoke schema mismatch")
        require(smoke.get("passed") is True, "hotfix smoke did not pass")
        require(smoke.get("manifest_sha256") == tokens[0], "smoke used a different hotfix digest")
        require(smoke.get("complex_count") == 1 and smoke.get("pose_count") == 10, "smoke cardinality mismatch")
        require(smoke.get("canonical_processed_receptor_observed") is True, "smoke did not exercise canonical processed receptor")
        require(smoke.get("all_metrics_finite") is True, "smoke contains non-finite metrics")
    print(json.dumps({"passed": True, "manifest_sha256": tokens[0], "receptors": 89}, sort_keys=True))


if __name__ == "__main__":
    main()
