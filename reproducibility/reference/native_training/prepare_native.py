"""CPU-only native-coordinate copies of the pinned L2 graph cache.

No PDBBind constructor, graph matching, surface recomputation, or source writes.
Only the unused receptor.atoms_pos padding archive is discarded. Nonfinite
actual model inputs cause failure rather than masking or dropping a member.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
import traceback
from pathlib import Path

from native_contract import (CHEMISTRY_COMPARISON_POLICY, RECIPE, atomic_json, atomic_torch, checked_file,
    event, exclusive_json, key, read_config, sha)


def native_coordinates(value):
    import numpy as np
    import torch
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    while array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise RuntimeError("Native orig_pos must be one finite [atoms,3] conformation")
    return array.copy()


def single_name(graph):
    name = graph.name
    if isinstance(name, (tuple, list)) and len(name) == 1:
        name = name[0]
    if not isinstance(name, str):
        raise RuntimeError("Graph has no unambiguous complex identity")
    return name


def heavy_molecule(molecule):
    from rdkit import Chem
    if molecule is None:
        raise RuntimeError("Null native/cached molecule")
    molecule = Chem.RemoveHs(Chem.Mol(molecule), sanitize=True)
    if molecule.GetNumConformers() != 1 or any(a.GetAtomicNum() == 1 for a in molecule.GetAtoms()):
        raise RuntimeError("Expected a single-conformer heavy-atom molecule")
    return molecule


def rdkit_runtime_audit():
    """Record imported code, not only distribution metadata (which can differ)."""
    import importlib.metadata
    import rdkit
    from rdkit import Chem
    distributions = {}
    for package in ("rdkit", "rdkit-pypi"):
        try:
            distributions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            distributions[package] = None
    return {"runtime_version": rdkit.__version__, "rdBase_version": Chem.rdBase.rdkitVersion,
            "module_path": rdkit.__file__, "Chem_module_path": Chem.__file__,
            "distribution_metadata": distributions}


def compare_native_cache_chemistry(native, cached):
    """Normalize comparison-only copies and require actual stereo identity.

Raw RDKit chiral tags can survive 3D assignment on non-stereogenic atoms.
For example a symmetric quaternary N can retain CHI_TETRAHEDRAL_CCW while
the SDF roundtrip correctly reports CHI_UNSPECIFIED. Standard stereo cleanup
removes these invalid annotations, but preserves meaningful CIP and E/Z.
The supplied molecules, frozen cache, graph.x and coordinates are untouched.
"""
    from rdkit import Chem

    def atoms(mol):
        return [(atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetIsotope(), str(atom.GetChiralTag()))
                for atom in mol.GetAtoms()]

    def cips(mol):
        return [(atom.GetIdx(), atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None)
                for atom in mol.GetAtoms()]

    def bonds(mol):
        values = []
        for bond in mol.GetBonds():
            first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            stereo_atoms = tuple(bond.GetStereoAtoms())
            if first > second:
                stereo_atoms = stereo_atoms[::-1]
            values.append((min(first, second), max(first, second), str(bond.GetBondType()),
                           str(bond.GetStereo()), stereo_atoms))
        return sorted(values)

    def directions(mol):
        return {(min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx())):
                str(b.GetBondDir()) for b in mol.GetBonds()}

    before_native, before_cached = atoms(native), atoms(cached)
    raw_bonds_native, raw_bonds_cached = bonds(native), bonds(cached)
    raw_dirs_native, raw_dirs_cached = directions(native), directions(cached)
    native_copy, cached_copy = Chem.Mol(native), Chem.Mol(cached)
    for molecule in (native_copy, cached_copy):
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    after_native, after_cached = atoms(native_copy), atoms(cached_copy)
    cip_native, cip_cached = cips(native_copy), cips(cached_copy)
    bonds_native, bonds_cached = bonds(native_copy), bonds(cached_copy)
    smiles_native = Chem.MolToSmiles(native_copy, canonical=True, isomericSmiles=True)
    smiles_cached = Chem.MolToSmiles(cached_copy, canonical=True, isomericSmiles=True)
    gates = {"indexed_atom_signature": after_native == after_cached, "indexed_CIP": cip_native == cip_cached,
             "canonical_isomeric_SMILES": smiles_native == smiles_cached,
             "indexed_bond_type_and_stereo": bonds_native == bonds_cached}
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError("Frozen native/cache normalized chemistry differs: " + ", ".join(failed))
    raw_atom_differences = [{"atom_index": index, "native": left, "cached": right}
                            for index, (left, right) in enumerate(zip(before_native, before_cached)) if left != right]
    raw_direction_differences = [{"bond_atoms": list(edge), "native": raw_dirs_native.get(edge),
                                  "cached": raw_dirs_cached.get(edge)}
                                 for edge in sorted(set(raw_dirs_native) | set(raw_dirs_cached))
                                 if raw_dirs_native.get(edge) != raw_dirs_cached.get(edge)]
    return {"policy_revision": CHEMISTRY_COMPARISON_POLICY["revision"],
            "normalization": CHEMISTRY_COMPARISON_POLICY["normalize_copies_only"],
            "normalization_for_comparison_only": True, "source_molecules_modified": False,
            "graph_features_modified": False, "atom_count": len(after_native), "bond_count": len(bonds_native),
            "raw_atom_signature_equal": before_native == before_cached,
            "raw_atom_signature_differences": raw_atom_differences,
            "raw_bond_type_and_stereo_equal": raw_bonds_native == raw_bonds_cached,
            "raw_bond_direction_differences": raw_direction_differences,
            "native_normalized_tag_changes": [i for i, (a, b) in enumerate(zip(before_native, after_native)) if a != b],
            "cached_normalized_tag_changes": [i for i, (a, b) in enumerate(zip(before_cached, after_cached)) if a != b],
            "normalized_gates": gates, "canonical_isomeric_SMILES_native": smiles_native,
            "canonical_isomeric_SMILES_cached": smiles_cached,
            "assigned_CIP_centers": sum(value is not None for _, value in cip_native),
            "assigned_EZ_bonds": sum(bond[3] not in ("STEREONONE", "STEREOANY") for bond in bonds_native)}


def check_native_chemistry(path, cached_molecule, coordinates):
    import numpy as np
    from rdkit import Chem
    supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
    if len(supplier) != 1:
        raise RuntimeError("Frozen native SDF must contain exactly one molecule")
    native, cached = heavy_molecule(supplier[0]), heavy_molecule(cached_molecule)
    chemistry_audit = compare_native_cache_chemistry(native, cached)
    native_xyz = native.GetConformer().GetPositions()
    if native_xyz.shape != coordinates.shape or not np.isfinite(native_xyz).all() or not np.isfinite(coordinates).all():
        raise RuntimeError("Frozen native/cache atom count or coordinate finite gate failed")
    maximum = float(np.max(np.abs(native_xyz - coordinates)))
    if maximum > 0.002:
        raise RuntimeError(f"Native orig_pos/global SDF coordinates differ by {maximum:.6g} A")
    chemistry_audit["rdkit_runtime"] = rdkit_runtime_audit()
    return maximum, chemistry_audit


def center_auxiliary_once(graph, center):
    """Pinned L2 raw-cache frames, not an adaptive/retry repair heuristic.

The G2.2 get_complex function has already centered nucleic anchors. The
loader separately centers center_pos. Record both plausible errors to expose
a contradicted source frame, including centers near zero where both coincide.
"""
    import torch
    if getattr(graph, "native_mdn_prepared", False):
        raise RuntimeError("Graph was already native-prepared; refusing double centering")
    receptor = graph["receptor"]
    result = {}
    if "center_pos" in receptor:
        raw = receptor.center_pos
        shifted = raw - center.to(raw)
        if raw.shape != receptor.pos.shape:
            raise RuntimeError("Receptor center_pos / pos row identity mismatch")
        finite = torch.isfinite(raw).all() and torch.isfinite(shifted).all()
        if not finite:
            raise RuntimeError("Nonfinite receptor center coordinates")
        before = float(torch.linalg.vector_norm(raw - receptor.pos.to(raw), dim=-1).median())
        after = float(torch.linalg.vector_norm(shifted - receptor.pos.to(raw), dim=-1).median())
        if after > before + 1.0:
            raise RuntimeError("Raw center_pos contradicts the pinned global-coordinate policy")
        receptor.center_pos = shifted
        result["center_pos"] = {"input_frame": "global", "output_frame": "centered", "subtractions": 1,
                                "parent_median_before_A": before, "parent_median_after_A": after}
    if "nucleic_anchor_pos" in receptor:
        raw = receptor.nucleic_anchor_pos
        parents = receptor.nucleic_anchor_parent_index.long().reshape(-1)
        if raw.shape != (len(parents), 3) or not torch.isfinite(raw).all():
            raise RuntimeError("Invalid nucleic anchor shape/finite gate")
        if len(parents):
            if int(parents.min()) < 0 or int(parents.max()) >= len(receptor.pos):
                raise RuntimeError("Invalid nucleic anchor parent index")
            target = receptor.pos[parents].to(raw)
            before = float(torch.linalg.vector_norm(raw - target, dim=-1).median())
            if_shifted = float(torch.linalg.vector_norm(raw - center.to(raw) - target, dim=-1).median())
            if if_shifted + 1.0 < before:
                raise RuntimeError("Raw anchors contradict pinned already-centered L2 cache; no implicit repair")
        else:
            before = if_shifted = None
        # Intentionally leave anchors unchanged; do not copy the old V3 rule.
        result["nucleic_anchor_pos"] = {"input_frame": "centered", "output_frame": "centered", "subtractions": 0,
                                       "parent_median_before_A": before, "if_subtracted_again_A": if_shifted}
    graph.native_mdn_prepared = True
    return result


def prepare_one(raw_graph, cached_molecule, member, config, output):
    import numpy as np
    import torch
    graph = raw_graph.clone()
    if single_name(graph) != member["name"]:
        raise RuntimeError("Graph/member identity mismatch")
    native_pin = member["native_sdf"]
    checked_file(native_pin["path"], native_pin["sha256"])
    xyz = native_coordinates(graph["ligand"].orig_pos)
    max_error, chemistry_audit = check_native_chemistry(native_pin["path"], cached_molecule, xyz)
    center = torch.as_tensor(graph.original_center).detach().cpu().reshape(1, 3).clone()
    if not torch.isfinite(center).all() or tuple(graph["ligand"].pos.shape) != xyz.shape:
        raise RuntimeError("Original center or native ligand shape invalid")
    # Raw caches contain the matched conformer. Restore the deposited native,
    # not that conformer, without mutating the frozen raw graph.
    global_tensor = torch.as_tensor(xyz, dtype=graph["ligand"].pos.dtype).clone()
    graph["ligand"].pos = global_tensor - center.to(global_tensor)
    graph.original_center = center
    removed = {}
    if "atoms_pos" in graph["receptor"]:
        value = graph["receptor"].atoms_pos
        removed["receptor.atoms_pos"] = {"shape": list(value.shape),
                                       "nonfinite_count": int((~torch.isfinite(value)).sum())}
        del graph["receptor"].atoms_pos
    coordinate_frame_audit = center_auxiliary_once(graph, center)
    for store in graph.stores:
        for field, value in store.items():
            if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
                if not torch.isfinite(value).all():
                    raise RuntimeError(f"Nonfinite prepared graph field: {store._key}/{field}")
    ns = int(graph["surface"].pos.shape[0])
    if not 1 <= ns <= RECIPE["max_surface_vertices"] or tuple(graph["surface"].x.shape) != (ns, 8):
        raise RuntimeError("L2 graph surface must be full8 and <=512 vertices")
    if not torch.allclose(graph["ligand"].pos + center, global_tensor, atol=0.002, rtol=0):
        raise RuntimeError("Native centering failed")
    distances = torch.cdist(graph["ligand"].pos.float(), graph["surface"].pos.float())
    if not torch.isfinite(distances).all():
        raise RuntimeError("Nonfinite native distances")
    contacts = torch.nonzero(distances <= RECIPE["contact_cutoff_A"], as_tuple=False)
    if not len(contacts):
        raise RuntimeError("Native graph has no <=7A contact; no sample skipping allowed")
    histogram = np.histogram(distances.numpy(), bins=np.linspace(0, 20, 201))[0]
    provenance = {"native_sdf": native_pin["sha256"],
                  "heterographs": config["graph_sources"][member["split"]]["heterographs"]["sha256"],
                  "rdkit_ligands": config["graph_sources"][member["split"]]["rdkit_ligands"]["sha256"],
                  "surface_scaler": config["dataset"]["surface_scaler"]["sha256"]}
    destination = output / "prepared" / (key(member["name"]) + ".pt")
    if destination.exists():
        raise RuntimeError("Prepared member already exists; no resume")
    artifact = {"name": member["name"], "split": member["split"], "graph": graph,
                "contact_index": contacts, "contact_distance": distances[contacts[:, 0], contacts[:, 1]],
                "native_global_coordinates": global_tensor, "original_center": center,
                "input_sha256": provenance, "removed_unused_metadata": removed,
                "native_global_max_abs_error_A": max_error, "chemistry_comparison_audit": chemistry_audit,
                "coordinate_policy": config["coordinate_policy"], "coordinate_frame_audit": coordinate_frame_audit,
                "surface_scaler_already_applied": True, "test_opened": False}
    atomic_torch(destination, artifact)
    return {"name": member["name"], "split": member["split"], "path": str(destination),
            "artifact_sha256": sha(destination), "ligand_atoms": len(xyz), "surface_vertices": ns,
            "contact_pairs": len(contacts), "input_sha256": provenance,
            "native_global_max_abs_error_A": max_error, "removed_unused_metadata": removed,
            "coordinate_frame_audit": coordinate_frame_audit,
            "chemistry_comparison_audit": chemistry_audit}, histogram


def prepare(root):
    root = Path(root).resolve()
    config = read_config(root)
    output = root / "native"
    exclusive_json(output / "PREPARATION_ATTEMPT.json", {
        "pid": os.getpid(), "started_at": time.time(), "contract_sha256": sha(root / "native_contract.json"),
        "test_opened": False, "resume": False, "graph_rebuild": False})
    stage, current = "imports", None
    try:
        import numpy as np
        import torch
        torch.set_num_threads(2)
        runtime = rdkit_runtime_audit()
        entries, train_histograms = [], []
        start = time.monotonic()
        for split in ("train", "val"):
            stage = "load_pinned_cache_" + split
            paths = config["graph_sources"][split]
            # Trusted, hash-pinned release artifacts only. Do not use untrusted pickle.
            with Path(paths["heterographs"]["path"]).open("rb") as handle:
                graphs = pickle.load(handle)
            with Path(paths["rdkit_ligands"]["path"]).open("rb") as handle:
                molecules = pickle.load(handle)
            members = [m for m in config["members"] if m["split"] == split]
            if not len(graphs) == len(molecules) == len(members) == config["dataset"][split]["count"]:
                raise RuntimeError("Raw graph/molecule/split counts do not agree")
            by_name = {single_name(g): (g, m) for g, m in zip(graphs, molecules)}
            if len(by_name) != len(graphs) or set(by_name) != {m["name"] for m in members}:
                raise RuntimeError("Raw graph identities differ from frozen split")
            stage = "prepare_native_" + split
            for member in sorted(members, key=lambda m: m["name"]):
                current = member["name"]
                row, histogram = prepare_one(*by_name[current], member, config, output)
                entries.append(row)
                if split == "train":
                    train_histograms.append(histogram)
                if len(entries) % 25 == 0 or len(entries) == len(config["members"]):
                    event("native_graphs", complete=len(entries), total=len(config["members"]),
                          seconds=time.monotonic() - start, failed=0)
            del by_name, graphs, molecules
        stage, current = "reference_train_only", None
        counts = {s: config["dataset"][s]["count"] for s in ("train", "val")}
        index = {"schema_version": "surfna-l2-native-graph-index-v1",
                 "contract_sha256": sha(root / "native_contract.json"),
                 "root_contract_sha256": config["root_contract_sha256"], "split_counts": counts,
                 "entries": entries, "source_graphs_modified": False, "test_opened": False,
                 "chemistry_comparison_policy": CHEMISTRY_COMPARISON_POLICY,
                 "rdkit_runtime": runtime}
        atomic_json(output / "graph_index.json", index)
        hist = np.asarray(train_histograms, dtype=np.float64)
        if hist.shape != (counts["train"], 200) or not np.isfinite(hist).all():
            raise RuntimeError("Training reference histogram shape/finite failure")
        density = ((hist + 0.5) / (hist.sum(axis=1, keepdims=True) + 100)).mean(axis=0) / 0.1
        if not np.isfinite(density).all() or (density <= 0).any() or not np.isclose(density.sum() * .1, 1):
            raise RuntimeError("Invalid native training reference density")
        atomic_json(output / "native_train_reference.json", {
            "bin_centers": ((np.arange(200) + 0.5) * 0.1).tolist(), "log_density": np.log(density).tolist(),
            "floor": 1e-8, "fit_split": "train", "fit_pairs": "native_all_pairs_0_to_20A_complex_balanced",
            "n_complexes": counts["train"], "source_sha256": sha(output / "graph_index.json"),
            "train_split_sha256": config["dataset"]["train"]["sha256"],
            "validation_used": False, "test_opened": False})
        chemistry_summary = {}
        for split in ("train", "val"):
            audit_rows = [entry["chemistry_comparison_audit"] for entry in entries if entry["split"] == split]
            chemistry_summary[split] = {
                "members": len(audit_rows), "normalized_chemistry_pass": len(audit_rows),
                "raw_atom_tag_difference_members": sum(not row["raw_atom_signature_equal"] for row in audit_rows),
                "raw_atom_tag_difference_atoms": sum(len(row["raw_atom_signature_differences"]) for row in audit_rows),
                "raw_bond_stereo_difference_members": sum(not row["raw_bond_type_and_stereo_equal"] for row in audit_rows),
                "raw_bond_drawing_direction_difference_members": sum(bool(row["raw_bond_direction_differences"]) for row in audit_rows),
                "assigned_CIP_centers": sum(row["assigned_CIP_centers"] for row in audit_rows),
                "assigned_EZ_bonds": sum(row["assigned_EZ_bonds"] for row in audit_rows)}
        exclusive_json(output / "PREPARED_READY.json", {
            "status": "PASS", "contract_sha256": sha(root / "native_contract.json"),
            "root_contract_sha256": config["root_contract_sha256"], "split_counts": counts,
            "graph_index_sha256": sha(output / "graph_index.json"),
            "reference_sha256": sha(output / "native_train_reference.json"),
            "failed": 0, "skipped": 0, "test_opened": False, "gpu_used": False,
            "chemistry_comparison_summary": chemistry_summary, "rdkit_runtime": runtime,
            "chemistry_comparison_policy": CHEMISTRY_COMPARISON_POLICY,
            "seconds": time.monotonic() - start})
        event("native_prepared", status="PASS", split_counts=counts, test_opened=False)
    except Exception:
        exclusive_json(output / "PREPARATION_FAILED.json", {
            "stage": stage, "member": current, "error": traceback.format_exc(),
            "time": time.time(), "no_retry": True, "test_opened": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    prepare(parser.parse_args().root)
