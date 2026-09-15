#!/usr/bin/env python3
"""Strict, label-independent atom-order checks and unaligned pose RMSD.

This module never reads a split or a test set.  RDKit and spyrmsd imports are
lazy so the orchestration/gate tests can run without a chemistry runtime.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


RMSD_METHOD = "symmetry_aware_no_alignment"


def remove_all_hs(molecule):
    from rdkit import Chem

    parameters = Chem.RemoveHsParameters()
    for field in (
        "removeAndTrackIsotopes", "removeDefiningBondStereo", "removeDegreeZero",
        "removeDummyNeighbors", "removeHigherDegrees", "removeHydrides",
        "removeInSGroups", "removeIsotopes", "removeMapped", "removeNonimplicit",
        "removeOnlyHNeighbors", "removeWithQuery", "removeWithWedgedBond",
    ):
        setattr(parameters, field, True)
    parameters.sanitize = True
    result = Chem.RemoveHs(molecule, parameters)
    if any(atom.GetAtomicNum() == 1 for atom in result.GetAtoms()):
        raise ValueError("explicit hydrogen survived the frozen heavy-atom policy")
    return result


def read_molecule(path: str | Path):
    """Read exactly one sanitized SDF molecule, never skip a failed record."""
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    if len(supplier) != 1 or supplier[0] is None:
        raise ValueError(f"expected exactly one readable molecule in {path}")
    result = remove_all_hs(supplier[0])
    if result.GetNumConformers() != 1 or result.GetNumAtoms() == 0:
        raise ValueError(f"expected one nonempty conformer in {path}")
    return result


def atom_order_signature(molecule) -> str:
    """Same ordered chemistry signature as the established V2 manifest builder.

    Chiral tags are not included: SDF round trips can change those tags without
    changing heavy-atom order or geometry. No geometrical quality filtering or
    chirality-based pose selection is performed here.
    """
    atoms = [
        (atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge(), atom.GetAtomMapNum())
        for atom in molecule.GetAtoms()
    ]
    bonds = sorted(
        (min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
         max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()), str(bond.GetBondType()))
        for bond in molecule.GetBonds()
    )
    return hashlib.sha256(json.dumps({"atoms": atoms, "bonds": bonds}, sort_keys=True).encode()).hexdigest()


def check_geometry(reference, poses: list) -> list[dict]:
    """Verify common atom order and recompute symmetry RMSD without alignment.

    The fixed evaluator writes molecules in the native/cache order. A different
    order is an error, not an invitation to choose a RMSD-minimizing atom map.
    Symmetry-equivalent maps are used *only* inside the reported RMSD metric.
    No coordinates or pose files are rewritten.
    """
    import numpy as np
    from spyrmsd import molecule as spyrmsd_molecule
    from spyrmsd.rmsd import symmrmsd

    reference_signature = atom_order_signature(reference)
    native_xyz = np.asarray(reference.GetConformer().GetPositions(), dtype=np.float64)
    if native_xyz.shape != (reference.GetNumAtoms(), 3) or not np.isfinite(native_xyz).all():
        raise ValueError("nonfinite or malformed native coordinates")
    if not poses:
        raise ValueError("empty pose group")
    coordinates = []
    for pose in poses:
        xyz = np.asarray(pose.GetConformer().GetPositions(), dtype=np.float64)
        if xyz.shape != native_xyz.shape or not np.isfinite(xyz).all():
            raise ValueError("nonfinite coordinates or heavy-atom count mismatch")
        if atom_order_signature(pose) != reference_signature:
            raise ValueError("ordered atom identity/bond connectivity mismatch; automatic remapping is forbidden")
        coordinates.append(xyz)
    graph = spyrmsd_molecule.Molecule.from_rdkit(reference)
    # Explicit center=False/minimize=False is essential. A translated ligand
    # must remain displaced; receptor and native coordinates define the frame.
    values = symmrmsd(native_xyz, coordinates, graph.atomicnums, graph.atomicnums,
                     graph.adjacency_matrix, graph.adjacency_matrix,
                     center=False, minimize=False, cache=True)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape != (len(poses),) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("symmetry RMSD returned invalid values; no coordinate-RMSD fallback is allowed")
    result = []
    for xyz, value in zip(coordinates, values):
        direct = float(np.sqrt(np.square(xyz - native_xyz).sum(axis=1).mean()))
        if float(value) > direct + 1e-5:
            raise ValueError("symmetry RMSD exceeds the verified identity-map RMSD")
        result.append({
            "rmsd": float(value), "rmsd_method": RMSD_METHOD,
            "rmsd_direct": direct,
            "centroid_distance_recomputed": float(np.linalg.norm(xyz.mean(axis=0) - native_xyz.mean(axis=0))),
            "atom_count": reference.GetNumAtoms(), "atom_order_sha256": reference_signature,
            "atom_mapping": "identity_order_verified", "coordinates_frame": "global_receptor_frame",
        })
    return result


def self_test() -> None:
    """Synthetic in-memory CPU checks; no structures, test data or files read."""
    import math
    from rdkit import Chem

    def molecule(smiles, xyz):
        mol = Chem.MolFromSmiles(smiles)
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for index, point in enumerate(xyz):
            conformer.SetAtomPosition(index, point)
        mol.AddConformer(conformer)
        return mol

    reference = molecule("CO", [(0., 0., 0.), (1.4, 0., 0.)])
    translated = molecule("CO", [(4., 0., 0.), (5.4, 0., 0.)])
    assert abs(check_geometry(reference, [reference])[0]["rmsd"]) < 1e-12
    assert abs(check_geometry(reference, [translated])[0]["rmsd"] - 4.0) < 1e-8
    # Equivalent terminal atoms may exchange through metric symmetry only.
    symmetric = molecule("CC", [(0., 0., 0.), (1.5, 0., 0.)])
    swapped = molecule("CC", [(1.5, 0., 0.), (0., 0., 0.)])
    assert check_geometry(symmetric, [swapped])[0]["rmsd"] < 1e-8
    reordered = Chem.RenumberAtoms(reference, [1, 0])
    try:
        check_geometry(reference, [reordered])
    except ValueError as exc:
        assert "ordered atom" in str(exc)
    else:
        raise AssertionError("atom-order mismatch was accepted")
    bad = Chem.Mol(reference)
    bad.GetConformer().SetAtomPosition(0, (math.nan, 0., 0.))
    try:
        check_geometry(reference, [bad])
    except ValueError as exc:
        assert "nonfinite" in str(exc)
    else:
        raise AssertionError("nonfinite pose was accepted")
    charged = molecule("C[O-]", [(0., 0., 0.), (1.4, 0., 0.)])
    try:
        check_geometry(reference, [charged])
    except ValueError as exc:
        assert "ordered atom" in str(exc)
    else:
        raise AssertionError("chemistry mismatch was accepted")
    print(json.dumps({"status": "PASS", "checks": 6, "data": "synthetic_in_memory_only", "rigid_alignment": False}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    self_test()
