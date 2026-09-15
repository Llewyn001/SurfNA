#!/usr/bin/env python3
"""Audit where PDB2PQR-added heavy atoms lie relative to the ligand."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.neighbors import KDTree

import generate_surfaces as legacy
import generate_surfaces_v2 as v2


def pqr_heavy_atoms(path: Path) -> tuple[np.ndarray, list[str]]:
    coords: list[list[float]] = []
    labels: list[str] = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        fields = line.split()
        if len(fields) < 10 or v2.is_pqr_hydrogen(fields[2]):
            continue
        coords.append([float(value) for value in fields[-5:-2]])
        labels.append("/".join(fields[2:-5]))
    return np.asarray(coords, dtype=float), labels


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def audit_row(
    row: dict[str, str], repo_root: Path, exclusion_radius: float
) -> dict[str, object]:
    target_kind = row["target_kind"]
    input_coords = v2.pdb_heavy_coordinates(
        resolve_path(repo_root, row["apbs_pdb"]), target_kind
    )
    pqr_coords, pqr_labels = pqr_heavy_atoms(resolve_path(repo_root, row["pqr_file"]))
    ligand_coords = legacy.read_ligand_coords(resolve_path(repo_root, row["ligand_file"]))

    distances, indices = KDTree(pqr_coords).query(input_coords)
    matched = set(int(index) for index in indices[:, 0])
    added_indices = sorted(set(range(len(pqr_coords))) - matched)
    added_coords = pqr_coords[added_indices]
    if len(added_coords):
        ligand_distances = KDTree(ligand_coords).query(added_coords)[0][:, 0]
        minimum = float(np.min(ligand_distances))
        nearest_order = np.argsort(ligand_distances)[:10]
        nearest = " | ".join(
            f"{pqr_labels[added_indices[int(index)]]}:{ligand_distances[int(index)]:.3f}A"
            for index in nearest_order
        )
    else:
        ligand_distances = np.asarray([], dtype=float)
        minimum = float("inf")
        nearest = ""

    return {
        "name": row["name"],
        "target_kind": target_kind,
        "input_heavy_atoms": len(input_coords),
        "pqr_heavy_atoms": len(pqr_coords),
        "added_heavy_atoms": len(added_indices),
        "mapping_max_distance": float(np.max(distances[:, 0])),
        "added_min_ligand_distance": minimum,
        "added_within_4A": int(np.count_nonzero(ligand_distances <= 4.0)),
        "added_within_6A": int(np.count_nonzero(ligand_distances <= 6.0)),
        "added_within_8A": int(np.count_nonzero(ligand_distances <= 8.0)),
        "added_within_15A": int(np.count_nonzero(ligand_distances <= 15.0)),
        "repair_exclusion_radius": exclusion_radius,
        "added_within_exclusion_radius": int(
            np.count_nonzero(ligand_distances <= exclusion_radius)
        ),
        "nearest_added_atoms": nearest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_csv", type=Path)
    parser.add_argument("--repo_root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusion_radius", type=float, default=12.0)
    args = parser.parse_args()

    rows = [
        row for row in csv.DictReader(args.audit_csv.open())
        if row.get("status") == "success"
    ]
    audited = [audit_row(row, args.repo_root.resolve(), args.exclusion_radius) for row in rows]
    fields = list(audited[0]) if audited else ["name"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audited)

    near_exclusion = sum(int(row["added_within_exclusion_radius"]) > 0 for row in audited)
    near_4a = sum(int(row["added_within_4A"]) > 0 for row in audited)
    print(
        f"audited={len(audited)} complexes_with_added_atoms_within_{args.exclusion_radius:g}A="
        f"{near_exclusion} within_4A={near_4a} output={args.output}"
    )


if __name__ == "__main__":
    main()
