#!/usr/bin/env python3
"""Validate Surface-v2 PLY topology, features and generation audit invariants."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from plyfile import PlyData


REQUIRED = [
    "x", "y", "z", "nx", "ny", "nz",
    "hbond", "hphob", "charge", "si",
    "donor", "acceptor", "apolar", "boundary",
]


def topology_boundary_mask(faces: np.ndarray, vertex_count: int) -> tuple[np.ndarray, int]:
    edge_counts: Counter[tuple[int, int]] = Counter()
    for face in faces:
        a, b, c = (int(value) for value in face)
        for u, v in ((a, b), (b, c), (c, a)):
            edge_counts[tuple(sorted((u, v)))] += 1
    edges = [edge for edge, count in edge_counts.items() if count == 1]
    mask = np.zeros(vertex_count, dtype=bool)
    for edge in edges:
        mask[list(edge)] = True
    return mask, len(edges)


def check_ply(path: Path, max_boundary_si_saturation: float) -> dict[str, object]:
    row: dict[str, object] = {"path": str(path), "status": "success", "reason": ""}
    try:
        data = PlyData.read(str(path))
        properties = {prop.name for prop in data["vertex"].properties}
        missing = [name for name in REQUIRED if name not in properties]
        if missing:
            raise ValueError("missing properties: " + ",".join(missing))
        values = {
            name: np.asarray(data["vertex"][name], dtype=float)
            for name in REQUIRED
        }
        vertex_count = len(values["x"])
        faces = np.asarray([face[0] for face in data["face"].data], dtype=np.int64)
        if vertex_count <= 9 or len(faces) <= 9:
            raise ValueError(f"mesh too small: vertices={vertex_count}, faces={len(faces)}")
        stack = np.column_stack([values[name] for name in REQUIRED])
        nan_count = int(np.count_nonzero(~np.isfinite(stack)))
        if nan_count:
            raise ValueError(f"non-finite values: {nan_count}")
        normals = np.column_stack([values["nx"], values["ny"], values["nz"]])
        normal_norms = np.linalg.norm(normals, axis=1)
        if np.max(np.abs(normal_norms - 1.0)) > 0.05:
            raise ValueError("surface normals are not unit length")
        for name in ("donor", "acceptor", "apolar", "boundary"):
            if np.min(values[name]) < -1e-5 or np.max(values[name]) > 1.00001:
                raise ValueError(f"{name} is outside [0,1]")
        if not np.allclose(values["hbond"], values["donor"] - values["acceptor"], atol=2e-5):
            raise ValueError("hbond is inconsistent with donor-acceptor")
        if not np.allclose(values["hphob"], 2.0 * values["apolar"] - 1.0, atol=2e-5):
            raise ValueError("hphob is inconsistent with the shared apolar channel")
        if np.isclose(np.std(values["charge"]), 0.0):
            raise ValueError("electrostatic potential has zero variance")
        topology_boundary, boundary_edges = topology_boundary_mask(faces, vertex_count)
        stored_boundary = values["boundary"] > 0.5
        if not np.array_equal(topology_boundary, stored_boundary):
            raise ValueError("stored boundary mask does not match mesh topology")
        saturation = float(np.mean(np.abs(values["si"][topology_boundary]) >= 0.99)) if np.any(topology_boundary) else 0.0
        if saturation > max_boundary_si_saturation:
            raise ValueError(
                f"boundary shape-index saturation {saturation:.3f} exceeds {max_boundary_si_saturation:.3f}"
            )
        row.update({
            "vertices": vertex_count,
            "faces": len(faces),
            "boundary_edges": boundary_edges,
            "boundary_vertices": int(np.count_nonzero(topology_boundary)),
            "boundary_si_saturation": saturation,
            "charge_mean": float(np.mean(values["charge"])),
            "charge_std": float(np.std(values["charge"])),
            "nan_count": nan_count,
        })
    except Exception as exc:  # noqa: BLE001
        row["status"] = "failed"
        row["reason"] = f"{type(exc).__name__}: {exc}"
    return row


def check_audit(path: Path) -> list[str]:
    errors = []
    for row in csv.DictReader(path.open()):
        name = row.get("name", "<unknown>")
        if row.get("status") not in {"success", "exists"}:
            errors.append(f"{name}: generation status={row.get('status')} reason={row.get('reason')}")
            continue
        if row.get("status") == "exists":
            continue
        checks = {
            "full_boundary_edges": 0,
            "apbs_outside_vertex_count": 0,
            "nan_count": 0,
        }
        for field, expected in checks.items():
            if int(float(row.get(field) or -1)) != expected:
                errors.append(f"{name}: {field}={row.get(field)} expected={expected}")
        input_heavy = int(row.get("pqr_input_heavy_atoms") or -1)
        output_heavy = int(row.get("pqr_output_heavy_atoms") or -2)
        added_heavy = int(row.get("pqr_added_heavy_atoms") or 0)
        if output_heavy < input_heavy or output_heavy - input_heavy != added_heavy:
            errors.append(f"{name}: inconsistent PDB/PQR heavy atom accounting")
        added_near_ligand = int(row.get("pqr_added_heavy_atoms_within_exclusion_radius") or -1)
        if added_near_ligand != 0:
            errors.append(
                f"{name}: pqr_added_heavy_atoms_within_exclusion_radius={added_near_ligand} expected=0"
            )
        if (row.get("hydrogen_policy") or "") != "heavy_only":
            errors.append(f"{name}: unexpected hydrogen policy")
        if (row.get("charge_units") or "") != "kBT/e":
            errors.append(f"{name}: unexpected charge units")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("surface_dir")
    parser.add_argument("--glob", default="*/*_protein_8A.ply")
    parser.add_argument("--audit_csv", default=None)
    parser.add_argument("--max_boundary_si_saturation", type=float, default=0.50)
    args = parser.parse_args()
    surface_dir = Path(args.surface_dir)
    rows = [
        check_ply(path, args.max_boundary_si_saturation)
        for path in sorted(surface_dir.glob(args.glob))
    ]
    fields = [
        "path", "status", "reason", "vertices", "faces", "boundary_edges",
        "boundary_vertices", "boundary_si_saturation", "charge_mean", "charge_std", "nan_count",
    ]
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    audit_path = Path(args.audit_csv) if args.audit_csv else surface_dir / "surface_v2_audit.csv"
    audit_errors = check_audit(audit_path) if audit_path.exists() else [f"missing audit CSV: {audit_path}"]
    for error in audit_errors:
        print(f"# AUDIT_ERROR {error}")
    failures = sum(row["status"] == "failed" for row in rows) + len(audit_errors)
    print(f"# Surface-v2 checked={len(rows)} failures={failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
