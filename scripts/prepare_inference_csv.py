#!/usr/bin/env python3
"""Create the CSV consumed by ``src/inference_accelerate.py``.

The script supports two common release workflows:

1. Benchmark-style folders:
   data_dir/<complex>/<complex>_protein.pdb and <complex>_ligand.sdf
   surface_dir/<complex>/*.pdb and *.ply
2. Single-target virtual screening:
   --protein_path, --pocket_path, --surface_ply, --ligand_library
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def build_rows_from_dirs(data_dir: Path, surface_dir: Path, ligand_library: str | None):
    rows = []
    for complex_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        name = complex_dir.name
        pdb_code = name.split("_")[0]
        protein_path = first_existing(
            [
                complex_dir / f"{name}_protein_processed.pdb",
                complex_dir / f"{name}_protein.pdb",
                complex_dir / f"{pdb_code}_protein_processed.pdb",
                complex_dir / f"{pdb_code}_protein.pdb",
            ]
        )
        ref_ligand = first_existing(
            [
                complex_dir / f"{name}_ligand.sdf",
                complex_dir / f"{pdb_code}_ligand.sdf",
                complex_dir / f"{name}_ligand.mol2",
                complex_dir / f"{pdb_code}_ligand.mol2",
            ]
        )
        surface_subdir = surface_dir / name
        if not surface_subdir.exists():
            surface_subdir = surface_dir / pdb_code
        pocket_path = first_existing(sorted(surface_subdir.glob("*_pocket*.pdb")) + sorted(surface_subdir.glob("*.pdb")))
        surface_ply = first_existing(sorted(surface_subdir.glob("*.ply")))
        ligand_path = Path(ligand_library) if ligand_library else ref_ligand
        if protein_path and ref_ligand and pocket_path and surface_ply and ligand_path:
            rows.append(
                {
                    "protein_path": str(protein_path.resolve()),
                    "pocket_path": str(pocket_path.resolve()),
                    "ref_ligand": str(ref_ligand.resolve()),
                    "ligand_path": str(Path(ligand_path).resolve()),
                    "protein_surface": str(surface_ply.resolve()),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--surface_dir", type=Path, default=None)
    parser.add_argument("--protein_path", type=Path, default=None)
    parser.add_argument("--pocket_path", type=Path, default=None)
    parser.add_argument("--surface_ply", type=Path, default=None)
    parser.add_argument("--ligand_library", type=Path, default=None)
    parser.add_argument("--ref_ligand", type=Path, default=None)
    parser.add_argument("--out_csv", type=Path, required=True)
    args = parser.parse_args()

    if args.protein_path:
        if not (args.pocket_path and args.surface_ply and args.ligand_library):
            raise SystemExit("--protein_path mode requires --pocket_path, --surface_ply and --ligand_library")
        rows = [
            {
                "protein_path": str(args.protein_path.resolve()),
                "pocket_path": str(args.pocket_path.resolve()),
                "ref_ligand": str((args.ref_ligand or args.ligand_library).resolve()),
                "ligand_path": str(args.ligand_library.resolve()),
                "protein_surface": str(args.surface_ply.resolve()),
            }
        ]
    else:
        if not (args.data_dir and args.surface_dir):
            raise SystemExit("folder mode requires --data_dir and --surface_dir")
        rows = build_rows_from_dirs(args.data_dir, args.surface_dir, str(args.ligand_library) if args.ligand_library else None)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["protein_path", "pocket_path", "ref_ligand", "ligand_path", "protein_surface"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
