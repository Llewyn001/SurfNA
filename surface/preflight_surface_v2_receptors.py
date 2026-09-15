#!/usr/bin/env python3
"""Preflight PDB2PQR invariants and select clean Surface-v2 pilot receptors."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import generate_surfaces as legacy
import generate_surfaces_v2 as v2


AUDIT_FIELDS = [
    "name", "status", "reason", "receptor_file", "ligand_file", "apbs_residues",
    "apbs_chain_count", "pdb2pqr_version", "pqr_input_heavy_atoms",
    "pqr_output_heavy_atoms", "pqr_added_heavy_atoms", "pqr_max_coordinate_delta", "pqr_total_charge",
    "pqr_added_heavy_atoms_within_exclusion_radius", "pqr_added_heavy_atom_min_ligand_distance",
    "pqr_repair_exclusion_radius", "pdb2pqr_warning_count", "pdb2pqr_warnings",
]


def make_config(args: argparse.Namespace) -> v2.SurfaceV2Config:
    return v2.SurfaceV2Config(
        data_dir=str(Path(args.data_dir).resolve()), out_dir="", target_kind=args.target_kind,
        surface_vertex_cutoff=8.0, pocket_residue_cutoff=15.0,
        apbs_context_cutoff=args.chain_selection_cutoff, min_vertices=20,
        overwrite=False, mesh_res=1.0, fix_mesh_backend="none", pymesh_python="",
        density=4.0, hdensity=4.0, probe_radius=1.4, apbs_salt_conc=0.15,
        pdb2pqr_ff="AMBER", pdb2pqr_bin=str(Path(args.pdb2pqr_bin).resolve()),
        apbs_bin="", msms_bin="", charge_scale=1.0,
        pqr_coordinate_tolerance=args.pqr_coordinate_tolerance,
        pqr_repair_exclusion_radius=args.pqr_repair_exclusion_radius,
        strict_closed_full_mesh=True, keep_failed_temp=False,
    )


def preflight(job: tuple[str, Path, Path], cfg: v2.SurfaceV2Config) -> dict[str, object]:
    name, receptor_file, ligand_file = job
    temp_dir = Path(tempfile.mkdtemp(prefix=f"surfna_v2_preflight_{name}_"))
    row: dict[str, object] = {
        "name": name, "status": "failed", "reason": "",
        "receptor_file": str(receptor_file), "ligand_file": str(ligand_file),
    }
    try:
        ligand_coords = legacy.read_ligand_coords(ligand_file)
        cleaned = temp_dir / f"{name}_apbs_chains.pdb"
        residue_count, chain_count = v2.write_clean_receptor_chains(
            receptor_file, ligand_coords, cleaned, cfg.target_kind, cfg.apbs_context_cutoff
        )
        _pqr, _apbs_input, stats = v2.prepare_pqr_strict(cleaned, temp_dir, cfg, ligand_coords)
        row.update({
            "status": "success", "apbs_residues": residue_count,
            "apbs_chain_count": chain_count, **stats,
        })
    except Exception as exc:  # noqa: BLE001
        row["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return row


def load_checkpoint(audit_output: Path, names: list[str]) -> dict[str, dict[str, str]]:
    """Load a prior complete-batch audit and reject stale/malformed checkpoints."""
    if not audit_output.exists():
        return {}
    allowed = set(names)
    rows: dict[str, dict[str, str]] = {}
    for row in csv.DictReader(audit_output.open()):
        name = row.get("name", "")
        if name not in allowed:
            raise ValueError(f"checkpoint contains name absent from current manifest: {name}")
        if name in rows:
            raise ValueError(f"checkpoint contains duplicate name: {name}")
        if row.get("status") not in {"success", "failed"}:
            raise ValueError(f"checkpoint has invalid status for {name}: {row.get('status')}")
        rows[name] = row
    return rows


def write_checkpoint(
    rows_by_name: dict[str, dict[str, object]], names: list[str], args: argparse.Namespace,
) -> int:
    """Atomically persist every completed batch, so a requeued job can resume exactly."""
    rows = [rows_by_name[name] for name in names if name in rows_by_name]
    successful = [row["name"] for row in rows if row["status"] == "success"]
    selected = successful[: args.required] if args.required else successful
    selected = sorted(selected)

    selected_output = Path(args.selected_output)
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    selected_tmp = selected_output.with_name(selected_output.name + ".tmp")
    selected_tmp.write_text("\n".join(selected) + ("\n" if selected else ""))
    selected_tmp.replace(selected_output)

    audit_output = Path(args.audit_output)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_tmp = audit_output.with_name(audit_output.name + ".tmp")
    with audit_tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in AUDIT_FIELDS} for row in rows)
    audit_tmp.replace(audit_output)
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy_audit_csv")
    parser.add_argument(
        "--names_file",
        help="Explicit complex names (one per line). Required for sharded full preflight.",
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--target_kind", choices=["protein", "na"], required=True)
    parser.add_argument("--pdb2pqr_bin", required=True)
    parser.add_argument("--selected_output", required=True)
    parser.add_argument("--audit_output", required=True)
    parser.add_argument("--required", type=int, default=50)
    parser.add_argument("--candidate_count", type=int, default=500,
                        help="Legacy-audit candidates to sample; 0 means all. Ignored with --names_file.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--chain_selection_cutoff", type=float, default=30.0)
    parser.add_argument("--pqr_coordinate_tolerance", type=float, default=0.05)
    parser.add_argument("--pqr_repair_exclusion_radius", type=float, default=12.0)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from a batch-atomic audit file written to --audit_output.")
    args = parser.parse_args()
    if not args.names_file and not args.legacy_audit_csv:
        parser.error("one of --names_file or --legacy_audit_csv is required")
    if args.required < 0 or args.candidate_count < 0:
        parser.error("--required and --candidate_count must be non-negative")
    cfg = make_config(args)

    data_dir = Path(args.data_dir)
    if args.names_file:
        names = [line.strip() for line in Path(args.names_file).read_text().splitlines() if line.strip()]
        if len(names) != len(set(names)):
            raise ValueError("--names_file must not contain duplicate complex names")
    else:
        successful_names = []
        for row in csv.DictReader(Path(args.legacy_audit_csv).open()):
            name = row.get("name", "").strip()
            if row.get("status") == "success" and name and (data_dir / name).is_dir():
                successful_names.append(name)
        names = sorted(set(successful_names))
        random.Random(args.seed).shuffle(names)
        if args.candidate_count:
            names = names[: min(args.candidate_count, len(names))]
    jobs_by_name = {
        job[0]: job for job in legacy.discover_complexes(data_dir, names, limit=0)
    }
    missing = [name for name in names if name not in jobs_by_name]
    if missing:
        raise ValueError(f"{len(missing)} requested complexes are not discoverable, e.g. {missing[:5]}")
    audit_output = Path(args.audit_output)
    rows_by_name: dict[str, dict[str, object]] = {}
    if args.resume:
        rows_by_name.update(load_checkpoint(audit_output, names))
    jobs = [jobs_by_name[name] for name in names if name not in rows_by_name]

    batch_size = max(args.workers * 2, 1)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for start in range(0, len(jobs), batch_size):
            selected_count = sum(
                row["status"] == "success" for row in rows_by_name.values()
            )
            if args.required and selected_count >= args.required:
                break
            batch = jobs[start:start + batch_size]
            batch_rows = list(executor.map(lambda job: preflight(job, cfg), batch))
            rows_by_name.update({row["name"]: row for row in batch_rows})
            selected_count = write_checkpoint(rows_by_name, names, args)
            print(
                f"preflight tested={len(rows_by_name)} passed={selected_count} "
                f"required={args.required} pending={len(names) - len(rows_by_name)}"
            )
            if args.required and selected_count >= args.required:
                break

    selected_count = write_checkpoint(rows_by_name, names, args)
    if not args.required and len(rows_by_name) != len(names):
        raise RuntimeError("full preflight ended before every manifest name was audited")
    print(f"selected={selected_count} list={args.selected_output} audit={args.audit_output}")
    if selected_count < args.required:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
