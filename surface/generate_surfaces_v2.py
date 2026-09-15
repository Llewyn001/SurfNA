#!/usr/bin/env python3
"""Generate audited, cross-domain-consistent SurfNA Surface-v2 pocket meshes.

This module intentionally leaves ``generate_surfaces.py`` unchanged.  It
reuses its stable parsing, MSMS reading, mesh-regularization and DX parsing
helpers, while enforcing the invariants that the legacy data did not enforce:

* MSMS geometry is heavy-atom-only and therefore independent of input H atoms;
* one explicit modern PDB2PQR executable is used, without silent fallback;
* input/PQR heavy-atom coordinates must match one-to-one;
* curvature is computed on the regularized full molecular surface before crop;
* donor, acceptor and apolar propensities use the same atom-level definition in
  proteins and nucleic acids;
* open-boundary and electrostatic-grid diagnostics are written to the audit.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import math
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from Bio.PDB import NeighborSearch, PDBIO, PDBParser, Select
from sklearn.neighbors import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_surfaces as legacy  # noqa: E402


NA_BASE = {
    "A": "A", "DA": "A", "ADE": "A",
    "C": "C", "DC": "C", "CYT": "C",
    "G": "G", "DG": "G", "GUA": "G",
    "U": "U", "DU": "U", "URA": "U",
    "T": "T", "DT": "T", "THY": "T",
}
NA_DONORS = {
    "A": {"N6"},
    "C": {"N4"},
    "G": {"N1", "N2"},
    "U": {"N3"},
    "T": {"N3"},
}
NA_ACCEPTORS = {
    "A": {"N1", "N3", "N7"},
    "C": {"N3", "O2"},
    "G": {"N3", "N7", "O6"},
    "U": {"O2", "O4"},
    "T": {"O2", "O4"},
}
NA_BACKBONE_ACCEPTORS = {
    "OP1", "OP2", "OP3", "O1P", "O2P", "O3P",
    "O2'", "O3'", "O4'", "O5'",
}
RNA_RESIDUES = {"A", "C", "G", "U", "ADE", "CYT", "GUA", "URA"}

PROTEIN_SIDECHAIN_DONORS = {
    "ARG": {"NE", "NH1", "NH2"},
    "ASN": {"ND2"},
    "GLN": {"NE2"},
    "HIS": {"ND1", "NE2"}, "HID": {"ND1"}, "HIE": {"NE2"},
    "HIP": {"ND1", "NE2"},
    "LYS": {"NZ"},
    "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
    "TRP": {"NE1"}, "CYS": {"SG"}, "CYX": {"SG"},
}
PROTEIN_SIDECHAIN_ACCEPTORS = {
    "ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"},
    "ASN": {"OD1"}, "GLN": {"OE1"},
    "HIS": {"ND1", "NE2"}, "HID": {"NE2"}, "HIE": {"ND1"},
    "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
    "CYS": {"SG"}, "CYM": {"SG"}, "MET": {"SD"},
}


@dataclass(frozen=True)
class SurfaceV2Config:
    data_dir: str
    out_dir: str
    target_kind: str
    surface_vertex_cutoff: float
    pocket_residue_cutoff: float
    apbs_context_cutoff: float
    min_vertices: int
    overwrite: bool
    mesh_res: float
    fix_mesh_backend: str
    pymesh_python: str
    density: float
    hdensity: float
    probe_radius: float
    apbs_salt_conc: float
    pdb2pqr_ff: str
    pdb2pqr_bin: str
    apbs_bin: str
    msms_bin: str
    charge_scale: float
    pqr_coordinate_tolerance: float
    pqr_repair_exclusion_radius: float
    allow_remote_terminal_phosphate_trim: bool
    remote_terminal_phosphate_trim_radius: float
    complex_timeout_seconds: int
    split_polymer_backbone_gaps: bool
    strict_closed_full_mesh: bool
    keep_failed_temp: bool


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    tools_dir = repo_root / "tools" / "transfer"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--audit_path",
        default="",
        help="Optional explicit audit CSV path; defaults to out_dir/surface_v2_audit.csv.",
    )
    parser.add_argument("--target_kind", choices=["protein", "na"], required=True)
    parser.add_argument("--surface_vertex_cutoff", type=float, default=8.0)
    parser.add_argument("--pocket_residue_cutoff", type=float, default=15.0)
    parser.add_argument(
        "--apbs_context_cutoff",
        type=float,
        default=30.0,
        help=(
            "Select complete receptor chains having an allowed atom within this distance of the ligand. "
            "Negative includes every cleaned receptor chain."
        ),
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--complexes", nargs="*", default=None)
    parser.add_argument("--complexes_file", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min_vertices", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mesh_res", type=float, default=1.0)
    parser.add_argument("--fix_mesh_backend", choices=["pymesh", "builtin", "none"], default="pymesh")
    parser.add_argument("--pymesh_python", default=legacy.default_pymesh_python())
    parser.add_argument("--density", type=float, default=4.0)
    parser.add_argument("--hdensity", type=float, default=4.0)
    parser.add_argument("--probe_radius", type=float, default=1.4)
    parser.add_argument("--apbs_salt_conc", type=float, default=0.15)
    parser.add_argument("--pdb2pqr_ff", default="AMBER")
    parser.add_argument("--pdb2pqr_bin", default=shutil.which("pdb2pqr") or "")
    parser.add_argument(
        "--apbs_bin",
        default=str(repo_root / "tools" / "conda_apbs_1.5" / "bin" / "apbs"),
    )
    parser.add_argument(
        "--msms_bin",
        default=str(tools_dir / "APBS-3.4.1.Linux" / "bin" / "msms"),
    )
    parser.add_argument(
        "--charge_scale",
        type=float,
        default=1.0,
        help="Multiplier applied to raw APBS kBT/e values. V2 defaults to physical APBS units.",
    )
    parser.add_argument("--pqr_coordinate_tolerance", type=float, default=0.05)
    parser.add_argument(
        "--pqr_repair_exclusion_radius",
        type=float,
        default=12.0,
        help=(
            "Reject receptors when PDB2PQR adds a heavy atom within this distance of the ligand. "
            "Negative disables this chemical-integrity gate."
        ),
    )
    parser.add_argument(
        "--allow_remote_terminal_phosphate_trim",
        action="store_true",
        help=(
            "Rescue only PDB2PQR omissions that are canonical phosphate atoms outside the "
            "ligand exclusion radius. Disabled by default."
        ),
    )
    parser.add_argument(
        "--remote_terminal_phosphate_trim_radius",
        type=float,
        default=12.0,
        help="Minimum ligand distance for the optional remote-terminal-phosphate rescue.",
    )
    parser.add_argument(
        "--complex_timeout_seconds",
        type=int,
        default=0,
        help="Hard wall-clock cap per complex; 0 disables the cap. Timed-out samples are audited as failures.",
    )
    parser.add_argument(
        "--split_polymer_backbone_gaps",
        action="store_true",
        help=(
            "Write observed polymer fragments as separate PDB chains before PDB2PQR. "
            "This preserves every observed heavy-atom coordinate but prevents artificial bonds across gaps."
        ),
    )
    parser.add_argument(
        "--allow_open_full_mesh",
        dest="strict_closed_full_mesh",
        action="store_false",
        default=True,
    )
    parser.add_argument("--keep_failed_temp", action="store_true")
    return parser.parse_args()


def legacy_mesh_config(cfg: SurfaceV2Config) -> legacy.SurfaceConfig:
    """Build the legacy helper config without enabling legacy processing."""
    return legacy.SurfaceConfig(
        data_dir=cfg.data_dir,
        out_dir=cfg.out_dir,
        target_kind=cfg.target_kind,
        surface_vertex_cutoff=cfg.surface_vertex_cutoff,
        pocket_residue_cutoff=cfg.pocket_residue_cutoff,
        min_vertices=cfg.min_vertices,
        overwrite=cfg.overwrite,
        no_fix_mesh=cfg.fix_mesh_backend == "none",
        mesh_res=cfg.mesh_res,
        msms_one_cavity=True,
        fix_mesh_backend=cfg.fix_mesh_backend,
        pymesh_python=cfg.pymesh_python,
        density=cfg.density,
        hdensity=cfg.hdensity,
        probe_radius=cfg.probe_radius,
        output_cutoff_label=cfg.surface_vertex_cutoff,
        apbs_context_cutoff=cfg.apbs_context_cutoff,
        apbs_salt_conc=cfg.apbs_salt_conc,
        pdb2pqr_ff=cfg.pdb2pqr_ff,
        ply_schema="surfna",
        tools_dir=str(Path(cfg.msms_bin).resolve().parents[2]),
    )


def heavy_surface_atoms(pdb_file: Path, target_kind: str) -> list:
    """Return allowed heavy atoms only; input hydrogen state cannot affect SES."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("surface_v2", str(pdb_file))
    atoms = []
    for atom in structure.get_atoms():
        residue = atom.get_parent()
        if residue.get_id()[0] != " ":
            continue
        if not legacy.residue_allowed(residue.get_resname(), target_kind):
            continue
        element = legacy.infer_element(atom)
        if element == "H" or element not in legacy.RADII:
            continue
        atoms.append(atom)
    return atoms


def write_clean_receptor_chains(
    receptor_file: Path,
    ligand_coords: np.ndarray,
    output_file: Path,
    target_kind: str,
    chain_selection_cutoff: float,
) -> tuple[int, int]:
    """Write complete relevant chains without ligand/water/unknown residues."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("surface_v2_apbs", str(receptor_file))
    atoms = [
        atom for atom in structure.get_atoms()
        if atom.get_parent().get_id()[0] == " "
        and legacy.residue_allowed(atom.get_parent().get_resname(), target_kind)
    ]
    if not atoms:
        raise ValueError("No allowed atoms in the APBS receptor")
    if chain_selection_cutoff < 0:
        selected_chains = {atom.get_parent().get_parent() for atom in atoms}
    else:
        search = NeighborSearch(atoms)
        selected_chains = {
            residue.get_parent()
            for coord in ligand_coords
            for residue in search.search(coord, chain_selection_cutoff, level="R")
        }
    if not selected_chains:
        raise ValueError("No complete receptor chain selected for APBS")

    class AllowedPolymerSelect(Select):
        def accept_residue(self, residue):  # noqa: D401
            return int(
                residue.get_parent() in selected_chains
                and
                residue.get_id()[0] == " "
                and legacy.residue_allowed(residue.get_resname(), target_kind)
            )

    count = sum(
        residue.get_parent() in selected_chains
        and residue.get_id()[0] == " "
        and legacy.residue_allowed(residue.get_resname(), target_kind)
        for residue in structure.get_residues()
    )
    if count == 0:
        raise ValueError("No allowed residues in the full APBS receptor")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(output_file), AllowedPolymerSelect())
    return int(count), len(selected_chains)


def split_observed_polymer_backbone_gaps(pdb_file: Path, target_kind: str) -> dict[str, int]:
    """Represent observed discontinuous polymer fragments as distinct PDB chains.

    No coordinate is reconstructed or removed.  A break is introduced only
    when two adjacent observed residues lack a peptide/backbone connecting atom
    or their C--N distance is incompatible with a covalent link.  This stops
    PDB2PQR from inventing a bond across a crystallographic/model gap.
    """
    if target_kind not in {"protein", "na"}:
        raise ValueError(target_kind)
    structure = PDBParser(QUIET=True).get_structure("surface_v2_fragment", str(pdb_file))
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
    next_label = 0
    residue_labels: dict[tuple[str, int, str], str] = {}
    fragments = 0
    breaks = 0
    for chain in structure.get_chains():
        residues = [
            residue for residue in chain.get_residues()
            if residue.get_id()[0] == " " and legacy.residue_allowed(residue.get_resname(), target_kind)
        ]
        previous = None
        current_label = ""
        for residue in residues:
            should_break = previous is None
            if previous is not None:
                if "C" not in previous or "N" not in residue:
                    should_break = True
                else:
                    should_break = float(np.linalg.norm(previous["C"].get_coord() - residue["N"].get_coord())) > 2.0
            if should_break:
                if previous is not None:
                    breaks += 1
                if next_label >= len(labels):
                    raise ValueError("Too many polymer fragments for one-character PDB chain identifiers")
                current_label = labels[next_label]
                next_label += 1
                fragments += 1
            residue_labels[(chain.id.strip() or " ", int(residue.get_id()[1]), residue.get_id()[2].strip() or " ")] = current_label
            previous = residue
    if not breaks:
        return {"polymer_fragment_count": fragments, "polymer_backbone_break_count": 0}

    rewritten = []
    for line in pdb_file.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 27:
            chain = line[21].strip() or " "
            try:
                resid = int(line[22:26])
            except ValueError:
                rewritten.append(line)
                continue
            insertion = line[26].strip() or " "
            label = residue_labels.get((chain, resid, insertion))
            if label is not None:
                line = line[:21] + label + line[22:]
        rewritten.append(line)
    pdb_file.write_text("\n".join(rewritten) + "\n")
    return {"polymer_fragment_count": fragments, "polymer_backbone_break_count": breaks}


def write_xyzrn_heavy(pdb_file: Path, xyzrn_file: Path, target_kind: str) -> tuple[int, list]:
    atoms = heavy_surface_atoms(pdb_file, target_kind)
    if not atoms:
        raise ValueError("No allowed heavy atoms available for Surface-v2 MSMS")
    with xyzrn_file.open("w") as handle:
        for atom in atoms:
            residue = atom.get_parent()
            resname = legacy.normalize_resname(residue.get_resname())
            atom_name = legacy.normalize_atom_name(atom.get_name())
            element = legacy.infer_element(atom)
            chain = residue.get_parent().get_id().strip() or "X"
            insertion = residue.get_id()[2].strip() or "x"
            full_id = f"{chain}_{residue.get_id()[1]}_{insertion}_{resname}_{atom_name}_V2"
            coord = atom.get_coord()
            handle.write(
                f"{coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f} "
                f"{legacy.RADII[element]:.6f} 1 {full_id}\n"
            )
    return len(atoms), atoms


def run_msms_v2(
    pocket_pdb: Path,
    ligand_coords: np.ndarray,
    tmp_dir: Path,
    cfg: SurfaceV2Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], int, str]:
    msms_bin = Path(cfg.msms_bin)
    if not msms_bin.exists():
        raise FileNotFoundError(f"MSMS binary not found: {msms_bin}")
    xyzrn_file = tmp_dir / "surface_v2.xyzrn"
    atom_count, atoms = write_xyzrn_heavy(pocket_pdb, xyzrn_file, cfg.target_kind)
    coords = np.asarray([atom.get_coord() for atom in atoms], dtype=float)
    ligand_center = np.mean(ligand_coords, axis=0)
    center_idx = int(np.argmin(np.linalg.norm(coords - ligand_center, axis=1)))
    ligand_dists, _ = KDTree(ligand_coords).query(coords)
    nearest_idx = int(np.argmin(ligand_dists[:, 0]))
    attempts = []
    for label, idx in (("one_cavity_center", center_idx), ("one_cavity_ligand_nearest", nearest_idx)):
        if not any(old_idx == idx for _, old_idx in attempts):
            attempts.append((label, idx))

    errors = []
    for attempt_number, (label, atom_idx) in enumerate(attempts):
        root = tmp_dir / f"msms_v2_{attempt_number}"
        cmd = [
            str(msms_bin),
            "-density", str(cfg.density),
            "-hdensity", str(cfg.hdensity),
            "-probe", str(cfg.probe_radius),
            "-one_cavity", "1", str(atom_idx),
            "-if", str(xyzrn_file),
            "-of", str(root),
            "-af", str(root),
        ]
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if completed.returncode != 0:
            errors.append(f"{label}: {(completed.stderr + completed.stdout)[-500:]}")
            continue
        try:
            vertices, faces, normals, names = legacy.read_msms(root)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            continue
        if len(vertices) and len(faces):
            return vertices, faces, normals, names, atom_count, label
    raise RuntimeError("MSMS one-cavity failed: " + " | ".join(errors)[-1500:])


def atom_propensities(names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return compatibility hbond/hphob plus explicit donor/acceptor/apolar."""
    donor = np.zeros(len(names), dtype=float)
    acceptor = np.zeros(len(names), dtype=float)
    apolar = np.zeros(len(names), dtype=float)
    for i, name in enumerate(names):
        resname, atom_name = legacy.atom_from_msms_name(name)
        atom_name = legacy.normalize_atom_name(atom_name)
        element = "".join(ch for ch in atom_name if ch.isalpha())[:1]
        apolar[i] = 1.0 if element in {"C", "S"} else 0.0
        if resname in NA_BASE:
            base = NA_BASE[resname]
            donor[i] = float(atom_name in NA_DONORS[base] or (resname in RNA_RESIDUES and atom_name == "O2'"))
            acceptor[i] = float(atom_name in NA_ACCEPTORS[base] or atom_name in NA_BACKBONE_ACCEPTORS)
        else:
            donor_atoms = PROTEIN_SIDECHAIN_DONORS.get(resname, set())
            acceptor_atoms = PROTEIN_SIDECHAIN_ACCEPTORS.get(resname, set())
            donor[i] = float((atom_name == "N" and resname != "PRO") or atom_name in donor_atoms)
            acceptor[i] = float(atom_name in {"O", "OXT"} or atom_name in acceptor_atoms)
    hbond = donor - acceptor
    hphob = 2.0 * apolar - 1.0
    return hbond, hphob, donor, acceptor, apolar


def assign_vector_to_new_mesh(new_vertices: np.ndarray, old_vertices: np.ndarray, values: np.ndarray) -> np.ndarray:
    assigned = np.column_stack(
        [legacy.assign_to_new_mesh(new_vertices, old_vertices, values[:, i]) for i in range(values.shape[1])]
    )
    norms = np.linalg.norm(assigned, axis=1)
    norms[norms < 1e-8] = 1.0
    return assigned / norms[:, None]


def mesh_boundary(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, int]:
    del vertices
    edge_counts: Counter[tuple[int, int]] = Counter()
    for face in faces:
        a, b, c = (int(x) for x in face)
        for u, v in ((a, b), (b, c), (c, a)):
            edge_counts[tuple(sorted((u, v)))] += 1
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    mask = np.zeros(max((max(edge) for edge in edge_counts), default=-1) + 1, dtype=float)
    for edge in boundary_edges:
        mask[list(edge)] = 1.0
    return mask, len(boundary_edges)


def crop_mesh_with_indices(
    vertices: np.ndarray,
    faces: np.ndarray,
    ligand_coords: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances, _ = KDTree(ligand_coords).query(vertices)
    keep = set(np.flatnonzero(distances[:, 0] <= cutoff).tolist())
    kept_faces = [face for face in faces if all(int(vertex) in keep for vertex in face)]
    if not kept_faces:
        raise ValueError("No full-surface faces remained after ligand-distance crop")
    used = np.asarray(sorted({int(v) for face in kept_faces for v in face}), dtype=np.int64)
    remap = {int(old): new for new, old in enumerate(used.tolist())}
    cropped_faces = np.asarray([[remap[int(v)] for v in face] for face in kept_faces], dtype=np.int64)
    return vertices[used], cropped_faces, used


def pdb_heavy_coordinates(path: Path, target_kind: str) -> np.ndarray:
    atoms = heavy_surface_atoms(path, target_kind)
    return np.asarray([atom.get_coord() for atom in atoms], dtype=float)


REMOTE_TRIMMABLE_PHOSPHATE_ATOMS = {"P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P"}


def remote_terminal_phosphate_trim(
    apbs_pdb: Path,
    pqr_coords: np.ndarray,
    ligand_coords: np.ndarray | None,
    cfg: SurfaceV2Config,
) -> tuple[np.ndarray, dict[str, object]] | None:
    """Virtually remove PDB2PQR-unparameterized remote terminal phosphate atoms.

    The PQR output already excludes these atoms.  This helper only admits that
    output if the corresponding *observed* PDB atoms are canonical phosphate
    atoms beyond the ligand safety radius and every retained heavy atom has an
    exact, one-to-one PQR coordinate match.  The mesh is subsequently built
    from the PQR-standardized receptor, so the saved surface and APBS input are
    chemically and geometrically consistent.
    """
    if not cfg.allow_remote_terminal_phosphate_trim or ligand_coords is None or not len(pqr_coords):
        return None
    atoms = heavy_surface_atoms(apbs_pdb, cfg.target_kind)
    input_coords = np.asarray([atom.get_coord() for atom in atoms], dtype=float)
    distances, indices = KDTree(pqr_coords).query(input_coords)
    matched = distances[:, 0] <= cfg.pqr_coordinate_tolerance
    missing_indices = np.flatnonzero(~matched)
    if not len(missing_indices):
        return None
    retained_indices = np.flatnonzero(matched)
    # Every retained raw atom must map one-to-one.  Extra PQR atoms remain
    # subject to the existing ligand-distance gate below; we do not silently
    # admit additions in the binding neighbourhood.
    if len(np.unique(indices[retained_indices, 0])) != len(retained_indices):
        return None
    missing_atoms = [atoms[int(index)] for index in missing_indices]
    atom_names = [legacy.normalize_atom_name(atom.get_name()).upper() for atom in missing_atoms]
    if any(name not in REMOTE_TRIMMABLE_PHOSPHATE_ATOMS for name in atom_names):
        return None
    missing_coords = input_coords[missing_indices]
    ligand_distances = KDTree(ligand_coords).query(missing_coords)[0][:, 0]
    minimum_distance = float(np.min(ligand_distances))
    if minimum_distance <= cfg.remote_terminal_phosphate_trim_radius:
        return None
    return input_coords[retained_indices], {
        "pqr_rescue_mode": "remote_terminal_phosphate_trim",
        "pqr_raw_input_heavy_atoms": len(input_coords),
        "pqr_remote_trimmed_heavy_atoms": len(missing_indices),
        "pqr_remote_trimmed_min_ligand_distance": minimum_distance,
        "pqr_remote_trimmed_atom_names": ",".join(sorted(set(atom_names))),
    }


def is_pqr_hydrogen(atom_name: str) -> bool:
    letters = "".join(ch for ch in atom_name.upper() if ch.isalpha())
    return letters.startswith(("H", "D"))


def read_pqr(path: Path) -> tuple[np.ndarray, float]:
    coords = []
    total_charge = 0.0
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        fields = line.split()
        if len(fields) < 10:
            continue
        atom_name = fields[2]
        total_charge += float(fields[-2])
        if is_pqr_hydrogen(atom_name):
            continue
        coords.append([float(value) for value in fields[-5:-2]])
    return np.asarray(coords, dtype=float), total_charge


def write_standardized_pdb_from_pqr(pqr_file: Path, pdb_file: Path) -> int:
    """Convert the parameterized PQR coordinates into an MSMS-ready PDB."""
    residue_aliases = {"RA": "A", "RC": "C", "RG": "G", "RU": "U"}
    lines = []
    for line in pqr_file.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        fields = line.split()
        if len(fields) < 10:
            continue
        atom_name = fields[2]
        resname = residue_aliases.get(fields[3].upper(), fields[3].upper())
        if len(fields) >= 11:
            chain = fields[4][:1] or "X"
            residue_token = fields[5]
        else:
            combined = fields[4]
            combined_match = re.fullmatch(r"([A-Za-z])(-?\d+)([A-Za-z]?)", combined)
            if combined_match is not None:
                chain = combined_match.group(1)
                residue_token = combined_match.group(2) + combined_match.group(3)
            else:
                chain = "X"
                residue_token = combined
        match = re.match(r"(-?\d+)([A-Za-z]?)", residue_token)
        if match is None:
            raise ValueError(f"Could not parse PQR residue identifier: {residue_token}")
        residue_number = int(match.group(1))
        insertion = match.group(2)[:1] or " "
        x, y, z = (float(value) for value in fields[-5:-2])
        letters = "".join(ch for ch in atom_name.upper() if ch.isalpha())
        element = "CL" if letters.startswith("CL") else "BR" if letters.startswith("BR") else letters[:1]
        serial = len(lines) + 1
        lines.append(
            f"ATOM  {serial:5d} {atom_name:>4s} {resname:>3s} {chain:1s}"
            f"{residue_number:4d}{insertion:1s}   {x:8.3f}{y:8.3f}{z:8.3f}"
            f"  1.00  0.00          {element:>2s}\n"
        )
    if not lines:
        raise ValueError("PQR did not contain atoms for standardized PDB")
    pdb_file.parent.mkdir(parents=True, exist_ok=True)
    pdb_file.write_text("".join(lines) + "END\n")
    return len(lines)


def executable_version(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable), "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
    )
    text = (completed.stdout + completed.stderr).strip().replace("\n", " ")
    return text[:200]


def write_apbs_compatible_pqr(pqr_file: Path, apbs_pqr_file: Path, apbs_input: Path) -> int:
    """Strip whitespace-PQR chain columns for the APBS 1.5 parser.

    PDB2PQR's ``--keep-chain --whitespace`` output has an extra chain token.
    APBS 1.5 treats a numeric chain id as the x-coordinate, which can shift the
    entire grid by hundreds of Angstrom.  The original PQR remains the strict
    coordinate-audited record; only this APBS input copy uses classic PQR
    columns.
    """
    output, stripped = [], 0
    for line in pqr_file.read_text(errors="ignore").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            fields = line.split()
            if len(fields) >= 11:
                fields = fields[:4] + fields[5:]
                stripped += 1
            output.append(" ".join(fields))
        else:
            output.append(line)
    if not stripped:
        return 0
    apbs_pqr_file.write_text("\n".join(output) + "\n")
    input_text = apbs_input.read_text()
    updated, substitutions = re.subn(
        r"(?im)^(\s*mol\s+pqr\s+)\S+", rf"\1{apbs_pqr_file.name}", input_text
    )
    if substitutions != 1:
        raise RuntimeError(f"Could not uniquely update APBS PQR reference: substitutions={substitutions}")
    apbs_input.write_text(updated)
    return stripped


def prepare_pqr_strict(
    apbs_pdb: Path,
    tmp_dir: Path,
    cfg: SurfaceV2Config,
    ligand_coords: np.ndarray | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    pdb2pqr = Path(cfg.pdb2pqr_bin)
    if not pdb2pqr.exists():
        raise FileNotFoundError(f"Fixed PDB2PQR executable not found: {pdb2pqr}")
    pqr_file = tmp_dir / "apbs_surface.pqr"
    apbs_input = tmp_dir / "apbs_surface.in"
    cmd = [
        str(pdb2pqr),
        f"--ff={cfg.pdb2pqr_ff.upper()}",
        "--whitespace",
        "--nodebump",
        "--noopt",
        "--keep-chain",
        "--apbs-input", str(apbs_input),
        str(apbs_pdb), str(pqr_file),
    ]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    log = completed.stdout + "\n" + completed.stderr
    if completed.returncode != 0 or not pqr_file.exists() or not apbs_input.exists():
        raise RuntimeError(f"PDB2PQR failed with code {completed.returncode}: {log[-3000:]}")

    pqr_text = pqr_file.read_text(errors="ignore")
    omitted = len(re.findall(r"^REMARK\s+5\s+\d+\s+", pqr_text, flags=re.MULTILINE))
    assignment_warning = "unable to assign charges" in (pqr_text + log).lower()
    if assignment_warning:
        raise ValueError(f"PDB2PQR omitted/unassigned atoms: omitted={omitted}")
    lower_log = log.lower()
    if "gap in backbone" in lower_log or "found gap in biomolecule" in lower_log:
        raise ValueError("PDB2PQR detected an unresolved polymer-backbone gap")

    raw_input_coords = pdb_heavy_coordinates(apbs_pdb, cfg.target_kind)
    input_coords = raw_input_coords
    pqr_coords, pqr_total_charge = read_pqr(pqr_file)
    rescue_stats: dict[str, object] = {
        "pqr_rescue_mode": "",
        "pqr_raw_input_heavy_atoms": len(raw_input_coords),
        "pqr_remote_trimmed_heavy_atoms": 0,
        "pqr_remote_trimmed_min_ligand_distance": "",
        "pqr_remote_trimmed_atom_names": "",
        "pqr_observed_5prime_phosphate_count": 0,
        "pqr_fragment_noninteger_charge_allowed": 0,
        "pqr_terminal_monophosphate_count": 0,
    }
    observed_5prime_count = log.count("SURFNA_PRESERVE_5PRIME_PHOSPHATE")
    if observed_5prime_count:
        rescue_stats.update({
            "pqr_rescue_mode": "preserve_observed_5prime_phosphate",
            "pqr_observed_5prime_phosphate_count": observed_5prime_count,
            "pqr_fragment_noninteger_charge_allowed": 1,
        })
    terminal_match = re.search(r"SURFNA_TERMINAL_MONOPHOSPHATE\s+count=(\d+)", log)
    if terminal_match is not None:
        rescue_stats.update({
            "pqr_rescue_mode": "amber_terminal_monophosphate",
            "pqr_terminal_monophosphate_count": int(terminal_match.group(1)),
        })
    rescued = remote_terminal_phosphate_trim(apbs_pdb, pqr_coords, ligand_coords, cfg)
    if rescued is not None:
        input_coords, rescue_stats = rescued
    elif len(pqr_coords) < len(input_coords):
        if rescued is None:
            raise ValueError(
                f"PDB2PQR omitted heavy atoms: input={len(input_coords)}, pqr={len(pqr_coords)}"
            )
    if omitted and not rescue_stats["pqr_rescue_mode"]:
        raise ValueError(f"PDB2PQR omitted/unassigned atoms: omitted={omitted}")
    distances, indices = KDTree(pqr_coords).query(input_coords)
    max_distance = float(np.max(distances[:, 0])) if len(distances) else math.inf
    unique_matches = len(np.unique(indices[:, 0]))
    if unique_matches != len(input_coords) or max_distance > cfg.pqr_coordinate_tolerance:
        raise ValueError(
            "PDB/PQR heavy atom coordinate mismatch: "
            f"unique={unique_matches}/{len(input_coords)}, max_distance={max_distance:.4f} A"
        )
    matched_pqr_indices = set(int(index) for index in indices[:, 0])
    added_indices = sorted(set(range(len(pqr_coords))) - matched_pqr_indices)
    if cfg.pqr_repair_exclusion_radius >= 0 and ligand_coords is None:
        raise ValueError("Ligand coordinates are required for the PDB2PQR repair-distance gate")
    if added_indices and ligand_coords is not None:
        added_coords = pqr_coords[added_indices]
        added_ligand_distances = KDTree(ligand_coords).query(added_coords)[0][:, 0]
        minimum_added_distance: float | str = float(np.min(added_ligand_distances))
        additions_in_exclusion_radius = int(
            np.count_nonzero(added_ligand_distances <= cfg.pqr_repair_exclusion_radius)
        ) if cfg.pqr_repair_exclusion_radius >= 0 else 0
    else:
        minimum_added_distance = ""
        additions_in_exclusion_radius = 0
    if additions_in_exclusion_radius:
        raise ValueError(
            "PDB2PQR added heavy atoms within the ligand exclusion radius: "
            f"count={additions_in_exclusion_radius}, "
            f"radius={cfg.pqr_repair_exclusion_radius:.1f} A, "
            f"minimum_distance={minimum_added_distance:.3f} A"
        )
    apbs_pqr_file = tmp_dir / "apbs_surface_apbs_compat.pqr"
    apbs_chain_columns_stripped = write_apbs_compatible_pqr(pqr_file, apbs_pqr_file, apbs_input)
    warnings = [line.strip() for line in log.splitlines() if "WARNING" in line.upper()]
    stats: dict[str, object] = {
        "pdb2pqr_version": executable_version(pdb2pqr),
        "pqr_input_heavy_atoms": len(input_coords),
        "pqr_output_heavy_atoms": len(pqr_coords),
        "pqr_added_heavy_atoms": len(pqr_coords) - len(input_coords),
        "pqr_added_heavy_atoms_within_exclusion_radius": additions_in_exclusion_radius,
        "pqr_added_heavy_atom_min_ligand_distance": minimum_added_distance,
        "pqr_repair_exclusion_radius": cfg.pqr_repair_exclusion_radius,
        "pqr_max_coordinate_delta": max_distance,
        "pqr_total_charge": pqr_total_charge,
        "apbs_pqr_chain_columns_stripped": apbs_chain_columns_stripped,
        "pdb2pqr_warning_count": len(warnings),
        "pdb2pqr_warnings": " | ".join(warnings)[:2000],
        **rescue_stats,
    }
    return pqr_file, apbs_input, stats


def interpolate_dx_strict(
    dx_file: Path,
    vertices: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    origin, deltas, grid = legacy.read_dx(dx_file)
    transform = np.linalg.inv(deltas.T)
    fractional = (vertices - origin) @ transform
    max_index = np.asarray(grid.shape, dtype=float) - 1.000001
    below = np.maximum(-fractional, 0.0)
    above = np.maximum(fractional - max_index, 0.0)
    excursion = np.maximum(below, above)
    outside = np.any(excursion > 1e-6, axis=1)
    outside_count = int(np.count_nonzero(outside))
    max_excursion = float(np.max(excursion)) if excursion.size else 0.0
    if outside_count:
        raise ValueError(
            f"APBS grid does not cover {outside_count}/{len(vertices)} surface vertices; "
            f"max grid-index excursion={max_excursion:.4f}"
        )
    fractional = np.clip(fractional, 0.0, max_index)
    base = np.floor(fractional).astype(int)
    upper = np.minimum(base + 1, np.asarray(grid.shape, dtype=int) - 1)
    frac = fractional - base
    x0, y0, z0 = base[:, 0], base[:, 1], base[:, 2]
    x1, y1, z1 = upper[:, 0], upper[:, 1], upper[:, 2]
    xd, yd, zd = frac[:, 0], frac[:, 1], frac[:, 2]
    c000 = grid[x0, y0, z0]
    c100 = grid[x1, y0, z0]
    c010 = grid[x0, y1, z0]
    c110 = grid[x1, y1, z0]
    c001 = grid[x0, y0, z1]
    c101 = grid[x1, y0, z1]
    c011 = grid[x0, y1, z1]
    c111 = grid[x1, y1, z1]
    c00 = c000 * (1 - xd) + c100 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c11 = c011 * (1 - xd) + c111 * xd
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd
    return c0 * (1 - zd) + c1 * zd, outside_count, max_excursion


def compute_apbs_v2(
    vertices: np.ndarray,
    apbs_pdb: Path,
    tmp_dir: Path,
    cfg: SurfaceV2Config,
    prepared: tuple[Path, Path, dict[str, object]] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    apbs = Path(cfg.apbs_bin)
    if not apbs.exists():
        raise FileNotFoundError(f"APBS executable not found: {apbs}")
    if prepared is None:
        _pqr_file, apbs_input, stats = prepare_pqr_strict(apbs_pdb, tmp_dir, cfg)
    else:
        _pqr_file, apbs_input, stats = prepared
    legacy.add_apbs_ions(apbs_input, cfg.apbs_salt_conc)
    completed = subprocess.run(
        [str(apbs), apbs_input.name], cwd=str(tmp_dir),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"APBS failed: {(completed.stdout + completed.stderr)[-3000:]}")
    dx_candidates = [tmp_dir / "apbs_surface.dx", tmp_dir / "apbs_surface.pqr.dx"]
    dx_file = next((path for path in dx_candidates if path.exists()), None)
    if dx_file is None:
        candidates = sorted(tmp_dir.glob("*.dx"))
        dx_file = candidates[0] if candidates else None
    if dx_file is None:
        raise RuntimeError("APBS did not create a DX potential map")
    raw, outside_count, max_excursion = interpolate_dx_strict(dx_file, vertices)
    if not np.isfinite(raw).all() or np.allclose(raw, 0.0, atol=1e-8):
        raise ValueError("APBS potential is non-finite or all-zero")
    stats.update({
        "apbs_version": executable_version(apbs),
        "apbs_outside_vertex_count": outside_count,
        "apbs_max_grid_excursion": max_excursion,
        "charge_units": "kBT/e",
        "charge_scale": cfg.charge_scale,
    })
    return raw * cfg.charge_scale, stats


def write_ply_v2(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    hbond: np.ndarray,
    hphob: np.ndarray,
    charge: np.ndarray,
    si: np.ndarray,
    donor: np.ndarray,
    acceptor: np.ndarray,
    apolar: np.ndarray,
    boundary: np.ndarray,
) -> None:
    props = [
        "x", "y", "z", "nx", "ny", "nz",
        "hbond", "hphob", "charge", "si",
        "donor", "acceptor", "apolar", "boundary",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"comment SurfNA Surface-v2; charge units kBT/e\n")
        handle.write(f"element vertex {len(vertices)}\n")
        for prop in props:
            handle.write(f"property float {prop}\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        columns = np.column_stack(
            [vertices, normals, hbond, hphob, charge, si, donor, acceptor, apolar, boundary]
        )
        for row in columns:
            handle.write(" ".join(f"{float(value):.6f}" for value in row) + "\n")
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


class ComplexWallClockTimeout(TimeoutError):
    """Raised when a single complex exceeds an explicitly configured wall-clock cap."""


@contextlib.contextmanager
def complex_wall_clock_limit(seconds: int):
    """Interrupt one worker cleanly without stalling the surrounding batch."""
    if seconds <= 0:
        yield
        return
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def on_alarm(_signum, _frame):
        raise ComplexWallClockTimeout(f"Surface-v2 complex exceeded {seconds}s wall-clock limit")

    old_handler = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def process_complex(job: tuple[str, Path, Path], cfg: SurfaceV2Config) -> dict[str, object]:
    name, receptor_file, ligand_file = job
    out_dir = Path(cfg.out_dir) / name
    label = int(cfg.surface_vertex_cutoff)
    pocket_pdb = out_dir / f"{name}_pocket_{label}A.pdb"
    apbs_pdb = (
        out_dir / f"{name}_apbs_all_chains.pdb"
        if cfg.apbs_context_cutoff < 0
        else out_dir / f"{name}_apbs_chains_{int(cfg.apbs_context_cutoff)}A.pdb"
    )
    standardized_pdb = out_dir / f"{name}_standardized_receptor.pdb"
    saved_pqr = out_dir / f"{name}_apbs.pqr"
    ply_file = out_dir / f"{name}_protein_{label}A.ply"
    result: dict[str, object] = {
        "name": name, "status": "failed", "reason": "",
        "receptor_file": str(receptor_file), "ligand_file": str(ligand_file),
        "pocket_pdb": str(pocket_pdb), "apbs_pdb": str(apbs_pdb), "ply_file": str(ply_file),
        "standardized_pdb": str(standardized_pdb), "pqr_file": str(saved_pqr),
        "pipeline_version": "surface-v2", "target_kind": cfg.target_kind,
        "hydrogen_policy": "heavy_only", "chemistry_version": "residue_atom_v2",
        "surface_vertex_cutoff": cfg.surface_vertex_cutoff,
        "pocket_residue_cutoff": cfg.pocket_residue_cutoff,
        "apbs_context_cutoff": cfg.apbs_context_cutoff,
        "probe_radius": cfg.probe_radius, "density": cfg.density, "hdensity": cfg.hdensity,
        "mesh_res": cfg.mesh_res, "fix_mesh_backend": cfg.fix_mesh_backend,
        "apbs_salt_conc": cfg.apbs_salt_conc, "pdb2pqr_ff": cfg.pdb2pqr_ff,
        "polymer_fragment_count": "", "polymer_backbone_break_count": "",
    }
    if ply_file.exists() and not cfg.overwrite:
        result["status"] = "exists"
        return result

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"surfna_v2_{name}_"))
    remove_tmp = True
    timeout_context = complex_wall_clock_limit(cfg.complex_timeout_seconds)
    timeout_context.__enter__()
    try:
        ligand_coords = legacy.read_ligand_coords(ligand_file)
        apbs_residues, apbs_chain_count = write_clean_receptor_chains(
            receptor_file, ligand_coords, apbs_pdb, cfg.target_kind, cfg.apbs_context_cutoff
        )
        fragment_stats = (
            split_observed_polymer_backbone_gaps(apbs_pdb, cfg.target_kind)
            if cfg.split_polymer_backbone_gaps else {}
        )
        result.update(fragment_stats)
        prepared_pqr = prepare_pqr_strict(apbs_pdb, tmp_dir, cfg, ligand_coords)
        temp_pqr, _apbs_input, preparation_stats = prepared_pqr
        shutil.copy2(temp_pqr, saved_pqr)
        standardized_atoms = write_standardized_pdb_from_pqr(temp_pqr, standardized_pdb)
        pocket_residues = legacy.select_pocket_pdb(
            standardized_pdb, ligand_coords, pocket_pdb, cfg.target_kind, cfg.pocket_residue_cutoff,
            filter_for_apbs=False,
        )
        vertices, faces, msms_normals, names, xyzrn_atoms, msms_mode = run_msms_v2(
            pocket_pdb, ligand_coords, tmp_dir, cfg
        )
        old_hbond, old_hphob, old_donor, old_acceptor, old_apolar = atom_propensities(names)

        full_vertices, full_faces, fix_backend, fix_note, pymesh_si, pymesh_si_bad = legacy.regularize_mesh(
            vertices, faces, tmp_dir, legacy_mesh_config(cfg)
        )
        full_boundary, full_boundary_edges = mesh_boundary(full_vertices, full_faces)
        if cfg.strict_closed_full_mesh and full_boundary_edges:
            raise ValueError(f"Regularized full molecular surface is open: {full_boundary_edges} boundary edges")
        if pymesh_si is not None and len(pymesh_si) == len(full_vertices):
            full_si = np.nan_to_num(pymesh_si, nan=0.0, posinf=0.0, neginf=0.0)
            si_bad = pymesh_si_bad
        else:
            full_si, si_bad = legacy.compute_shape_index(full_vertices, full_faces)
        full_normals = assign_vector_to_new_mesh(full_vertices, vertices, msms_normals)
        full_hbond = legacy.assign_to_new_mesh(full_vertices, vertices, old_hbond)
        full_hphob = legacy.assign_to_new_mesh(full_vertices, vertices, old_hphob)
        full_donor = legacy.assign_to_new_mesh(full_vertices, vertices, old_donor)
        full_acceptor = legacy.assign_to_new_mesh(full_vertices, vertices, old_acceptor)
        full_apolar = legacy.assign_to_new_mesh(full_vertices, vertices, old_apolar)

        final_vertices, final_faces, used = crop_mesh_with_indices(
            full_vertices, full_faces, ligand_coords, cfg.surface_vertex_cutoff
        )
        if len(final_vertices) < cfg.min_vertices or len(final_faces) < cfg.min_vertices:
            raise ValueError(f"Final V2 mesh too small: {len(final_vertices)} vertices, {len(final_faces)} faces")
        patch_boundary, patch_boundary_edges = mesh_boundary(final_vertices, final_faces)
        charge, electrostatic_stats = compute_apbs_v2(
            final_vertices, apbs_pdb, tmp_dir, cfg, prepared=prepared_pqr
        )
        feature_parts = {
            "vertex_xyz": final_vertices,
            "normal_xyz": full_normals[used],
            "hbond": full_hbond[used],
            "hphob": full_hphob[used],
            "charge": charge,
            "shape_index": full_si[used],
            "donor": full_donor[used],
            "acceptor": full_acceptor[used],
            "apolar": full_apolar[used],
            "boundary": patch_boundary,
        }
        feature_stack = np.column_stack(list(feature_parts.values()))
        nonfinite_by_feature = {
            name: int(np.count_nonzero(~np.isfinite(values)))
            for name, values in feature_parts.items()
        }
        nonfinite_by_feature = {name: count for name, count in nonfinite_by_feature.items() if count}
        nan_count = int(sum(nonfinite_by_feature.values())) + int(si_bad)
        if nan_count or not np.isfinite(feature_stack).all():
            raise ValueError(
                f"Non-finite Surface-v2 features: count={nan_count}; "
                f"by_feature={nonfinite_by_feature}; shape_index_bad={int(si_bad)}"
            )
        write_ply_v2(
            ply_file, final_vertices, final_faces, full_normals[used],
            full_hbond[used], full_hphob[used], charge, full_si[used],
            full_donor[used], full_acceptor[used], full_apolar[used], patch_boundary,
        )
        boundary_si_saturation = float(
            np.mean(np.abs(full_si[used][patch_boundary > 0.5]) >= 0.99)
        ) if np.any(patch_boundary > 0.5) else 0.0
        result.update({
            "status": "success", "reason": "",
            "pocket_residues": pocket_residues, "apbs_residues": apbs_residues,
            "apbs_chain_count": apbs_chain_count,
            **fragment_stats,
            "standardized_receptor_atoms": standardized_atoms,
            "xyzrn_heavy_atoms": xyzrn_atoms, "msms_mode": msms_mode,
            "msms_vertices": len(vertices), "msms_faces": len(faces),
            "full_vertices": len(full_vertices), "full_faces": len(full_faces),
            "full_boundary_edges": full_boundary_edges,
            "full_boundary_vertices": int(np.count_nonzero(full_boundary)),
            "fix_mesh_backend_used": fix_backend, "fix_mesh_note": fix_note,
            "final_vertices": len(final_vertices), "final_faces": len(final_faces),
            "patch_boundary_edges": patch_boundary_edges,
            "patch_boundary_vertices": int(np.count_nonzero(patch_boundary)),
            "patch_boundary_si_saturation": boundary_si_saturation,
            "charge_mean": float(np.mean(charge)), "charge_std": float(np.std(charge)),
            "nan_count": nan_count,
            **electrostatic_stats,
        })
    except Exception as exc:  # noqa: BLE001
        result["status"] = "failed"
        result["reason"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=8)
        if ply_file.exists():
            ply_file.unlink()
        for partial in (standardized_pdb, saved_pqr):
            if partial.exists():
                partial.unlink()
        if cfg.keep_failed_temp:
            kept = Path(cfg.out_dir) / "failed_tmp" / name
            kept.parent.mkdir(parents=True, exist_ok=True)
            if kept.exists():
                shutil.rmtree(kept)
            shutil.move(str(tmp_dir), str(kept))
            result["failed_tmp"] = str(kept)
            remove_tmp = False
    finally:
        timeout_context.__exit__(None, None, None)
        if remove_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return result


AUDIT_FIELDS = [
    "name", "status", "reason", "pipeline_version", "target_kind", "hydrogen_policy",
    "chemistry_version", "receptor_file", "ligand_file", "pocket_pdb", "apbs_pdb", "ply_file",
    "standardized_pdb", "pqr_file", "standardized_receptor_atoms",
    "pocket_residues", "apbs_residues", "apbs_chain_count", "xyzrn_heavy_atoms", "msms_mode", "msms_vertices",
    "msms_faces", "full_vertices", "full_faces", "full_boundary_edges", "full_boundary_vertices",
    "fix_mesh_backend", "fix_mesh_backend_used", "fix_mesh_note", "final_vertices", "final_faces",
    "patch_boundary_edges", "patch_boundary_vertices", "patch_boundary_si_saturation",
    "polymer_fragment_count", "polymer_backbone_break_count",
    "charge_mean", "charge_std", "charge_units", "charge_scale", "nan_count",
    "surface_vertex_cutoff", "pocket_residue_cutoff", "apbs_context_cutoff", "probe_radius",
    "density", "hdensity", "mesh_res", "apbs_salt_conc", "pdb2pqr_ff", "pdb2pqr_version",
    "pqr_input_heavy_atoms", "pqr_output_heavy_atoms", "pqr_added_heavy_atoms", "pqr_max_coordinate_delta",
    "pqr_added_heavy_atoms_within_exclusion_radius", "pqr_added_heavy_atom_min_ligand_distance",
    "pqr_repair_exclusion_radius",
    "pqr_rescue_mode", "pqr_raw_input_heavy_atoms", "pqr_remote_trimmed_heavy_atoms",
    "pqr_remote_trimmed_min_ligand_distance", "pqr_remote_trimmed_atom_names",
    "pqr_observed_5prime_phosphate_count", "pqr_fragment_noninteger_charge_allowed",
    "pqr_terminal_monophosphate_count",
    "pqr_total_charge", "apbs_pqr_chain_columns_stripped", "pdb2pqr_warning_count", "pdb2pqr_warnings", "apbs_version",
    "apbs_outside_vertex_count", "apbs_max_grid_excursion", "failed_tmp", "traceback",
]


def write_audit(audit_path: Path, rows: list[dict[str, object]]) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in AUDIT_FIELDS})


def main() -> None:
    args = parse_args()
    requested = args.complexes
    if args.complexes_file:
        requested = [line.strip() for line in Path(args.complexes_file).read_text().splitlines() if line.strip()]
    jobs = legacy.discover_complexes(Path(args.data_dir), requested, args.limit)
    cfg = SurfaceV2Config(
        data_dir=args.data_dir, out_dir=args.out_dir, target_kind=args.target_kind,
        surface_vertex_cutoff=args.surface_vertex_cutoff,
        pocket_residue_cutoff=args.pocket_residue_cutoff,
        apbs_context_cutoff=args.apbs_context_cutoff,
        min_vertices=args.min_vertices, overwrite=args.overwrite,
        mesh_res=args.mesh_res, fix_mesh_backend=args.fix_mesh_backend,
        pymesh_python=str(Path(args.pymesh_python).resolve()), density=args.density, hdensity=args.hdensity,
        probe_radius=args.probe_radius, apbs_salt_conc=args.apbs_salt_conc,
        pdb2pqr_ff=args.pdb2pqr_ff, pdb2pqr_bin=str(Path(args.pdb2pqr_bin).resolve()),
        apbs_bin=str(Path(args.apbs_bin).resolve()), msms_bin=str(Path(args.msms_bin).resolve()),
        charge_scale=args.charge_scale,
        pqr_coordinate_tolerance=args.pqr_coordinate_tolerance,
        pqr_repair_exclusion_radius=args.pqr_repair_exclusion_radius,
        allow_remote_terminal_phosphate_trim=args.allow_remote_terminal_phosphate_trim,
        remote_terminal_phosphate_trim_radius=args.remote_terminal_phosphate_trim_radius,
        complex_timeout_seconds=args.complex_timeout_seconds,
        split_polymer_backbone_gaps=args.split_polymer_backbone_gaps,
        strict_closed_full_mesh=args.strict_closed_full_mesh,
        keep_failed_temp=args.keep_failed_temp,
    )
    if not jobs:
        raise SystemExit("No complexes discovered")
    if args.num_workers > 1:
        with Pool(processes=args.num_workers) as pool:
            rows = list(pool.starmap(process_complex, [(job, cfg) for job in jobs]))
    else:
        rows = [process_complex(job, cfg) for job in jobs]
    audit_path = Path(args.audit_path) if args.audit_path else Path(cfg.out_dir) / "surface_v2_audit.csv"
    write_audit(audit_path, rows)
    successes = sum(row["status"] in {"success", "exists"} for row in rows)
    failures = sum(row["status"] == "failed" for row in rows)
    print(f"Surface-v2 processed={len(rows)} success/existing={successes} failed={failures}")
    if failures:
        print(f"Inspect {audit_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
