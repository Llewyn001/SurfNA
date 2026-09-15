#!/usr/bin/env python3
"""Generate aligned protein/NA pocket surfaces with hbond,hphob,charge,si features."""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import tempfile
import traceback
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

import numpy as np
from Bio.PDB import NeighborSearch, PDBIO, PDBParser, Select, Selection
from rdkit import Chem
from sklearn.neighbors import KDTree


def default_pymesh_python() -> str:
    configured = os.environ.get("SURFNA_PYMESH_PYTHON", "").strip()
    if configured:
        return configured
    candidate = Path.home() / ".conda" / "envs" / "surfna_pymesh_py36" / "bin" / "python"
    return str(candidate) if candidate.exists() else ""


PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "HIP",
    "HIE", "HID", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR",
    "TRP", "TYR", "VAL", "TPO", "PTR", "SEP", "CYX", "CYM",
}
NUCLEIC_RESIDUES = {
    "A", "C", "G", "U", "T", "DA", "DC", "DG", "DT", "DU",
    "ADE", "CYT", "GUA", "THY", "URA",
}
PURINE_RESIDUES = {"A", "G", "DA", "DG", "ADE", "GUA"}
PYRIMIDINE_RESIDUES = {"C", "U", "T", "DC", "DT", "DU", "CYT", "THY", "URA"}
RADII = {
    "H": 1.20,
    "C": 1.74,
    "N": 1.54,
    "O": 1.40,
    "P": 1.80,
    "S": 1.80,
}
KD_SCALE = {
    "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5,
    "MET": 1.9, "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8,
    "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "HID": -3.2,
    "HIE": -3.2, "HIP": -3.2, "GLU": -3.5, "GLN": -3.5, "ASP": -3.5,
    "ASN": -3.5, "LYS": -3.9, "ARG": -4.5,
}
NUCLEIC_ACCEPTORS = {
    "OP1", "OP2", "OP3", "O1P", "O2P", "O3P", "O5'", "O4'", "O3'",
    "O2'", "O2", "O4", "O6", "N1", "N3", "N7",
}
NUCLEIC_DONORS = {"N2", "N4", "N6"}


@dataclass(frozen=True)
class SurfaceConfig:
    data_dir: str
    out_dir: str
    target_kind: str
    surface_vertex_cutoff: float
    pocket_residue_cutoff: float
    min_vertices: int
    overwrite: bool
    no_fix_mesh: bool
    mesh_res: float
    msms_one_cavity: bool
    fix_mesh_backend: str
    pymesh_python: str
    density: float
    hdensity: float
    probe_radius: float
    output_cutoff_label: float
    apbs_context_cutoff: float
    apbs_salt_conc: float
    pdb2pqr_ff: str
    ply_schema: str
    tools_dir: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--target_kind", choices=["protein", "na", "auto"], default="auto")
    parser.add_argument("--surface_vertex_cutoff", type=float, default=8.0)
    parser.add_argument("--pocket_residue_cutoff", type=float, default=13.0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--complexes", nargs="*", default=None)
    parser.add_argument("--complexes_file", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min_vertices", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_fix_mesh", action="store_true")
    parser.add_argument(
        "--mesh_res",
        type=float,
        default=1.0,
        help="Target mesh resolution used by the PyMesh-like regularizer. SurfDock/MaSIF used 1.0.",
    )
    parser.add_argument(
        "--no_msms_one_cavity",
        dest="msms_one_cavity",
        action="store_false",
        default=True,
        help="Disable MSMS -one_cavity and fall back to -all_components.",
    )
    parser.add_argument(
        "--fix_mesh_backend",
        choices=["auto", "pymesh", "builtin", "none"],
        default="auto",
        help="Mesh regularization backend. auto uses PyMesh if available and falls back to builtin.",
    )
    parser.add_argument(
        "--pymesh_python",
        default=default_pymesh_python(),
        help="Python executable for the isolated PyMesh environment.",
    )
    parser.add_argument("--density", type=float, default=4.0)
    parser.add_argument("--hdensity", type=float, default=4.0)
    parser.add_argument("--probe_radius", type=float, default=1.4)
    parser.add_argument(
        "--output_cutoff_label",
        type=float,
        default=0.0,
        help="Label used in output PDB/PLY filenames. 0 reuses --surface_vertex_cutoff.",
    )
    parser.add_argument(
        "--apbs_context_cutoff",
        type=float,
        default=0.0,
        help="Ligand-distance residue cutoff for APBS charge context. 0 reuses the surface pocket; negative uses the full receptor.",
    )
    parser.add_argument("--apbs_salt_conc", type=float, default=0.15)
    parser.add_argument(
        "--pdb2pqr_ff",
        default="amber",
        help="PDB2PQR force field, e.g. amber or parse. SurfDock v0.0.1 used parse.",
    )
    parser.add_argument(
        "--ply_schema",
        choices=["surfna", "surfdock"],
        default="surfna",
        help="PLY property order. Dataset reads by name, but surfdock matches the paper code layout.",
    )
    parser.add_argument("--tools_dir", default=str(repo_root / "tools" / "transfer"))
    return parser.parse_args()


def residue_allowed(resname: str, target_kind: str) -> bool:
    resname = normalize_resname(resname)
    if target_kind == "protein":
        return resname in PROTEIN_RESIDUES
    if target_kind == "na":
        return resname in NUCLEIC_RESIDUES
    return resname in PROTEIN_RESIDUES or resname in NUCLEIC_RESIDUES


def normalize_resname(resname: str) -> str:
    return resname.strip().upper()


def infer_element(atom) -> str:
    element = (getattr(atom, "element", "") or "").strip().upper()
    if element:
        return element
    name = atom.get_name().strip().upper()
    letters = "".join(ch for ch in name if ch.isalpha())
    if letters.startswith("CL"):
        return "CL"
    if letters.startswith("BR"):
        return "BR"
    return letters[0] if letters else ""


def normalize_atom_name(atom_name: str) -> str:
    return atom_name.strip().upper().replace("*", "'")


def residue_complete_for_apbs(residue, target_kind: str) -> bool:
    """Drop badly incomplete residues before PDB2PQR/APBS charge assignment."""
    resname = normalize_resname(residue.get_resname())
    if not residue_allowed(resname, target_kind):
        return False
    atom_names = {
        normalize_atom_name(atom.get_name())
        for atom in residue
        if infer_element(atom) != "H"
    }
    if resname in NUCLEIC_RESIDUES:
        if len(atom_names) < 8:
            return False
        if not {"C1'", "C3'", "C4'", "O4'"}.issubset(atom_names):
            return False
        if resname in PURINE_RESIDUES:
            return "N9" in atom_names
        if resname in PYRIMIDINE_RESIDUES:
            return "N1" in atom_names
        return True
    if resname in PROTEIN_RESIDUES:
        return {"N", "CA", "C", "O"}.issubset(atom_names)
    return True


def read_ligand_coords(ligand_file: Path) -> np.ndarray:
    suffix = ligand_file.suffix.lower()
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(ligand_file), sanitize=False, removeHs=False)
        mol = supplier[0] if supplier and len(supplier) else None
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(ligand_file), sanitize=False, cleanupSubstructures=False)
    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(ligand_file), sanitize=False, removeHs=False)
    else:
        raise ValueError(f"Unsupported ligand format: {ligand_file}")
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"RDKit could not read ligand: {ligand_file}")
    conformer = mol.GetConformer()
    return np.array(
        [[conformer.GetAtomPosition(i).x, conformer.GetAtomPosition(i).y, conformer.GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())],
        dtype=float,
    )


def discover_complexes(data_dir: Path, requested: Iterable[str] | None, limit: int) -> list[tuple[str, Path, Path]]:
    names = list(requested) if requested else sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    jobs: list[tuple[str, Path, Path]] = []
    for name in names:
        complex_dir = data_dir / name
        if not complex_dir.is_dir():
            continue
        receptor_candidates = [
            complex_dir / f"{name}_protein.pdb",
            complex_dir / f"{name}_protein_processed.pdb",
            complex_dir / f"{name}_pocket.pdb",
        ]
        ligand_candidates = [
            complex_dir / f"{name}_ligand.sdf",
            complex_dir / f"{name}_ligand.mol2",
        ]
        receptor = next((p for p in receptor_candidates if p.exists()), None)
        ligand = next((p for p in ligand_candidates if p.exists()), None)
        if receptor is None:
            receptor = next(iter(sorted(complex_dir.glob("*_protein.pdb"))), None)
        if ligand is None:
            ligand = next(iter(sorted(complex_dir.glob("*_ligand.sdf"))), None)
        if ligand is None:
            ligand = next(iter(sorted(complex_dir.glob("*_ligand.mol2"))), None)
        if receptor is not None and ligand is not None:
            jobs.append((name, receptor, ligand))
        if limit and len(jobs) >= limit:
            break
    return jobs


def select_pocket_pdb(receptor_file: Path, ligand_coords: np.ndarray, pocket_pdb: Path, target_kind: str,
                      cutoff: float, filter_for_apbs: bool = False) -> int:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", str(receptor_file))
    atoms = [atom for atom in Selection.unfold_entities(structure, "A")
             if atom.get_parent().get_id()[0] == " "
             and residue_allowed(atom.get_parent().get_resname(), target_kind)]
    if not atoms:
        raise ValueError("No allowed receptor atoms found for pocket selection")
    ns = NeighborSearch(atoms)
    close_residues = []
    for coord in ligand_coords:
        close_residues.extend(ns.search(coord, cutoff, level="R"))
    close_residues = set(Selection.uniqueify(close_residues))
    if filter_for_apbs:
        close_residues = {
            residue for residue in close_residues
            if residue_complete_for_apbs(residue, target_kind)
        }
    if not close_residues:
        raise ValueError("No receptor residues within pocket cutoff")

    class PocketSelect(Select):
        def accept_residue(self, residue):  # noqa: D401
            if residue not in close_residues:
                return 0
            if residue.get_id()[0] != " ":
                return 0
            if filter_for_apbs and not residue_complete_for_apbs(residue, target_kind):
                return 0
            return int(residue_allowed(residue.get_resname(), target_kind))

    pocket_pdb.parent.mkdir(parents=True, exist_ok=True)
    pdbio = PDBIO()
    pdbio.set_structure(structure)
    pdbio.save(str(pocket_pdb), PocketSelect())
    return len(close_residues)


def count_allowed_residues(pdb_file: Path, target_kind: str) -> int:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", str(pdb_file))
    residues = {
        residue
        for residue in Selection.unfold_entities(structure, "R")
        if residue.get_id()[0] == " " and residue_allowed(residue.get_resname(), target_kind)
    }
    return len(residues)


def surface_atoms_for_xyzrn(pdb_file: Path, target_kind: str) -> list:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pocket", str(pdb_file))
    atoms = []
    for atom in structure.get_atoms():
        residue = atom.get_parent()
        if residue.get_id()[0] != " ":
            continue
        resname = normalize_resname(residue.get_resname())
        if not residue_allowed(resname, target_kind):
            continue
        if infer_element(atom) not in RADII:
            continue
        atoms.append(atom)
    return atoms


def write_xyzrn(pdb_file: Path, xyzrn_file: Path, target_kind: str) -> int:
    atoms = surface_atoms_for_xyzrn(pdb_file, target_kind)
    count = 0
    with xyzrn_file.open("w") as handle:
        for atom in atoms:
            residue = atom.get_parent()
            resname = normalize_resname(residue.get_resname())
            element = infer_element(atom)
            atom_name = atom.get_name().strip()
            color = {"O": "Red", "N": "Blue", "H": "Blue", "P": "Orange"}.get(element, "Green")
            chain = residue.get_parent().get_id().strip() or "X"
            insertion = residue.get_id()[2].strip() or "x"
            full_id = f"{chain}_{residue.get_id()[1]}_{insertion}_{resname}_{atom_name}_{color}"
            coord = atom.get_coord()
            handle.write(
                f"{coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f} {RADII[element]:.6f} 1 {full_id}\n"
            )
            count += 1
    if count == 0:
        raise ValueError("No atoms written to XYZRN")
    return count


def msms_cavity_atom_indices(pocket_pdb: Path, ligand_coords: np.ndarray, target_kind: str) -> list[int]:
    atoms = surface_atoms_for_xyzrn(pocket_pdb, target_kind)
    if not atoms:
        raise ValueError("No atoms available for MSMS -one_cavity")
    coords = np.asarray([atom.get_coord() for atom in atoms], dtype=float)
    ligand_center = np.mean(ligand_coords, axis=0)
    center_idx = int(np.argmin(np.linalg.norm(coords - ligand_center, axis=1)))
    dists, _ = KDTree(ligand_coords).query(coords)
    nearest_idx = int(np.argmin(dists[:, 0]))
    ordered = []
    for idx in (center_idx, nearest_idx):
        if idx not in ordered:
            ordered.append(idx)
    return ordered


def read_msms(file_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    vert_lines = (file_root.with_suffix(".vert")).read_text().rstrip().splitlines()
    face_lines = (file_root.with_suffix(".face")).read_text().rstrip().splitlines()
    vertex_count = int(vert_lines[2].split()[0])
    face_count = int(face_lines[2].split()[0])
    vertices = np.zeros((vertex_count, 3), dtype=float)
    normals = np.zeros((vertex_count, 3), dtype=float)
    names: list[str] = [""] * vertex_count
    for i, line in enumerate(vert_lines[3:]):
        fields = line.split()
        vertices[i] = [float(fields[0]), float(fields[1]), float(fields[2])]
        normals[i] = [float(fields[3]), float(fields[4]), float(fields[5])]
        names[i] = fields[9]
    faces = np.zeros((face_count, 3), dtype=np.int64)
    for i, line in enumerate(face_lines[3:]):
        fields = line.split()
        faces[i] = [int(fields[0]) - 1, int(fields[1]) - 1, int(fields[2]) - 1]
    return vertices, faces, normals, names


def run_msms(
    pocket_pdb: Path,
    tmp_dir: Path,
    cfg: SurfaceConfig,
    ligand_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], int, str]:
    tools = Path(cfg.tools_dir)
    msms_bin = tools / "APBS-3.4.1.Linux" / "bin" / "msms"
    if not msms_bin.exists():
        raise FileNotFoundError(f"MSMS binary not found: {msms_bin}")
    xyzrn_file = tmp_dir / "surface.xyzrn"
    xyzrn_count = write_xyzrn(pocket_pdb, xyzrn_file, cfg.target_kind)
    attempts: list[tuple[str, list[str]]] = []
    if cfg.msms_one_cavity:
        for i, atom_idx in enumerate(msms_cavity_atom_indices(pocket_pdb, ligand_coords, cfg.target_kind)):
            label = "one_cavity_center" if i == 0 else "one_cavity_ligand_nearest"
            attempts.append((label, ["-one_cavity", "1", str(atom_idx)]))
    attempts.append(("all_components", ["-all_components"]))

    errors = []
    for attempt_idx, (mode, mode_args) in enumerate(attempts):
        file_root = tmp_dir / f"msms_surface_{attempt_idx}"
        cmd = [
            str(msms_bin),
            "-density", str(cfg.density),
            "-hdensity", str(cfg.hdensity),
            "-probe", str(cfg.probe_radius),
            *mode_args,
            "-if", str(xyzrn_file),
            "-of", str(file_root),
            "-af", str(file_root),
        ]
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if completed.returncode != 0:
            errors.append(f"{mode}: {completed.stderr[-500:]}")
            continue
        try:
            vertices, faces, normals, names = read_msms(file_root)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{mode}: {type(exc).__name__}: {exc}")
            continue
        if len(vertices) == 0 or len(faces) == 0:
            errors.append(f"{mode}: empty mesh")
            continue
        return vertices, faces, normals, names, xyzrn_count, mode
    raise RuntimeError(f"MSMS failed for all modes: {' | '.join(errors)[-1500:]}")


def atom_from_msms_name(name: str) -> tuple[str, str]:
    fields = name.split("_")
    if len(fields) < 5:
        return "", ""
    return normalize_resname(fields[3]), fields[4].strip().upper()


def compute_hbond(names: list[str]) -> np.ndarray:
    values = np.zeros(len(names), dtype=float)
    for i, name in enumerate(names):
        resname, atom_name = atom_from_msms_name(name)
        element = atom_name[0] if atom_name else ""
        if resname in NUCLEIC_RESIDUES:
            if atom_name in NUCLEIC_DONORS:
                values[i] = 1.0
            elif atom_name in NUCLEIC_ACCEPTORS or element == "O":
                values[i] = -1.0
            elif element == "N":
                values[i] = -0.5
        else:
            if element == "H":
                values[i] = 1.0
            elif atom_name.startswith(("OD", "OE")) or atom_name in {"O", "OXT"}:
                values[i] = -1.0
            elif atom_name in {"OG", "OG1", "OH"}:
                values[i] = -0.5
            elif resname in {"HIS", "HID", "HIE"} and atom_name in {"ND1", "NE2"}:
                values[i] = -0.5
    return values


def compute_hphob(names: list[str]) -> np.ndarray:
    values = np.zeros(len(names), dtype=float)
    for i, name in enumerate(names):
        resname, atom_name = atom_from_msms_name(name)
        if resname in KD_SCALE:
            values[i] = KD_SCALE[resname]
        elif resname in NUCLEIC_RESIDUES:
            if atom_name.startswith(("P", "OP", "O")):
                values[i] = -2.0
            elif atom_name.startswith("C"):
                values[i] = 0.5
            elif atom_name.startswith("N"):
                values[i] = -0.5
            else:
                values[i] = 0.0
    return values


def submesh_near_ligand(vertices: np.ndarray, faces: np.ndarray, ligand_coords: np.ndarray,
                        cutoff: float) -> tuple[np.ndarray, np.ndarray, int]:
    dists, _ = KDTree(ligand_coords).query(vertices)
    keep_vertices = set(np.where(dists[:, 0] <= cutoff)[0].tolist())
    keep_faces = [face for face in faces if all(int(v) in keep_vertices for v in face)]
    if not keep_faces:
        raise ValueError("No MSMS faces remained after ligand-distance cropping")
    used = sorted({int(v) for face in keep_faces for v in face})
    remap = {old: new for new, old in enumerate(used)}
    new_vertices = vertices[used]
    new_faces = np.array([[remap[int(v)] for v in face] for face in keep_faces], dtype=np.int64)
    return new_vertices, new_faces, len(keep_vertices)


def compute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices)
    tris = vertices[faces]
    face_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(face_normals, axis=1)
    lengths[lengths < 1e-8] = 1.0
    face_normals = face_normals / lengths[:, None]
    for face, normal in zip(faces, face_normals):
        normals[face] += normal
    n = np.linalg.norm(normals, axis=1)
    n[n < 1e-8] = 1.0
    normals = normals / n[:, None]
    return np.nan_to_num(normals)


def assign_to_new_mesh(new_vertices: np.ndarray, old_vertices: np.ndarray, old_values: np.ndarray) -> np.ndarray:
    k = min(4, len(old_vertices))
    dists, indices = KDTree(old_vertices).query(new_vertices, k=k)
    if k == 1:
        return old_values[indices[:, 0]]
    dists = np.square(dists)
    values = np.zeros(len(new_vertices), dtype=float)
    for i in range(len(new_vertices)):
        if dists[i, 0] == 0.0:
            values[i] = old_values[indices[i, 0]]
            continue
        weights = 1.0 / np.maximum(dists[i], 1e-12)
        weights = weights / weights.sum()
        values[i] = float(np.sum(old_values[indices[i]] * weights))
    return values


def compact_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Mesh is empty")
    unique_mask = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 0] != faces[:, 2])
        & (faces[:, 1] != faces[:, 2])
    )
    faces = faces[unique_mask]
    if len(faces) == 0:
        raise ValueError("Mesh has no non-degenerate faces")
    tris = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    faces = faces[areas > 1e-10]
    if len(faces) == 0:
        raise ValueError("Mesh has no finite-area faces")
    used = np.unique(faces.reshape(-1))
    remap = {int(old): new for new, old in enumerate(used.tolist())}
    new_vertices = vertices[used]
    new_faces = np.asarray([[remap[int(v)] for v in face] for face in faces], dtype=np.int64)
    return new_vertices, new_faces


def largest_face_component(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = compact_mesh(vertices, faces)
    vertex_to_faces: list[list[int]] = [[] for _ in range(len(vertices))]
    for face_idx, face in enumerate(faces):
        for vertex_idx in face:
            vertex_to_faces[int(vertex_idx)].append(face_idx)
    visited = np.zeros(len(faces), dtype=bool)
    components: list[list[int]] = []
    for start in range(len(faces)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            face_idx = stack.pop()
            component.append(face_idx)
            for vertex_idx in faces[face_idx]:
                for neighbor_face in vertex_to_faces[int(vertex_idx)]:
                    if not visited[neighbor_face]:
                        visited[neighbor_face] = True
                        stack.append(neighbor_face)
        components.append(component)
    largest = max(components, key=len)
    return compact_mesh(vertices, faces[np.asarray(largest, dtype=np.int64)])


def regularize_mesh_builtin(vertices: np.ndarray, faces: np.ndarray, mesh_res: float) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = largest_face_component(vertices, faces)
    if mesh_res <= 0:
        return vertices, faces
    origin = np.min(vertices, axis=0)
    keys = np.floor((vertices - origin) / mesh_res + 0.5).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    collapsed = np.zeros((int(inverse.max()) + 1, 3), dtype=float)
    counts = np.bincount(inverse)
    np.add.at(collapsed, inverse, vertices)
    collapsed = collapsed / np.maximum(counts[:, None], 1)
    collapsed_faces = inverse[faces]
    return largest_face_component(collapsed, collapsed_faces)


def run_pymesh_fix_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    tmp_dir: Path,
    cfg: SurfaceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if not cfg.pymesh_python:
        raise FileNotFoundError("No PyMesh Python executable configured")
    pymesh_python = Path(cfg.pymesh_python)
    if not pymesh_python.exists():
        raise FileNotFoundError(f"PyMesh Python executable not found: {pymesh_python}")
    helper = Path(__file__).resolve().with_name("pymesh_fix_mesh.py")
    if not helper.exists():
        raise FileNotFoundError(f"PyMesh helper not found: {helper}")
    input_npz = tmp_dir / "fix_mesh_input.npz"
    output_npz = tmp_dir / "fix_mesh_output.npz"
    np.savez(input_npz, vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=np.int64))
    cmd = [
        str(pymesh_python),
        str(helper),
        "--input", str(input_npz),
        "--output", str(output_npz),
        "--mesh_res", str(cfg.mesh_res),
    ]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"PyMesh fix_mesh failed: {completed.stderr[-1500:]}")
    if not output_npz.exists():
        raise RuntimeError("PyMesh fix_mesh did not write an output file")
    with np.load(output_npz) as data:
        new_vertices = np.asarray(data["vertices"], dtype=float)
        new_faces = np.asarray(data["faces"], dtype=np.int64)
        si = np.asarray(data["si"], dtype=float)
        si_bad = int(np.asarray(data["si_bad"]).reshape(-1)[0])
    if len(new_vertices) == 0 or len(new_faces) == 0:
        raise ValueError("PyMesh fix_mesh produced an empty mesh")
    return new_vertices, new_faces, si, si_bad


def regularize_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    tmp_dir: Path,
    cfg: SurfaceConfig,
) -> tuple[np.ndarray, np.ndarray, str, str, np.ndarray | None, int]:
    if cfg.no_fix_mesh or cfg.fix_mesh_backend == "none":
        new_vertices, new_faces = largest_face_component(vertices, faces)
        return new_vertices, new_faces, "none", "", None, 0

    if cfg.fix_mesh_backend in {"auto", "pymesh"}:
        try:
            new_vertices, new_faces, si, si_bad = run_pymesh_fix_mesh(vertices, faces, tmp_dir, cfg)
            return new_vertices, new_faces, "pymesh", "", si, si_bad
        except Exception as exc:  # noqa: BLE001
            if cfg.fix_mesh_backend == "pymesh":
                raise
            fallback_note = f"PyMesh fallback: {type(exc).__name__}: {exc}"
            new_vertices, new_faces = regularize_mesh_builtin(vertices, faces, cfg.mesh_res)
            return new_vertices, new_faces, "builtin", fallback_note[:1000], None, 0

    new_vertices, new_faces = regularize_mesh_builtin(vertices, faces, cfg.mesh_res)
    return new_vertices, new_faces, "builtin", "", None, 0


def compute_shape_index(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, int]:
    """Discrete shape-index approximation without PyMesh.

    PyMesh is not ABI-compatible with the available GLIBC on this cluster. This
    keeps the SurfNA surface pipeline independent of PyMesh while preserving the
    same bounded shape-index feature expected by the model.
    """
    n_vertices = len(vertices)
    neighbors = [set() for _ in range(n_vertices)]
    angle_sum = np.zeros(n_vertices, dtype=float)
    area = np.zeros(n_vertices, dtype=float)
    for face in faces:
        tri = vertices[face]
        face_area = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
        if face_area <= 1e-12:
            continue
        for local_idx, vi in enumerate(face):
            v = tri[local_idx]
            a = tri[(local_idx + 1) % 3] - v
            b = tri[(local_idx + 2) % 3] - v
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom > 1e-12:
                cosine = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
                angle_sum[int(vi)] += math.acos(cosine)
            area[int(vi)] += face_area / 3.0
        a, b, c = [int(x) for x in face]
        neighbors[a].update([b, c])
        neighbors[b].update([a, c])
        neighbors[c].update([a, b])

    h = np.zeros(n_vertices, dtype=float)
    for i, neigh in enumerate(neighbors):
        if not neigh:
            continue
        lap = np.mean(vertices[list(neigh)] - vertices[i], axis=0)
        h[i] = 0.5 * np.linalg.norm(lap)
    safe_area = np.maximum(area, 1e-8)
    k = (2.0 * np.pi - angle_sum) / safe_area
    elem = np.square(h) - k
    elem[elem < 0] = 1e-8
    k1 = h + np.sqrt(elem)
    k2 = h - np.sqrt(elem)
    denom = k1 - k2
    denom[np.abs(denom) < 1e-8] = 1e-8
    raw_si = np.arctan((k1 + k2) / denom) * (2.0 / np.pi)
    bad = int(np.count_nonzero(~np.isfinite(raw_si)))
    return np.nan_to_num(raw_si, nan=0.0, posinf=0.0, neginf=0.0), bad


def read_dx(dx_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = None
    origin = None
    deltas = []
    values = []
    reading_values = False
    with dx_file.open() as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            if fields[:5] == ["object", "1", "class", "gridpositions", "counts"]:
                counts = np.array([int(fields[5]), int(fields[6]), int(fields[7])], dtype=int)
            elif fields[0] == "origin":
                origin = np.array([float(fields[1]), float(fields[2]), float(fields[3])], dtype=float)
            elif fields[0] == "delta":
                deltas.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif "data" in fields and "follows" in fields:
                reading_values = True
            elif reading_values:
                try:
                    values.extend(float(field) for field in fields)
                except ValueError:
                    reading_values = False
    if counts is None or origin is None or len(deltas) != 3:
        raise RuntimeError(f"Could not parse DX grid metadata from {dx_file}")
    values_array = np.asarray(values, dtype=float)
    expected = int(np.prod(counts))
    if values_array.size < expected:
        raise RuntimeError(f"DX value count too small: {values_array.size} < {expected}")
    grid = values_array[:expected].reshape(tuple(counts), order="C")
    return origin, np.asarray(deltas, dtype=float), grid


def interpolate_dx(dx_file: Path, vertices: np.ndarray) -> np.ndarray:
    origin, deltas, grid = read_dx(dx_file)
    transform = np.linalg.inv(deltas.T)
    fractional = (vertices - origin) @ transform
    max_index = np.asarray(grid.shape, dtype=float) - 1.000001
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
    return c0 * (1 - zd) + c1 * zd


def add_apbs_ions(apbs_input: Path, salt_conc: float) -> None:
    if salt_conc <= 0.0:
        return
    lines = apbs_input.read_text().splitlines()
    if any(line.strip().lower().startswith("ion charge") for line in lines):
        return
    ion_lines = [
        f"    ion charge 1 conc {salt_conc:.3f} radius 2.0",
        f"    ion charge -1 conc {salt_conc:.3f} radius 2.0",
    ]
    insert_at = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("sdie"):
            insert_at = i + 1
            break
    if insert_at is None:
        for i, line in enumerate(lines):
            if line.strip().lower().startswith("srfm"):
                insert_at = i
                break
    if insert_at is None:
        raise RuntimeError(f"Could not find APBS solvent block in {apbs_input}")
    lines[insert_at:insert_at] = ion_lines
    apbs_input.write_text("\n".join(lines) + "\n")


def compute_apbs(vertices: np.ndarray, apbs_pdb: Path, tmp_dir: Path, cfg: SurfaceConfig) -> np.ndarray:
    tools = Path(cfg.tools_dir)
    bundled_pdb2pqr_bin = tools / "pdb2pqr-linux-bin64-2.1.1" / "pdb2pqr"
    apbs_candidates = [
        Path(__file__).resolve().parents[1] / "tools" / "conda_apbs_1.5" / "bin" / "apbs",
        tools / "APBS-3.4.1.Linux" / "bin" / "apbs",
    ]
    apbs_bin = next((path for path in apbs_candidates if path.exists()), apbs_candidates[-1])
    if not apbs_bin.exists():
        raise FileNotFoundError(f"Required APBS tool missing: {apbs_bin}")

    base = tmp_dir / "apbs_surface"
    pdb2pqr_errors = []
    old_style_input = base.with_suffix(".in")
    if bundled_pdb2pqr_bin.exists():
        cmd = [
            str(bundled_pdb2pqr_bin),
            f"--ff={cfg.pdb2pqr_ff.lower()}", "--whitespace", "--noopt", "--apbs-input",
            str(apbs_pdb), str(base),
        ]
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if completed.returncode == 0 and old_style_input.exists():
            apbs_input = old_style_input
        else:
            pdb2pqr_errors.append((completed.stderr + completed.stdout)[-2000:])
            apbs_input = None
    else:
        pdb2pqr_errors.append(f"Bundled pdb2pqr not found: {bundled_pdb2pqr_bin}")
        apbs_input = None

    if apbs_input is None:
        conda_pdb2pqr = shutil.which("pdb2pqr") or "/public/home/luoyuxuan/.conda/envs/SurfDock/bin/pdb2pqr"
        conda_pdb2pqr = Path(conda_pdb2pqr)
        if not conda_pdb2pqr.exists():
            raise FileNotFoundError(f"Required pdb2pqr fallback missing: {conda_pdb2pqr}")
        conda_input = base.with_suffix(".in")
        conda_pqr = base.with_suffix(".pqr")
        cmd = [
            str(conda_pdb2pqr),
            f"--ff={cfg.pdb2pqr_ff.upper()}", "--whitespace", "--noopt",
            "--apbs-input", str(conda_input),
            str(apbs_pdb), str(conda_pqr),
        ]
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if completed.returncode != 0 or not conda_input.exists():
            pdb2pqr_errors.append((completed.stderr + completed.stdout)[-3000:])
            raise RuntimeError(f"pdb2pqr {cfg.pdb2pqr_ff} failed:\n" + "\n--- fallback ---\n".join(pdb2pqr_errors))
        apbs_input = conda_input

    add_apbs_ions(apbs_input, cfg.apbs_salt_conc)
    completed = subprocess.run([str(apbs_bin), apbs_input.name],
                               cwd=str(tmp_dir),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"APBS failed: {(completed.stderr + completed.stdout)[-2000:]}")
    dx_file = base.with_suffix(".dx")
    if not dx_file.exists():
        pqr_dx = base.with_suffix(".pqr.dx")
        if pqr_dx.exists():
            dx_file = pqr_dx
    if not dx_file.exists():
        dx_candidates = sorted(tmp_dir.glob("*.dx"))
        if dx_candidates:
            dx_file = dx_candidates[0]
    if not dx_file.exists():
        raise RuntimeError("APBS did not create DX output")

    charges = interpolate_dx(dx_file, vertices)
    if len(charges) != len(vertices):
        raise RuntimeError(f"APBS charge count mismatch: {len(charges)} vs {len(vertices)}")
    if not np.isfinite(charges).all():
        raise RuntimeError("APBS charges contain NaN or inf")
    if np.allclose(charges, 0.0, atol=1e-8):
        raise RuntimeError("APBS charges are all zero")
    return charges / 10.0


def write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray,
              hbond: np.ndarray, hphob: np.ndarray, charge: np.ndarray, si: np.ndarray,
              schema: str = "surfna") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if schema == "surfdock":
        props = ["x", "y", "z", "charge", "hbond", "hphob", "nx", "ny", "nz", "si"]
    else:
        props = ["x", "y", "z", "nx", "ny", "nz", "hbond", "hphob", "charge", "si"]
    with path.open("w") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        for prop in props:
            handle.write(f"property float {prop}\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for i, vertex in enumerate(vertices):
            values = {
                "x": vertex[0], "y": vertex[1], "z": vertex[2],
                "nx": normals[i, 0], "ny": normals[i, 1], "nz": normals[i, 2],
                "hbond": hbond[i], "hphob": hphob[i], "charge": charge[i], "si": si[i],
            }
            row = [values[prop] for prop in props]
            handle.write(" ".join(f"{float(v):.6f}" for v in row) + "\n")
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def process_complex(job: tuple[str, Path, Path], cfg: SurfaceConfig) -> dict[str, object]:
    name, receptor_file, ligand_file = job
    out_complex_dir = Path(cfg.out_dir) / name
    cutoff_label = cfg.output_cutoff_label if cfg.output_cutoff_label > 0 else cfg.surface_vertex_cutoff
    cutoff_label_int = int(cutoff_label)
    pocket_pdb = out_complex_dir / f"{name}_pocket_{cutoff_label_int}A.pdb"
    if cfg.apbs_context_cutoff > 0.0:
        apbs_pdb = out_complex_dir / f"{name}_apbs_{int(cfg.apbs_context_cutoff)}A.pdb"
    elif cfg.apbs_context_cutoff < 0.0:
        apbs_pdb = receptor_file
    else:
        apbs_pdb = pocket_pdb
    ply_file = out_complex_dir / f"{name}_protein_{cutoff_label_int}A.ply"
    result: dict[str, object] = {
        "name": name,
        "status": "failed",
        "reason": "",
        "receptor_file": str(receptor_file),
        "ligand_file": str(ligand_file),
        "pocket_pdb": str(pocket_pdb),
        "apbs_pdb": str(apbs_pdb),
        "ply_file": str(ply_file),
        "pocket_residues": 0,
        "apbs_residues": 0,
        "xyzrn_atoms": 0,
        "msms_mode": "",
        "msms_vertices": 0,
        "msms_faces": 0,
        "cropped_vertices": 0,
        "cropped_faces": 0,
        "regularized_vertices": 0,
        "regularized_faces": 0,
        "final_vertices": 0,
        "final_faces": 0,
        "charge_mean": "",
        "charge_std": "",
        "nan_count": 0,
        "no_fix_mesh": cfg.no_fix_mesh,
        "mesh_res": cfg.mesh_res,
        "msms_one_cavity": cfg.msms_one_cavity,
        "fix_mesh_backend": cfg.fix_mesh_backend,
        "fix_mesh_backend_used": "",
        "fix_mesh_note": "",
        "probe_radius": cfg.probe_radius,
        "density": cfg.density,
        "hdensity": cfg.hdensity,
        "surface_vertex_cutoff": cfg.surface_vertex_cutoff,
        "pocket_residue_cutoff": cfg.pocket_residue_cutoff,
        "output_cutoff_label": cutoff_label,
        "apbs_context_cutoff": cfg.apbs_context_cutoff,
        "apbs_salt_conc": cfg.apbs_salt_conc,
        "pdb2pqr_ff": cfg.pdb2pqr_ff,
        "ply_schema": cfg.ply_schema,
    }
    if ply_file.exists() and not cfg.overwrite:
        result["status"] = "exists"
        return result

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"surfna_{name}_"))
    try:
        ligand_coords = read_ligand_coords(ligand_file)
        pocket_residues = select_pocket_pdb(
            receptor_file, ligand_coords, pocket_pdb, cfg.target_kind, cfg.pocket_residue_cutoff
        )
        if cfg.apbs_context_cutoff < 0.0:
            apbs_residues = count_allowed_residues(receptor_file, cfg.target_kind)
        elif cfg.apbs_context_cutoff > 0.0 and not math.isclose(
            cfg.apbs_context_cutoff, cfg.pocket_residue_cutoff
        ):
            apbs_residues = select_pocket_pdb(
                receptor_file, ligand_coords, apbs_pdb, cfg.target_kind, cfg.apbs_context_cutoff,
                filter_for_apbs=True,
            )
        else:
            apbs_pdb = pocket_pdb
            apbs_residues = pocket_residues
            result["apbs_pdb"] = str(apbs_pdb)
        vertices, faces, _normals, names, xyzrn_atoms, msms_mode = run_msms(
            pocket_pdb, tmp_dir, cfg, ligand_coords
        )
        result.update({
            "pocket_residues": pocket_residues,
            "apbs_residues": apbs_residues,
            "xyzrn_atoms": xyzrn_atoms,
            "msms_mode": msms_mode,
            "msms_vertices": len(vertices),
            "msms_faces": len(faces),
        })
        old_hbond = compute_hbond(names)
        old_hphob = compute_hphob(names)
        cropped_vertices, cropped_faces, iface_vertices = submesh_near_ligand(
            vertices, faces, ligand_coords, cfg.surface_vertex_cutoff
        )
        result.update({"cropped_vertices": len(cropped_vertices), "cropped_faces": len(cropped_faces)})
        if len(cropped_vertices) < cfg.min_vertices or len(cropped_faces) < cfg.min_vertices:
            raise ValueError(f"Cropped mesh too small: {len(cropped_vertices)} vertices, {len(cropped_faces)} faces")

        final_vertices, final_faces, fix_backend_used, fix_note, pymesh_si, pymesh_si_bad = regularize_mesh(
            cropped_vertices, cropped_faces, tmp_dir, cfg
        )
        result.update({
            "regularized_vertices": len(final_vertices),
            "regularized_faces": len(final_faces),
            "fix_mesh_backend_used": fix_backend_used,
            "fix_mesh_note": fix_note,
        })
        if len(final_vertices) < cfg.min_vertices or len(final_faces) < cfg.min_vertices:
            raise ValueError(f"Final mesh too small: {len(final_vertices)} vertices, {len(final_faces)} faces")
        hbond = assign_to_new_mesh(final_vertices, vertices, old_hbond)
        hphob = assign_to_new_mesh(final_vertices, vertices, old_hphob)
        charge = compute_apbs(final_vertices, apbs_pdb, tmp_dir, cfg)
        normals = compute_normals(final_vertices, final_faces)
        if pymesh_si is not None and len(pymesh_si) == len(final_vertices):
            si = np.nan_to_num(pymesh_si, nan=0.0, posinf=0.0, neginf=0.0)
            si_bad = pymesh_si_bad
        else:
            si, si_bad = compute_shape_index(final_vertices, final_faces)
        feature_stack = np.column_stack([final_vertices, normals, hbond, hphob, charge, si])
        nan_count = int(np.count_nonzero(~np.isfinite(feature_stack))) + si_bad
        if not np.isfinite(feature_stack).all():
            raise ValueError("Final surface features contain NaN or inf")
        write_ply(ply_file, final_vertices, final_faces, normals, hbond, hphob, charge, si, cfg.ply_schema)
        result.update({
            "status": "success",
            "reason": "",
            "final_vertices": len(final_vertices),
            "final_faces": len(final_faces),
            "charge_mean": float(np.mean(charge)),
            "charge_std": float(np.std(charge)),
            "nan_count": nan_count,
        })
    except Exception as exc:  # noqa: BLE001
        result["status"] = "failed"
        result["reason"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=5)
        if ply_file.exists():
            ply_file.unlink()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return result


def write_audit(out_dir: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "name", "status", "reason", "receptor_file", "ligand_file", "pocket_pdb", "apbs_pdb",
        "ply_file", "pocket_residues", "apbs_residues", "xyzrn_atoms", "msms_mode", "msms_vertices",
        "msms_faces", "cropped_vertices", "cropped_faces", "regularized_vertices", "regularized_faces",
        "final_vertices", "final_faces", "charge_mean", "charge_std", "nan_count",
        "no_fix_mesh", "mesh_res", "msms_one_cavity", "fix_mesh_backend", "fix_mesh_backend_used",
        "fix_mesh_note", "probe_radius", "density", "hdensity", "surface_vertex_cutoff", "pocket_residue_cutoff",
        "output_cutoff_label", "apbs_context_cutoff", "apbs_salt_conc", "pdb2pqr_ff", "ply_schema",
        "traceback",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "surface_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    args = parse_args()
    requested = args.complexes
    if args.complexes_file:
        requested = [line.strip() for line in Path(args.complexes_file).read_text().splitlines() if line.strip()]
    jobs = discover_complexes(Path(args.data_dir), requested, args.limit)
    cfg = SurfaceConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        target_kind=args.target_kind,
        surface_vertex_cutoff=args.surface_vertex_cutoff,
        pocket_residue_cutoff=args.pocket_residue_cutoff,
        min_vertices=args.min_vertices,
        overwrite=args.overwrite,
        no_fix_mesh=args.no_fix_mesh,
        mesh_res=args.mesh_res,
        msms_one_cavity=args.msms_one_cavity,
        fix_mesh_backend=args.fix_mesh_backend,
        pymesh_python=args.pymesh_python,
        density=args.density,
        hdensity=args.hdensity,
        probe_radius=args.probe_radius,
        output_cutoff_label=args.output_cutoff_label,
        apbs_context_cutoff=args.apbs_context_cutoff,
        apbs_salt_conc=args.apbs_salt_conc,
        pdb2pqr_ff=args.pdb2pqr_ff,
        ply_schema=args.ply_schema,
        tools_dir=args.tools_dir,
    )
    if args.num_workers > 1:
        with Pool(processes=args.num_workers) as pool:
            rows = list(pool.starmap(process_complex, [(job, cfg) for job in jobs]))
    else:
        rows = [process_complex(job, cfg) for job in jobs]
    write_audit(Path(args.out_dir), rows)
    successes = sum(1 for row in rows if row["status"] in {"success", "exists"})
    failures = sum(1 for row in rows if row["status"] == "failed")
    print(f"Processed {len(rows)} complexes: success/existing={successes}, failed={failures}")
    if rows and failures / len(rows) > 0.05:
        print("WARNING: surface failure rate exceeds 5%; inspect surface_audit.csv before training.")


if __name__ == "__main__":
    main()
