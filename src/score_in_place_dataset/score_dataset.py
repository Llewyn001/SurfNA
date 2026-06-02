"""Build SurfNA graphs from explicit receptor, ligand, and surface files.

This module replaces the historical SurfDock ``score_in_place_dataset`` import
with a small inference-only implementation. It is intentionally narrower than
the training dataset: each row is an already prepared pocket PDB, a ligand or
ligand library, and one SurfNA surface PLY.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Iterable

import MDAnalysis as mda
import numpy as np
import torch
from loguru import logger
from plyfile import PlyData
from rdkit import Chem
from rdkit.Chem import AddHs
from torch_geometric.data import Data, Dataset, HeteroData
from torch_geometric.transforms import Cartesian, FaceToEdge

from datasets.pdbbind import cap_surface_vertices, sanitize_surface_features
from datasets.process_mols import (
    extract_receptor_structure,
    generate_conformer,
    get_lig_graph_with_matching,
    get_rec_graph,
    parse_pdb_from_path,
    read_molecule,
)


def _safe_stem(path: str | Path) -> str:
    stem = Path(path).stem
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)


def _split_sdf_library(path: Path, out_dir: Path) -> list[Chem.Mol]:
    suppl = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
    mols = [mol for mol in suppl if mol is not None]
    if not mols:
        raise ValueError(f"RDKit could not read any molecule from {path}")
    return mols


def _read_ligands(ligands_path: str | Path) -> list[tuple[str, Chem.Mol]]:
    path = Path(ligands_path)
    if path.is_dir():
        files: Iterable[Path] = sorted(
            p for p in path.iterdir() if p.suffix.lower() in {".sdf", ".mol2", ".pdb"}
        )
        ligands = []
        for file_path in files:
            mol = read_molecule(str(file_path), sanitize=True, remove_hs=False)
            if mol is not None:
                ligands.append((_safe_stem(file_path), mol))
        return ligands
    if path.suffix.lower() == ".sdf":
        mols = _split_sdf_library(path, path.parent)
        return [(f"{_safe_stem(path)}_{idx:05d}", mol) for idx, mol in enumerate(mols)]
    mol = read_molecule(str(path), sanitize=True, remove_hs=False)
    if mol is None:
        raise ValueError(f"RDKit could not read ligand {path}")
    return [(_safe_stem(path), mol)]


def _ensure_conformer(mol: Chem.Mol, keep_input_pose: bool) -> Chem.Mol:
    mol = copy.deepcopy(mol)
    if keep_input_pose and mol.GetNumConformers() > 0:
        return mol
    mol.RemoveAllConformers()
    mol = AddHs(mol, addCoords=True)
    generate_conformer(mol)
    return mol


def _load_surface_graph(surface_path: str | Path, center: torch.Tensor, ligand_pos: torch.Tensor, max_vertices: int) -> Data:
    with open(surface_path, "rb") as handle:
        ply = PlyData.read(handle)
    vertex_properties = {prop.name for prop in ply["vertex"].properties}
    coord_names = ["x", "y", "z"]
    feature_names = ["hbond", "hphob", "charge", "si"]
    missing = [name for name in coord_names + feature_names if name not in vertex_properties]
    if missing:
        raise ValueError(f"surface PLY {surface_path} is missing required properties: {missing}")

    pos = torch.stack([torch.tensor(ply["vertex"][axis]).float() for axis in coord_names], dim=-1)
    pos -= center
    features = torch.stack([torch.tensor(ply["vertex"][name]).float() for name in feature_names], dim=-1)
    if not torch.isfinite(pos).all() or not torch.isfinite(features).all():
        raise ValueError(f"surface PLY {surface_path} contains NaN or inf values")
    features = sanitize_surface_features(features)

    face = None
    if "face" in ply:
        faces = [torch.tensor(face, dtype=torch.long) for face in ply["face"]["vertex_indices"]]
        if faces:
            face = torch.stack(faces, dim=-1)
    pos, features, face = cap_surface_vertices(pos, features, face, ligand_pos, max_vertices)
    data = Data(x=features, pos=pos, face=face)
    data = FaceToEdge()(data)
    return Cartesian(cat=False)(data)


class ScreenDataset(Dataset):
    """Inference-only graph dataset for one receptor against one ligand library."""

    def __init__(
        self,
        pocket_path,
        ligands_path,
        ref_ligand=None,
        surface_path=None,
        pocket_center=None,
        transform=None,
        receptor_radius=30,
        cache_path=None,
        split_path=None,
        remove_hs=False,
        max_lig_size=None,
        c_alpha_max_neighbors=None,
        matching=False,
        keep_original=True,
        popsize=15,
        maxiter=15,
        all_atoms=False,
        atom_radius=5,
        atom_max_neighbors=None,
        esm_embeddings=None,
        require_ligand=False,
        num_workers=1,
        keep_input_pose=False,
        save_dir=None,
        inference_mode="Screen",
        ligandsMaxAtoms=80,
        max_surface_vertices=0,
        **_,
    ):
        super().__init__(None, transform)
        if surface_path is None:
            raise ValueError("ScreenDataset requires an explicit SurfNA surface PLY path.")
        self.graphs = []
        self.ligands = []
        self.pocket_path = str(pocket_path)
        self.surface_path = str(surface_path)
        self.save_dir = Path(save_dir or Path.cwd() / "screen_cache")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        rec_model = parse_pdb_from_path(self.pocket_path)
        ligands = _read_ligands(ligands_path)
        if max_lig_size is None:
            max_lig_size = ligandsMaxAtoms

        for ligand_name, mol in ligands:
            try:
                if max_lig_size is not None and mol.GetNumHeavyAtoms() > int(max_lig_size):
                    logger.info(f"Skipping {ligand_name}: heavy atoms exceed {max_lig_size}")
                    continue
                mol = _ensure_conformer(mol, keep_input_pose)
                graph = HeteroData()
                graph["name"] = ligand_name
                pure_pocket_path = self.save_dir / f"{_safe_stem(self.pocket_path)}_{ligand_name}_pure.pdb"
                get_lig_graph_with_matching(
                    mol,
                    graph,
                    popsize=popsize,
                    maxiter=maxiter,
                    matching=matching,
                    keep_original=keep_original,
                    num_conformers=1,
                    remove_hs=remove_hs,
                )
                rec, rec_coords, c_alpha_coords, n_coords, c_coords, lm_embeddings = extract_receptor_structure(
                    copy.deepcopy(rec_model), mol, save_file=str(pure_pocket_path), lm_embedding_chains=esm_embeddings
                )
                mda_rec_model = mda.Universe(str(pure_pocket_path))
                get_rec_graph(
                    mda_rec_model,
                    rec_coords,
                    c_alpha_coords,
                    n_coords,
                    c_coords,
                    graph,
                    rec_radius=receptor_radius,
                    c_alpha_max_neighbors=c_alpha_max_neighbors,
                    all_atoms=all_atoms,
                    atom_radius=atom_radius,
                    atom_max_neighbors=atom_max_neighbors,
                    remove_hs=remove_hs,
                    lm_embeddings=lm_embeddings,
                )

                receptor_center = torch.mean(graph["receptor"].pos, dim=0, keepdim=True)
                graph["receptor"].pos -= receptor_center
                if all_atoms and "atom" in graph.node_types:
                    graph["atom"].pos -= receptor_center
                graph["ligand"].pos -= receptor_center
                graph.original_center = receptor_center
                graph.original_ligand_center = torch.mean(graph["ligand"].pos, dim=0, keepdim=True) + receptor_center

                surface_graph = _load_surface_graph(
                    self.surface_path,
                    receptor_center,
                    graph["ligand"].pos,
                    int(max_surface_vertices or 0),
                )
                graph["surface"].pos = surface_graph.pos
                graph["surface"].x = surface_graph.x
                graph["surface", "surface_edge", "surface"].edge_index = surface_graph.edge_index
                graph["surface", "surface_edge", "surface"].edge_attr = surface_graph.edge_attr
                graph.mol = copy.deepcopy(mol)
                graph["protein_path"] = self.pocket_path
                graph["pocket_path"] = self.pocket_path
                graph["ref_ligand"] = ref_ligand
                self.graphs.append(graph)
                self.ligands.append(mol)
            except Exception as exc:
                logger.exception(f"Skipping ligand {ligand_name} because graph construction failed: {exc}")

        if not self.graphs:
            logger.warning(f"No valid ligands were loaded from {ligands_path}")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return copy.deepcopy(self.graphs[idx])
