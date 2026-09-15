import copy
import csv
import os

import numpy as np
import torch
from loguru import logger
from rdkit.Chem import RemoveHs
from torch_geometric.data import Dataset

from datasets.pdbbind import PDBBind, read_abs_file_mol, sanitize_cached_surface_graph


class DecoyPDBBind(Dataset):
    """Dataset of generated ligand poses with RMSD labels for MDN reranking."""

    def __init__(self, manifest_path, base_split_path, root, transform=None, cache_path='data/cache',
                 limit_complexes=0, receptor_radius=30, num_workers=1, c_alpha_max_neighbors=None,
                 popsize=15, maxiter=15, matching=True, keep_original=True, max_lig_size=None,
                 remove_hs=False, num_conformers=1, all_atoms=False, atom_radius=5,
                 atom_max_neighbors=None, esm_embeddings_path=None, surface_path=None,
                 per_complex_timeout_sec=180, max_surface_vertices=0,
                 surface_feature_schema='legacy4', surface_scaler_json=None):
        super().__init__(root, transform)
        self.manifest_path = manifest_path
        self.base_split_path = base_split_path
        self.remove_hs = remove_hs
        self.rows = self._read_manifest(manifest_path, limit_complexes)
        complex_names = []
        seen = set()
        for row in self.rows:
            name = row['complex_name']
            if name not in seen:
                seen.add(name)
                complex_names.append(name)
        self.group_id_by_name = {name: i for i, name in enumerate(complex_names)}

        self.base_dataset = PDBBind(
            root=root, transform=None, cache_path=cache_path, split_path=base_split_path,
            limit_complexes=0, receptor_radius=receptor_radius, num_workers=num_workers,
            c_alpha_max_neighbors=c_alpha_max_neighbors, popsize=popsize, maxiter=maxiter,
            matching=matching, keep_original=keep_original, max_lig_size=max_lig_size,
            remove_hs=remove_hs, num_conformers=num_conformers, all_atoms=all_atoms,
            atom_radius=atom_radius, atom_max_neighbors=atom_max_neighbors,
            esm_embeddings_path=esm_embeddings_path, require_ligand=True,
            surface_path=surface_path, per_complex_timeout_sec=per_complex_timeout_sec,
            max_surface_vertices=max_surface_vertices,
            surface_feature_schema=surface_feature_schema,
            surface_scaler_json=surface_scaler_json,
        )
        self.base_index = {}
        for idx in range(len(self.base_dataset)):
            graph = self.base_dataset.get(idx)
            name = graph.name[0] if isinstance(graph.name, (list, tuple)) else graph.name
            self.base_index[str(name)] = idx
        missing = sorted(set(complex_names) - set(self.base_index))
        if missing:
            raise ValueError(f'{len(missing)} decoy complexes are missing from base cache; first examples: {missing[:10]}')
        logger.info(f'Loaded DecoyPDBBind manifest={manifest_path} poses={len(self.rows)} complexes={len(complex_names)}')

    @staticmethod
    def _read_manifest(path, limit_complexes):
        rows = []
        seen_complexes = []
        seen = set()
        with open(path) as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            for row in reader:
                name = row['complex_name']
                if limit_complexes and name not in seen and len(seen_complexes) >= limit_complexes:
                    continue
                if name not in seen:
                    seen.add(name)
                    seen_complexes.append(name)
                if not os.path.exists(row['pose_sdf']):
                    logger.info(f"Skipping missing decoy pose {row['pose_sdf']}")
                    continue
                rows.append(row)
        if not rows:
            raise ValueError(f'No usable decoy rows found in {path}')
        rows.sort(key=lambda r: (r['complex_name'], int(float(r.get('sample_idx', 0)))))
        return rows

    def len(self):
        return len(self.rows)

    def get(self, idx):
        row = self.rows[idx]
        complex_name = row['complex_name']
        graph = self.base_dataset.get(self.base_index[complex_name])
        graph = copy.deepcopy(graph)
        sanitize_cached_surface_graph(graph)

        mol = read_abs_file_mol(row['pose_sdf'], remove_hs=self.remove_hs, sanitize=True)
        if mol is None:
            raise ValueError(f"Could not read decoy pose {row['pose_sdf']}")
        if self.remove_hs:
            mol = RemoveHs(mol, sanitize=True)
        pos = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
        expected = int(graph['ligand'].pos.shape[0])
        if pos.shape[0] != expected:
            raise ValueError(f"Atom count mismatch for {complex_name}: pose={pos.shape[0]} graph={expected} path={row['pose_sdf']}")
        center = graph.original_center
        if torch.is_tensor(center):
            center_np = center.detach().cpu().numpy().reshape(1, 3)
        else:
            center_np = np.asarray(center, dtype=np.float32).reshape(1, 3)
        graph['ligand'].pos = torch.from_numpy(pos - center_np).float()
        graph.decoy_rmsd = torch.tensor([float(row['rmsd'])], dtype=torch.float32)
        graph.decoy_sample_idx = torch.tensor([int(float(row.get('sample_idx', idx)))], dtype=torch.long)
        graph.decoy_group_id = torch.tensor([self.group_id_by_name[complex_name]], dtype=torch.long)
        graph.decoy_complex_name = complex_name
        graph.decoy_pose_sdf = row['pose_sdf']
        graph.name = complex_name
        return graph
