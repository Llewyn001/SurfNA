#!/usr/bin/env python3
"""Candidate-only nucleic-acid geometry/chemistry features, schema NA_GEOM_V1.

API
---
context = prepare_receptor(receptor_path, metadata=None)
features, names = candidate_features(candidate_mol, context)

``features`` is finite float32 [n_heavy_atoms, 103] in the input molecule's
heavy-atom order. Coordinates must ALREADY share the receptor's global frame.
Scalar features are invariant to a joint rigid transformation; rows are
equivariant to ligand atom renumbering. No model, training statistics, native
coordinates, RMSD, reference contacts, sample ID or molecule properties are read.
This module never writes files and never imports torch or accesses a GPU.

Receptor source: frozen *_protein_processed.pdb, not a graph containing native
ligand positions. Canonical ATOM nucleotides with sugar/base atom signatures are
polymer anchors. Other complete sugar/base residues (including HETATM modified
nucleotides) require a same-chain O3'-P link of 1.1..2.2 A to an anchor/component.
Isolated HET ligands, waters, ions and proteins are excluded WITHOUT native-based
self-clash removal. ``metadata`` may contain only ``sha256`` and/or a restrictive
``residue_keys`` list of [chain, resseq, icode, resname]. It cannot promote a HET
ligand into a polymer. Missing/partial polymer atoms are audited, never imputed.

Features are geometric/chemical PROXIES, not binding energies or hydrogen-bond
counts. Receptor donor/acceptor tags assume conventional neutral canonical bases
plus sugar/phosphate oxygen chemistry; modification/protonation uncertainty is
not silently treated as atomically complete chemistry. Ligand tags use RDKit's
installed BaseFeatures.fdef; pin the RDKit environment when freezing a run.

Fixed layout: 22 ligand chemistry + 11*5 local radial channels + 8 vdW/intraligand
features + 10 complementary-polar/direction proxies + 8 ring-plane proxies=103.
All constants below are a priori feature definitions, NOT fitted on test data.
No per-dataset normalization is learned here; any learned scaler must be trained
outside this module on the authorized TRAIN split only.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

SCHEMA_VERSION = 'NA_GEOM_V1'
LOCAL_CUTOFF_A = 8.0
RADIAL_CENTERS_A = (2.5, 3.5, 5.0)
RADIAL_WIDTH_A = 0.7
POLYMER_LINK_MAX_A = 2.2
SOFT_OVERLAP_WIDTH_A = 0.2
EPS = 1e-10
PT = Chem.GetPeriodicTable()

ELEMENTS = (6, 7, 8, 16, 15, 9, 17, 35, 53)
ELEMENT_NAMES = ('C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'other')
CANONICAL_BASE = {'A': 'A', 'DA': 'A', 'G': 'G', 'DG': 'G', 'C': 'C', 'DC': 'C',
                  'U': 'U', 'DU': 'U', 'T': 'T', 'DT': 'T', 'I': 'I', 'DI': 'I'}
BASE_LABELS = ('A', 'G', 'C', 'U', 'T', 'I', 'modified')
REGIONS = ('base', 'sugar', 'phosphate')
CHANNELS = ('all', *REGIONS, *('base_' + name for name in BASE_LABELS))
PHOSPHATE_ATOMS = {'P', 'OP1', 'OP2', 'OP3', 'O1P', 'O2P', 'O3P'}
SUGAR_SIGNATURE = {"C1'", "C3'", "C4'"}
BASE_RING_TEMPLATES = (('N1', 'C2', 'N3', 'C4', 'C5', 'C6'),
                       ('N9', 'C8', 'N7', 'C5', 'C4'))
BASE_DONORS = {'A': {'N6'}, 'G': {'N1', 'N2'}, 'C': {'N4'},
               'U': {'N3'}, 'T': {'N3'}, 'I': {'N1'}}
BASE_ACCEPTORS = {'A': {'N1', 'N3', 'N7'}, 'G': {'O6', 'N3', 'N7'},
                  'C': {'O2', 'N3'}, 'U': {'O2', 'O4'}, 'T': {'O2', 'O4'},
                  'I': {'O6', 'N3', 'N7'}}

CHEMICAL_NAMES = tuple('lig_element_' + s for s in ELEMENT_NAMES) + (
    'lig_aromatic', 'lig_in_ring', 'lig_donor', 'lig_acceptor',
    'lig_formal_charge_clip4_div4', 'lig_total_H_clip4_div4',
    'lig_degree_clip6_div6', 'lig_hybrid_sp', 'lig_hybrid_sp2',
    'lig_hybrid_sp3', 'lig_hybrid_other', 'lig_vdw_radius_div3',
)
RADIAL_NAMES = tuple(f'{channel}_{suffix}' for channel in CHANNELS for suffix in (
    'local_present', 'nearest_distance_div8', 'rbf2p5_logcount',
    'rbf3p5_logcount', 'rbf5p0_logcount'))
OVERLAP_NAMES = (
    'vdw_all_soft_max_div4', 'vdw_all_soft_sum_logcount',
    'vdw_all_soft_squared_sum_logcount', 'vdw_base_soft_max_div4',
    'vdw_sugar_soft_max_div4', 'vdw_phosphate_soft_max_div4',
    'vdw_local_min_clearance_clip_div8', 'lig_nonbonded_soft_overlap_logcount',
)
POLAR_NAMES = (
    'lig_donor_rec_acceptor_distance_proxy',
    'lig_acceptor_rec_donor_distance_proxy',
    'lig_donor_rec_donor_distance_proxy',
    'lig_acceptor_rec_acceptor_distance_proxy',
    'complementary_polar_nearest_distance_div8', 'complementary_polar_local_present',
    'complementary_lig_away_from_bonds_proxy',
    'complementary_rec_away_from_bonds_proxy',
    'complementary_joint_away_from_bonds_proxy',
    'complementary_joint_direction_present',
)
RING_NAMES = (
    'aromatic_plane_valid', 'nearby_base_ring_present',
    'base_ring_nearest_center_distance_div8', 'base_ring_weighted_abs_normal_cos',
    'base_ring_weighted_height_div8', 'base_ring_weighted_lateral_div8',
    'base_ring_parallel_geometry_soft_max', 'base_ring_parallel_geometry_logcount',
)
FEATURE_NAMES = CHEMICAL_NAMES + RADIAL_NAMES + OVERLAP_NAMES + POLAR_NAMES + RING_NAMES
FEATURE_DIM = len(FEATURE_NAMES)
assert FEATURE_DIM == 103 and len(set(FEATURE_NAMES)) == FEATURE_DIM
FEATURE_NAMES_SHA256 = hashlib.sha256('\n'.join(FEATURE_NAMES).encode()).hexdigest()


def _readonly(value, dtype=np.float64):
    out = np.array(value, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class ReceptorContext:
    positions: np.ndarray
    atomic_numbers: np.ndarray
    vdw_radii: np.ndarray
    regions: np.ndarray
    base_types: np.ndarray
    donor_proxy: np.ndarray
    acceptor_proxy: np.ndarray
    away_from_bonds: np.ndarray
    direction_valid: np.ndarray
    ring_centers: np.ndarray
    ring_normals: np.ndarray
    audit: Mapping


def _plane(x):
    center = x.mean(axis=0)
    _, singular, vt = np.linalg.svd(x-center, full_matrices=False)
    if len(singular) < 2 or singular[1] <= EPS:
        return center, np.zeros(3), False
    return center, vt[-1], True


def _away_vectors(x, adjacency):
    """Heavy-neighbor angular proxy; not explicit hydrogens/lone-pair positions."""
    difference = x[None, :, :] - x[:, None, :]
    distance = np.linalg.norm(difference, axis=-1)
    direction = -(difference / np.maximum(distance[..., None], EPS) * adjacency[..., None]).sum(axis=1)
    norm = np.linalg.norm(direction, axis=1)
    valid = (norm > EPS) & adjacency.any(axis=1)
    direction = np.where(valid[:, None], direction / np.maximum(norm[:, None], EPS), 0.0)
    return direction, valid


def _parse_polymer(text, metadata):
    residues = {}
    for line in text.splitlines():
        record = line[:6].strip()
        if record == 'ENDMDL':
            break
        if record not in ('ATOM', 'HETATM'):
            continue
        if len(line) < 54:
            raise ValueError('Truncated PDB atom record')
        name = line[12:16].strip().replace('*', "'")
        resname = line[17:20].strip()
        key = (line[21:22].strip(), line[22:26].strip(), line[26:27].strip(), resname)
        element = line[76:78].strip().capitalize() if len(line) >= 78 else ''
        # Standard NA atom names are unambiguous; do not parse a protein CA as calcium.
        if not element:
            element = name.lstrip('0123456789')[:1].capitalize()
        if element in ('H', 'D'):
            continue
        try:
            atomic_number = PT.GetAtomicNumber(element)
        except Exception as exc:
            raise ValueError('Unknown PDB element') from exc
        if atomic_number <= 1:
            continue
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        if not np.isfinite(xyz).all():
            raise ValueError('Nonfinite receptor atom coordinate')
        alt = line[16:17].strip()
        occupancy = float(line[54:60].strip() or '0')
        if not math.isfinite(occupancy):
            raise ValueError('Nonfinite PDB occupancy')
        priority = (alt == '', occupancy, alt == 'A', alt)
        item = (xyz, atomic_number, record, priority)
        table = residues.setdefault(key, {})
        if name not in table or priority > table[name][3]:
            table[name] = item
    candidates = {key for key, table in residues.items()
                  if SUGAR_SIGNATURE <= table.keys() and ('N1' in table or 'N9' in table)}
    anchors = {key for key in candidates if key[3] in CANONICAL_BASE
               and any(item[2] == 'ATOM' for item in residues[key].values())}
    selected = set(anchors)
    # Only topology can add modified/HET nucleotides, never proximity to ligand/native.
    remaining = candidates-selected
    while remaining:
        added = set()
        for key in remaining:
            table = residues[key]
            for other in selected:
                if other[0] != key[0]:
                    continue
                target = residues[other]
                linked = any(a in table and b in target and
                             1.1 <= np.linalg.norm(table[a][0]-target[b][0]) <= POLYMER_LINK_MAX_A
                             for a, b in (("O3'", 'P'), ('P', "O3'")))
                if linked:
                    added.add(key)
                    break
        if not added:
            break
        selected.update(added)
        remaining.difference_update(added)
    unlinked_count = len(candidates-selected)
    restricted_count = 0
    if 'residue_keys' in metadata:
        raw_keys = metadata['residue_keys']
        if not isinstance(raw_keys, (list, tuple)) or any(not isinstance(k, (list, tuple)) or len(k) != 4 for k in raw_keys):
            raise ValueError('residue_keys must contain chain/resseq/icode/resname quadruples')
        whitelist = {tuple(str(v).strip() for v in key) for key in raw_keys}
        if not whitelist <= residues.keys():
            raise ValueError('Metadata names absent receptor residues')
        restricted_count = len(selected-whitelist)
        selected.intersection_update(whitelist)
    if not selected:
        raise ValueError('No safely identified nucleic-acid polymer receptor atoms')
    excluded = {}
    for key, table in residues.items():
        if key not in selected:
            excluded[key[3]] = excluded.get(key[3], 0) + len(table)
    return [(key, table) for key, table in residues.items() if key in selected], {
        'polymer_residues': len(selected), 'canonical_atom_anchors': len(anchors & selected),
        'linked_nonanchor_residues': len(selected-anchors),
        'excluded_residue_heavy_atoms': dict(sorted(excluded.items())),
        'unlinked_sugar_base_residues_excluded': unlinked_count,
        'polymer_residues_excluded_by_metadata': restricted_count,
    }


def prepare_receptor(receptor_path, metadata=None):
    """Parse frozen global-frame receptor PDB; no native or sample label allowed.

    ``metadata`` is optional and restrictive: only sha256 and residue_keys.
    Source digest is retained in audit but never used in candidate features.
    """
    metadata = {} if metadata is None else dict(metadata)
    if set(metadata)-{'sha256', 'residue_keys'}:
        raise ValueError('Unsupported receptor metadata; labels/native data are forbidden')
    raw = Path(receptor_path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if 'sha256' in metadata and metadata['sha256'] != digest:
        raise ValueError('Frozen receptor SHA256 mismatch')
    residues, audit = _parse_polymer(raw.decode('utf-8', errors='strict'), metadata)
    positions, atomic_numbers, regions, types = [], [], [], []
    donors, acceptors, away, direction_valid = [], [], [], []
    ring_centers, ring_normals = [], []
    degenerate_rings = 0
    for key, table in residues:
        base = CANONICAL_BASE.get(key[3], 'modified')
        x = np.array([item[0] for item in table.values()])
        z = np.array([item[1] for item in table.values()])
        distances = np.linalg.norm(x[:, None, :]-x[None, :, :], axis=-1)
        covalent = np.array([PT.GetRcovalent(int(n)) for n in z])
        adjacency = ((distances > EPS) &
                     (distances < covalent[:, None]+covalent[None, :]+0.45))
        np.fill_diagonal(adjacency, False)
        directions, valid = _away_vectors(x, adjacency)
        away.extend(directions); direction_valid.extend(valid)
        for name, (xyz, number, _, _) in table.items():
            region = 'phosphate' if name in PHOSPHATE_ATOMS else 'sugar' if "'" in name else 'base'
            positions.append(xyz); atomic_numbers.append(number)
            regions.append(REGIONS.index(region)); types.append(BASE_LABELS.index(base))
            is_donor = name in BASE_DONORS.get(base, set()) if region == 'base' else name == "O2'"
            is_acceptor = name in BASE_ACCEPTORS.get(base, set()) if region == 'base' else number == 8
            # Do not invent sugar donor chemistry for chemically modified residues.
            if base == 'modified' and region == 'sugar':
                is_donor = False
            donors.append(is_donor); acceptors.append(is_acceptor)
        for template in BASE_RING_TEMPLATES:
            if set(template) <= table.keys():
                center, normal, valid = _plane(np.array([table[n][0] for n in template]))
                if valid:
                    ring_centers.append(center); ring_normals.append(normal)
                else:
                    degenerate_rings += 1
    audit.update({'schema_version': SCHEMA_VERSION, 'source_sha256': digest,
                  'coordinate_frame': 'unchanged_global_PDB',
                  'selected_heavy_atoms': len(positions), 'base_ring_planes': len(ring_centers),
                  'degenerate_base_ring_planes_masked': degenerate_rings,
                  'native_used': False, 'proxy_chemistry': True})
    context = ReceptorContext(
        _readonly(positions), _readonly(atomic_numbers, np.int16),
        _readonly([PT.GetRvdw(int(n)) for n in atomic_numbers]),
        _readonly(regions, np.int8), _readonly(types, np.int8),
        _readonly(donors, bool), _readonly(acceptors, bool),
        _readonly(away), _readonly(direction_valid, bool),
        _readonly(np.array(ring_centers).reshape(-1, 3)),
        _readonly(np.array(ring_normals).reshape(-1, 3)), MappingProxyType(audit))
    _validate_context(context)
    return context


def _validate_context(context):
    if not isinstance(context, ReceptorContext):
        raise TypeError('Expected ReceptorContext, not arbitrary graph/metadata')
    n = len(context.positions)
    if n == 0 or context.positions.shape != (n, 3) or context.away_from_bonds.shape != (n, 3):
        raise ValueError('Invalid/empty receptor array shape')
    for key in ('atomic_numbers', 'vdw_radii', 'regions', 'base_types', 'donor_proxy',
                'acceptor_proxy', 'direction_valid'):
        if getattr(context, key).shape != (n,):
            raise ValueError('Receptor feature length mismatch')
    for a in (context.positions, context.vdw_radii, context.away_from_bonds,
              context.ring_centers, context.ring_normals):
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite receptor context')
    if context.ring_centers.shape != context.ring_normals.shape or context.ring_centers.shape[1:] != (3,):
        raise ValueError('Ring context shape mismatch')
    if np.any(context.vdw_radii <= 0) or np.any(context.regions < 0) or np.any(context.regions >= len(REGIONS)):
        raise ValueError('Invalid receptor radii/region')
    if np.any(context.base_types < 0) or np.any(context.base_types >= len(BASE_LABELS)):
        raise ValueError('Invalid receptor base type')


@lru_cache(maxsize=1)
def _feature_factory():
    return ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir)/'BaseFeatures.fdef'))


def _heavy_molecule(mol):
    if not isinstance(mol, Chem.Mol):
        raise TypeError('Expected RDKit Mol candidate')
    # Copy before sanitizing: never change the caller's molecule/properties/order.
    out = Chem.Mol(mol)
    Chem.SanitizeMol(out)
    out = Chem.RemoveHs(out)
    if out.GetNumAtoms() == 0 or out.GetNumConformers() != 1:
        raise ValueError('Candidate must have exactly one nonempty heavy-atom conformer')
    x = np.array(out.GetConformer().GetPositions(), dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite candidate coordinates')
    if any(a.GetAtomicNum() <= 1 or a.HasQuery() for a in out.GetAtoms()):
        raise ValueError('Unsupported heavy/query atom')
    return out, x


def _ligand_chemistry(mol):
    n = mol.GetNumAtoms()
    donor, acceptor = np.zeros(n, bool), np.zeros(n, bool)
    for feature in _feature_factory().GetFeaturesForMol(mol):
        if feature.GetFamily() == 'Donor':
            donor[list(feature.GetAtomIds())] = True
        elif feature.GetFamily() == 'Acceptor':
            acceptor[list(feature.GetAtomIds())] = True
    values = np.zeros((n, len(CHEMICAL_NAMES)), np.float64)
    radii = np.zeros(n)
    hybrids = (Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3)
    for i, atom in enumerate(mol.GetAtoms()):
        z = atom.GetAtomicNum()
        values[i, ELEMENTS.index(z) if z in ELEMENTS else len(ELEMENTS)] = 1
        hybrid = atom.GetHybridization()
        hs = [float(hybrid == kind) for kind in hybrids] + [float(hybrid not in hybrids)]
        radii[i] = PT.GetRvdw(z)
        values[i, len(ELEMENT_NAMES):] = [
            float(atom.GetIsAromatic()), float(atom.IsInRing()), float(donor[i]), float(acceptor[i]),
            float(np.clip(atom.GetFormalCharge(), -4, 4))/4,
            min(atom.GetTotalNumHs(), 4)/4, min(atom.GetDegree(), 6)/6, *hs, radii[i]/3,
        ]
    return values, radii, donor, acceptor


def _cutoff(d):
    return np.where(d < LOCAL_CUTOFF_A, .5*(1+np.cos(np.minimum(d, LOCAL_CUTOFF_A)*np.pi/LOCAL_CUTOFF_A)), 0.)


def _soft_overlap(clearance):
    return SOFT_OVERLAP_WIDTH_A*np.logaddexp(0., -clearance/SOFT_OVERLAP_WIDTH_A)


def _radial(d, context, cutoff):
    pieces = []
    for channel in CHANNELS:
        if channel == 'all':
            mask = np.ones(len(context.positions), bool)
        elif channel in REGIONS:
            mask = context.regions == REGIONS.index(channel)
        else:
            mask = ((context.regions == 0) &
                    (context.base_types == BASE_LABELS.index(channel.removeprefix('base_'))))
        out = np.zeros((len(d), 5))
        if mask.any():
            distance = d[:, mask]
            nearest = distance.min(axis=1)
            present = nearest < LOCAL_CUTOFF_A
            out[:, 0] = present
            out[:, 1] = np.where(present, nearest/LOCAL_CUTOFF_A, 0.)
            for j, center in enumerate(RADIAL_CENTERS_A, 2):
                count = (np.exp(-.5*((distance-center)/RADIAL_WIDTH_A)**2)*cutoff[:, mask]).sum(axis=1)
                out[:, j] = np.log1p(count)/math.log(33)
        pieces.append(out)
    return np.concatenate(pieces, axis=1)


def _overlap_features(mol, x, radii, d, context, cutoff):
    clearance = d-radii[:, None]-context.vdw_radii[None, :]
    soft = _soft_overlap(clearance)*cutoff
    out = np.zeros((len(x), 8))
    out[:, 0] = np.minimum(soft.max(axis=1), 4)/4
    out[:, 1] = np.log1p(soft.sum(axis=1))/math.log(33)
    out[:, 2] = np.log1p((soft**2).sum(axis=1))/math.log(33)
    for j in range(3):
        mask = context.regions == j
        if mask.any():
            out[:, 3+j] = np.minimum(soft[:, mask].max(axis=1), 4)/4
    masked = np.where(d < LOCAL_CUTOFF_A, clearance, np.inf)
    nearest = masked.min(axis=1)
    out[:, 6] = np.where(np.isfinite(nearest), np.clip(nearest, -4, 8)/8, 0)
    intra_d = np.linalg.norm(x[:, None, :]-x[None, :, :], axis=-1)
    nonbonded = Chem.GetDistanceMatrix(mol) > 2
    intra_soft = _soft_overlap(intra_d-radii[:, None]-radii[None, :])*nonbonded
    out[:, 7] = np.log1p(intra_soft.sum(axis=1))/math.log(33)
    return out


def _polar_features(mol, x, ligand_donor, ligand_acceptor, d, context, cutoff):
    out = np.zeros((len(x), 10))
    radial = np.exp(-.5*((d-2.9)/.6)**2)*cutoff
    pairs = ((ligand_donor, context.acceptor_proxy), (ligand_acceptor, context.donor_proxy),
             (ligand_donor, context.donor_proxy), (ligand_acceptor, context.acceptor_proxy))
    for j, (lig_mask, rec_mask) in enumerate(pairs):
        count = (radial*lig_mask[:, None]*rec_mask[None, :]).sum(axis=1)
        out[:, j] = np.log1p(count)/math.log(17)
    complementary = ((ligand_donor[:, None] & context.acceptor_proxy[None, :]) |
                     (ligand_acceptor[:, None] & context.donor_proxy[None, :]))
    local = complementary & (d < LOCAL_CUTOFF_A)
    nearest = np.where(local, d, np.inf).min(axis=1)
    present = np.isfinite(nearest)
    out[:, 4] = np.where(present, nearest/LOCAL_CUTOFF_A, 0)
    out[:, 5] = present
    adjacency = np.asarray(Chem.GetAdjacencyMatrix(mol), bool)
    lig_away, lig_valid = _away_vectors(x, adjacency)
    # Candidate-to-receptor approach vs outward heavy-bond bisectors; no H angles.
    lig_cos = (lig_away@context.positions.T-(lig_away*x).sum(axis=1)[:, None])/np.maximum(d, EPS)
    rec_cos = (x@context.away_from_bonds.T-(context.positions*context.away_from_bonds).sum(axis=1)[None, :])/np.maximum(d, EPS)
    lig_cos = np.clip(lig_cos, 0, 1)*lig_valid[:, None]
    rec_cos = np.clip(rec_cos, 0, 1)*context.direction_valid[None, :]
    weight = radial*local
    out[:, 6] = np.log1p((weight*lig_cos).sum(axis=1))/math.log(17)
    out[:, 7] = np.log1p((weight*rec_cos).sum(axis=1))/math.log(17)
    out[:, 8] = np.log1p((weight*lig_cos*rec_cos).sum(axis=1))/math.log(17)
    out[:, 9] = (local & lig_valid[:, None] & context.direction_valid[None, :]).any(axis=1)
    return out


def _ring_features(mol, x, context):
    out = np.zeros((len(x), 8))
    membership_count = np.zeros(len(x))
    # Symmetrized ring sets, not order-dependent arbitrary SSSR choices.
    rings = {tuple(sorted(map(int, ring))) for ring in Chem.GetSymmSSSR(mol)
             if len(ring) >= 5 and all(mol.GetAtomWithIdx(int(i)).GetIsAromatic() for i in ring)}
    for ring in sorted(rings):
        center, normal, valid = _plane(x[list(ring)])
        feature = np.zeros(8)
        if valid:
            feature[0] = 1
            if len(context.ring_centers):
                delta = center-context.ring_centers
                distance = np.linalg.norm(delta, axis=1)
                near = distance < LOCAL_CUTOFF_A
                if near.any():
                    distance = distance[near]; delta = delta[near]
                    rec_normal = context.ring_normals[near]
                    cosine = np.clip(np.abs(rec_normal@normal), 0, 1)
                    height = np.abs((delta*rec_normal).sum(axis=1))
                    lateral = np.sqrt(np.maximum(distance**2-height**2, 0))
                    weight = np.exp(-.5*(distance/4)**2)*_cutoff(distance)
                    denominator = max(weight.sum(), EPS)
                    parallel = (cosine**4*np.exp(-.5*((height-3.4)/.7)**2)*
                                np.exp(-.5*(lateral/2)**2)*_cutoff(distance))
                    feature[1:] = [1., distance.min()/LOCAL_CUTOFF_A,
                                   float(weight@cosine)/denominator,
                                   float(weight@height)/denominator/LOCAL_CUTOFF_A,
                                   float(weight@lateral)/denominator/LOCAL_CUTOFF_A,
                                   parallel.max(), np.log1p(parallel.sum())/math.log(9)]
        out[list(ring)] += feature
        membership_count[list(ring)] += 1
    return out/np.maximum(membership_count[:, None], 1)


def candidate_features(mol, context):
    """Return (float32[n_heavy,103], FEATURE_NAMES); no labels or native accepted."""
    _validate_context(context)
    mol, x = _heavy_molecule(mol)
    chemical, radii, donor, acceptor = _ligand_chemistry(mol)
    with np.errstate(invalid='raise', divide='raise', over='raise'):
        d = np.linalg.norm(x[:, None, :]-context.positions[None, :, :], axis=-1)
        if not np.isfinite(d).all():
            raise ValueError('Nonfinite candidate/receptor distances')
        cutoff = _cutoff(d)
        result = np.concatenate((chemical, _radial(d, context, cutoff),
            _overlap_features(mol, x, radii, d, context, cutoff),
            _polar_features(mol, x, donor, acceptor, d, context, cutoff),
            _ring_features(mol, x, context)), axis=1)
    if result.shape != (mol.GetNumAtoms(), FEATURE_DIM) or not np.isfinite(result).all():
        raise ValueError('Geometry feature shape/finiteness gate failed')
    result = result.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Geometry features overflow float32')
    return result, FEATURE_NAMES
