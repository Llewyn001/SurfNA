#!/usr/bin/env python3
"""Shared, frozen helpers for the SurfNA LB200 mechanism experiment.

The module intentionally reuses the production G2.2 model, graph loader,
diffusion schedule, conformer update, and symmetry-aware RMSD implementation.
It refuses to rebuild a missing cache and contains no training code.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import tempfile
from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


EXPERIMENT_ROOT = Path(
    "/public/home/luoyuxuan/SurfNA_V2/development/"
    "surfna_physchem_transfer_experiment_20260902"
)
FROZEN_CODE_ROOT = Path(
    "/public/home/luoyuxuan/SurfNA_V2/development/"
    "l2_generator_scratch_ddp2_20260901_v5_4090d/frozen/code/src"
)
L2_RELEASE_ROOT = Path(
    "/public/home/luoyuxuan/SurfNA_V2/data/na_dataset_homologyclean_l2/"
    "releases/surfna_homologyclean_l2_fullsnapshot_20260831_v2"
)
EXPECTED_SCALER_SHA256 = (
    "ded3a097344b6d943411dd2466f70fb0f798bb1b72f283fa8cd6a1fdd847abff"
)
EXPECTED_SOURCE_SHA256 = (
    "c90a95381926534dcdddc7699ba3d5cc400afd23f9b07cbc7b614c4b9987239f"
)
EXPECTED_COUNTS = {"train": 801, "val": 89, "test": 128}
ARMS = ("Scratch-Full8", "Transfer-All", "Transfer-NonSurface")
SEEDS = (0, 1, 2)
EPOCHS = (25, 50, 75, 100, 125, 150, 175, 200)
NOISE_LEVELS = (0.10, 0.30, 0.60, 0.90)
POSE_REPEATS = 5
POSE_BANK_SEED = 20260903
CHEMISTRY_CHANNELS = (0, 1, 2, 4, 5, 6)
SHUFFLE_SEEDS = (20260911, 20260912, 20260913)
MAIN_GATE_ALPHAS = (0.0, 0.5, 1.0)
DOSE_GATE_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DOSE_EPOCHS = (50, 100, 200)
INFERENCE_STEPS = 20
INFERENCE_SEED = 20260826
NMSE_EPS = 1e-8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode())
    digest.update(str(value.dtype).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def stable_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def arm_slug(arm: str) -> str:
    return arm.lower().replace("-", "_")


def model_dir(arm: str, seed: int) -> Path:
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError(f"unsupported arm/seed: {arm}/{seed}")
    if arm == "Scratch-Full8" and seed == 0:
        return Path(
            "/public/home/luoyuxuan/SurfNA_V2/development/"
            "l2_generator_scratch_ddp2_20260901_v5_4090d/runs/"
            "surfna_homologyclean_l2_fullsnapshot_20260831_v2_"
            "g22_scratch_seed0_e500_ddp2_b8"
        )
    if arm == "Scratch-Full8" and seed == 1:
        return EXPERIMENT_ROOT / "03_training/scratch_full8_seed1_e500_ddp2_b8"
    if arm == "Scratch-Full8":
        return EXPERIMENT_ROOT / f"03_training/scratch_full8_seed{seed}_e200_ddp2_b8"
    if arm == "Transfer-All":
        return EXPERIMENT_ROOT / f"03_training/transfer_all_seed{seed}_e200_ddp2_b8"
    if seed == 0:
        return EXPERIMENT_ROOT / "03_training/transfer_nonsurface_seed0_e500_ddp2_b8"
    return EXPERIMENT_ROOT / f"03_training/transfer_nonsurface_seed{seed}_e200_ddp2_b8"


def checkpoint_path(analysis_root: Path, arm: str, seed: int, epoch: int) -> Path:
    if epoch not in EPOCHS:
        raise ValueError(f"unsupported epoch: {epoch}")
    return model_dir(arm, seed) / f"ema_epoch_{epoch:04d}.pt"


def load_model_args(arm: str, seed: int) -> Namespace:
    path = model_dir(arm, seed) / "model_parameters.yml"
    with path.open() as handle:
        args = Namespace(**yaml.safe_load(handle))
    return args


def deny_rebuild(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(
        "Frozen validation graph cache is missing or mismatched; automatic rebuilding is forbidden"
    )


def load_validation_dataset(model_args: Namespace):
    from datasets.pdbbind import PDBBind

    PDBBind.preprocessing = deny_rebuild
    PDBBind.inference_preprocessing = deny_rebuild
    dataset = PDBBind(
        transform=None,
        root=model_args.data_dir,
        limit_complexes=0,
        receptor_radius=model_args.receptor_radius,
        cache_path=model_args.cache_path,
        split_path=model_args.split_val,
        remove_hs=model_args.remove_hs,
        max_lig_size=None,
        c_alpha_max_neighbors=model_args.c_alpha_max_neighbors,
        matching=not model_args.no_torsion,
        keep_original=True,
        popsize=model_args.matching_popsize,
        maxiter=model_args.matching_maxiter,
        all_atoms=model_args.all_atoms,
        atom_radius=model_args.atom_radius,
        atom_max_neighbors=model_args.atom_max_neighbors,
        esm_embeddings_path=getattr(model_args, "esm_embeddings_path", None),
        require_ligand=True,
        num_workers=1,
        surface_path=model_args.surface_path,
        per_complex_timeout_sec=getattr(model_args, "per_complex_timeout_sec", 180),
        max_surface_vertices=getattr(model_args, "max_surface_vertices", 0),
        surface_feature_schema=getattr(model_args, "surface_feature_schema", "legacy4"),
        surface_scaler_json=getattr(model_args, "surface_scaler_json", None),
    )
    if len(dataset) != EXPECTED_COUNTS["val"]:
        raise RuntimeError(f"validation cohort drift: expected 89, observed {len(dataset)}")
    split_names = [line.strip() for line in Path(model_args.split_val).read_text().splitlines() if line.strip()]
    observed = [complex_name(dataset[index]) for index in range(len(dataset))]
    if split_names != observed:
        raise RuntimeError("validation cache order/identity differs from the frozen val split")
    return dataset


def load_score_model(
    analysis_root: Path,
    arm: str,
    seed: int,
    epoch: int,
    device: torch.device,
):
    from utils.diffusion_utils import t_to_sigma as t_to_sigma_impl
    from utils.utils import get_model

    args = load_model_args(arm, seed)
    sigma_fn = partial(t_to_sigma_impl, args=args)
    model = get_model(
        args,
        device,
        t_to_sigma=sigma_fn,
        no_parallel=True,
        model_type=args.model_type,
    )
    checkpoint = checkpoint_path(analysis_root, arm, seed, epoch)
    raw = torch.load(checkpoint, map_location="cpu")
    state = raw.get("model", raw) if isinstance(raw, dict) else raw
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint is not a state dict: {checkpoint}")
    normalized = {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}
    incompatible = model.load_state_dict(normalized, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"strict checkpoint load failed: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model = model.to(device)
    model.eval()
    return model, args, checkpoint, stable_state_sha256(normalized)


def complex_name(graph: Any) -> str:
    name = getattr(graph, "name", None)
    if isinstance(name, (list, tuple)):
        name = name[0]
    return str(name)


def native_ligand_coordinates(graph: Any) -> np.ndarray:
    native = graph["ligand"].orig_pos
    if isinstance(native, list):
        native = native[0]
    if torch.is_tensor(native):
        native = native.detach().cpu().numpy()
    return np.asarray(native, dtype=np.float32)


def heavy_atom_mask(graph: Any) -> np.ndarray:
    return torch.not_equal(graph["ligand"].x[:, 0], 0).detach().cpu().numpy()


def evaluator_rmsd(graph: Any, coordinates: torch.Tensor | np.ndarray) -> tuple[float, str]:
    from utils.utils import get_symmetry_rmsd, remove_all_hs

    coords = coordinates.detach().cpu().numpy() if torch.is_tensor(coordinates) else np.asarray(coordinates)
    mask = heavy_atom_mask(graph)
    coords = coords[mask]
    native = native_ligand_coordinates(graph)[mask]
    try:
        mol = remove_all_hs(graph.mol[0])
        value = get_symmetry_rmsd(mol, native, [coords])
        return float(np.asarray(value).reshape(-1)[0]), "symmetry_aware"
    except Exception:
        value = float(np.sqrt(((coords - native) ** 2).sum(axis=1).mean()))
        return value, "coordinate_rmsd_fallback"


def tensor_cosine(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.detach().float().reshape(-1).cpu()
    target = target.detach().float().reshape(-1).cpu()
    if prediction.numel() == 0 or target.numel() == 0:
        return math.nan
    denom = float(torch.linalg.vector_norm(prediction) * torch.linalg.vector_norm(target))
    if denom <= NMSE_EPS:
        return math.nan
    return float(torch.dot(prediction, target) / denom)


def normalized_mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.detach().float().reshape(-1).cpu()
    target = target.detach().float().reshape(-1).cpu()
    if prediction.numel() == 0 or target.numel() == 0:
        return math.nan
    numerator = torch.sum((prediction - target) ** 2)
    denominator = torch.sum(target ** 2).clamp_min(NMSE_EPS)
    return float(numerator / denominator)


def equal_component_mean(values: Iterable[float]) -> tuple[float, int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return (float(np.mean(finite)), len(finite)) if finite else (math.nan, 0)


def clone_with_bank_entry(dataset: Any, entry: dict[str, Any]):
    graph = dataset[int(entry["dataset_index"])]
    if complex_name(graph) != entry["complex_id"]:
        raise RuntimeError("pose-bank entry no longer maps to the same validation graph")
    graph["ligand"].pos = entry["initial_ligand_coordinates"].clone()
    return graph


def make_joint_shuffle_permutation(complex_id: str, n_vertices: int, seed: int) -> tuple[torch.Tensor, int]:
    if n_vertices <= 1:
        raise ValueError(f"surface has too few vertices for a joint shuffle: {complex_id}/{n_vertices}")
    attempt = 0
    while True:
        token = f"surfna-joint-shuffle-v1|{complex_id}|{seed}|{attempt}".encode()
        derived = int.from_bytes(hashlib.sha256(token).digest()[:8], "little", signed=False)
        permutation = np.random.default_rng(derived).permutation(n_vertices)
        fixed_ratio = float(np.mean(permutation == np.arange(n_vertices)))
        if fixed_ratio < 0.05:
            return torch.as_tensor(permutation, dtype=torch.long), attempt + 1
        attempt += 1
        if attempt > 10000:
            raise RuntimeError(f"could not construct a sufficiently non-identity permutation: {complex_id}")


def apply_joint_chemistry_shuffle(graph: Any, permutation: torch.Tensor) -> None:
    features = graph["surface"].x
    if features.ndim != 2 or features.shape[1] != 8:
        raise RuntimeError(f"expected v2_full8 surface features, observed {tuple(features.shape)}")
    permutation = permutation.to(features.device)
    shuffled = features.clone()
    channels = torch.as_tensor(CHEMISTRY_CHANNELS, dtype=torch.long, device=features.device)
    shuffled[:, channels] = features[permutation][:, channels]
    graph["surface"].x = shuffled


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def graph_invariant_hashes(graph: Any) -> dict[str, str]:
    payload = {
        "ligand_x": graph["ligand"].x,
        "ligand_pos": graph["ligand"].pos,
        "receptor_x": graph["receptor"].x,
        "receptor_pos": graph["receptor"].pos,
        "surface_pos": graph["surface"].pos,
    }
    for edge_type in graph.edge_types:
        payload[f"edge_index::{edge_type}"] = graph[edge_type].edge_index
    return {key: stable_tensor_sha256(value) for key, value in payload.items()}


def json_safe_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None
