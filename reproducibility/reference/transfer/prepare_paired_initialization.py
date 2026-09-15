#!/usr/bin/env python3
"""Freeze paired initializations and the transfer manifest for SurfNA L2.

This program is deliberately CPU-only.  It constructs one target-model random
state per paired seed, records source/target compatibility, and writes one
immutable initialization recipe for each requested experiment arm.  It never
loads a training optimizer, performs data preprocessing, or writes into the
frozen dataset or source-code directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from functools import partial
from pathlib import Path
from typing import Any


EXPECTED_SOURCE_SHA256 = "c90a95381926534dcdddc7699ba3d5cc400afd23f9b07cbc7b614c4b9987239f"
EXPECTED_SCALER_SHA256 = "ded3a097344b6d943411dd2466f70fb0f798bb1b72f283fa8cd6a1fdd847abff"
EXPECTED_COUNTS = {"train": 801, "val": 89, "test": 128}
ARMS = ("Scratch-Full8", "Transfer-NonSurface", "Transfer-All")

# These are roots of modules that are both physically surface-facing and
# executed by SurfaceScoreModelV3's ligand/receptor/surface forward pathway.
# The smoke test records actual calls before this provisional manifest is
# promoted to the final manifest.
SURFACE_ROOTS = (
    "surface_node_embedding",
    "surface_edge_embedding",
    "surface_rec_cross_edge_embedding",
    "surface_distance_expansion",
    "cross_edge_embedding",
    "ligand_patch_tokenizer",
    "surface_conv_layers",
    "lig_to_surface_conv_layers",
    "surface_to_lig_conv_layers",
    "residue_to_surface_conv_layers",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp = Path(handle.name)
    os.replace(temp, path)


def atomic_torch_save(torch: Any, path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def tree_hash(root: Path) -> dict[str, Any]:
    records: list[dict[str, str]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel = str(path.relative_to(root))
        records.append({"path": rel, "sha256": sha256_file(path)})
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"root": str(root), "tree_sha256": digest, "files": records}


def state_digest(torch: Any, state: dict[str, Any], keys: list[str] | None = None) -> str:
    digest = hashlib.sha256()
    requested = sorted(state if keys is None else keys)
    for key in requested:
        value = state[key]
        digest.update(key.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        data = value.detach().cpu().contiguous().numpy().tobytes()
        digest.update(data)
    return digest.hexdigest()


def same_shape(left: Any, right: Any) -> bool:
    return tuple(left.shape) == tuple(right.shape)


def surface_key(key: str) -> bool:
    return any(key == root or key.startswith(root + ".") for root in SURFACE_ROOTS)


def training_args(release: Path) -> list[str]:
    """The architectural/training arguments of the frozen L2 DDP2 protocol."""
    layout = json.loads((release / "PORTABLE_RUNTIME.json").read_text())["layout"]
    splits = json.loads((release / "PORTABLE_RUNTIME.json").read_text())["splits"]
    inside = lambda rel: str(release / rel)
    return [
        "--data_dir", inside(layout["data_dir"]), "--cache_path", inside(layout["cache_base"]),
        "--surface_path", inside(layout["surface_dir"]), "--surface_feature_schema", "v2_full8",
        "--surface_feature_dim", "8", "--surface_scaler_json", inside(layout["scaler"]),
        "--max_surface_vertices", "512", "--split_train", inside(splits["train"]["path"]),
        "--split_val", inside(splits["val"]["path"]), "--split_test", inside(splits["val"]["path"]),
        "--train_sampling_weights", inside(layout["weights"]), "--model_type", "surface_score_model",
        "--model_version", "version3", "--transformStyle", "diffdock", "--lr", "2e-4",
        "--w_decay", "0.01", "--batch_size", "8", "--n_epochs", "500", "--checkpoint_interval", "25",
        "--tr_weight", "0.33", "--rot_weight", "0.33", "--tor_weight", "0.33",
        "--g22_rot_low_noise_boost", "0.50", "--g22_tor_low_noise_boost", "1.00",
        "--g22_low_noise_decay", "4.0", "--tr_sigma_min", "0.1", "--tr_sigma_max", "5.0",
        "--rot_sigma_min", "0.03", "--rot_sigma_max", "1.55", "--tor_sigma_min", "0.0314",
        "--tor_sigma_max", "3.14", "--ns", "48", "--nv", "10", "--distance_embed_dim", "32",
        "--cross_distance_embed_dim", "32", "--sigma_embed_dim", "32", "--num_conv_layers", "6",
        "--dynamic_max_cross", "--scale_by_sigma", "--dropout", "0.1", "--remove_hs",
        "--c_alpha_max_neighbors", "24", "--receptor_radius", "15.0", "--matching_popsize", "20",
        "--matching_maxiter", "20", "--num_conformers", "1", "--num_workers", "4",
        "--num_dataloader_workers", "0", "--use_ema", "--cudnn_benchmark", "--scheduler", "plateau",
        "--scheduler_patience", "8", "--val_inference_freq", "25", "--skip_inference_freq", "0",
        "--num_inference_complexes", "89", "--inference_steps", "20", "--val_samples_per_complex", "10",
        "--val_inference_seed", "20260826", "--inference_earlystop_metric", "valinf_pose_density_lt2",
        "--inference_earlystop_goal", "max", "--use_nucleic_feat_fusion", "--g21_validity_aware_nucleic",
        "--use_ligand_patch_tokenizer", "--precision_patch_tokenizer", "--patch_temperature", "2.5",
        "--patch_cutoff", "8.0", "--patch_multiscale_cutoffs", "4.5,8.0,12.0", "--patch_topk", "32",
        "--patch_time_gate_center", "0.5", "--patch_time_gate_width", "0.1", "--patch_realign_layers", "3",
        "--use_nusurf_fusion", "--precision_nusurf", "--g22_directional_nusurf",
        "--nusurf_num_blocks", "3", "--nusurf_distance_scale", "6.0", "--nusurf_cutoff", "10.0",
        "--nusurf_realign_layers", "2,4", "--task_aligned_aux", "--task_aux_weight", "0.30",
        "--task_aux_warmup_epochs", "50", "--task_aux_contact_radius", "4.5", "--task_aux_pair_weight", "1.0",
        "--task_aux_ligand_weight", "0.0", "--task_aux_anchor_weight", "0.10",
        "--task_aux_anchor_radius", "6.0", "--task_aux_contrastive_weight", "0.0",
        "--task_aux_noise_decay", "1.5", "--task_aux_clearance_weight", "0.15",
        "--task_aux_clearance_cutoff", "1.0", "--task_aux_clearance_margin", "0.2",
        "--task_aux_clearance_noise_decay", "4.0",
    ]


def make_target_model(code_root: Path, release: Path, seed: int):
    sys.path.insert(0, str(code_root))
    from accelerate.utils import set_seed
    from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
    from utils.parsing import parse_train_args
    from utils.utils import get_model
    import torch

    previous = sys.argv[:]
    try:
        sys.argv = ["paired-init", *training_args(release), "--seed", str(seed)]
        args = parse_train_args()
    finally:
        sys.argv = previous
    set_seed(seed)
    model = get_model(args, torch.device("cpu"), t_to_sigma=partial(t_to_sigma_compl, args=args), model_type=args.model_type)
    return torch, args, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    release = args.release.resolve(strict=True)
    code_root = args.code_root.resolve(strict=True)
    source_path = args.source_checkpoint.resolve(strict=True)
    output = args.output_root.resolve()
    scaffold_names = {"00_audit", "01_initialization", "02_smoke_tests", "03_configs", "03_training", "04_validation", "05_counterfactual", "06_test", "07_statistics", "08_figures"}
    if output.exists():
        # The runtime and an empty directory scaffold may be staged first.
        # Any material experiment artifact would make this an overwrite and
        # is therefore refused.
        unexpected = sorted(path.name for path in output.iterdir() if path.name not in scaffold_names | {"runtime"})
        populated = sorted(path.name for path in output.iterdir() if path.name in scaffold_names and any(path.iterdir()))
        if unexpected or populated:
            raise SystemExit(f"Refusing to overwrite an existing experiment root: {output}; unexpected={unexpected} populated={populated}")
    if sorted(set(args.seeds)) != [0, 1, 2]:
        raise SystemExit("This pre-registered protocol requires exactly the paired seeds 0, 1, and 2")

    portable = json.loads((release / "PORTABLE_RUNTIME.json").read_text())
    if portable.get("counts") != EXPECTED_COUNTS:
        raise SystemExit(f"Unexpected frozen L2 counts: {portable.get('counts')}")
    scaler = release / portable["layout"]["scaler"]
    if sha256_file(source_path) != EXPECTED_SOURCE_SHA256:
        raise SystemExit("Protein source SHA-256 mismatch")
    if sha256_file(scaler) != EXPECTED_SCALER_SHA256:
        raise SystemExit("Frozen L2 train-only scaler SHA-256 mismatch")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("00_audit", "01_initialization", "02_smoke_tests", "03_configs", "03_training", "04_validation", "05_counterfactual", "06_test", "07_statistics", "08_figures", "runtime"):
        (output / name).mkdir(exist_ok=True)

    import torch
    raw_source = torch.load(source_path, map_location="cpu")
    source_state = raw_source.get("model", raw_source) if isinstance(raw_source, dict) else raw_source
    source_state = {(key[7:] if key.startswith("module.") else key): value for key, value in source_state.items()}
    code_provenance = tree_hash(code_root)
    atomic_json(output / "00_audit" / "frozen_inputs.json", {
        "schema_version": "surfna-physchem-transfer-input-audit-v1",
        "release": {"path": str(release), "portable_runtime_sha256": sha256_file(release / "PORTABLE_RUNTIME.json"), "release_name": portable["release_name"], "counts": portable["counts"]},
        "scaler": {"path": str(scaler), "sha256": sha256_file(scaler), "train_only": True},
        "source_checkpoint": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "code": code_provenance,
        "protocol": {"epochs": 500, "world_size": 2, "batch_per_rank": 8, "global_batch": 16, "lr": 2e-4, "weight_decay": 0.01, "ema": 0.999, "gradient_clip": 1.0, "val_every_epochs": 25, "val_k": 10, "val_seed": 20260826, "selection_metric": "valinf_pose_density_lt2", "fixed_test_used_for_selection": False},
        "known_loader_behavior": "Negative surface shape indices are preserved; this experiment does not alter the existing clipping behavior.",
    })

    protocol_args = training_args(release)
    atomic_json(output / "03_configs" / "frozen_training_args.json", {"args": protocol_args})
    first_manifest: dict[str, Any] | None = None
    for seed in args.seeds:
        torch, model_args, model = make_target_model(code_root, release, seed)
        target = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        compatible = sorted(key for key, value in source_state.items() if key in target and same_shape(value, target[key]))
        shape_mismatch = sorted(key for key, value in source_state.items() if key in target and not same_shape(value, target[key]))
        target_missing = sorted(key for key in target if key not in compatible)
        ignored_source = sorted(key for key in source_state if key not in target)
        compatibility = {"loaded": len(compatible), "missing": len(target_missing), "unexpected": 0, "skipped_shape": len(shape_mismatch)}
        expected = {"loaded": 383, "missing": 80, "unexpected": 0, "skipped_shape": 0}
        if compatibility != expected:
            raise SystemExit(f"Unexpected source/target compatibility for seed {seed}: {compatibility} != {expected}")
        parameter_records = []
        for key in sorted(target):
            parameter_records.append({
                "key": key,
                "shape": list(target[key].shape),
                "dtype": str(target[key].dtype),
                "source_compatible": key in compatible,
                "surface_path_candidate": surface_key(key),
                "transfer_group": "SurfacePath" if surface_key(key) else "NonSurfacePath",
            })
        if first_manifest is None:
            first_manifest = {
                "schema_version": "surfna-physchem-transfer-manifest-v1",
                "status": "provisional_pending_real_graph_smoke_trace",
                "surface_path_roots": list(SURFACE_ROOTS),
                "surface_path_definition": "Candidate roots are promoted only if their modules are observed during the real-graph DDP smoke forward trace.",
                "source_compatibility": compatibility,
                "ignored_source_state_keys": ignored_source,
                "shape_mismatch_source_keys": shape_mismatch,
                "records": parameter_records,
            }
        base_path = output / "01_initialization" / f"base_target_seed{seed}.pt"
        atomic_torch_save(torch, base_path, {"schema_version": "surfna-paired-target-init-v1", "seed": seed, "model": target})
        base_sha = sha256_file(base_path)
        atomic_json(output / "01_initialization" / f"base_target_seed{seed}.json", {
            "seed": seed, "path": str(base_path), "sha256": base_sha, "model_state_sha256": state_digest(torch, target), "model_key_count": len(target),
            "source_compatibility": compatibility,
        })
    assert first_manifest is not None
    provisional_path = output / "01_initialization" / "parameter_transfer_manifest_provisional.json"
    atomic_json(provisional_path, first_manifest)
    manifest_sha = sha256_file(provisional_path)
    records = {item["key"]: item for item in first_manifest["records"]}
    for seed in args.seeds:
        base_path = output / "01_initialization" / f"base_target_seed{seed}.pt"
        for arm in ARMS:
            transfer_keys = [] if arm == "Scratch-Full8" else [
                key for key, item in records.items()
                if item["source_compatible"] and (arm == "Transfer-All" or not item["surface_path_candidate"])
            ]
            config = {
                "schema_version": "surfna-paired-arm-init-v1", "arm": arm, "seed": seed,
                "base_init": {"path": str(base_path), "sha256": sha256_file(base_path)},
                "source_checkpoint": {"path": str(source_path), "sha256": EXPECTED_SOURCE_SHA256},
                "manifest": {"path": str(provisional_path), "sha256": manifest_sha, "status_required": "final_after_smoke"},
                "transfer_keys": sorted(transfer_keys), "fresh_optimizer_scheduler": True, "full_model_finetuning": True,
            }
            atomic_json(output / "01_initialization" / f"init_{arm.lower().replace('-', '_')}_seed{seed}.json", config)
    atomic_json(output / "03_configs" / "experiment_matrix.json", {"arms": list(ARMS), "seeds": args.seeds, "planned_training_runs": len(ARMS) * len(args.seeds), "smoke_first": "Transfer-All seed0"})
    print(json.dumps({"output_root": str(output), "source_compatibility": first_manifest["source_compatibility"], "base_seeds": args.seeds, "manifest": str(provisional_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
