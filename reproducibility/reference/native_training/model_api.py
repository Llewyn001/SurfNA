"""Shared native/rank model API, with explicit source and checkpoint gates.

Numerical model/transfer code is the pinned native_prior_v3 source snapshot;
no old NA-trained head, V3 scaler, or cached V3 ligand encoding is loaded here.
"""
from __future__ import annotations

import hashlib
import os
import random
import sys
from functools import partial
from pathlib import Path

from native_contract import RECIPE, checked_file, load_graph_index, read_config, read_json, sha, within


def finite(*values):
    import torch
    for value in values:
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError("NaN/Inf detected before numerical masking")


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def static_prior_state(model):
    """Exact exported state scope, also used for bundle/readback parity."""
    from models.surfna_v2_transfer import STATIC_PREFIXES
    selected = {}
    for name, tensor in model.state_dict().items():
        canonical = name.removeprefix("backbone.")
        if name.startswith("prior_head.") or (name.startswith("backbone.") and any(
                canonical == prefix or canonical.startswith(prefix + ".") for prefix in STATIC_PREFIXES)):
            finite(tensor)
            selected[name] = tensor.detach().cpu().clone()
    if not selected or not any(name.startswith("prior_head.") for name in selected):
        raise RuntimeError("Static/native-prior export scope is empty")
    return selected


def seed_all(seed=RECIPE["seed"]):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False


def activate_source(contract):
    source = Path(contract["source_snapshot"]["root"]).resolve()
    # This override would silently swap the scorer architecture for generator
    # code, so it must be cleared before the model package is first imported.
    os.environ.pop("SURFNA_V2_G22_SOURCE_ROOT", None)
    os.environ["precomputed_arrays"] = contract["precomputed_dir"]
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] in {"models", "utils", "datasets"}:
            path = getattr(module, "__file__", None)
            if path is not None and not within(path, source):
                raise RuntimeError(f"Model dependency already imported from another tree: {name}: {path}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return source


def clean_graph(graph):
    graph = graph.clone()
    forbidden = {"orig_pos", "original_center", "original_ligand_center", "rmsd", "rmsd_matching",
                 "name", "complex_id", "parent_pdb", "pdb_id", "split", "split_group", "pose_uid",
                 "pose_path", "native_path", "native_global_coordinates", "candidate_rank_input",
                 "ordinal", "atoms_pos"}
    for store in graph.stores:
        for field in list(store.keys()):
            if field in forbidden or field.startswith(("reference_", "native_", "label_")):
                del store[field]
    return graph


def build_model(contract, reference_path, device="cuda"):
    """Return (model, metadata). Only prior_head parameters require gradients."""
    source = activate_source(contract)
    import torch
    from models.surface_interaction_backbone_v2 import BACKBONE_SOURCE_FILE
    from models.surfna_v2_transfer import load_protein_mdn_scorer_transfer, load_v2_diffusion_backbone
    from utils.diffusion_utils import t_to_sigma
    from utils.scorer_config_v2 import load_scorer_model_args
    from utils.scorer_factory_v2 import build_surfna_v2_scorer
    if Path(BACKBONE_SOURCE_FILE).resolve() != source / "models/surface_score_model_v3.py":
        raise RuntimeError("Scorer loaded a different backbone implementation")
    seed_all()
    for field in ("protein_model_config", "protein_diffusion_checkpoint", "protein_mdn_checkpoint"):
        checked_file(contract[field]["path"], contract[field]["sha256"])
    checked_file(contract["dataset"]["surface_scaler"]["path"],
                 contract["dataset"]["surface_scaler"]["sha256"])
    config = load_scorer_model_args(contract["protein_model_config"]["path"])
    for field in ("scorer_variant", "n_gaussians", "scorer_hidden_dim", "mdn_dropout",
                  "mdn_score_topk_per_atom", "max_surface_vertices", "use_nusurf_fusion",
                  "use_nucleic_feat_fusion", "surface_feature_schema"):
        setattr(config, field, RECIPE[field])
    config.surface_scaler_json = contract["dataset"]["surface_scaler"]["path"]
    config.surface_feature_dim = 8
    config.mdn_reference_json = str(Path(reference_path).resolve())
    config.scorer_equivariant_rms_cap = RECIPE["equivariant_graph_rms_cap"]
    model = build_surfna_v2_scorer(config, torch.device(device), partial(t_to_sigma, args=config))
    load = lambda record: torch.load(record["path"], map_location="cpu", weights_only=False)
    diffusion = load_v2_diffusion_backbone(model, load(contract["protein_diffusion_checkpoint"]),
        surface_feature_schema="v2_full8", distance_embed_dim=config.distance_embed_dim,
        require_dynamic_ranker=False).to_dict()
    # The transfer helper strictly copies prior/static parameters while keeping
    # the target (new L2 train-only) reference_prior buffers intact.
    reference_before = {k: v.detach().cpu().clone() for k, v in model.prior_head.state_dict().items()
                        if k.startswith("reference_prior.")}
    mdn = load_protein_mdn_scorer_transfer(model, load(contract["protein_mdn_checkpoint"])).to_dict()
    reference_after = {k: v.detach().cpu().clone() for k, v in model.prior_head.state_dict().items()
                       if k.startswith("reference_prior.")}
    if not reference_before or state_digest(reference_before) != state_digest(reference_after):
        raise RuntimeError("Protein transfer changed the new L2 distance reference")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.prior_head.parameters():
        parameter.requires_grad_(True)
    model.eval()
    finite(*model.state_dict().values())
    metadata = {"arm": "protein_mdn_transfer", "seed": RECIPE["seed"], "native_only": True,
        "frozen_backbone": True, "fresh_optimizer": True, "scheduler": None,
        "diffusion_sha256": contract["protein_diffusion_checkpoint"]["sha256"],
        "protein_mdn_sha256": contract["protein_mdn_checkpoint"]["sha256"],
        "source_config_sha256": contract["protein_model_config"]["sha256"],
        "source_snapshot_sha256": contract["source_snapshot"]["manifest"]["sha256"],
        "surface_scaler_sha256": contract["dataset"]["surface_scaler"]["sha256"],
        "reference_sha256": sha(reference_path), "diffusion_transfer": diffusion, "protein_mdn_transfer": mdn,
        "initial_head_sha256": state_digest(model.prior_head.state_dict()),
        "initial_backbone_sha256": state_digest(model.backbone.state_dict()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_names": [name for name, p in model.named_parameters() if p.requires_grad],
        "test_opened": False, "architecture_args": vars(config)}
    return model, metadata


def load_trained_model(root, device="cuda"):
    """Rank-stage entry: verifies final gates, loads L2 best head, freezes all."""
    import torch
    root = Path(root).resolve()
    config = read_config(root)
    load_graph_index(root, config)
    native = root / "native"
    ready = read_json(native / "READY.json")
    if ready.get("status") != "PASS" or ready["contract_sha256"] != sha(root / "native_contract.json"):
        raise RuntimeError("Native stage is not ready for ranking")
    if ready.get("test_opened") is not False or ready["root_contract_sha256"] != config["root_contract_sha256"]:
        raise RuntimeError("Native final contract mismatch")
    for path, field in (("best_head.pt", "best_head_sha256"), ("native_prior_static_bundle.pt", "bundle_sha256"),
                        ("graph_index.json", "graph_index_sha256")):
        checked_file(native / path, ready[field])
    model, metadata = build_model(config, native / "native_train_reference.json", device)
    best = torch.load(native / "best_head.pt", map_location="cpu", weights_only=False)
    if metadata["initial_backbone_sha256"] != best["metadata"]["initial_backbone_sha256"]:
        raise RuntimeError("Reconstructed backbone differs from the trained frozen backbone")
    model.prior_head.load_state_dict(best["prior_head"], strict=True)
    finite(*model.state_dict().values())
    bundle = torch.load(native / "native_prior_static_bundle.pt", map_location="cpu", weights_only=False)
    selected = static_prior_state(model)
    if set(selected) != set(bundle["model"]) or state_digest(selected) != state_digest(bundle["model"]):
        raise RuntimeError("Best-head/static-backbone reconstruction differs from the exported native bundle")
    if bundle["metadata"]["source_head_sha256"] != ready["best_head_sha256"]:
        raise RuntimeError("Native bundle was not exported from the pinned best head")
    if bundle["metadata"]["contract_sha256"] != ready["contract_sha256"]:
        raise RuntimeError("Native bundle belongs to another native contract")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model
