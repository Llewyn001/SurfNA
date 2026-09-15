"""Configuration helpers shared by SurfNA V2 scorer entrypoints."""

from argparse import Namespace
from pathlib import Path

import yaml


def load_scorer_model_args(path: str) -> Namespace:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("model config must be a YAML mapping")
    flattened = {}
    for key, value in payload.items():
        if isinstance(value, dict) and set(value) == {"value"}:
            value = value["value"]
        flattened[key] = value
    defaults = {
        "all_atoms": False,
        "surface_feature_schema": "v2_full8",
        "surface_feature_dim": 8,
        "use_adaptive_transfer": False,
        "use_nucleic_feat_fusion": False,
        "use_nusurf_fusion": False,
        "use_ligand_patch_tokenizer": True,
        "n_gaussians": 20,
        "mdn_dropout": 0.1,
        "mdn_score_topk_per_atom": 8,
        "scorer_hidden_dim": 192,
        "scorer_cross_distance": 12.0,
        "scorer_clash_distance": 2.0,
        "scorer_variant": "aligned_a",
    }
    for key, value in defaults.items():
        flattened.setdefault(key, value)
    return Namespace(**flattened)
