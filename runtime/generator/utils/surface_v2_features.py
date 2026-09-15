"""Schema-aware Surface-v2 feature loading and shared-scale normalization."""

from __future__ import annotations

import json
from pathlib import Path

import torch


SURFACE_FEATURE_SCHEMAS = {
    "legacy4": ("hbond", "hphob", "charge", "si"),
    "v2_shared4": ("hbond", "hphob", "charge", "si"),
    "v2_full8": ("hbond", "hphob", "charge", "si", "donor", "acceptor", "apolar", "boundary"),
}


def feature_names(schema: str) -> tuple[str, ...]:
    try:
        return SURFACE_FEATURE_SCHEMAS[schema]
    except KeyError as exc:
        raise ValueError(f"Unknown surface feature schema: {schema}") from exc


def load_scaler(path: str | None) -> dict | None:
    if path is None:
        return None
    scaler_path = Path(path)
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Surface scaler JSON not found: {scaler_path}")
    payload = json.loads(scaler_path.read_text())
    charge = payload.get("transforms", {}).get("charge", {})
    if charge.get("kind") != "domain_balanced_robust_zscore":
        raise ValueError("Surface scaler does not define a domain-balanced charge transform")
    if float(charge.get("scale", 0.0)) <= 0:
        raise ValueError("Surface scaler charge scale must be positive")
    return payload


def normalize(features: torch.Tensor, names: tuple[str, ...], schema: str, scaler: dict | None) -> torch.Tensor:
    """Apply the one shared V2 transform; legacy inputs retain legacy handling."""
    values = torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0).clone()
    if schema == "legacy4":
        values[:, 0] = values[:, 0].clamp(-1.0, 1.0)
        values[:, 1] = values[:, 1].clamp(-4.5, 4.5)
        values[:, 2] = values[:, 2].clamp(-10.0, 10.0)
        values[:, 3] = values[:, 3].clamp(0.0, 1.0)
        return values
    if scaler is None:
        raise ValueError(f"{schema} requires --surface_scaler_json fitted on training surfaces")
    charge_cfg = scaler["transforms"]["charge"]
    charge_index = names.index("charge")
    center, scale = float(charge_cfg["center"]), float(charge_cfg["scale"])
    low, high = (float(value) for value in charge_cfg["clip_after_zscore"])
    values[:, charge_index] = ((values[:, charge_index] - center) / scale).clamp(low, high)
    for name in ("hbond", "hphob", "si"):
        if name in names:
            values[:, names.index(name)] = values[:, names.index(name)].clamp(-1.0, 1.0)
    for name in ("donor", "acceptor", "apolar", "boundary"):
        if name in names:
            values[:, names.index(name)] = values[:, names.index(name)].clamp(0.0, 1.0)
    return values
