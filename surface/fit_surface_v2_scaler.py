#!/usr/bin/env python3
"""Fit one domain-balanced, train-time scaler for Surface-v2 features.

The generator deliberately stores physical/semantic values rather than
domain-specific normalized values.  This utility fits the sole learnable-scale
transform on training surfaces only, with equal vertex budgets for protein and
nucleic-acid domains.  It therefore retains a real domain shift (for example,
the phosphate-driven NA electrostatic shift) while preventing a numerically
larger domain from defining the model input scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData


FEATURES = ("hbond", "hphob", "charge", "si", "donor", "acceptor", "apolar", "boundary")
FIXED_RANGES = {
    "hbond": [-1.0, 1.0],
    "hphob": [-1.0, 1.0],
    "si": [-1.0, 1.0],
    "donor": [0.0, 1.0],
    "acceptor": [0.0, 1.0],
    "apolar": [0.0, 1.0],
    "boundary": [0.0, 1.0],
}


def read_features(paths: list[Path]) -> dict[str, np.ndarray]:
    values = {feature: [] for feature in FEATURES}
    for path in paths:
        data = PlyData.read(str(path))
        for feature in FEATURES:
            values[feature].append(np.asarray(data["vertex"][feature], dtype=np.float64))
    if not paths:
        raise ValueError("No PLY files found")
    return {feature: np.concatenate(chunks) for feature, chunks in values.items()}


def balanced_sample(values: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
    if len(values) <= budget:
        return values
    return values[rng.choice(len(values), size=budget, replace=False)]


def summary(values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(values, [0.01, 0.25, 0.5, 0.75, 0.99])
    return {
        "vertices": int(len(values)), "mean": float(np.mean(values)), "std": float(np.std(values)),
        "q01": float(quantiles[0]), "q25": float(quantiles[1]), "median": float(quantiles[2]),
        "q75": float(quantiles[3]), "q99": float(quantiles[4]),
    }


def paths_from_train_manifest(surface_dir: Path, manifest: Path) -> list[Path]:
    """Resolve exactly the training complexes; validation/test meshes cannot enter a fit."""
    names = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"training manifest must be non-empty and unique: {manifest}")
    paths = [surface_dir / name / f"{name}_protein_8A.ply" for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} training surfaces missing under {surface_dir}, e.g. {missing[:3]}"
        )
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protein_dir", type=Path, required=True)
    parser.add_argument("--na_dir", type=Path, required=True)
    parser.add_argument("--protein_train_manifest", type=Path, required=True)
    parser.add_argument("--na_train_manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vertex_budget_per_domain", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--charge_clip_mad", type=float, default=5.0)
    args = parser.parse_args()
    if args.vertex_budget_per_domain < 1 or args.charge_clip_mad <= 0:
        raise SystemExit("vertex budget and charge_clip_mad must be positive")

    domain_paths = {
        "protein": paths_from_train_manifest(args.protein_dir, args.protein_train_manifest),
        "na": paths_from_train_manifest(args.na_dir, args.na_train_manifest),
    }
    domain = {kind: read_features(paths) for kind, paths in domain_paths.items()}
    rng = np.random.default_rng(args.seed)
    sampled_charge = np.concatenate([
        balanced_sample(domain[kind]["charge"], args.vertex_budget_per_domain, rng)
        for kind in ("protein", "na")
    ])
    center = float(np.median(sampled_charge))
    mad = float(np.median(np.abs(sampled_charge - center)))
    scale = max(1.4826 * mad, 1e-6)

    transforms: dict[str, dict[str, object]] = {
        feature: {
            "kind": "fixed_range_identity",
            "physical_range": FIXED_RANGES[feature],
            "output_range": FIXED_RANGES[feature],
        }
        for feature in FIXED_RANGES
    }
    transforms["charge"] = {
        "kind": "domain_balanced_robust_zscore",
        "input_units": "kBT/e",
        "center": center,
        "scale": scale,
        "clip_after_zscore": [-args.charge_clip_mad, args.charge_clip_mad],
        "formula": "clip((charge_kBT_per_e - center) / scale, low, high)",
    }
    payload = {
        "schema_version": "surface-v2-scaler-1",
        "fit_policy": {
            "fit_split": "training_only",
            "domain_balance": "equal vertex budget per domain before fitting charge",
            "vertex_budget_per_domain": args.vertex_budget_per_domain,
            "seed": args.seed,
            "warning": "Do not refit on validation/test surfaces or fit separate protein/NA scalers.",
        },
        "source": {
            kind: {
                "ply_count": len(paths), "directory": str(getattr(args, f"{kind}_dir")),
                "training_manifest": str(getattr(args, f"{kind}_train_manifest")),
            }
            for kind, paths in domain_paths.items()
        },
        "transforms": transforms,
        "domain_feature_summary": {
            kind: {feature: summary(values) for feature, values in features.items()}
            for kind, features in domain.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {args.output}; protein_ply={len(domain_paths['protein'])}; "
        f"na_ply={len(domain_paths['na'])}; charge_center={center:.6g}; charge_scale={scale:.6g}"
    )


if __name__ == "__main__":
    main()
