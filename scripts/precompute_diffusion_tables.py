#!/usr/bin/env python3
"""Precompute SO(3) and torus lookup tables used by SurfNA sampling."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src" / "precomputed_arrays",
        help="Directory where lookup tables will be saved.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["precomputed_arrays"] = str(args.out_dir.resolve())

    # Importing these modules triggers the same cache-building logic used by
    # DiffDock/SurfDock/SurfNA when the .npy files are absent.
    import utils.so3  # noqa: F401
    import utils.torus  # noqa: F401

    expected = [
        ".so3_omegas_array2.npy",
        ".so3_cdf_vals2.npy",
        ".so3_score_norms2.npy",
        ".so3_exp_score_norms2.npy",
        ".p.npy",
        ".score.npy",
    ]
    missing = [name for name in expected if not (args.out_dir / name).exists()]
    if missing:
        raise SystemExit(f"Missing precomputed tables: {missing}")
    print(f"Precomputed SurfNA diffusion lookup tables in {args.out_dir}")


if __name__ == "__main__":
    main()
