#!/usr/bin/env python3
"""Validate SurfNA PLY files use the fixed 4-feature surface schema."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from plyfile import PlyData


REQUIRED = ["x", "y", "z", "nx", "ny", "nz", "hbond", "hphob", "charge", "si"]


def check_ply(path: Path) -> dict[str, object]:
    data = PlyData.read(str(path))
    props = {p.name for p in data["vertex"].properties}
    missing = [name for name in REQUIRED if name not in props]
    row: dict[str, object] = {"path": str(path), "status": "success", "missing": ",".join(missing)}
    if missing:
        row["status"] = "failed"
        return row
    features = np.stack([np.asarray(data["vertex"][name], dtype=float) for name in ["hbond", "hphob", "charge", "si"]], axis=-1)
    row["vertices"] = features.shape[0]
    row["nan_count"] = int(np.count_nonzero(~np.isfinite(features)))
    row["charge_std"] = float(np.nanstd(features[:, 2])) if len(features) else 0.0
    if row["vertices"] <= 9 or row["nan_count"] or np.isclose(row["charge_std"], 0.0):
        row["status"] = "failed"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("surface_dir")
    parser.add_argument("--glob", default="*/*.ply")
    args = parser.parse_args()
    rows = [check_ply(path) for path in sorted(Path(args.surface_dir).glob(args.glob))]
    fields = ["path", "status", "missing", "vertices", "nan_count", "charge_std"]
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    failed = sum(row["status"] == "failed" for row in rows)
    print(f"# checked={len(rows)} failed={failed}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
