#!/usr/bin/env python3
"""Materialize a deterministic, success-only canonical Surface-v2 subset."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from generate_surfaces_v2 import AUDIT_FIELDS


def rebase_paths(row: dict[str, str], source_dir: Path, output_dir: Path) -> dict[str, str]:
    old_prefix = str(source_dir).rstrip("/")
    new_prefix = str(output_dir).rstrip("/")
    result = dict(row)
    for field, value in result.items():
        if value.startswith(old_prefix + "/"):
            result[field] = new_prefix + value[len(old_prefix):]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    if args.count < 1:
        raise SystemExit("count must be positive")
    audit_path = args.source_dir / "surface_v2_audit.csv"
    rows = list(csv.DictReader(audit_path.open()))
    successes = sorted((row for row in rows if row.get("status") == "success"), key=lambda row: row["name"])
    if len(successes) < args.count:
        raise SystemExit(f"Only {len(successes)} successful rows; need {args.count}")
    if args.output_dir.exists():
        raise SystemExit(f"Output directory already exists: {args.output_dir}")
    selected = successes[:args.count]
    args.output_dir.mkdir(parents=True)
    for row in selected:
        shutil.copytree(args.source_dir / row["name"], args.output_dir / row["name"])
    rebased = [rebase_paths(row, args.source_dir, args.output_dir) for row in selected]
    with (args.output_dir / "surface_v2_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in AUDIT_FIELDS} for row in rebased)
    (args.output_dir / "selected_complexes.txt").write_text("\n".join(row["name"] for row in selected) + "\n")
    print(f"Selected {len(selected)}/{len(successes)} successful surfaces into {args.output_dir}")


if __name__ == "__main__":
    main()
