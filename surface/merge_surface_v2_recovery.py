#!/usr/bin/env python3
"""Merge successful recovery jobs into a Surface-v2 pilot with an audit backup."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from generate_surfaces_v2 import AUDIT_FIELDS


def read_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open()))


def rebase_recovery_paths(
    row: dict[str, str], recovery_dir: Path, main_dir: Path
) -> dict[str, str]:
    """Point copied recovery artifacts at their canonical location in main_dir."""
    old_prefix = str(recovery_dir).rstrip("/")
    new_prefix = str(main_dir).rstrip("/")
    rebased = dict(row)
    for field, value in rebased.items():
        if isinstance(value, str) and value.startswith(old_prefix + "/"):
            rebased[field] = new_prefix + value[len(old_prefix):]
    return rebased


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("main_dir")
    parser.add_argument("recovery_dir")
    parser.add_argument("--include", nargs="*", default=None)
    parser.add_argument("--drop", nargs="*", default=[])
    parser.add_argument("--allow_new", action="store_true")
    args = parser.parse_args()
    main_dir = Path(args.main_dir)
    recovery_dir = Path(args.recovery_dir)
    main_audit = main_dir / "surface_v2_audit.csv"
    recovery_audit = recovery_dir / "surface_v2_audit.csv"
    main_rows = read_rows(main_audit)
    recovery_rows = read_rows(recovery_audit)
    included = set(args.include) if args.include is not None else None
    replacements = {
        row["name"]: rebase_recovery_paths(row, recovery_dir, main_dir) for row in recovery_rows
        if row["status"] == "success" and (included is None or row["name"] in included)
    }
    if not replacements:
        raise SystemExit("Recovery audit contains no successful rows")
    missing = sorted(set(replacements) - {row["name"] for row in main_rows})
    if missing and not args.allow_new:
        raise SystemExit(f"Recovery names not present in main audit: {missing}")
    unknown_drop = sorted(set(args.drop) - {row["name"] for row in main_rows})
    if unknown_drop:
        raise SystemExit(f"Dropped names not present in main audit: {unknown_drop}")

    backup = main_dir / "surface_v2_audit.pre_recovery.csv"
    shutil.copy2(main_audit, backup)
    for name in replacements:
        shutil.copytree(recovery_dir / name, main_dir / name, dirs_exist_ok=True)
    dropped = set(args.drop)
    merged = [
        replacements.get(row["name"], row)
        for row in main_rows
        if row["name"] not in dropped
    ]
    old_names = {row["name"] for row in main_rows}
    merged.extend(replacements[name] for name in sorted(set(replacements) - old_names))
    with main_audit.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in AUDIT_FIELDS} for row in merged)
    print(
        f"Merged {len(replacements)} recovery rows; dropped={len(dropped)}; "
        f"final={len(merged)}; backup={backup}"
    )


if __name__ == "__main__":
    main()
