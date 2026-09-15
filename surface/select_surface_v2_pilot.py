#!/usr/bin/env python3
"""Select a deterministic pilot list from successful legacy surface jobs."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_csv", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    successful = []
    for row in csv.DictReader(Path(args.audit_csv).open()):
        name = row.get("name", "").strip()
        if row.get("status") == "success" and name and (data_dir / name).is_dir():
            successful.append(name)
    names = sorted(set(successful))
    if len(names) < args.count:
        raise SystemExit(f"Only {len(names)} successful complexes available; requested {args.count}")
    selected = sorted(random.Random(args.seed).sample(names, args.count))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(selected) + "\n")
    print(f"Wrote {len(selected)} complexes to {output}")


if __name__ == "__main__":
    main()
