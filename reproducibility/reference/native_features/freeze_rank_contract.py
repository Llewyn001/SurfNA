"""Freeze new L2 inputs only after native MDN-B and all K8 candidates are ready."""
from __future__ import annotations

import argparse
from pathlib import Path

from rank_common import (COUNTS, FEATURE_NAMES, RECIPE, SOURCE_RECIPE, VERSION, atomic_json,
                         event, sha, upstream_inputs)


def main(args):
    root = Path(args.root).resolve(strict=True)
    if any((root / name).exists() for name in ("rank_contract.json", "RANK_CONTRACT_READY.json")):
        raise RuntimeError("Rank contract already exists; no overwrite or implicit retry")
    _parent, _groups, _rows, pins, distribution = upstream_inputs(root)
    directories = {"rank": Path(__file__).resolve().parent,
                   "native": Path(args.native_tools).resolve(strict=True),
                   "pose": Path(args.pose_tools).resolve(strict=True)}
    for name, required in (("native", "model_api.py"), ("pose", "pose_quality.py")):
        if not (directories[name] / required).is_file():
            raise RuntimeError(f"Missing explicit {name} API: {required}")
    code = {str(path.resolve()): sha(path) for directory in directories.values() for path in sorted(directory.glob("*.py"))}
    contract = {"version": VERSION, "recipe": RECIPE, "counts": COUNTS, "total_poses": 7120,
                "feature_names": FEATURE_NAMES, "native_tools": str(directories["native"]),
                "pose_tools": str(directories["pose"]), "pinned_files": {str(path): digest for path, digest in pins.items()},
                "code_files_sha256": code, "root_contract_sha256": sha(root / "contract.json"),
                "source_recipe_directory": "implementation/surfna_v2_mdn_dev/iterations/mdn_rank_k8_v3_20260829",
                "source_recipe_sha256": SOURCE_RECIPE, "old_V3_model_cache_or_statistics_reused": False,
                "source_recipe_note": "Algorithm provenance only; all NA model fitting, reference density, K8 features and rank statistics are new L2 outputs.",
                "good_pose_count_histograms": distribution, "drop_zero_good_groups": False,
                "test_opened": False, "baseline_absolute_tolerance": .0005,
                "native_static_bundle_sha256": sha(root / "native/native_prior_static_bundle.pt"),
                "native_head_sha256": sha(root / "native/best_head.pt")}
    atomic_json(root / "rank_contract.json", contract, new=True)
    atomic_json(root / "RANK_CONTRACT_READY.json", {"status": "PASS", "rank_contract_sha256": sha(root / "rank_contract.json"),
                                                    "root_contract_sha256": contract["root_contract_sha256"],
                                                    "test_opened": False}, new=True)
    event("rank_contract_ready", counts=COUNTS, poses=7120, old_V3_statistics_used=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--native-tools", required=True)
    parser.add_argument("--pose-tools", required=True)
    main(parser.parse_args())
