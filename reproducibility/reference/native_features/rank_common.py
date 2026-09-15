"""Fail-closed L2-only contracts and artifact IO; no torch/model imports here."""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
VERSION = "surfna-l2-native-mdn-b-rank-k8-v1"
COUNTS = {"train": 801, "val": 89}
SPLIT_SHAS = {"train": "fbf47922f7bb99e19c795e7957424fc3a82aea19e64bd14f8b1e96c5d30e35ba",
              "val": "cb9ca586abe37e29fc8ab4036528c1a4d8427e52cf6a045646266769d3147a23"}
SEED = 20260829
FEATURE_NAMES = [f"{kind}_neighbor_{i}" for kind in ("distance", "logp", "mixture_mean", "mixture_sd") for i in range(8)]
RECIPE = {"arm": "protein_mdn_transfer", "feature_dimension": 32, "nearest_surface_vertices": 8,
          "candidate_poses": 8, "trainable_parameters": 4225, "epochs": 30, "min_epochs": 5,
          "patience": 5, "batch_groups": 16, "lr": .001, "weight_decay": .001,
          "residual_l2": .01, "residual_cap": 2., "dropout": .1, "hidden_dim": 32,
          "pair_gap": .25, "pair_weight": "min(RMSD_gap/2,1) * (2 if crossing_2A else 1)",
          "gradient_clip_norm": 1., "seed": SEED, "input_clip": 8,
          "selection": ["val_top1_lt2_max", "val_regret_min", "val_ndcg_max"],
          "baseline_selectable": True, "baseline_epoch": -1,
          "baseline": "nearest8_surface_mean_log_likelihood / train_score_scale",
          "statistics_fit": "new_L2_train_only_complex_balanced",
          "drop_zero_good_groups": False, "test_opened": False}
SOURCE_RECIPE = {
    "head.py": "7e00503016f58583f4194a3777e5dbe446ef5f8e0215cb7d7bb119b86f492267",
    "train_head.py": "121fb15d09fc16768ef9eb8fd3c78ab02115abe7ac93f1306186c7f6bf5a9bd7",
    "extract_features.py": "2e182220ef8e8111a805a509dcfd9122397fb9831266420eb9b6c3233ebdc8c5",
    "rank_common.py": "881199e2389ee477a31c73a65a8d19f107108559acdbb492286ddeb5daa1e849",
    "reports_contract.json": "30f2f6a981be1724a11c8a92a9693689ae2c6c622eab4cf5453eac6400df96c8"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_hash(path, expected):
    if not re.fullmatch(r"[a-f0-9]{64}", str(expected)) or sha(path) != expected:
        raise RuntimeError(f"Pinned artifact SHA-256 mismatch: {path}")


def read_json(path):
    return json.loads(Path(path).read_text())


def key(name):
    return hashlib.sha256(name.encode()).hexdigest()[:24]


def safe_path(root, path):
    root = Path(root).resolve(strict=True)
    path = Path(path)
    path = (path if path.is_absolute() else root / path).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise RuntimeError(f"Artifact is not a regular file inside its new L2 stage: {path}")
    return path


def read_ids(path):
    values = [value.strip() for value in Path(path).read_text().splitlines() if value.strip()]
    if not values or len(values) != len(set(values)):
        raise RuntimeError("Split membership empty or duplicated")
    return values


def atomic_json(path, payload, new=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if new:
        with path.open("x") as stream:
            stream.write(raw)
        return
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("x") as stream:
        stream.write(raw)
    os.replace(tmp, path)


def atomic_torch(path, value):
    import torch
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    if tmp.exists():
        raise RuntimeError("Unresolved previous temporary checkpoint; no implicit retry")
    torch.save(value, tmp)
    os.replace(tmp, path)


def event(stage, **kwargs):
    print(json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), stage=stage, **kwargs), allow_nan=False), flush=True)


def parent_inputs(root):
    root = Path(root).resolve(strict=True)
    ready = read_json(root / "CONTRACT_READY.json")
    if ready.get("status") != "PASS":
        raise RuntimeError("Parent experiment contract is not ready")
    check_hash(root / "contract.json", ready["contract_sha256"])
    contract = read_json(root / "contract.json")
    data = contract["dataset"]
    ids = {}
    for split in COUNTS:
        spec = data[split]
        if spec["sha256"] != SPLIT_SHAS[split] or spec["count"] != COUNTS[split]:
            raise RuntimeError("Not the approved fixed L2 801/89 split")
        check_hash(spec["path"], spec["sha256"])
        ids[split] = read_ids(spec["path"])
        if len(ids[split]) != COUNTS[split]:
            raise RuntimeError("Frozen L2 split count mismatch")
    if set(ids["train"]) & set(ids["val"]):
        raise RuntimeError("Train/validation membership overlap")
    check_hash(data["membership_path"], data["membership_sha256"])
    check_hash(data["manifest_path"], data["manifest_sha256"])
    manifest = read_json(data["manifest_path"])
    if (manifest.get("policy_id") != "surfna_l2_baseqc_seq70_rm075_cov080_v1"
            or manifest.get("c2_used") is not True or manifest.get("membership_ready") is not True):
        raise RuntimeError("L2 scientific membership gate is not ready")
    return contract, ids


def validate_groups(groups, split_ids):
    if not isinstance(groups, list) or len(groups) != sum(COUNTS.values()):
        raise RuntimeError("K8 pool must retain all 890 L2 train/val groups")
    expected = {name: split for split, names in split_ids.items() for name in names}
    seen, pose_uids, group_splits, pdb_splits = set(), set(), {}, {}
    distribution = {split: [0] * 9 for split in COUNTS}
    for group in groups:
        name, split = group["name"], group["split"]
        if name in seen or expected.get(name) != split or split not in COUNTS:
            raise RuntimeError("K8 group is duplicated, omitted, held-out, or assigned to the wrong split")
        seen.add(name)
        for field, assignments in (("split_group", group_splits), ("parent_pdb", pdb_splits)):
            value = group[field]
            if not value or (value in assignments and assignments[value] != split):
                raise RuntimeError(f"Empty or train/val-overlapping {field}")
            assignments[value] = split
        poses = group["poses"]
        if len(poses) != 8 or sorted(p["ordinal"] for p in poses) != list(range(8)):
            raise RuntimeError("Every group must preserve exactly ordinals 0..7")
        signatures = set()
        for pose in poses:
            if (type(pose["ordinal"]) is not int or pose["pose_uid"] in pose_uids
                    or not isinstance(pose["pose_uid"], str) or not pose["pose_uid"]):
                raise RuntimeError("Duplicate/invalid candidate identity or ordinal")
            pose_uids.add(pose["pose_uid"])
            rmsd = pose["rmsd"]
            if isinstance(rmsd, bool) or not isinstance(rmsd, (int, float)) or not math.isfinite(rmsd) or rmsd < 0:
                raise RuntimeError("Every RMSD label must be finite and nonnegative")
            if pose["rmsd_method"] != "symmetry_aware_no_alignment" or pose["sampling_seed"] != 20260826:
                raise RuntimeError("Candidate metric/generation protocol drift")
            if type(pose["atom_count"]) is not int or pose["atom_count"] <= 0:
                raise RuntimeError("Invalid candidate heavy-atom count")
            if not re.fullmatch(r"[a-f0-9]{64}", pose["atom_order_sha256"]):
                raise RuntimeError("Missing ordered heavy-atom graph signature")
            signatures.add((pose["atom_order_sha256"], pose["atom_count"]))
        if len(signatures) != 1:
            raise RuntimeError("Candidate atom identity/order differs within one target")
        distribution[split][sum(p["rmsd"] < 2 for p in poses)] += 1
    if seen != set(expected) or len(pose_uids) != 7120:
        raise RuntimeError("Incomplete 801/89 groups or 7120 candidates; no subset/curation allowed")
    return distribution


def upstream_inputs(root):
    root = Path(root).resolve(strict=True)
    parent, split_ids = parent_inputs(root)
    parent_sha = sha(root / "contract.json")
    pose_ready = read_json(root / "poses/POSE_READY.json")
    native_ready = read_json(root / "native/READY.json")
    for label, ready in (("poses", pose_ready), ("native", native_ready)):
        if ready.get("status") != "PASS" or ready.get("test_opened") is not False or ready["split_counts"] != COUNTS:
            raise RuntimeError(f"{label} is not a complete test-blind L2 801/89 stage")
    if (pose_ready["contract_sha256"] != parent_sha or native_ready["root_contract_sha256"] != parent_sha
            or pose_ready["groups"] != 890 or pose_ready["poses"] != 7120
            or pose_ready["train_val_split_group_overlap"] != 0):
        raise RuntimeError("Upstream ready gates disagree with the root experiment contract")
    if pose_ready["generator_checkpoint_sha256"] != parent["generator"]["checkpoint_sha256"]:
        raise RuntimeError("K8 pool was not generated by the frozen generator")
    for split in COUNTS:
        if pose_ready[f"{split}_split_sha256"] != SPLIT_SHAS[split]:
            raise RuntimeError("K8 pool uses another dataset split")
    pins = {root / "contract.json": parent_sha, root / "CONTRACT_READY.json": sha(root / "CONTRACT_READY.json"),
            root / "native_contract.json": native_ready["contract_sha256"],
            root / "poses/groups.json": pose_ready["groups_sha256"],
            root / "native/best_head.pt": native_ready["best_head_sha256"],
            root / "native/native_prior_static_bundle.pt": native_ready["bundle_sha256"],
            root / "native/graph_index.json": native_ready["graph_index_sha256"]}
    for path, expected in pins.items():
        check_hash(path, expected)
    for relative in ("poses/POSE_READY.json", "native/READY.json", "native/TRAIN_READY.json",
                     "native/native_prior_static_bundle.json", "NATIVE_CONTRACT_READY.json",
                     "native/PREPARED_READY.json", "native/native_train_reference.json"):
        pins[root / relative] = sha(root / relative)
    preparation = read_json(root / "native/PREPARED_READY.json")
    native_contract_ready = read_json(root / "NATIVE_CONTRACT_READY.json")
    if (native_contract_ready.get("status") != "PASS"
            or native_contract_ready["contract_sha256"] != native_ready["contract_sha256"]
            or preparation.get("status") != "PASS" or preparation["split_counts"] != COUNTS
            or preparation["contract_sha256"] != native_ready["contract_sha256"]
            or preparation["root_contract_sha256"] != parent_sha
            or preparation.get("test_opened") is not False
            or preparation["graph_index_sha256"] != native_ready["graph_index_sha256"]):
        raise RuntimeError("Native preparation contract does not match the trained MDN")
    check_hash(root / "native/native_train_reference.json", preparation["reference_sha256"])
    reference = read_json(root / "native/native_train_reference.json")
    if (reference.get("fit_split") != "train" or reference["n_complexes"] != COUNTS["train"]
            or reference["source_sha256"] != native_ready["graph_index_sha256"]
            or reference["train_split_sha256"] != SPLIT_SHAS["train"]
            or reference.get("validation_used") is not False or reference.get("test_opened") is not False):
        raise RuntimeError("Native reference was not fitted only on this new L2 training set")
    for split in COUNTS:
        spec = parent["dataset"][split]
        pins[Path(spec["path"])] = spec["sha256"]
    for key_name in ("membership", "manifest"):
        pins[Path(parent["dataset"][f"{key_name}_path"])] = parent["dataset"][f"{key_name}_sha256"]
    groups = read_json(root / "poses/groups.json")
    distribution = validate_groups(groups, split_ids)
    index = read_json(root / "native/graph_index.json")
    if (index["contract_sha256"] != native_ready["contract_sha256"] or index["root_contract_sha256"] != parent_sha
            or index["split_counts"] != COUNTS):
        raise RuntimeError("Prepared native graph index belongs to another experiment")
    rows = {row["name"]: row for row in index["entries"]}
    if len(rows) != 890 or len(index["entries"]) != len(rows) or set(rows) != {group["name"] for group in groups}:
        raise RuntimeError("Native graph index does not exactly cover the new K8 pool")
    for group in groups:
        row = rows[group["name"]]
        if row["split"] != group["split"] or row["ligand_atoms"] != group["poses"][0]["atom_count"]:
            raise RuntimeError("Native graph identity/split/atom count mismatch")
        graph_path = safe_path(root / "native", row["path"])
        pins[graph_path] = row["artifact_sha256"]
        for pose in group["poses"]:
            pins[safe_path(root / "poses", pose["pose_path"])] = pose["pose_file_sha256"]
    for path, expected in pins.items():
        check_hash(path, expected)
    return parent, groups, rows, pins, distribution


def validate_contract(root):
    root = Path(root).resolve(strict=True)
    ready = read_json(root / "RANK_CONTRACT_READY.json")
    check_hash(root / "rank_contract.json", ready["rank_contract_sha256"])
    contract = read_json(root / "rank_contract.json")
    if (ready.get("status") != "PASS" or contract.get("version") != VERSION
            or contract["recipe"] != RECIPE or contract["counts"] != COUNTS
            or contract["feature_names"] != FEATURE_NAMES or contract["total_poses"] != 7120):
        raise RuntimeError("Fixed MDN-B rank recipe was changed")
    for filename, expected in contract["pinned_files"].items():
        check_hash(filename, expected)
    for filename, expected in contract["code_files_sha256"].items():
        check_hash(filename, expected)
    _parent, groups, rows, _pins, distribution = upstream_inputs(root)
    if distribution != contract["good_pose_count_histograms"]:
        raise RuntimeError("The full, uncurated K8 label population changed")
    return contract, groups, rows


def load_api(contract, name):
    directory = Path(contract[f"{name}_tools"])
    filename = "model_api.py" if name == "native" else "pose_quality.py"
    path = (directory / filename).resolve(strict=True)
    check_hash(path, contract["code_files_sha256"][str(path)])
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(f"surfna_l2_rank_{name}_api", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def reserve_stage(root, stage):
    root = Path(root).resolve(strict=True)
    directory = root / "rank" / stage
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / "STARTED.json", {"stage": stage, "pid": os.getpid(), "time": time.time(),
                                            "automatic_retry": False, "test_opened": False}, new=True)
    return directory


def validate_features(root, contract, groups):
    """Validate bytes/identities without importing a model or reading test assets."""
    root = Path(root).resolve(strict=True)
    directory = root / "rank/features"
    ready = read_json(directory / "FEATURES_READY.json")
    if (ready.get("status") != "PASS" or ready["rank_contract_sha256"] != sha(root / "rank_contract.json")
            or ready["split_counts"] != COUNTS or ready["groups"] != 890 or ready["poses"] != 7120
            or ready.get("labels_in_feature_cache") is not False or ready.get("test_opened") is not False
            or ready["source_bundle_sha256"] != contract["native_static_bundle_sha256"]
            or ready["source_head_sha256"] != contract["native_head_sha256"]
            or ready["candidate_ligand_encoder_calls"] != 7120
            or ready["surface_encoder_calls"] != 890 or ready["old_V3_cache_used"] is not False):
        raise RuntimeError("Candidate-only feature stage is not ready or violates the fixed recipe")
    for filename, field in (("feature_index.json", "feature_index_sha256"),
                            ("baseline_metrics.json", "baseline_metrics_sha256")):
        check_hash(directory / filename, ready[field])
    entries = read_json(directory / "feature_index.json")
    expected = {g["name"]: g for g in groups}
    if (len(entries) != 890 or len({row["name"] for row in entries}) != 890
            or {row["name"] for row in entries} != set(expected)):
        raise RuntimeError("Feature cache is not exactly all 890 approved targets")
    for row in entries:
        if row["split"] != expected[row["name"]]["split"]:
            raise RuntimeError("Feature cache split mismatch")
        check_hash(safe_path(directory, row["path"]), row["sha256"])
    return ready, entries
