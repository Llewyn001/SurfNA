"""Traceable, group-aware pose dataset for the SurfNA V2 scorer."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import torch

from datasets.mdn_rerank import DecoyPDBBind


POSE_SOURCE_ID = {
    "native": 0,
    "native_perturb": 1,
    "generator": 2,
    "hard_negative": 3,
}
STRATUM_ID = {
    "native": 0,
    "near": 1,
    "plausible": 2,
    "far": 3,
    "very_far": 4,
    "clash": 5,
}

REQUIRED_RUNTIME_FIELDS = {
    "manifest_version",
    "pose_uid",
    "complex_name",
    "split",
    "pose_group_uid",
    "candidate_rank_input",
    "pose_source",
    "is_prior_valid",
    "rmsd",
    "pose_sdf",
    "pose_sha256",
    "atom_count",
    "atom_order_sha256",
    "stratum",
}


def stable_group_integer(group_uid: str) -> int:
    """Map a stable string UID to a positive signed-int64-compatible value."""
    return int(hashlib.sha256(group_uid.encode("utf-8")).hexdigest()[:15], 16)


def _parse_bool(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes"}:
        return True
    if lowered in {"0", "false", "no"}:
        return False
    raise ValueError(f"not a boolean: {value!r}")


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def align_legacy_nucleic_anchor_coordinates(complex_graph) -> bool:
    """Center legacy NA anchors when the cache stored absolute PDB coordinates.

    The cache receptor, ligand and surface positions are receptor-centered, but
    historical custom anchor tensors can remain in the original PDB frame.  We
    compare both representations against each anchor's parent receptor node and
    only subtract ``original_center`` when it improves the median parent error
    by more than one Angstrom.  The margin makes this idempotent.

    Returns ``True`` when a correction was applied.  Graphs without NA anchors
    are left untouched; partially populated NA anchor records fail closed.
    """
    if "receptor" not in getattr(complex_graph, "node_types", []):
        return False
    receptor = complex_graph["receptor"]
    anchor_required = (
        "nucleic_anchor_pos",
        "nucleic_anchor_parent_index",
        "nucleic_anchor_type",
    )
    present = [hasattr(receptor, attr) for attr in anchor_required]
    if not any(present):
        return False
    if not all(present):
        missing = [
            attr for attr, exists in zip(anchor_required, present) if not exists
        ]
        raise ValueError(f"incomplete nucleic anchor record; missing {missing}")
    if not hasattr(receptor, "pos"):
        raise ValueError("nucleic anchor record requires receptor.pos")

    anchor_pos = receptor.nucleic_anchor_pos
    parent_index = receptor.nucleic_anchor_parent_index
    anchor_type = receptor.nucleic_anchor_type
    receptor_pos = receptor.pos
    original_center = getattr(complex_graph, "original_center", None)
    tensors = (anchor_pos, parent_index, anchor_type, receptor_pos)
    if not all(torch.is_tensor(value) for value in tensors):
        raise ValueError("nucleic anchor record must contain tensors")
    if anchor_pos.ndim != 2 or anchor_pos.shape[-1] != 3:
        raise ValueError("nucleic_anchor_pos must have shape [A,3]")
    if anchor_pos.shape[0] != parent_index.numel() or anchor_pos.shape[0] != anchor_type.numel():
        raise ValueError("nucleic anchor tensors have inconsistent row counts")
    if anchor_pos.shape[0] == 0:
        return False
    if not torch.is_tensor(original_center):
        raise ValueError("legacy nucleic anchors require original_center for frame audit")

    anchor_pos = anchor_pos.float()
    receptor_pos = receptor_pos.float()
    original_center = original_center.float()
    if not bool(torch.isfinite(anchor_pos).all()):
        raise ValueError("nucleic_anchor_pos contains non-finite coordinates")
    if not bool(torch.isfinite(receptor_pos).all()):
        raise ValueError("receptor.pos contains non-finite coordinates")
    if not bool(torch.isfinite(original_center).all()):
        raise ValueError("original_center contains non-finite coordinates")
    parent_index = parent_index.long().view(-1)
    valid = (parent_index >= 0) & (parent_index < receptor_pos.shape[0])
    if not bool(valid.all()):
        raise ValueError("nucleic anchor parent index is out of receptor range")
    if original_center.numel() != 3:
        raise ValueError("original_center must contain exactly three coordinates")
    center = original_center.to(anchor_pos).reshape(3)
    centered_candidate = anchor_pos - center
    parent_pos = receptor_pos[parent_index]
    current_error = torch.linalg.vector_norm(anchor_pos - parent_pos, dim=-1).median()
    centered_error = torch.linalg.vector_norm(
        centered_candidate - parent_pos, dim=-1
    ).median()
    if bool(centered_error + 1.0 < current_error):
        receptor.nucleic_anchor_pos = centered_candidate
        complex_graph.nucleic_anchor_parent_error_before_A = current_error.view(1)
        complex_graph.nucleic_anchor_parent_error_after_A = centered_error.view(1)
        return True
    receptor.nucleic_anchor_pos = anchor_pos
    complex_graph.nucleic_anchor_parent_error_before_A = current_error.view(1)
    complex_graph.nucleic_anchor_parent_error_after_A = current_error.view(1)
    return False


class DecoyPDBBindV2(DecoyPDBBind):
    """V2 manifest reader with stable group IDs and provenance attributes.

    A passing report from ``scripts/audit_mdn_v2_manifest.py`` is required by
    default.  This makes it difficult to accidentally train on a historical
    smoke manifest, fixed-test overlap, or a group without near-native poses.
    """

    def __init__(
        self,
        manifest_path,
        base_split_path,
        root,
        *,
        gate_report_path=None,
        require_gate_report=True,
        expected_profile="curated",
        **kwargs,
    ):
        self.gate_report_path = gate_report_path
        self.require_gate_report = bool(require_gate_report)
        self.expected_profile = expected_profile
        self._verify_gate_report(manifest_path)
        super().__init__(
            manifest_path=manifest_path,
            base_split_path=base_split_path,
            root=root,
            **kwargs,
        )
        self.pose_group_uids = tuple(row["pose_group_uid"] for row in self.rows)
        self.stable_group_ids = tuple(stable_group_integer(uid) for uid in self.pose_group_uids)
        self.expected_group_counts = dict(Counter(self.pose_group_uids))

    def _verify_gate_report(self, manifest_path: str) -> None:
        if not self.gate_report_path:
            if self.require_gate_report:
                raise ValueError("V2 scorer dataset requires --mdn_manifest_gate_report")
            return
        with open(self.gate_report_path) as handle:
            report = json.load(handle)
        if report.get("status") != "pass":
            raise ValueError(f"manifest gate report did not pass: {self.gate_report_path}")
        if report.get("profile") != self.expected_profile:
            raise ValueError(
                f"manifest profile mismatch: expected={self.expected_profile} got={report.get('profile')}"
            )
        actual_hash = _file_sha256(manifest_path)
        if report.get("manifest_sha256") != actual_hash:
            raise ValueError("manifest changed after audit; SHA256 no longer matches gate report")

    @staticmethod
    def _read_manifest(path, limit_complexes):
        rows = []
        seen_complexes = []
        seen_complex_set = set()
        pose_uids = set()
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            missing = REQUIRED_RUNTIME_FIELDS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"manifest V2 missing fields: {sorted(missing)}")
            for row in reader:
                name = row["complex_name"]
                if limit_complexes and name not in seen_complex_set and len(seen_complexes) >= limit_complexes:
                    continue
                if name not in seen_complex_set:
                    seen_complex_set.add(name)
                    seen_complexes.append(name)
                if row["manifest_version"] != "v2":
                    raise ValueError(f"unsupported manifest version {row['manifest_version']!r}")
                if row["pose_uid"] in pose_uids:
                    raise ValueError(f"duplicate pose_uid {row['pose_uid']}")
                pose_uids.add(row["pose_uid"])
                if row["pose_source"] not in POSE_SOURCE_ID:
                    raise ValueError(f"unknown pose_source {row['pose_source']!r}")
                if row["stratum"] not in STRATUM_ID:
                    raise ValueError(f"unknown stratum {row['stratum']!r}")
                _parse_bool(row["is_prior_valid"])
                if not os.path.exists(row["pose_sdf"]):
                    raise ValueError(f"missing pose SDF {row['pose_sdf']}")
                row["sample_idx"] = row["candidate_rank_input"]
                rows.append(row)
        if not rows:
            raise ValueError(f"no usable V2 decoy rows found in {path}")
        rows.sort(key=lambda row: (row["pose_group_uid"], int(row["candidate_rank_input"])))
        return rows

    def get(self, idx):
        graph = super().get(idx)
        alignment_applied = align_legacy_nucleic_anchor_coordinates(graph)
        graph.nucleic_anchor_alignment_applied = torch.tensor(
            [alignment_applied], dtype=torch.bool
        )
        row = self.rows[idx]
        group_uid = row["pose_group_uid"]
        graph.decoy_group_id = torch.tensor([stable_group_integer(group_uid)], dtype=torch.long)
        graph.decoy_candidate_rank_input = torch.tensor(
            [int(row["candidate_rank_input"])], dtype=torch.long
        )
        graph.decoy_pose_source_id = torch.tensor(
            [POSE_SOURCE_ID[row["pose_source"]]], dtype=torch.long
        )
        graph.decoy_stratum_id = torch.tensor([STRATUM_ID[row["stratum"]]], dtype=torch.long)
        graph.decoy_is_prior_valid = torch.tensor(
            [_parse_bool(row["is_prior_valid"])], dtype=torch.bool
        )
        graph.decoy_pose_uid = row["pose_uid"]
        graph.decoy_pose_group_uid = group_uid
        graph.decoy_pose_source = row["pose_source"]
        graph.decoy_stratum = row["stratum"]
        graph.decoy_pose_sha256 = row["pose_sha256"]
        graph.decoy_atom_order_sha256 = row["atom_order_sha256"]
        graph.decoy_split = row["split"]
        return graph
