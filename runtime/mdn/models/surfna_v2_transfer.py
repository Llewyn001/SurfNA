"""Explicit, auditable diffusion-to-scorer weight transfer for SurfNA V2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

import torch


STATIC_PREFIXES = (
    "lig_node_embedding",
    "lig_edge_embedding",
    "lig_conv_layers",
    "rec_node_embedding",
    "rec_edge_embedding",
    "rec_conv_layers",
    "surface_node_embedding",
    "surface_edge_embedding",
    "surface_rec_cross_edge_embedding",
    "residue_to_surface_conv_layers",
    "surface_conv_layers",
    "lig_distance_expansion",
    "rec_distance_expansion",
    "surface_distance_expansion",
    "cross_distance_expansion",
)

DYNAMIC_PREFIXES = (
    "cross_edge_embedding",
    "surface_to_lig_conv_layers",
    "lig_to_surface_conv_layers",
    "ligand_patch_tokenizer",
)

NA_PREFIXES = (
    "nusurf_fusion",
    "nusurf_refine_blocks",
    "nuc_feat_proj",
    "nuc_feat_gate",
)

INTENTIONALLY_EXCLUDED_PREFIXES = (
    "tr_final_layer",
    "rot_final_layer",
    "tor_final_layer",
    "final_conv",
    "final_edge_embedding",
    "final_tp_tor",
    "tor_bond_conv",
    "center_distance_expansion",
    "center_edge_embedding",
    "pretrain_",
    "task_",
    "queue",
)


@dataclass
class TransferReport:
    exact_loaded: list[str] = field(default_factory=list)
    time_columns_projected: list[str] = field(default_factory=list)
    na_adapter_initialized: list[str] = field(default_factory=list)
    new_head_initialized: list[str] = field(default_factory=list)
    intentionally_excluded: list[str] = field(default_factory=list)
    unexpected_or_shape_error: dict[str, str] = field(default_factory=dict)
    prefix_coverage: dict[str, dict[str, float]] = field(default_factory=dict)
    prior_only: bool = False
    na_required_tensors: int = 0
    na_loaded_tensors: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProteinMDNTransferReport:
    """Audit record for the shared protein-scorer initialization layer."""

    shared_static_loaded: list[str] = field(default_factory=list)
    prior_head_loaded: list[str] = field(default_factory=list)
    reference_prior_preserved: list[str] = field(default_factory=list)
    unexpected_or_shape_error: dict[str, str] = field(default_factory=dict)
    prefix_coverage: dict[str, dict[str, float]] = field(default_factory=dict)
    source_epoch: int | None = None
    source_selection_key: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _strip_wrappers(key: str) -> str:
    for prefix in ("module.", "model."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def extract_state_dict(checkpoint: Mapping) -> dict[str, torch.Tensor]:
    state = checkpoint
    for candidate in ("model", "state_dict", "model_state_dict"):
        if isinstance(state, Mapping) and candidate in state and isinstance(state[candidate], Mapping):
            state = state[candidate]
            break
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint does not contain a state dictionary")
    return {_strip_wrappers(str(key)): value for key, value in state.items() if torch.is_tensor(value)}


def load_protein_mdn_scorer_transfer(
    target_model: torch.nn.Module,
    checkpoint: Mapping,
) -> ProteinMDNTransferReport:
    """Overlay a trained protein MDN prior on either NA scorer variant.

    Only the shared static surface/ligand backbone and the trainable MDN head
    are transferred.  The target's NA reference density is deliberately kept:
    it is fitted on native contacts from the NA training split and is not a
    learned protein parameter.  Dynamic G2.2/NuSurf modules already loaded
    into variant B are likewise preserved.
    """

    source_state = extract_state_dict(checkpoint)
    target_state = target_model.state_dict()
    report = ProteinMDNTransferReport()
    if isinstance(checkpoint, Mapping):
        if checkpoint.get("epoch") is not None:
            report.source_epoch = int(checkpoint["epoch"])
        selection = checkpoint.get("selection_key")
        if isinstance(selection, (list, tuple)):
            report.source_selection_key = [float(value) for value in selection]

    reference_keys = [
        key for key in target_state if key.startswith("prior_head.reference_prior.")
    ]
    reference_before = {key: target_state[key].detach().clone() for key in reference_keys}
    required_static = [
        key
        for key in target_state
        if key.startswith("backbone.")
        and _belongs(_canonical_target_key(key), STATIC_PREFIXES)
    ]
    required_prior = [
        key
        for key in target_state
        if key.startswith("prior_head.")
        and not key.startswith("prior_head.reference_prior.")
    ]
    loadable: dict[str, torch.Tensor] = {}
    for family, required, loaded in (
        ("shared_static", required_static, report.shared_static_loaded),
        ("prior_head", required_prior, report.prior_head_loaded),
    ):
        for key in required:
            source_tensor = source_state.get(key)
            if source_tensor is None:
                report.unexpected_or_shape_error[key] = f"{family}: missing from protein scorer"
                continue
            if source_tensor.shape != target_state[key].shape:
                report.unexpected_or_shape_error[key] = (
                    f"{family}: source_shape={tuple(source_tensor.shape)} "
                    f"target_shape={tuple(target_state[key].shape)}"
                )
                continue
            if not bool(torch.isfinite(source_tensor).all()):
                report.unexpected_or_shape_error[key] = f"{family}: source is non-finite"
                continue
            loadable[key] = source_tensor.to(dtype=target_state[key].dtype)
            loaded.append(key)

    if report.unexpected_or_shape_error:
        raise RuntimeError(
            "protein MDN scorer transfer coverage gate failed: "
            f"{report.unexpected_or_shape_error}"
        )
    incompatible = target_model.load_state_dict(loadable, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"protein MDN transfer constructed unexpected keys: {incompatible.unexpected_keys}"
        )
    target_after = target_model.state_dict()
    for key, before in reference_before.items():
        if not torch.equal(before, target_after[key]):
            raise RuntimeError(f"protein MDN transfer overwrote NA reference prior buffer: {key}")
        report.reference_prior_preserved.append(key)

    loaded_keys = set(loadable)
    report.prefix_coverage = _coverage(
        list(target_state), loaded_keys, STATIC_PREFIXES
    )
    failed_static = {
        prefix: stats
        for prefix, stats in report.prefix_coverage.items()
        if stats["required_tensors"] > 0 and stats["ratio"] < 1.0
    }
    if failed_static or len(report.prior_head_loaded) != len(required_prior):
        raise RuntimeError(
            "protein MDN scorer transfer did not fully cover shared static/prior tensors: "
            f"static={failed_static} prior={len(report.prior_head_loaded)}/{len(required_prior)}"
        )
    return report


def _belongs(key: str, prefixes: tuple[str, ...]) -> bool:
    return any(
        key.startswith(prefix)
        if prefix.endswith("_")
        else key == prefix or key.startswith(prefix + ".")
        for prefix in prefixes
    )


def _canonical_target_key(key: str) -> str:
    return key[len("backbone.") :] if key.startswith("backbone.") else key


def _project_time_columns(
    key: str,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    distance_embed_dim: int,
) -> torch.Tensor | None:
    """Remove diffusion-time columns while preserving physical/RBF columns."""
    if source.ndim != 2 or target.ndim != 2 or source.shape[0] != target.shape[0]:
        return None
    if key == "surface_node_embedding.linear.weight":
        if source.shape[1] > target.shape[1]:
            return source[:, : target.shape[1]]
    if key in {"lig_edge_embedding.0.weight", "surface_edge_embedding.0.weight"}:
        physical_dim = target.shape[1] - int(distance_embed_dim)
        if physical_dim < 0 or source.shape[1] < physical_dim + int(distance_embed_dim):
            return None
        return torch.cat(
            [source[:, :physical_dim], source[:, -int(distance_embed_dim) :]], dim=1
        )
    return None


def _coverage(
    target_keys: list[str],
    loaded_keys: set[str],
    prefixes: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    output = {}
    for prefix in prefixes:
        required = [key for key in target_keys if _belongs(_canonical_target_key(key), (prefix,))]
        loaded = [key for key in required if key in loaded_keys]
        output[prefix] = {
            "required_tensors": float(len(required)),
            "loaded_tensors": float(len(loaded)),
            "ratio": float(len(loaded) / len(required)) if required else 1.0,
        }
    return output


def load_v2_diffusion_backbone(
    target_model: torch.nn.Module,
    checkpoint: Mapping,
    *,
    surface_feature_schema: str,
    expected_surface_feature_schema: str = "v2_full8",
    distance_embed_dim: int = 32,
    require_dynamic_ranker: bool = True,
    require_static_full_coverage: bool = True,
    require_na_transfer: bool = False,
) -> TransferReport:
    """Load only scientifically valid modules and fail on unexplained skips.

    The target dual-branch backbone should intentionally retain diffusion-time
    input shapes for the dynamic cross branch.  Three static input projections
    remove time columns from the ligand edge, surface node and surface edge
    embeddings.  No generic ``strict=False`` success message is allowed.
    """
    if surface_feature_schema != expected_surface_feature_schema:
        raise ValueError(
            f"surface schema mismatch: {surface_feature_schema!r} != "
            f"{expected_surface_feature_schema!r}"
        )
    source_state = extract_state_dict(checkpoint)
    target_state = target_model.state_dict()
    report = TransferReport()
    loadable: dict[str, torch.Tensor] = {}

    allowed_prefixes = STATIC_PREFIXES + DYNAMIC_PREFIXES + NA_PREFIXES
    for key, target_tensor in target_state.items():
        source_key = _canonical_target_key(key)
        if _belongs(source_key, INTENTIONALLY_EXCLUDED_PREFIXES):
            report.intentionally_excluded.append(key)
            continue
        if not _belongs(source_key, allowed_prefixes):
            report.new_head_initialized.append(key)
            continue
        source_tensor = source_state.get(source_key)
        if source_tensor is None:
            if _belongs(source_key, NA_PREFIXES):
                report.na_adapter_initialized.append(key)
            else:
                report.unexpected_or_shape_error[key] = "missing from checkpoint"
            continue
        if source_tensor.shape == target_tensor.shape:
            loadable[key] = source_tensor.to(dtype=target_tensor.dtype)
            report.exact_loaded.append(key)
            continue
        projected = _project_time_columns(
            source_key,
            source_tensor,
            target_tensor,
            distance_embed_dim=distance_embed_dim,
        )
        if projected is not None and projected.shape == target_tensor.shape:
            loadable[key] = projected.to(dtype=target_tensor.dtype)
            report.time_columns_projected.append(key)
        else:
            report.unexpected_or_shape_error[key] = (
                f"source_shape={tuple(source_tensor.shape)} target_shape={tuple(target_tensor.shape)}"
            )

    for key in source_state:
        if _belongs(key, INTENTIONALLY_EXCLUDED_PREFIXES):
            report.intentionally_excluded.append(key)

    incompatible = target_model.load_state_dict(loadable, strict=False)
    unexpected_loaded = list(incompatible.unexpected_keys)
    if unexpected_loaded:
        raise RuntimeError(f"transfer constructed unexpected target keys: {unexpected_loaded}")

    loaded_keys = set(loadable)
    target_keys = list(target_state)
    report.prefix_coverage = _coverage(
        target_keys,
        loaded_keys,
        STATIC_PREFIXES + DYNAMIC_PREFIXES + NA_PREFIXES,
    )
    report.na_required_tensors = int(
        sum(
            stats["required_tensors"]
            for prefix, stats in report.prefix_coverage.items()
            if prefix in NA_PREFIXES
        )
    )
    report.na_loaded_tensors = int(
        sum(
            stats["loaded_tensors"]
            for prefix, stats in report.prefix_coverage.items()
            if prefix in NA_PREFIXES
        )
    )
    dynamic_required = [
        prefix
        for prefix in DYNAMIC_PREFIXES
        if report.prefix_coverage[prefix]["required_tensors"] > 0
    ]
    report.prior_only = any(report.prefix_coverage[prefix]["ratio"] < 1.0 for prefix in dynamic_required)

    if require_static_full_coverage:
        failed_static = {
            prefix: stats
            for prefix, stats in report.prefix_coverage.items()
            if prefix in STATIC_PREFIXES
            and stats["required_tensors"] > 0
            and stats["ratio"] < 1.0
        }
        if failed_static:
            raise RuntimeError(f"static backbone transfer coverage gate failed: {failed_static}")
    if require_dynamic_ranker:
        missing_dynamic_module = [
            prefix
            for prefix in DYNAMIC_PREFIXES
            if report.prefix_coverage[prefix]["required_tensors"] == 0
        ]
        failed_dynamic = {
            prefix: report.prefix_coverage[prefix]
            for prefix in dynamic_required
            if report.prefix_coverage[prefix]["ratio"] < 1.0
        }
        if missing_dynamic_module or failed_dynamic:
            raise RuntimeError(
                "dynamic ranker transfer gate failed; "
                f"missing_target_modules={missing_dynamic_module} incomplete={failed_dynamic}"
            )
    if require_na_transfer:
        failed_na = {
            prefix: stats
            for prefix, stats in report.prefix_coverage.items()
            if prefix in NA_PREFIXES
            and stats["required_tensors"] > 0
            and stats["ratio"] < 1.0
        }
        if report.na_required_tensors <= 0:
            raise RuntimeError(
                "NA transfer gate failed: target was not instantiated with any NuSurf/nucleic tensors"
            )
        if report.na_loaded_tensors != report.na_required_tensors or failed_na:
            raise RuntimeError(
                "NA transfer gate failed: G2.2 NuSurf/nucleic weights are incomplete; "
                f"required={report.na_required_tensors} loaded={report.na_loaded_tensors} "
                f"incomplete={failed_na}"
            )
    unexplained = {
        key: reason
        for key, reason in report.unexpected_or_shape_error.items()
        if not _belongs(_canonical_target_key(key), NA_PREFIXES)
    }
    if unexplained:
        raise RuntimeError(f"unexplained diffusion transfer skips: {unexplained}")
    return report
