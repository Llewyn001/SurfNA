"""Concrete SurfNA V2 scorer backbone refactored from the diffusion model.

The diffusion modules are reused at a fixed clean timestep (t=0).  The static
path excludes every ligand-surface cross operation; the dynamic path starts
from cached target/ligand base states and runs the transferred patch and cross
layers.  This preserves checkpoint behavior while making information flow
explicit and testable.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch
from e3nn import o3
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_add

_g22_source_root = os.environ.get("SURFNA_V2_G22_SOURCE_ROOT", "").strip()
if _g22_source_root:
    _g22_source_root = str(Path(_g22_source_root).resolve())
    _g22_model_dir = str(Path(_g22_source_root) / "models")
    _g22_surface_model = Path(_g22_model_dir) / "surface_score_model_v3.py"
    if not _g22_surface_model.is_file():
        raise ImportError(
            "SURFNA_V2_G22_SOURCE_ROOT does not contain models/surface_score_model_v3.py: "
            f"{_g22_source_root}"
        )
    # G2.2's surface model imports its sibling ``models.surfna_v2_modules``.
    # Put that exact source directory first in the namespace package before
    # executing the explicitly addressed module.
    import models as _models_package

    if _g22_model_dir not in _models_package.__path__:
        _models_package.__path__.insert(0, _g22_model_dir)
    if _g22_source_root not in sys.path:
        sys.path.insert(0, _g22_source_root)
    # Some dataset modules import ``models.surfna_v2_modules`` before the
    # scorer is constructed.  Merely prepending the namespace path would then
    # silently reuse that cached (main-tree) implementation.  Address and
    # install the G2.2 sibling explicitly so precision/multiscale constructor
    # arguments cannot be resolved against the wrong class definition.
    _g22_components = Path(_g22_model_dir) / "surfna_v2_modules.py"
    _components_spec = importlib.util.spec_from_file_location(
        "models.surfna_v2_modules", _g22_components
    )
    if _components_spec is None or _components_spec.loader is None:
        raise ImportError(f"could not load G2.2 component spec: {_g22_components}")
    _components_module = importlib.util.module_from_spec(_components_spec)
    sys.modules[_components_spec.name] = _components_module
    _components_spec.loader.exec_module(_components_module)
    _spec = importlib.util.spec_from_file_location(
        "models._surfna_v2_g22_surface_score_model_v3", _g22_surface_model
    )
    if _spec is None or _spec.loader is None:
        raise ImportError(f"could not load G2.2 surface model spec: {_g22_surface_model}")
    _g22_module = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _g22_module
    _spec.loader.exec_module(_g22_module)
    TensorProductScoreModel = _g22_module.TensorProductScoreModel
    BACKBONE_SOURCE_FILE = str(_g22_surface_model)
else:
    from models.surface_score_model_v3 import TensorProductScoreModel

    BACKBONE_SOURCE_FILE = str(Path(sys.modules[TensorProductScoreModel.__module__].__file__).resolve())
from models.surfna_v2_scorer import (
    CandidatePoseState,
    LigandIntraState,
    StaticSurfaceState,
    SurfaceInteractionBackboneV2,
)
from utils.diffusion_utils import set_time


# The transferred diffusion layers were trained jointly with ligand-surface
# messages.  A small number of scorer-only/OOD decoys can otherwise drive the
# multiplicative tensor-product stack far outside that training range.  This
# non-parametric, per-graph RMS ceiling is rotation invariant and is inactive
# on ordinary transferred representations (observed RMS is typically < 50).
EQUIVARIANT_GRAPH_RMS_CAP = 100.0


def cap_equivariant_graph_rms(
    node_attr: torch.Tensor,
    batch_index: torch.Tensor,
    *,
    cap: float = EQUIVARIANT_GRAPH_RMS_CAP,
) -> torch.Tensor:
    """Rescale only out-of-range graph states with one invariant scalar.

    A single gain is applied to every node/channel in a graph, so rotations of
    non-scalar irreps commute with the guard.  Inputs must already be finite;
    the guard is preventive and never masks a NaN/Inf produced upstream.
    """
    if node_attr.ndim != 2:
        raise ValueError("equivariant graph state must have shape [N,D]")
    batch_index = batch_index.long().view(-1)
    if batch_index.numel() != node_attr.shape[0]:
        raise ValueError("equivariant graph state and batch index row counts differ")
    if not bool(torch.isfinite(node_attr).all()):
        raise RuntimeError("equivariant graph state is non-finite before RMS guard")
    if not float(cap) > 0.0:
        raise ValueError("equivariant graph RMS cap must be positive")
    if node_attr.numel() == 0:
        return node_attr
    if int(batch_index.min()) < 0:
        raise ValueError("equivariant graph batch index is negative")

    graph_count = int(batch_index.max().item()) + 1
    gains = node_attr.new_ones(graph_count)
    for graph_index in range(graph_count):
        graph_values = node_attr[batch_index == graph_index]
        if graph_values.numel() == 0:
            continue
        # Scaled RMS avoids overflow in square() while remaining exactly the
        # same graphwise L2 statistic mathematically.
        scale = graph_values.abs().max()
        if bool(scale > 0):
            normalized = graph_values / scale
            rms = scale * normalized.square().mean().sqrt()
            gains[graph_index] = torch.clamp(
                node_attr.new_tensor(float(cap)) / rms.clamp_min(1e-30),
                max=1.0,
            )
    return node_attr * gains[batch_index].unsqueeze(-1)


def normalize_batched_anchor_parent_index(data) -> torch.Tensor:
    """Return receptor-global anchor parents for one graph or a PyG batch.

    Historical caches store ``nucleic_anchor_parent_index`` as a receptor
    attribute.  PyG concatenates that attribute with increment zero, so every
    multi-graph segment remains local to its source receptor.  Use PyG's slice
    metadata to offset those segments, while accepting an already-global
    future representation and failing closed on ambiguous/corrupt input.
    """
    receptor = data["receptor"]
    raw = getattr(receptor, "nucleic_anchor_parent_index", None)
    anchor_pos = getattr(receptor, "nucleic_anchor_pos", None)
    anchor_type = getattr(receptor, "nucleic_anchor_type", None)
    if raw is None or anchor_pos is None or anchor_type is None:
        raise ValueError("NuSurf requires anchor position, type and parent tensors")
    raw = raw.long().view(-1)
    if int(anchor_pos.shape[0]) != raw.numel() or int(anchor_type.shape[0]) != raw.numel():
        raise ValueError("NuSurf anchor tensors have inconsistent row counts")

    receptor_ptr = getattr(receptor, "ptr", None)
    if receptor_ptr is None:
        n_receptor = int(receptor.pos.shape[0])
        if raw.numel() and (int(raw.min()) < 0 or int(raw.max()) >= n_receptor):
            raise ValueError("single-graph anchor parent index is out of receptor range")
        return raw
    receptor_ptr = receptor_ptr.long().view(-1)
    num_graphs = int(receptor_ptr.numel()) - 1
    if num_graphs < 1:
        raise ValueError("receptor ptr does not describe any graph")

    slice_dict = getattr(data, "_slice_dict", None)
    receptor_slices = slice_dict.get("receptor") if isinstance(slice_dict, dict) else None
    if not isinstance(receptor_slices, dict):
        if num_graphs > 1:
            raise ValueError(
                "multi-graph NuSurf batching requires PyG receptor slice metadata"
            )
        anchor_slices = torch.tensor([0, raw.numel()], dtype=torch.long)
    else:
        anchor_slices = receptor_slices.get("nucleic_anchor_parent_index")
        if anchor_slices is None:
            raise ValueError("anchor parent slice metadata is missing")
        expected_slices = anchor_slices.tolist()
        for key in ("nucleic_anchor_pos", "nucleic_anchor_type"):
            other = receptor_slices.get(key)
            if other is None or other.tolist() != expected_slices:
                raise ValueError(f"{key} slices do not match anchor parent slices")
    boundaries = [int(value) for value in anchor_slices.tolist()]
    if (
        len(boundaries) != num_graphs + 1
        or boundaries[0] != 0
        or boundaries[-1] != raw.numel()
        or any(right < left for left, right in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError("invalid batched anchor slice boundaries")

    ptr = [int(value) for value in receptor_ptr.tolist()]
    inc_dict = getattr(data, "_inc_dict", None)
    receptor_incs = inc_dict.get("receptor") if isinstance(inc_dict, dict) else None
    anchor_incs = (
        receptor_incs.get("nucleic_anchor_parent_index")
        if isinstance(receptor_incs, dict)
        else None
    )
    if anchor_incs is None:
        if num_graphs > 1:
            raise ValueError(
                "multi-graph NuSurf batching requires PyG anchor increment metadata"
            )
        increments = [0]
    else:
        increments = [int(value) for value in anchor_incs.reshape(-1).tolist()]
    if len(increments) != num_graphs:
        raise ValueError("anchor parent increment metadata has the wrong length")
    if increments == [0] * num_graphs:
        parent_mode = "local"
    elif increments == ptr[:-1]:
        parent_mode = "global"
    else:
        raise ValueError(
            "anchor parent increments are neither historical-local nor receptor-global: "
            f"increments={increments} receptor_starts={ptr[:-1]}"
        )

    normalized = raw.clone()
    for graph_index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        parent = raw[start:end]
        if parent.numel() == 0:
            continue
        receptor_start, receptor_end = ptr[graph_index], ptr[graph_index + 1]
        local_count = receptor_end - receptor_start
        if parent_mode == "local" and bool(
            torch.all((parent >= 0) & (parent < local_count))
        ):
            normalized[start:end] = parent + receptor_start
        elif parent_mode == "global" and bool(
            torch.all((parent >= receptor_start) & (parent < receptor_end))
        ):
            normalized[start:end] = parent
        else:
            raise ValueError(
                f"anchor parent segment violates declared {parent_mode} batching mode: "
                f"graph={graph_index} receptor_range=[{receptor_start},{receptor_end})"
            )
    return normalized


def invariant_irrep_features(node_attr: torch.Tensor, irreps: o3.Irreps) -> torch.Tensor:
    """Convert equivariant irreps to rotation-invariant scalar channels."""
    irreps = o3.Irreps(irreps)
    outputs = []
    for mul_irrep, block_slice in zip(irreps, irreps.slices()):
        multiplicity, irrep = mul_irrep.mul, mul_irrep.ir
        block = node_attr[:, block_slice].reshape(node_attr.shape[0], multiplicity, irrep.dim)
        if irrep.l == 0:
            outputs.append(block.squeeze(-1))
        else:
            # ``sqrt(sum(x**2))`` overflows in float32 long before the true
            # vector norm leaves the representable range.  Large but finite
            # E(3) features occur for a small fraction of hard decoys, so use
            # the mathematically equivalent scaled norm instead.  Keeping the
            # zero branch explicit also gives a finite, zero gradient there.
            scale = block.abs().amax(dim=-1)
            safe_scale = scale.clamp_min(1e-12)
            normalized = block / safe_scale.unsqueeze(-1)
            normalized_norm = normalized.square().sum(dim=-1).clamp_min(1e-12).sqrt()
            norm = safe_scale * normalized_norm
            outputs.append(torch.where(scale > 0, norm, torch.zeros_like(norm)))
    if not outputs:
        return node_attr.new_zeros((node_attr.shape[0], 0))
    return torch.cat(outputs, dim=-1)


def invariant_irrep_dim(irreps: o3.Irreps) -> int:
    irreps = o3.Irreps(irreps)
    return sum(mul_irrep.mul for mul_irrep in irreps)


class SurfaceInteractionBackboneFromDiffusion(TensorProductScoreModel, SurfaceInteractionBackboneV2):
    """Six-layer t=0 static/dynamic backbone with diffusion-compatible weights."""

    def __init__(
        self,
        *args,
        scorer_clash_distance: float = 2.0,
        scorer_cross_distance: float = 12.0,
        scorer_equivariant_rms_cap: float = EQUIVARIANT_GRAPH_RMS_CAP,
        **kwargs,
    ) -> None:
        scorer_args = kwargs.get("args")
        super().__init__(*args, **kwargs)
        self.scorer_args = scorer_args
        self.scorer_clash_distance = float(scorer_clash_distance)
        self.scorer_cross_distance = float(scorer_cross_distance)
        self.scorer_equivariant_rms_cap = float(scorer_equivariant_rms_cap)
        if self.scorer_equivariant_rms_cap <= 0:
            raise ValueError("scorer_equivariant_rms_cap must be positive")
        self._ligand_terminal_irreps = o3.Irreps(self.lig_conv_layers[-1].out_irreps)
        self._surface_terminal_irreps = (
            o3.Irreps(self.surface_conv_layers[-1].out_irreps)
            if len(self.surface_conv_layers)
            else o3.Irreps(f"{self.ns}x0e")
        )
        self.ligand_invariant_dim = invariant_irrep_dim(self._ligand_terminal_irreps)
        self.surface_invariant_dim = invariant_irrep_dim(self._surface_terminal_irreps)
        self.cross_rank_edge_dim = (
            self.ligand_invariant_dim
            + self.surface_invariant_dim
            + self.ns
            + self.surface_feature_dim
        )
        self.anchor_rank_edge_dim = (
            self.ligand_invariant_dim + self.ns + self.cross_distance_embed_dim + 3
        )

    def _set_clean_time(self, data) -> None:
        num_graphs = int(getattr(data, "num_graphs", 0))
        if not num_graphs:
            num_graphs = int(data["ligand"].batch.max().item()) + 1
        set_time(
            data,
            0.0,
            0.0,
            0.0,
            num_graphs,
            bool(getattr(self.scorer_args, "all_atoms", False)),
            data["ligand"].pos.device,
        )

    @staticmethod
    def _pad_add(base: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        return F.pad(base, (0, update.shape[-1] - base.shape[-1])) + update

    def _prepared_nucleic_features(self, data, *, like: torch.Tensor) -> torch.Tensor | None:
        raw = getattr(data["receptor"], "nucleic_feat", None)
        if raw is None:
            return None
        raw = raw.to(like).float()
        prepare = getattr(self, "_prepare_nucleic_features", None)
        if prepare is not None:
            return prepare(raw)
        return torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_nucleic_receptor_features(self, data, rec_node_attr: torch.Tensor) -> torch.Tensor:
        nucleic_feat = self._prepared_nucleic_features(data, like=rec_node_attr)
        if self.use_nucleic_feat_fusion and nucleic_feat is not None:
            rec_node_attr = rec_node_attr + self.nuc_feat_gate(nucleic_feat) * self.nuc_feat_proj(
                nucleic_feat
            )
        return rec_node_attr

    def _shared_surface_anchor_state(
        self, data, surface_attr: torch.Tensor
    ) -> torch.Tensor | None:
        """Parameter-free NA anchor states pooled from the shared surface.

        S1 and S2 intentionally use the same protein-transfer backbone.  When
        NuSurf is absent, base/sugar/phosphate anchors still receive a local
        representation by pooling nearby *shared* surface scalar channels.
        The gated S2 head can therefore add NA chemistry without replacing the
        common protein/NA surface scale or changing the S1 interaction trunk.
        """
        variant = str(getattr(self.scorer_args, "scorer_variant", ""))
        if variant not in {"distribution_s1", "gated_na_s2"}:
            return None
        receptor = data["receptor"]
        if not hasattr(receptor, "nucleic_anchor_pos"):
            return None
        anchor_pos = receptor.nucleic_anchor_pos
        if anchor_pos.numel() == 0:
            return surface_attr.new_zeros((0, self.ns))
        parent = normalize_batched_anchor_parent_index(data)
        anchor_batch = receptor.batch[parent]
        anchor_index, surface_index = radius(
            data["surface"].pos,
            anchor_pos,
            8.0,
            batch_x=data["surface"].batch,
            batch_y=anchor_batch,
            max_num_neighbors=32,
        )
        state = surface_attr.new_zeros((anchor_pos.shape[0], self.ns))
        if anchor_index.numel() == 0:
            return state
        distance = torch.linalg.vector_norm(
            anchor_pos[anchor_index] - data["surface"].pos[surface_index], dim=-1
        )
        weight = torch.exp(-distance / 3.0)
        numerator = scatter_add(
            surface_attr[surface_index, : self.ns] * weight.unsqueeze(-1),
            anchor_index,
            dim=0,
            dim_size=anchor_pos.shape[0],
        )
        denominator = scatter_add(
            weight,
            anchor_index,
            dim=0,
            dim_size=anchor_pos.shape[0],
        ).clamp_min(1e-8)
        return numerator / denominator.unsqueeze(-1)

    def _initial_surface_state(self, data):
        self._set_clean_time(data)
        rec_node_input, rec_edge_index, rec_edge_input, rec_edge_sh = self.build_rec_conv_graph(data)
        rec_src, rec_dst = rec_edge_index
        rec_node_attr = self._apply_nucleic_receptor_features(
            data, self.rec_node_embedding(rec_node_input)
        )
        rec_edge_attr = self.rec_edge_embedding(rec_edge_input)
        rec_message_attr = torch.cat(
            [rec_edge_attr, rec_node_attr[rec_src, : self.ns], rec_node_attr[rec_dst, : self.ns]],
            dim=-1,
        )
        rec_update = self.rec_conv_layers[0](
            rec_node_attr, rec_edge_index, rec_message_attr, rec_edge_sh
        )
        rec_node_attr = self._pad_add(rec_node_attr, rec_update)
        rec_node_attr = cap_equivariant_graph_rms(
            rec_node_attr, data["receptor"].batch, cap=self.scorer_equivariant_rms_cap
        )

        surface_input, _, _, _, _, _ = self.build_surface_conv_graph(data, mode=False)
        surface_node_attr = self.surface_node_embedding(surface_input)
        if self.use_adaptive_transfer and self.surface_node_adapter is not None:
            surface_node_attr = self.surface_node_adapter(surface_node_attr, surface_node_attr)

        cross_index, cross_input, cross_sh = self.build_surface_rec_cross_conv_graph(data)
        cross_rec, cross_surface = cross_index
        cross_attr = self.surface_rec_cross_edge_embedding(cross_input)
        message_attr = torch.cat(
            [
                cross_attr,
                rec_node_attr[cross_rec, : self.ns],
                surface_node_attr[cross_surface, : self.ns],
            ],
            dim=-1,
        )
        surface_update = self.residue_to_surface_conv_layers[0](
            rec_node_attr,
            torch.flip(cross_index, dims=[0]),
            message_attr,
            cross_sh,
            out_nodes=surface_node_attr.shape[0],
        )
        if self.use_adaptive_transfer and self.residue_to_surface_adapter is not None:
            padded = F.pad(
                surface_node_attr,
                (0, self.residue_to_surface_adapter.feature_dim - surface_node_attr.shape[-1]),
            )
            padded_update = F.pad(
                surface_update,
                (0, self.residue_to_surface_adapter.feature_dim - surface_update.shape[-1]),
            )
            surface_update = self.residue_to_surface_adapter(padded, padded_update)[
                :, : surface_update.shape[-1]
            ]
        surface_node_attr = self._pad_add(surface_node_attr, surface_update)
        surface_node_attr = cap_equivariant_graph_rms(
            surface_node_attr, data["surface"].batch, cap=self.scorer_equivariant_rms_cap
        )

        anchor_state = None
        if self.nusurf_fusion is not None and hasattr(data["receptor"], "nucleic_anchor_pos"):
            anchor_pos = data["receptor"].nucleic_anchor_pos
            if anchor_pos.numel() > 0:
                anchor_parent = normalize_batched_anchor_parent_index(data)
                fused, anchor_state = self.nusurf_fusion(
                    surface_node_attr[:, : self.ns],
                    data["surface"].pos,
                    data["surface"].batch,
                    anchor_pos,
                    data["receptor"].nucleic_anchor_type,
                    anchor_parent,
                    data["receptor"].batch,
                    self._prepared_nucleic_features(data, like=surface_node_attr),
                )
                surface_node_attr = torch.cat([fused, surface_node_attr[:, self.ns :]], dim=-1)
                surface_node_attr = cap_equivariant_graph_rms(
                    surface_node_attr,
                    data["surface"].batch,
                    cap=self.scorer_equivariant_rms_cap,
                )
        return surface_node_attr, anchor_state

    def encode_surface_static(self, data) -> StaticSurfaceState:
        base_surface_attr, anchor_state = self._initial_surface_state(data)
        surface_attr = base_surface_attr
        _, edge_index, edge_input, edge_sh, _, _ = self.build_surface_conv_graph(data, mode=False)
        edge_src, edge_dst = edge_index
        edge_attr = self.surface_edge_embedding(edge_input)
        if self.use_adaptive_transfer and self.surface_edge_adapter is not None:
            edge_attr = self.surface_edge_adapter(edge_attr, edge_attr)
        for layer_index, layer in enumerate(self.surface_conv_layers):
            message_attr = torch.cat(
                [edge_attr, surface_attr[edge_src, : self.ns], surface_attr[edge_dst, : self.ns]],
                dim=-1,
            )
            update = layer(surface_attr, edge_index, message_attr, edge_sh)
            if (
                self.use_adaptive_transfer
                and layer_index < len(self.surface_conv_adapters)
                and self.surface_conv_adapters[layer_index] is not None
            ):
                adapter = self.surface_conv_adapters[layer_index]
                padded = F.pad(surface_attr, (0, max(0, adapter.feature_dim - surface_attr.shape[-1])))
                padded_update = F.pad(update, (0, max(0, adapter.feature_dim - update.shape[-1])))
                update = adapter(padded, padded_update)[:, : update.shape[-1]]
            surface_attr = self._pad_add(surface_attr, update)
            surface_attr = cap_equivariant_graph_rms(
                surface_attr, data["surface"].batch, cap=self.scorer_equivariant_rms_cap
            )

        if len(self.nusurf_refine_blocks) and hasattr(data["receptor"], "nucleic_anchor_pos"):
            anchor_pos = data["receptor"].nucleic_anchor_pos
            if anchor_pos.numel() > 0:
                anchor_parent = normalize_batched_anchor_parent_index(data)
                for block in self.nusurf_refine_blocks:
                    fused, anchor_state = block(
                        surface_attr[:, : self.ns],
                        data["surface"].pos,
                        data["surface"].batch,
                        anchor_pos,
                        data["receptor"].nucleic_anchor_type,
                        anchor_parent,
                        data["receptor"].batch,
                        self._prepared_nucleic_features(data, like=surface_attr),
                    )
                    surface_attr = torch.cat([fused, surface_attr[:, self.ns :]], dim=-1)
                    surface_attr = cap_equivariant_graph_rms(
                        surface_attr,
                        data["surface"].batch,
                        cap=self.scorer_equivariant_rms_cap,
                    )

        if anchor_state is None:
            anchor_state = self._shared_surface_anchor_state(data, surface_attr)

        scalar_flat = invariant_irrep_features(surface_attr, self._surface_terminal_irreps)
        scalar, mask = to_dense_batch(scalar_flat, data["surface"].batch, fill_value=0.0)
        position, _ = to_dense_batch(data["surface"].pos, data["surface"].batch, fill_value=0.0)
        anchor_scalar = anchor_position = anchor_mask = None
        if anchor_state is not None and anchor_state.numel() > 0:
            parent = normalize_batched_anchor_parent_index(data)
            anchor_batch = data["receptor"].batch[parent]
            anchor_scalar, anchor_mask = to_dense_batch(anchor_state, anchor_batch, fill_value=0.0)
            anchor_position, _ = to_dense_batch(
                data["receptor"].nucleic_anchor_pos, anchor_batch, fill_value=0.0
            )
        return StaticSurfaceState(
            scalar=scalar,
            position=position,
            mask=mask,
            anchor_scalar=anchor_scalar,
            anchor_position=anchor_position,
            anchor_mask=anchor_mask,
            equivariant_flat=surface_attr,
            base_equivariant_flat=base_surface_attr,
            batch_index=data["surface"].batch,
            anchor_state_flat=anchor_state,
        )

    def encode_ligand_intra(self, data) -> LigandIntraState:
        self._set_clean_time(data)
        node_input, edge_index, edge_input, edge_sh = self.build_lig_conv_graph(data)
        edge_src, edge_dst = edge_index
        base_ligand_attr = self.lig_node_embedding(node_input)
        if self.use_adaptive_transfer and self.lig_node_adapter is not None:
            base_ligand_attr = self.lig_node_adapter(base_ligand_attr, base_ligand_attr)
        edge_attr = self.lig_edge_embedding(edge_input)
        if self.use_adaptive_transfer and self.lig_edge_adapter is not None:
            edge_attr = self.lig_edge_adapter(edge_attr, edge_attr)
        ligand_attr = base_ligand_attr
        for layer_index, layer in enumerate(self.lig_conv_layers):
            message_attr = torch.cat(
                [edge_attr, ligand_attr[edge_src, : self.ns], ligand_attr[edge_dst, : self.ns]],
                dim=-1,
            )
            update = layer(ligand_attr, edge_index, message_attr, edge_sh)
            if (
                self.use_adaptive_transfer
                and layer_index < len(self.lig_conv_adapters)
                and self.lig_conv_adapters[layer_index] is not None
            ):
                adapter = self.lig_conv_adapters[layer_index]
                padded = F.pad(ligand_attr, (0, max(0, adapter.feature_dim - ligand_attr.shape[-1])))
                padded_update = F.pad(update, (0, max(0, adapter.feature_dim - update.shape[-1])))
                update = adapter(padded, padded_update)[:, : update.shape[-1]]
            ligand_attr = self._pad_add(ligand_attr, update)
            ligand_attr = cap_equivariant_graph_rms(
                ligand_attr, data["ligand"].batch, cap=self.scorer_equivariant_rms_cap
            )

        scalar_flat = invariant_irrep_features(ligand_attr, self._ligand_terminal_irreps)
        scalar, mask = to_dense_batch(scalar_flat, data["ligand"].batch, fill_value=0.0)
        position, _ = to_dense_batch(data["ligand"].pos, data["ligand"].batch, fill_value=0.0)
        return LigandIntraState(
            scalar=scalar,
            position=position,
            mask=mask,
            equivariant_flat=ligand_attr,
            base_equivariant_flat=base_ligand_attr,
            batch_index=data["ligand"].batch,
        )

    def _strain_penalty(self, data, num_graphs: int) -> torch.Tensor:
        native = self._native_ligand_positions(data)
        if native is None:
            return data["ligand"].pos.new_zeros(num_graphs)
        edge_index = data["ligand", "ligand"].edge_index.long()
        if edge_index.numel() == 0:
            return data["ligand"].pos.new_zeros(num_graphs)
        src, dst = edge_index
        current_length = (data["ligand"].pos[src] - data["ligand"].pos[dst]).norm(dim=-1)
        native_length = (native[src] - native[dst]).norm(dim=-1)
        deviation = (current_length - native_length).abs()
        edge_batch = data["ligand"].batch[src]
        total = scatter_add(deviation, edge_batch, dim=0, dim_size=num_graphs)
        count = scatter_add(torch.ones_like(deviation), edge_batch, dim=0, dim_size=num_graphs)
        return total / count.clamp_min(1.0)

    def encode_pose(
        self,
        data,
        surface_state: StaticSurfaceState,
        ligand_state: LigandIntraState,
    ) -> CandidatePoseState:
        if surface_state.base_equivariant_flat is None or ligand_state.base_equivariant_flat is None:
            raise ValueError("dynamic scoring requires cached base equivariant states")
        self._set_clean_time(data)
        num_graphs = int(getattr(data, "num_graphs", ligand_state.scalar.shape[0]))
        ligand_attr = ligand_state.base_equivariant_flat
        surface_attr = surface_state.base_equivariant_flat

        if self.ligand_patch_tokenizer is not None:
            patched, _ = self.ligand_patch_tokenizer(
                ligand_attr[:, : self.ns],
                data["ligand"].pos,
                data["ligand"].batch,
                surface_attr[:, : self.ns],
                data["surface"].pos,
                data["surface"].batch,
            )
            ligand_attr = torch.cat([patched, ligand_attr[:, self.ns :]], dim=-1)

        _, ligand_edge_index, ligand_edge_input, ligand_edge_sh = self.build_lig_conv_graph(data)
        ligand_src, ligand_dst = ligand_edge_index
        ligand_edge_attr = self.lig_edge_embedding(ligand_edge_input)
        _, surface_edge_index, surface_edge_input, surface_edge_sh, _, _ = self.build_surface_conv_graph(
            data, mode=False
        )
        surface_src, surface_dst = surface_edge_index
        surface_edge_attr = self.surface_edge_embedding(surface_edge_input)
        cross_index, cross_input, cross_sh = self.build_surface_cross_conv_graph(
            data, self.scorer_cross_distance
        )
        cross_ligand, cross_surface = cross_index
        cross_attr = self.cross_edge_embedding(cross_input)
        if self.use_adaptive_transfer and self.cross_edge_adapter is not None:
            cross_attr = self.cross_edge_adapter(cross_attr, cross_attr)

        for layer_index, ligand_layer in enumerate(self.lig_conv_layers):
            ligand_message = torch.cat(
                [
                    ligand_edge_attr,
                    ligand_attr[ligand_src, : self.ns],
                    ligand_attr[ligand_dst, : self.ns],
                ],
                dim=-1,
            )
            ligand_update = ligand_layer(
                ligand_attr, ligand_edge_index, ligand_message, ligand_edge_sh
            )
            cross_message = torch.cat(
                [
                    cross_attr,
                    ligand_attr[cross_ligand, : self.ns],
                    surface_attr[cross_surface, : self.ns],
                ],
                dim=-1,
            )
            surface_to_ligand = self.surface_to_lig_conv_layers[layer_index](
                surface_attr,
                cross_index,
                cross_message,
                cross_sh,
                out_nodes=ligand_attr.shape[0],
            )
            if layer_index < len(self.surface_conv_layers):
                surface_message = torch.cat(
                    [
                        surface_edge_attr,
                        surface_attr[surface_src, : self.ns],
                        surface_attr[surface_dst, : self.ns],
                    ],
                    dim=-1,
                )
                surface_update = self.surface_conv_layers[layer_index](
                    surface_attr, surface_edge_index, surface_message, surface_edge_sh
                )
                ligand_to_surface = self.lig_to_surface_conv_layers[layer_index](
                    ligand_attr,
                    torch.flip(cross_index, dims=[0]),
                    cross_message,
                    cross_sh,
                    out_nodes=surface_attr.shape[0],
                )
            ligand_attr = F.pad(
                ligand_attr, (0, ligand_update.shape[-1] - ligand_attr.shape[-1])
            )
            ligand_attr = ligand_attr + ligand_update + surface_to_ligand
            ligand_attr = cap_equivariant_graph_rms(
                ligand_attr, data["ligand"].batch, cap=self.scorer_equivariant_rms_cap
            )
            if layer_index < len(self.surface_conv_layers):
                surface_attr = F.pad(
                    surface_attr, (0, surface_update.shape[-1] - surface_attr.shape[-1])
                )
                surface_attr = surface_attr + surface_update + ligand_to_surface
                surface_attr = cap_equivariant_graph_rms(
                    surface_attr,
                    data["surface"].batch,
                    cap=self.scorer_equivariant_rms_cap,
                )

        anchor_state = surface_state.anchor_state_flat
        if len(self.nusurf_refine_blocks) and hasattr(data["receptor"], "nucleic_anchor_pos"):
            anchor_pos = data["receptor"].nucleic_anchor_pos
            if anchor_pos.numel() > 0:
                anchor_parent = normalize_batched_anchor_parent_index(data)
                for block in self.nusurf_refine_blocks:
                    fused, anchor_state = block(
                        surface_attr[:, : self.ns],
                        data["surface"].pos,
                        data["surface"].batch,
                        anchor_pos,
                        data["receptor"].nucleic_anchor_type,
                        anchor_parent,
                        data["receptor"].batch,
                        self._prepared_nucleic_features(data, like=surface_attr),
                    )
                    surface_attr = torch.cat([fused, surface_attr[:, self.ns :]], dim=-1)
                    surface_attr = cap_equivariant_graph_rms(
                        surface_attr,
                        data["surface"].batch,
                        cap=self.scorer_equivariant_rms_cap,
                    )

        ligand_invariant = invariant_irrep_features(ligand_attr, self._ligand_terminal_irreps)
        surface_invariant = invariant_irrep_features(surface_attr, self._surface_terminal_irreps)
        surface_physical = torch.nan_to_num(data["surface"].x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        cross_features = torch.cat(
            [
                ligand_invariant[cross_ligand],
                surface_invariant[cross_surface],
                cross_attr,
                surface_physical[cross_surface],
            ],
            dim=-1,
        )
        cross_batch = data["ligand"].batch[cross_ligand]
        cross_distance = (
            data["surface"].pos[cross_surface] - data["ligand"].pos[cross_ligand]
        ).norm(dim=-1)
        clash = F.relu(self.scorer_clash_distance - cross_distance).square()
        clash_total = scatter_add(clash, cross_batch, dim=0, dim_size=num_graphs)
        clash_count = scatter_add(
            torch.ones_like(clash), cross_batch, dim=0, dim_size=num_graphs
        )
        clash_penalty = clash_total / clash_count.clamp_min(1.0)

        anchor_features = cross_features.new_zeros((0, self.anchor_rank_edge_dim))
        anchor_edge_batch = cross_batch.new_zeros((0,), dtype=torch.long)
        anchor_edge_ligand = cross_batch.new_zeros((0,), dtype=torch.long)
        is_na = cross_features.new_zeros(num_graphs)
        if (
            anchor_state is not None
            and anchor_state.numel() > 0
            and hasattr(data["receptor"], "nucleic_anchor_pos")
        ):
            anchor_parent = normalize_batched_anchor_parent_index(data)
            anchor_batch = data["receptor"].batch[anchor_parent]
            is_na[torch.unique(anchor_batch)] = 1.0
            anchor_edges = radius(
                data["receptor"].nucleic_anchor_pos,
                data["ligand"].pos,
                self.scorer_cross_distance,
                batch_x=anchor_batch,
                batch_y=data["ligand"].batch,
                max_num_neighbors=30,
            )
            anchor_ligand, anchor_index = anchor_edges
            if anchor_ligand.numel() > 0:
                anchor_distance = (
                    data["receptor"].nucleic_anchor_pos[anchor_index]
                    - data["ligand"].pos[anchor_ligand]
                ).norm(dim=-1)
                role = F.one_hot(
                    data["receptor"].nucleic_anchor_type[anchor_index].long().clamp(0, 2),
                    num_classes=3,
                ).to(ligand_invariant)
                anchor_features = torch.cat(
                    [
                        ligand_invariant[anchor_ligand],
                        anchor_state[anchor_index],
                        self.cross_distance_expansion(anchor_distance),
                        role,
                    ],
                    dim=-1,
                )
                anchor_edge_batch = data["ligand"].batch[anchor_ligand]
                anchor_edge_ligand = anchor_ligand

        return CandidatePoseState(
            invariant_cross_edge_features=cross_features,
            cross_edge_batch=cross_batch,
            cross_edge_ligand_index=cross_ligand,
            ligand_atom_batch=data["ligand"].batch,
            invariant_anchor_edge_features=anchor_features,
            anchor_edge_batch=anchor_edge_batch,
            anchor_edge_ligand_index=anchor_edge_ligand,
            is_nucleic_acid=is_na,
            clash_penalty=clash_penalty,
            strain_penalty=self._strain_penalty(data, num_graphs),
        )
