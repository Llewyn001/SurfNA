"""Top-level dual-branch SurfNA V2 scorer contract.

The concrete E(3)-equivariant backbone will be refactored from
``surface_score_model_v3``.  This file fixes the public boundary first, so the
scientific no-leakage invariants and grouped losses can be tested separately
from the large docking model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
from torch import nn

from models.surfna_v2_scorer_components import (
    MDNPriorOutput,
    DistanceReferencePrior,
    GatedNucleicAcidResidualHead,
    HierarchicalContactCoverageRanker,
    NormalizedMDNPriorHead,
    NucleicAcidResidualHead,
    PoseConditionedCrossRanker,
    PoseScoreOutput,
    SurfacePoseScoreCombiner,
)


class StaticSurfaceState(NamedTuple):
    scalar: torch.Tensor
    position: torch.Tensor
    mask: torch.Tensor
    anchor_scalar: torch.Tensor | None = None
    anchor_position: torch.Tensor | None = None
    anchor_mask: torch.Tensor | None = None
    equivariant_flat: torch.Tensor | None = None
    base_equivariant_flat: torch.Tensor | None = None
    batch_index: torch.Tensor | None = None
    anchor_state_flat: torch.Tensor | None = None


class LigandIntraState(NamedTuple):
    scalar: torch.Tensor
    position: torch.Tensor
    mask: torch.Tensor
    equivariant_flat: torch.Tensor | None = None
    base_equivariant_flat: torch.Tensor | None = None
    batch_index: torch.Tensor | None = None


class CandidatePoseState(NamedTuple):
    invariant_cross_edge_features: torch.Tensor
    cross_edge_batch: torch.Tensor
    cross_edge_ligand_index: torch.Tensor
    ligand_atom_batch: torch.Tensor
    invariant_anchor_edge_features: torch.Tensor
    anchor_edge_batch: torch.Tensor
    anchor_edge_ligand_index: torch.Tensor
    is_nucleic_acid: torch.Tensor
    clash_penalty: torch.Tensor
    strain_penalty: torch.Tensor


class SurfNAV2ScorerOutput(NamedTuple):
    pose: PoseScoreOutput
    prior: MDNPriorOutput | torch.Tensor


class SurfaceInteractionBackboneV2(nn.Module, ABC):
    """Required static/dynamic boundary for a SurfNA V2 scorer backbone."""

    @abstractmethod
    def encode_surface_static(self, data) -> StaticSurfaceState:
        """Encode receptor, physical surface and optional NA anchors once."""

    @abstractmethod
    def encode_ligand_intra(self, data) -> LigandIntraState:
        """Encode ligand without ligand-surface cross edges or patch tokens."""

    @abstractmethod
    def encode_pose(
        self,
        data,
        surface_state: StaticSurfaceState,
        ligand_state: LigandIntraState,
    ) -> CandidatePoseState:
        """Run t=0 patch/cross modules for discriminative candidate ranking."""


class SurfNAV2Scorer(nn.Module):
    """Normalized MDN prior + diffusion-cross residual + NA residual."""

    def __init__(
        self,
        backbone: SurfaceInteractionBackboneV2,
        *,
        ligand_scalar_dim: int,
        surface_scalar_dim: int,
        cross_edge_feature_dim: int,
        anchor_edge_feature_dim: int,
        hidden_dim: int = 192,
        n_gaussians: int = 20,
        topk_surface_per_atom: int = 8,
        dropout: float = 0.1,
        reference_prior: DistanceReferencePrior | None = None,
        scorer_variant: str = "aligned_a",
    ) -> None:
        super().__init__()
        if scorer_variant not in {
            "aligned_a",
            "full_v2_b",
            "distribution_s1",
            "gated_na_s2",
        }:
            raise ValueError(f"unknown SurfNA V2 scorer variant {scorer_variant!r}")
        self.scorer_variant = scorer_variant
        self.backbone = backbone
        self.prior_head = NormalizedMDNPriorHead(
            pair_feature_dim=int(ligand_scalar_dim + surface_scalar_dim),
            hidden_dim=hidden_dim,
            n_gaussians=n_gaussians,
            topk_surface_per_atom=topk_surface_per_atom,
            dropout=dropout,
            reference_prior=reference_prior,
        )
        if scorer_variant == "aligned_a":
            self.cross_ranker = PoseConditionedCrossRanker(
                edge_feature_dim=cross_edge_feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            self.cross_ranker = HierarchicalContactCoverageRanker(
                edge_feature_dim=cross_edge_feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                zero_initialize_score=scorer_variant in {
                    "distribution_s1",
                    "gated_na_s2",
                },
            )
        if scorer_variant in {"distribution_s1", "gated_na_s2"}:
            self.na_residual = GatedNucleicAcidResidualHead(
                anchor_edge_feature_dim=anchor_edge_feature_dim,
                hidden_dim=max(64, hidden_dim // 2),
                dropout=dropout,
            )
        else:
            self.na_residual = NucleicAcidResidualHead(
                anchor_edge_feature_dim=anchor_edge_feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                zero_initialize_output=scorer_variant == "aligned_a",
            )
        self.combiner = SurfacePoseScoreCombiner()

    @classmethod
    def from_diffusion_backbone(
        cls,
        backbone: SurfaceInteractionBackboneV2,
        **kwargs,
    ) -> "SurfNAV2Scorer":
        """Construct heads from dimensions published by the concrete backbone."""
        return cls(
            backbone,
            ligand_scalar_dim=int(backbone.ligand_invariant_dim),
            surface_scalar_dim=int(backbone.surface_invariant_dim),
            cross_edge_feature_dim=int(backbone.cross_rank_edge_dim),
            anchor_edge_feature_dim=int(backbone.anchor_rank_edge_dim),
            **kwargs,
        )

    @staticmethod
    def build_static_pair_features(
        ligand_state: LigandIntraState,
        surface_state: StaticSurfaceState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ligand = ligand_state.scalar
        surface = surface_state.scalar
        if ligand.ndim != 3 or surface.ndim != 3:
            raise ValueError("dense static scalar states must have shape [B,N,D]")
        if ligand.shape[0] != surface.shape[0]:
            raise ValueError("ligand and surface batch sizes differ")
        batch, n_ligand, _ = ligand.shape
        n_surface = surface.shape[1]
        ligand_pairs = ligand.unsqueeze(2).expand(-1, -1, n_surface, -1)
        surface_pairs = surface.unsqueeze(1).expand(-1, n_ligand, -1, -1)
        pair_features = torch.cat([ligand_pairs, surface_pairs], dim=-1)
        pair_mask = ligand_state.mask.bool().unsqueeze(2) & surface_state.mask.bool().unsqueeze(1)
        distance = torch.cdist(ligand_state.position.float(), surface_state.position.float())
        if distance.shape != (batch, n_ligand, n_surface):
            raise RuntimeError("unexpected dense ligand-surface distance shape")
        return pair_features, distance, pair_mask

    def predict_prior_parameters(
        self,
        surface_state: StaticSurfaceState,
        ligand_state: LigandIntraState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair_features, _, _ = self.build_static_pair_features(ligand_state, surface_state)
        return self.prior_head.predict_parameters(pair_features)

    def forward_prior(
        self,
        surface_state: StaticSurfaceState,
        ligand_state: LigandIntraState,
    ) -> MDNPriorOutput:
        pair_features, distance, pair_mask = self.build_static_pair_features(
            ligand_state, surface_state
        )
        return self.prior_head(pair_features, distance, pair_mask)

    def score_pose(
        self,
        data,
        surface_state: StaticSurfaceState | None,
        ligand_state: LigandIntraState | None,
        prior: MDNPriorOutput | torch.Tensor,
        *,
        pose_state: CandidatePoseState | None = None,
    ) -> PoseScoreOutput:
        if pose_state is None:
            if surface_state is None or ligand_state is None:
                raise ValueError("uncached pose scoring requires surface and ligand states")
            pose_state = self.backbone.encode_pose(data, surface_state, ligand_state)
        prior_score = prior.score if isinstance(prior, MDNPriorOutput) else prior
        if not torch.is_tensor(prior_score) or prior_score.ndim != 1:
            raise ValueError("pose scoring requires one prior score per graph")
        num_graphs = int(prior_score.shape[0])
        ordinal_kwargs = {}
        na_gate = torch.zeros_like(prior_score)
        if self.scorer_variant == "aligned_a":
            cross_score = self.cross_ranker(
                pose_state.invariant_cross_edge_features,
                pose_state.cross_edge_batch,
                num_graphs,
            )
            # A is the strictly aligned protein-transfer baseline.  It must
            # not acquire a hidden NA-only branch merely because the target
            # batches contain nucleic-acid anchors.
            na_score = torch.zeros_like(prior_score)
        else:
            if not bool(torch.all(pose_state.is_nucleic_acid.bool())):
                raise RuntimeError(
                    f"{self.scorer_variant} requires a nucleic-acid graph in every batch item"
                )
            anchor_features = pose_state.invariant_anchor_edge_features
            anchor_edge_batch = pose_state.anchor_edge_batch.long().view(-1)
            if anchor_features.ndim != 2:
                raise RuntimeError("full_v2_b anchor-edge features must have shape [E,D]")
            if anchor_features.shape[0] != anchor_edge_batch.numel():
                raise RuntimeError(
                    "full_v2_b requires one graph index for every anchor edge"
                )
            if anchor_edge_batch.numel() and (
                int(anchor_edge_batch.min()) < 0
                or int(anchor_edge_batch.max()) >= num_graphs
            ):
                raise RuntimeError("full_v2_b anchor-edge graph index is out of range")
            cross_output = self.cross_ranker(
                pose_state.invariant_cross_edge_features,
                pose_state.cross_edge_ligand_index,
                pose_state.ligand_atom_batch,
                num_graphs,
            )
            cross_score = cross_output.score
            ordinal_kwargs = {
                "ordinal_logit_lt5": cross_output.ordinal_logit_lt5,
                "ordinal_logit_lt2_given_lt5": cross_output.ordinal_logit_lt2_given_lt5,
                "prob_lt2": cross_output.prob_lt2,
                "prob_lt5": cross_output.prob_lt5,
                "coverage_score": cross_output.coverage_score,
            }
            if self.scorer_variant == "full_v2_b":
                na_score = self.na_residual(
                    pose_state.invariant_anchor_edge_features,
                    pose_state.anchor_edge_batch,
                    num_graphs,
                    pose_state.is_nucleic_acid,
                )
            elif self.scorer_variant == "distribution_s1":
                na_score = torch.zeros_like(prior_score)
            else:
                if pose_state.anchor_edge_ligand_index.numel() != anchor_edge_batch.numel():
                    raise RuntimeError(
                        "gated_na_s2 requires one ligand index for every anchor edge"
                    )
                ligand_atoms = int(pose_state.ligand_atom_batch.numel())
                if pose_state.anchor_edge_ligand_index.numel() and (
                    int(pose_state.anchor_edge_ligand_index.min()) < 0
                    or int(pose_state.anchor_edge_ligand_index.max()) >= ligand_atoms
                ):
                    raise RuntimeError("gated_na_s2 anchor ligand index is out of range")
                na_score, na_gate = self.na_residual(
                    pose_state.invariant_anchor_edge_features,
                    pose_state.anchor_edge_batch,
                    pose_state.anchor_edge_ligand_index,
                    pose_state.ligand_atom_batch,
                    cross_output.coverage_score,
                    num_graphs,
                    pose_state.is_nucleic_acid,
                )
        components = {
            "prior": prior_score,
            "cross": cross_score,
            "na": na_score,
            "clash": pose_state.clash_penalty,
            "strain": pose_state.strain_penalty,
        }
        invalid = {
            name: tuple(value.shape)
            for name, value in components.items()
            if value.shape != prior_score.shape or not bool(torch.isfinite(value).all())
        }
        if invalid:
            raise RuntimeError(
                f"{self.scorer_variant} pose components must be finite with one score "
                f"per graph; invalid={invalid}"
            )
        if na_gate.shape != prior_score.shape or not bool(torch.isfinite(na_gate).all()):
            raise RuntimeError(
                f"{self.scorer_variant} NA gate must be finite with one value per graph"
            )
        return self.combiner(
            prior_score,
            cross_score,
            na_score,
            pose_state.clash_penalty,
            pose_state.strain_penalty,
            pose_state.is_nucleic_acid,
            na_gate=na_gate,
            **ordinal_kwargs,
        )

    def forward(
        self,
        data,
        *,
        cached_surface_state: StaticSurfaceState | None = None,
        cached_pose_state: CandidatePoseState | None = None,
        cached_prior_score: torch.Tensor | None = None,
        prior_only: bool = False,
    ) -> SurfNAV2ScorerOutput | MDNPriorOutput:
        if (cached_pose_state is None) != (cached_prior_score is None):
            raise ValueError("cached pose state and cached prior score must be provided together")
        if cached_pose_state is not None:
            if prior_only:
                raise ValueError("cached pose scoring is incompatible with prior_only")
            pose = self.score_pose(
                data,
                None,
                None,
                cached_prior_score,
                pose_state=cached_pose_state,
            )
            return SurfNAV2ScorerOutput(pose=pose, prior=cached_prior_score)
        surface_state = cached_surface_state or self.backbone.encode_surface_static(data)
        ligand_state = self.backbone.encode_ligand_intra(data)
        prior = self.forward_prior(surface_state, ligand_state)
        if prior_only:
            return prior
        pose = self.score_pose(data, surface_state, ligand_state, prior)
        return SurfNAV2ScorerOutput(pose=pose, prior=prior)
