"""Leakage-safe scoring heads for SurfNA V2.

This module deliberately separates two kinds of information:

* the MDN prior predicts preferred distances from pose-independent ligand and
  surface embeddings; candidate coordinates are used only to evaluate the
  predicted density;
* the cross and NA residual heads are discriminative and may consume the
  current candidate's geometry.

The backbone is expected to expose explicit ``encode_static`` and
``encode_pose`` methods.  Keeping these heads independent makes that contract
unit-testable before the large equivariant backbone refactor is activated.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add, scatter_max


class MDNPriorOutput(NamedTuple):
    score: torch.Tensor
    pi: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor
    pair_log_likelihood_ratio: torch.Tensor
    pair_mask: torch.Tensor
    per_ligand_score: torch.Tensor
    candidate_distances: torch.Tensor


class PoseScoreOutput(NamedTuple):
    score: torch.Tensor
    prior_score: torch.Tensor
    cross_score: torch.Tensor
    na_score: torch.Tensor
    clash_penalty: torch.Tensor
    strain_penalty: torch.Tensor
    ordinal_logit_lt5: torch.Tensor
    ordinal_logit_lt2_given_lt5: torch.Tensor
    prob_lt2: torch.Tensor
    prob_lt5: torch.Tensor
    coverage_score: torch.Tensor
    na_gate: torch.Tensor


class CrossRankerOutput(NamedTuple):
    """Pose readout plus monotone cumulative-threshold predictions."""

    score: torch.Tensor
    ordinal_logit_lt5: torch.Tensor
    ordinal_logit_lt2_given_lt5: torch.Tensor
    prob_lt2: torch.Tensor
    prob_lt5: torch.Tensor
    coverage_score: torch.Tensor


def mixture_log_probability(
    pi: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    distance: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Evaluate a Gaussian mixture in log space.

    ``pi``, ``sigma`` and ``mu`` have a final mixture dimension. ``distance``
    has the same leading shape and no mixture dimension.
    """
    distance = torch.nan_to_num(distance.float(), nan=1e4, posinf=1e4, neginf=0.0)
    pi = torch.nan_to_num(pi.float(), nan=0.0).clamp_min(eps)
    pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(eps)
    sigma = torch.nan_to_num(sigma.float(), nan=1.0, posinf=10.0).clamp_min(eps)
    mu = torch.nan_to_num(mu.float(), nan=0.0, posinf=1e4).clamp_min(0.0)
    z = (distance.unsqueeze(-1) - mu) / sigma
    component_log_prob = -0.5 * z.square() - sigma.log() - 0.5 * math.log(2.0 * math.pi)
    return torch.logsumexp(pi.log() + component_log_prob, dim=-1)


class DistanceReferencePrior(nn.Module):
    """Frozen empirical distance density used to turn MDN density into odds.

    The preferred constructor is :meth:`from_json`, using a histogram fitted
    only on training-split candidate distances.  The default distribution is
    the 3D spherical-shell density on ``[0, max_distance]`` and exists only for
    smoke tests; formal runs must record the fitted JSON hash.
    """

    def __init__(
        self,
        bin_centers: torch.Tensor | None = None,
        log_density: torch.Tensor | None = None,
        *,
        max_distance: float = 20.0,
        floor: float = 1e-8,
        source_sha256: str = "UNFITTED_SHELL_REFERENCE",
    ) -> None:
        super().__init__()
        if bin_centers is None or log_density is None:
            bin_centers = torch.linspace(0.05, float(max_distance), 256)
            density = 3.0 * bin_centers.square() / max(float(max_distance) ** 3, floor)
            log_density = density.clamp_min(floor).log()
        if bin_centers.ndim != 1 or log_density.ndim != 1:
            raise ValueError("reference centers and log_density must be one-dimensional")
        if bin_centers.numel() != log_density.numel() or bin_centers.numel() < 2:
            raise ValueError("reference centers and log_density must have the same length >=2")
        if not torch.all(bin_centers[1:] > bin_centers[:-1]):
            raise ValueError("reference bin centers must be strictly increasing")
        self.register_buffer("bin_centers", bin_centers.detach().float())
        self.register_buffer("log_density", log_density.detach().float())
        self.floor = float(floor)
        self.source_sha256 = str(source_sha256)

    @classmethod
    def from_json(cls, path: str | Path) -> "DistanceReferencePrior":
        path = Path(path)
        payload = json.loads(path.read_text())
        return cls(
            torch.tensor(payload["bin_centers"], dtype=torch.float32),
            torch.tensor(payload["log_density"], dtype=torch.float32),
            floor=float(payload.get("floor", 1e-8)),
            source_sha256=str(payload["source_sha256"]),
        )

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        distance = torch.nan_to_num(distance.float(), nan=float(self.bin_centers[-1]))
        distance = distance.clamp(float(self.bin_centers[0]), float(self.bin_centers[-1]))
        upper = torch.bucketize(distance.contiguous(), self.bin_centers)
        upper = upper.clamp(1, self.bin_centers.numel() - 1)
        lower = upper - 1
        x0 = self.bin_centers[lower]
        x1 = self.bin_centers[upper]
        y0 = self.log_density[lower]
        y1 = self.log_density[upper]
        fraction = (distance - x0) / (x1 - x0).clamp_min(self.floor)
        return y0 + fraction * (y1 - y0)


class NormalizedMDNPriorHead(nn.Module):
    """Pose-independent MDN parameters plus size-normalized pose likelihood."""

    def __init__(
        self,
        pair_feature_dim: int,
        hidden_dim: int,
        n_gaussians: int = 20,
        *,
        sigma_min: float = 0.25,
        mu_min: float = 0.0,
        topk_surface_per_atom: int = 8,
        contact_radius: float = 8.0,
        contact_temperature: float = 1.0,
        log_ratio_clip: float = 20.0,
        dropout: float = 0.1,
        reference_prior: DistanceReferencePrior | None = None,
    ) -> None:
        super().__init__()
        self.n_gaussians = int(n_gaussians)
        self.sigma_min = float(sigma_min)
        self.mu_min = float(mu_min)
        self.topk_surface_per_atom = int(topk_surface_per_atom)
        self.contact_radius = float(contact_radius)
        self.contact_temperature = float(contact_temperature)
        self.log_ratio_clip = float(log_ratio_clip)
        self.reference_prior = reference_prior or DistanceReferencePrior()
        self.pair_mlp = nn.Sequential(
            nn.LayerNorm(pair_feature_dim),
            nn.Linear(pair_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.pi_head = nn.Linear(hidden_dim, self.n_gaussians)
        self.mu_head = nn.Linear(hidden_dim, self.n_gaussians)
        self.sigma_head = nn.Linear(hidden_dim, self.n_gaussians)

    def predict_parameters(
        self,
        static_pair_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict mixture parameters without accepting candidate coordinates."""
        hidden = self.pair_mlp(static_pair_features)
        pi = F.softmax(self.pi_head(hidden), dim=-1)
        mu = F.softplus(self.mu_head(hidden)) + self.mu_min
        sigma = F.softplus(self.sigma_head(hidden)) + self.sigma_min
        return pi, mu, sigma

    def score_parameters(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        candidate_distances: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> MDNPriorOutput:
        """Evaluate and normalize a candidate without changing MDN parameters.

        Dense leading dimensions must be ``[batch, ligand_atom, surface_node]``.
        Invalid padded pairs are excluded.  Aggregation first takes a top-k mean
        per ligand atom and then averages atoms, removing raw pair-count bias.
        """
        if candidate_distances.ndim != 3:
            raise ValueError("candidate_distances must have shape [B, N_ligand, N_surface]")
        if pair_mask.shape != candidate_distances.shape:
            raise ValueError("pair_mask must match candidate_distances")
        if pi.shape[:-1] != candidate_distances.shape:
            raise ValueError("mixture parameter leading dimensions must match distances")
        pair_mask = pair_mask.bool()
        model_log_prob = mixture_log_probability(pi, sigma, mu, candidate_distances)
        reference_log_prob = self.reference_prior(candidate_distances)
        log_ratio = (model_log_prob - reference_log_prob).clamp(
            -self.log_ratio_clip, self.log_ratio_clip
        )
        contact_weight = torch.sigmoid(
            (self.contact_radius - candidate_distances) / max(self.contact_temperature, 1e-6)
        )
        utility = log_ratio * contact_weight
        utility = utility.masked_fill(~pair_mask, torch.finfo(utility.dtype).min)

        n_surface = candidate_distances.shape[-1]
        k = min(max(1, self.topk_surface_per_atom), n_surface)
        top_values, top_indices = torch.topk(utility, k=k, dim=-1)
        selected_valid = torch.gather(pair_mask, dim=-1, index=top_indices)
        top_values = torch.where(selected_valid, top_values, torch.zeros_like(top_values))
        selected_count = selected_valid.sum(dim=-1).clamp_min(1)
        per_ligand_score = top_values.sum(dim=-1) / selected_count
        ligand_valid = pair_mask.any(dim=-1)
        graph_score = (per_ligand_score * ligand_valid).sum(dim=-1) / ligand_valid.sum(dim=-1).clamp_min(1)
        return MDNPriorOutput(
            score=graph_score,
            pi=pi,
            mu=mu,
            sigma=sigma,
            pair_log_likelihood_ratio=log_ratio,
            pair_mask=pair_mask,
            per_ligand_score=per_ligand_score,
            candidate_distances=candidate_distances,
        )

    def forward(
        self,
        static_pair_features: torch.Tensor,
        candidate_distances: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> MDNPriorOutput:
        pi, mu, sigma = self.predict_parameters(static_pair_features)
        return self.score_parameters(pi, mu, sigma, candidate_distances, pair_mask)


def _segment_attention_pool(
    values: torch.Tensor,
    logits: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    if values.ndim != 2 or logits.ndim != 1 or batch.ndim != 1:
        raise ValueError("attention pool expects values[E,D], logits[E], batch[E]")
    if values.shape[0] == 0:
        return values.new_zeros((num_graphs, values.shape[-1]))
    maxima, _ = scatter_max(logits, batch, dim=0, dim_size=num_graphs)
    stabilized = logits - maxima[batch]
    weights = stabilized.exp()
    normalizer = scatter_add(weights, batch, dim=0, dim_size=num_graphs).clamp_min(1e-8)
    weights = weights / normalizer[batch]
    return scatter_add(values * weights.unsqueeze(-1), batch, dim=0, dim_size=num_graphs)


class PoseConditionedCrossRanker(nn.Module):
    """Discriminative residual over invariant ligand-surface cross-edge features."""

    def __init__(self, edge_feature_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(edge_feature_dim),
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        invariant_cross_edge_features: torch.Tensor,
        edge_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        hidden = self.edge_encoder(invariant_cross_edge_features)
        pooled = _segment_attention_pool(
            hidden, self.attention(hidden).squeeze(-1), edge_batch.long(), num_graphs
        )
        return self.output(pooled).squeeze(-1)


class HierarchicalContactCoverageRanker(nn.Module):
    """Edge -> ligand-atom -> whole-ligand contact/coverage readout.

    Every ligand atom participates in the whole-ligand pool, including atoms
    with no surface edge inside the cutoff.  This prevents a large ligand from
    receiving a high score solely because a small subset of its atoms forms
    attractive contacts.  The two ordinal logits parameterize
    ``P(RMSD<5)`` and ``P(RMSD<2 | RMSD<5)``; consequently ``P(<2)<=P(<5)``
    by construction.
    """

    def __init__(
        self,
        edge_feature_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        *,
        zero_initialize_score: bool = False,
    ) -> None:
        super().__init__()
        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(edge_feature_dim),
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.edge_attention = nn.Linear(hidden_dim, 1)
        self.edge_contact = nn.Linear(hidden_dim, 1)
        self.atom_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim + 3),
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.atom_attention = nn.Linear(hidden_dim, 1)
        self.pose_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim + 2),
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.score_head = nn.Linear(hidden_dim, 1)
        self.ordinal_head = nn.Linear(hidden_dim, 2)
        if zero_initialize_score:
            nn.init.zeros_(self.score_head.weight)
            nn.init.zeros_(self.score_head.bias)

    def forward(
        self,
        invariant_cross_edge_features: torch.Tensor,
        edge_ligand_index: torch.Tensor,
        ligand_atom_batch: torch.Tensor,
        num_graphs: int,
    ) -> CrossRankerOutput:
        if ligand_atom_batch.ndim != 1:
            raise ValueError("ligand_atom_batch must be one-dimensional")
        num_atoms = int(ligand_atom_batch.numel())
        if num_atoms == 0:
            raise ValueError("hierarchical ranker received no ligand atoms")
        edge_ligand_index = edge_ligand_index.long().view(-1)
        if edge_ligand_index.numel() != invariant_cross_edge_features.shape[0]:
            raise ValueError("one ligand index is required for every cross edge")
        if edge_ligand_index.numel() and (
            int(edge_ligand_index.min()) < 0 or int(edge_ligand_index.max()) >= num_atoms
        ):
            raise ValueError("cross-edge ligand index is out of range")

        hidden = self.edge_encoder(invariant_cross_edge_features)
        edge_logits = self.edge_attention(hidden).squeeze(-1)
        atom_contact = hidden.new_zeros(num_atoms)
        edge_count = hidden.new_zeros(num_atoms)
        if hidden.shape[0]:
            atom_hidden = _segment_attention_pool(
                hidden, edge_logits, edge_ligand_index, num_atoms
            )
            edge_contact = torch.sigmoid(self.edge_contact(hidden).squeeze(-1))
            max_contact, _ = scatter_max(
                edge_contact, edge_ligand_index, dim=0, dim_size=num_atoms
            )
            edge_count = scatter_add(
                torch.ones_like(edge_contact), edge_ligand_index, dim=0, dim_size=num_atoms
            )
            seen = edge_count > 0
            atom_contact = torch.where(seen, max_contact, torch.zeros_like(max_contact))
        else:
            atom_hidden = hidden.new_zeros((num_atoms, hidden.shape[-1]))
            seen = torch.zeros(num_atoms, dtype=torch.bool, device=hidden.device)

        atom_features = torch.cat(
            [
                atom_hidden,
                atom_contact.unsqueeze(-1),
                torch.log1p(edge_count).unsqueeze(-1),
                seen.to(hidden.dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        atom_hidden = self.atom_encoder(atom_features)
        ligand_atom_batch = ligand_atom_batch.long()
        graph_hidden = _segment_attention_pool(
            atom_hidden,
            self.atom_attention(atom_hidden).squeeze(-1),
            ligand_atom_batch,
            num_graphs,
        )
        atom_count = scatter_add(
            torch.ones_like(atom_contact), ligand_atom_batch, dim=0, dim_size=num_graphs
        ).clamp_min(1.0)
        mean_contact = scatter_add(
            atom_contact, ligand_atom_batch, dim=0, dim_size=num_graphs
        ) / atom_count
        covered_fraction = scatter_add(
            seen.to(hidden.dtype), ligand_atom_batch, dim=0, dim_size=num_graphs
        ) / atom_count
        pose_hidden = self.pose_encoder(
            torch.cat(
                [graph_hidden, mean_contact.unsqueeze(-1), covered_fraction.unsqueeze(-1)],
                dim=-1,
            )
        )
        score = self.score_head(pose_hidden).squeeze(-1)
        ordinal = self.ordinal_head(pose_hidden)
        logit_lt5 = ordinal[:, 0]
        logit_lt2_given_lt5 = ordinal[:, 1]
        prob_lt5 = torch.sigmoid(logit_lt5)
        prob_lt2 = prob_lt5 * torch.sigmoid(logit_lt2_given_lt5)
        return CrossRankerOutput(
            score=score,
            ordinal_logit_lt5=logit_lt5,
            ordinal_logit_lt2_given_lt5=logit_lt2_given_lt5,
            prob_lt2=prob_lt2,
            prob_lt5=prob_lt5,
            coverage_score=covered_fraction,
        )


class NucleicAcidResidualHead(nn.Module):
    """NA-only residual over ligand-to-base/sugar/phosphate anchor edges."""

    def __init__(
        self,
        anchor_edge_feature_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        *,
        zero_initialize_output: bool = True,
    ) -> None:
        super().__init__()
        self.anchor_encoder = nn.Sequential(
            nn.LayerNorm(anchor_edge_feature_dim),
            nn.Linear(anchor_edge_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        if zero_initialize_output:
            nn.init.zeros_(self.output[-1].weight)
            nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        invariant_anchor_edge_features: torch.Tensor,
        edge_batch: torch.Tensor,
        num_graphs: int,
        is_nucleic_acid: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.anchor_encoder(invariant_anchor_edge_features)
        pooled = _segment_attention_pool(
            hidden, self.attention(hidden).squeeze(-1), edge_batch.long(), num_graphs
        )
        score = self.output(pooled).squeeze(-1)
        return score * is_nucleic_acid.to(score.dtype).view(-1)


class GatedNucleicAcidResidualHead(nn.Module):
    """Small, bounded NA correction conditioned on explicit contact coverage.

    The residual output is exactly zero at initialization, while the gate
    starts small but non-zero so the residual head receives gradients on the
    first step.  No-contact graphs are forced to zero correction.  This makes
    the S2 scorer a strict functional extension of the S1 shared scorer.
    """

    def __init__(
        self,
        anchor_edge_feature_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        *,
        residual_cap: float = 2.0,
        initial_gate: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must lie strictly between zero and one")
        self.residual_cap = float(residual_cap)
        self.anchor_encoder = nn.Sequential(
            nn.LayerNorm(anchor_edge_feature_dim),
            nn.Linear(anchor_edge_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.gate = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, max(16, hidden_dim // 2)),
            nn.SiLU(),
            nn.Linear(max(16, hidden_dim // 2), 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(
            self.gate[-1].bias,
            math.log(initial_gate / (1.0 - initial_gate)),
        )

    def forward(
        self,
        invariant_anchor_edge_features: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_ligand_index: torch.Tensor,
        ligand_atom_batch: torch.Tensor,
        cross_coverage: torch.Tensor,
        num_graphs: int,
        is_nucleic_acid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_batch = edge_batch.long().view(-1)
        edge_ligand_index = edge_ligand_index.long().view(-1)
        ligand_atom_batch = ligand_atom_batch.long().view(-1)
        if invariant_anchor_edge_features.shape[0] != edge_batch.numel():
            raise ValueError("one graph index is required per anchor edge")
        if edge_ligand_index.numel() != edge_batch.numel():
            raise ValueError("one ligand index is required per anchor edge")
        if cross_coverage.shape != (num_graphs,):
            raise ValueError("cross coverage must contain one value per graph")
        hidden = self.anchor_encoder(invariant_anchor_edge_features)
        pooled = _segment_attention_pool(
            hidden, self.attention(hidden).squeeze(-1), edge_batch, num_graphs
        )
        edge_count = scatter_add(
            torch.ones_like(edge_batch, dtype=pooled.dtype),
            edge_batch,
            dim=0,
            dim_size=num_graphs,
        )
        atom_seen = pooled.new_zeros(ligand_atom_batch.numel())
        if edge_ligand_index.numel():
            atom_seen[torch.unique(edge_ligand_index)] = 1.0
        atom_count = scatter_add(
            torch.ones_like(ligand_atom_batch, dtype=pooled.dtype),
            ligand_atom_batch,
            dim=0,
            dim_size=num_graphs,
        ).clamp_min(1.0)
        anchor_atom_coverage = scatter_add(
            atom_seen,
            ligand_atom_batch,
            dim=0,
            dim_size=num_graphs,
        ) / atom_count
        role_coverage = pooled.new_zeros(num_graphs)
        if invariant_anchor_edge_features.shape[0]:
            role = invariant_anchor_edge_features[:, -3:].argmax(dim=-1)
            role_graph = edge_batch * 3 + role
            role_seen = scatter_max(
                torch.ones_like(role_graph, dtype=pooled.dtype),
                role_graph,
                dim=0,
                dim_size=num_graphs * 3,
            )[0].clamp_min(0.0)
            role_coverage = role_seen.view(num_graphs, 3).mean(dim=-1)
        has_contact = (edge_count > 0).to(pooled.dtype)
        gate_features = torch.stack(
            (
                cross_coverage.to(pooled),
                anchor_atom_coverage,
                role_coverage,
                torch.log1p(edge_count) / math.log(31.0),
                has_contact,
            ),
            dim=-1,
        )
        gate = torch.sigmoid(self.gate(gate_features).squeeze(-1)) * has_contact
        gate = gate * is_nucleic_acid.to(gate).view(-1)
        raw_residual = self.residual_head(pooled).squeeze(-1)
        residual = self.residual_cap * torch.tanh(raw_residual) * gate
        return residual, gate


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-4)
    return math.log(math.expm1(value))


class SurfacePoseScoreCombiner(nn.Module):
    """Calibrated combination of shared prior, dynamic residuals and penalties."""

    COMPONENTS = ("prior", "cross", "na", "clash", "strain")

    def __init__(
        self,
        *,
        prior_weight: float = 1.0,
        cross_weight: float = 1.0,
        na_weight: float = 1.0,
        clash_weight: float = 1.0,
        strain_weight: float = 0.25,
    ) -> None:
        super().__init__()
        initial = (prior_weight, cross_weight, na_weight, clash_weight, strain_weight)
        self.raw_weights = nn.Parameter(
            torch.tensor([_inverse_softplus(value) for value in initial], dtype=torch.float32)
        )
        self.register_buffer("component_mean", torch.zeros(len(self.COMPONENTS)))
        self.register_buffer("component_std", torch.ones(len(self.COMPONENTS)))

    def set_calibration(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.numel() != len(self.COMPONENTS) or std.numel() != len(self.COMPONENTS):
            raise ValueError(f"calibration expects {len(self.COMPONENTS)} components")
        self.component_mean.copy_(mean.detach().view(-1).to(self.component_mean))
        self.component_std.copy_(std.detach().view(-1).clamp_min(1e-6).to(self.component_std))

    def forward(
        self,
        prior_score: torch.Tensor,
        cross_score: torch.Tensor,
        na_score: torch.Tensor,
        clash_penalty: torch.Tensor,
        strain_penalty: torch.Tensor,
        is_nucleic_acid: torch.Tensor,
        *,
        ordinal_logit_lt5: torch.Tensor | None = None,
        ordinal_logit_lt2_given_lt5: torch.Tensor | None = None,
        prob_lt2: torch.Tensor | None = None,
        prob_lt5: torch.Tensor | None = None,
        coverage_score: torch.Tensor | None = None,
        na_gate: torch.Tensor | None = None,
    ) -> PoseScoreOutput:
        values = torch.stack(
            [prior_score, cross_score, na_score, clash_penalty, strain_penalty], dim=-1
        )
        normalized = (values - self.component_mean) / self.component_std
        weights = F.softplus(self.raw_weights)
        na_mask = is_nucleic_acid.to(values.dtype).view(-1)
        final = (
            weights[0] * normalized[:, 0]
            + weights[1] * normalized[:, 1]
            + weights[2] * normalized[:, 2] * na_mask
            - weights[3] * normalized[:, 3]
            - weights[4] * normalized[:, 4]
        )
        zeros = final.new_zeros(final.shape)
        return PoseScoreOutput(
            score=final,
            prior_score=prior_score,
            cross_score=cross_score,
            na_score=na_score * na_mask,
            clash_penalty=clash_penalty,
            strain_penalty=strain_penalty,
            ordinal_logit_lt5=(
                ordinal_logit_lt5 if ordinal_logit_lt5 is not None else zeros
            ),
            ordinal_logit_lt2_given_lt5=(
                ordinal_logit_lt2_given_lt5
                if ordinal_logit_lt2_given_lt5 is not None
                else zeros
            ),
            prob_lt2=prob_lt2 if prob_lt2 is not None else zeros,
            prob_lt5=prob_lt5 if prob_lt5 is not None else zeros,
            coverage_score=coverage_score if coverage_score is not None else zeros,
            na_gate=na_gate if na_gate is not None else zeros,
        )
