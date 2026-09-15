"""SurfNA G2.1/G2.2 interaction modules.

The ligand-patch block is receptor-domain agnostic and is therefore shared by
protein pretraining and NA fine-tuning.  NuSurf is an NA-only residual branch.
Both consume only invariant scalars, distances and equivariant local-frame
projections, preserving the SE(3) contract of the SurfDock backbone.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _masked_softmax(logits, mask, dim):
    """Softmax that returns zeros when an entire row is masked."""
    masked_logits = logits.masked_fill(~mask, -1e9)
    weights = torch.softmax(masked_logits, dim=dim) * mask.to(logits.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)


def _anchor_local_frames(anchor_pos, anchor_type, anchor_parent_index):
    """Build available base/sugar/phosphate frames and an explicit validity mask.

    Cached G2 graphs do not retain atom-level base planes.  G2.1 therefore keeps
    the legacy base--sugar/backbone frame for this run, but invalid terminal or
    incomplete residues are no longer silently interpreted as zero orientation.
    A future atom-enriched cache can replace the frame without changing NuSurf's
    interface.
    """
    frames = anchor_pos.new_zeros((anchor_pos.shape[0], 3, 3))
    valid = torch.zeros(anchor_pos.shape[0], dtype=torch.bool, device=anchor_pos.device)
    for parent in torch.unique(anchor_parent_index):
        parent_idx = torch.where(anchor_parent_index == parent)[0]
        base_idx = parent_idx[anchor_type[parent_idx] == 0]
        sugar_idx = parent_idx[anchor_type[parent_idx] == 1]
        phosphate_idx = parent_idx[anchor_type[parent_idx] == 2]
        if base_idx.numel() == 0 or sugar_idx.numel() == 0 or phosphate_idx.numel() == 0:
            continue
        base = anchor_pos[base_idx[0]]
        sugar = anchor_pos[sugar_idx[0]]
        phosphate = anchor_pos[phosphate_idx[0]]
        base_axis = F.normalize(base - sugar, dim=0, eps=1e-8)
        backbone_raw = phosphate - sugar
        backbone_axis = backbone_raw - torch.dot(backbone_raw, base_axis) * base_axis
        backbone_norm = torch.linalg.vector_norm(backbone_axis)
        if not torch.isfinite(backbone_norm) or float(backbone_norm.detach()) < 1e-6:
            continue
        backbone_axis = F.normalize(backbone_axis, dim=0, eps=1e-8)
        normal_axis = F.normalize(torch.cross(base_axis, backbone_axis, dim=0), dim=0, eps=1e-8)
        frame = torch.stack([base_axis, backbone_axis, normal_axis], dim=0)
        if not torch.isfinite(frame).all():
            continue
        frames[parent_idx] = frame
        valid[parent_idx] = True
    return frames, valid


class LigandAnchoredPatchTokenizer(nn.Module):
    """Multi-scale ligand-centred surface patches shared across receptor domains."""

    def __init__(self, scalar_dim: int, temperature: float = 2.5,
                 precision_mode: bool = False, cutoff: float = 8.0,
                 topk: int = 32, time_gate_center: float = 0.5,
                 time_gate_width: float = 0.1, multiscale_cutoffs=None):
        super().__init__()
        self.temperature = float(temperature)
        self.precision_mode = precision_mode
        self.cutoff = float(cutoff)
        self.topk = int(topk)
        self.time_gate_center = float(time_gate_center)
        self.time_gate_width = float(time_gate_width)
        cutoffs = list(multiscale_cutoffs or [cutoff])
        self.cutoffs = tuple(float(value) for value in cutoffs)
        self.surface_values = nn.ModuleList([
            nn.Linear(scalar_dim, scalar_dim, bias=False) for _ in self.cutoffs
        ])
        self.token_norms = nn.ModuleList([nn.LayerNorm(scalar_dim) for _ in self.cutoffs])
        self.ligand_updates = nn.ModuleList([
            nn.Linear(scalar_dim, scalar_dim, bias=False) for _ in self.cutoffs
        ])
        self.scale_gate = nn.Linear(2 * scalar_dim, len(self.cutoffs))
        self.gate = nn.Sequential(nn.Linear(2 * scalar_dim, scalar_dim), nn.Sigmoid())
        self.contact_head = nn.Linear(scalar_dim, 1)
        self.null_tokens = nn.Parameter(torch.zeros(len(self.cutoffs), scalar_dim))
        self.null_logits = nn.Parameter(torch.zeros(len(self.cutoffs)))
        self.last_metrics = {}

    def _scale_token(self, scale_index, ligand_count, distance, surface_scalar):
        surface_values = self.surface_values[scale_index](surface_scalar)
        cutoff = self.cutoffs[scale_index]
        if self.precision_mode:
            local_k = min(max(1, self.topk), surface_scalar.shape[0])
            local_distance, local_index = torch.topk(
                distance, k=local_k, dim=-1, largest=False, sorted=False
            )
            local_values = surface_values[local_index]
            valid = local_distance <= cutoff
            scale_temperature = max(self.temperature * cutoff / max(self.cutoff, 1e-6), 1e-6)
            local_logits = (-local_distance / scale_temperature).masked_fill(~valid, -1e9)
            null_logits = self.null_logits[scale_index].expand(ligand_count, 1)
            assignment = torch.softmax(torch.cat([local_logits, null_logits], dim=-1), dim=-1)
            token = (assignment[:, :-1, None] * local_values).sum(dim=1)
            token = token + assignment[:, -1, None] * self.null_tokens[scale_index]
            entropy = -(assignment * assignment.clamp_min(1e-8).log()).sum(dim=-1).mean()
            return self.token_norms[scale_index](token), assignment[:, -1].mean(), entropy
        assignment = torch.softmax(-distance / max(self.temperature, 1e-6), dim=-1)
        token = assignment @ surface_values
        entropy = -(assignment * assignment.clamp_min(1e-8).log()).sum(dim=-1).mean()
        return self.token_norms[scale_index](token), token.new_zeros(()), entropy

    def forward(self, ligand_scalar, ligand_pos, ligand_batch,
                surface_scalar, surface_pos, surface_batch, graph_t=None):
        updated = ligand_scalar.clone()
        contact_logits = ligand_scalar.new_zeros((ligand_scalar.shape[0],))
        null_weights = []
        patch_entropies = []
        time_gates = []
        scale_weight_sums = ligand_scalar.new_zeros((len(self.cutoffs),))
        scale_weight_count = 0
        for graph_id in torch.unique(ligand_batch):
            lig_idx = torch.where(ligand_batch == graph_id)[0]
            surf_idx = torch.where(surface_batch == graph_id)[0]
            if lig_idx.numel() == 0 or surf_idx.numel() == 0:
                continue
            distance = torch.cdist(ligand_pos[lig_idx], surface_pos[surf_idx])
            scale_tokens = []
            for scale_index in range(len(self.cutoffs)):
                token, null_weight, entropy = self._scale_token(
                    scale_index, lig_idx.numel(), distance, surface_scalar[surf_idx]
                )
                scale_tokens.append(token)
                null_weights.append(null_weight)
                patch_entropies.append(entropy)
            token_stack = torch.stack(scale_tokens, dim=1)
            token_summary = token_stack.mean(dim=1)
            scale_weights = torch.softmax(
                self.scale_gate(torch.cat([ligand_scalar[lig_idx], token_summary], dim=-1)), dim=-1
            )
            scale_weight_sums += scale_weights.detach().sum(dim=0)
            scale_weight_count += int(scale_weights.shape[0])
            fused_token = (scale_weights[..., None] * token_stack).sum(dim=1)
            fused_update = torch.stack([
                layer(scale_tokens[index]) for index, layer in enumerate(self.ligand_updates)
            ], dim=1)
            fused_update = (scale_weights[..., None] * fused_update).sum(dim=1)
            gate = self.gate(torch.cat([ligand_scalar[lig_idx], fused_token], dim=-1))
            if self.precision_mode and graph_t is not None:
                graph_index = int(graph_id.item())
                t_value = graph_t[graph_index].to(gate)
                width = max(self.time_gate_width, 1e-3)
                time_gate = torch.sigmoid((self.time_gate_center - t_value) / width)
                gate = gate * time_gate
                time_gates.append(time_gate)
            updated[lig_idx] = ligand_scalar[lig_idx] + gate * fused_update
            contact_logits[lig_idx] = self.contact_head(fused_token).squeeze(-1)
        self.last_metrics = {
            'null_weight': torch.stack(null_weights).mean().detach() if null_weights else ligand_scalar.new_zeros(()),
            'patch_entropy': torch.stack(patch_entropies).mean().detach() if patch_entropies else ligand_scalar.new_zeros(()),
            'time_gate': torch.stack(time_gates).mean().detach() if time_gates else ligand_scalar.new_ones(()),
            'scale_weights': (scale_weight_sums / max(scale_weight_count, 1)).detach(),
        }
        return updated, contact_logits


class NuSurfAnchorFusion(nn.Module):
    """Validity-aware bidirectional fusion between surface and NA anchors."""

    def __init__(self, scalar_dim: int, nucleic_feat_dim: int = 12,
                 distance_scale: float = 6.0, precision_mode: bool = False,
                 cutoff: float = 10.0, directional_value: bool = False):
        super().__init__()
        self.distance_scale = distance_scale
        self.precision_mode = precision_mode
        self.cutoff = cutoff
        self.directional_value = bool(directional_value)
        self.role_embedding = nn.Embedding(3, scalar_dim)
        self.nucleic_projection = nn.Linear(nucleic_feat_dim, scalar_dim, bias=False)
        self.surface_query = nn.Linear(scalar_dim, scalar_dim, bias=False)
        self.anchor_key = nn.Linear(scalar_dim, scalar_dim, bias=False)
        self.anchor_value = nn.Linear(scalar_dim, scalar_dim, bias=False)
        self.surface_value = nn.Linear(scalar_dim, scalar_dim, bias=False)
        self.surface_gate = nn.Sequential(nn.Linear(2 * scalar_dim, scalar_dim), nn.Sigmoid())
        self.anchor_gate = nn.Sequential(nn.Linear(2 * scalar_dim, scalar_dim), nn.Sigmoid())
        self.state_projection = nn.Linear(scalar_dim, scalar_dim, bias=False)
        orientation_hidden = max(8, scalar_dim // 2)
        self.orientation_bias = nn.Sequential(
            nn.Linear(3, orientation_hidden, bias=False), nn.SiLU(),
            nn.Linear(orientation_hidden, 1, bias=False),
        )
        self.orientation_value = None
        if self.directional_value:
            self.orientation_value = nn.Sequential(
                nn.Linear(3, orientation_hidden, bias=False), nn.SiLU(),
                nn.Linear(orientation_hidden, scalar_dim, bias=False),
            )
            # Start exactly at the G2.1 function and let training introduce the
            # direction-conditioned residual gradually.
            nn.init.zeros_(self.orientation_value[-1].weight)
        self.last_metrics = {}

    def forward(self, surface_scalar, surface_pos, surface_batch,
                anchor_pos, anchor_type, anchor_parent_index, receptor_batch,
                receptor_nucleic_feat, anchor_scalar_state=None):
        if anchor_pos.numel() == 0:
            self.last_metrics = {'frame_valid_fraction': surface_scalar.new_zeros(())}
            return surface_scalar, anchor_pos.new_zeros((0, surface_scalar.shape[-1]))
        anchor_scalar = self.role_embedding(anchor_type.long())
        if receptor_nucleic_feat is not None and receptor_nucleic_feat.numel() > 0:
            anchor_scalar = anchor_scalar + self.nucleic_projection(
                receptor_nucleic_feat[anchor_parent_index].float()
            )
        if self.precision_mode and anchor_scalar_state is not None:
            if anchor_scalar_state.shape != anchor_scalar.shape:
                raise ValueError(
                    f'NuSurf anchor state shape mismatch: {anchor_scalar_state.shape} != {anchor_scalar.shape}'
                )
            anchor_scalar = anchor_scalar + self.state_projection(anchor_scalar_state)
        surface_out = surface_scalar.clone()
        anchor_out = anchor_scalar.clone()
        anchor_batch = receptor_batch[anchor_parent_index]
        if self.precision_mode:
            anchor_frames, frame_valid = _anchor_local_frames(
                anchor_pos, anchor_type.long(), anchor_parent_index.long()
            )
        else:
            anchor_frames = None
            frame_valid = torch.ones(anchor_pos.shape[0], dtype=torch.bool, device=anchor_pos.device)
        directional_norms = []
        self.last_metrics = {'frame_valid_fraction': frame_valid.float().mean().detach()}
        scale = math.sqrt(surface_scalar.shape[-1])
        for graph_id in torch.unique(surface_batch):
            surf_idx = torch.where(surface_batch == graph_id)[0]
            anchor_idx = torch.where(anchor_batch == graph_id)[0]
            if surf_idx.numel() == 0 or anchor_idx.numel() == 0:
                continue
            distance = torch.cdist(surface_pos[surf_idx], anchor_pos[anchor_idx])
            distance_bias = -(distance / self.distance_scale).square()
            affinity = (
                self.surface_query(surface_scalar[surf_idx]) @
                self.anchor_key(anchor_scalar[anchor_idx]).transpose(0, 1) / scale + distance_bias
            )
            if self.precision_mode:
                direction = F.normalize(
                    surface_pos[surf_idx, None, :] - anchor_pos[anchor_idx][None, :, :],
                    dim=-1, eps=1e-8,
                )
                orientation = torch.einsum('sad,akd->sak', direction, anchor_frames[anchor_idx])
                orientation_bias = self.orientation_bias(orientation).squeeze(-1)
                orientation_bias = orientation_bias * frame_valid[anchor_idx][None, :].to(orientation_bias)
                affinity = affinity + orientation_bias
                surface_mask = distance <= float(self.cutoff)
                anchor_mask = surface_mask.transpose(0, 1)
                surface_attention = _masked_softmax(affinity, surface_mask, dim=-1)
            else:
                surface_attention = torch.softmax(affinity, dim=-1)
            surface_message = surface_attention @ self.anchor_value(anchor_scalar[anchor_idx])
            if self.directional_value and self.precision_mode:
                directional_pair = self.orientation_value(orientation)
                directional_pair = directional_pair * frame_valid[anchor_idx][None, :, None].to(
                    directional_pair
                )
                directional_message = (
                    surface_attention[..., None] * directional_pair
                ).sum(dim=1)
                surface_message = surface_message + directional_message
                directional_norms.append(
                    torch.linalg.vector_norm(directional_message, dim=-1).mean().detach()
                )
            surface_gate = self.surface_gate(torch.cat([surface_scalar[surf_idx], surface_message], dim=-1))
            surface_out[surf_idx] = surface_scalar[surf_idx] + surface_gate * surface_message

            anchor_attention = (
                _masked_softmax(affinity.transpose(0, 1), anchor_mask, dim=-1)
                if self.precision_mode else torch.softmax(affinity.transpose(0, 1), dim=-1)
            )
            anchor_message = anchor_attention @ self.surface_value(surface_scalar[surf_idx])
            anchor_gate = self.anchor_gate(torch.cat([anchor_scalar[anchor_idx], anchor_message], dim=-1))
            anchor_out[anchor_idx] = anchor_scalar[anchor_idx] + anchor_gate * anchor_message
        self.last_metrics['directional_value_norm'] = (
            torch.stack(directional_norms).mean()
            if directional_norms else surface_scalar.new_zeros(())
        )
        return surface_out, anchor_out
