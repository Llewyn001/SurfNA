"""Invariant SurfNA V2 interaction modules.

Both modules use only scalar features and inter-point distances, so their
outputs are invariant to a common SE(3) transform of ligand, surface and
nucleotide-anchor coordinates.  They deliberately augment, rather than
replace, the equivariant DiffDock/SurfDock backbone.
"""

import math

import torch
from torch import nn


class LigandAnchoredPatchTokenizer(nn.Module):
    """Pools ligand-centred soft surface patches into one token per ligand atom."""

    def __init__(self, scalar_dim: int, temperature: float = 2.5):
        super().__init__()
        self.temperature = temperature
        self.surface_value = nn.Linear(scalar_dim, scalar_dim, bias=False)
        self.token_norm = nn.LayerNorm(scalar_dim)
        self.ligand_update = nn.Linear(scalar_dim, scalar_dim, bias=False)
        self.gate = nn.Sequential(nn.Linear(2 * scalar_dim, scalar_dim), nn.Sigmoid())
        self.contact_head = nn.Linear(scalar_dim, 1)

    def forward(self, ligand_scalar, ligand_pos, ligand_batch,
                surface_scalar, surface_pos, surface_batch):
        updated = ligand_scalar.clone()
        contact_logits = ligand_scalar.new_zeros((ligand_scalar.shape[0],))
        for graph_id in torch.unique(ligand_batch):
            lig_idx = torch.where(ligand_batch == graph_id)[0]
            surf_idx = torch.where(surface_batch == graph_id)[0]
            if lig_idx.numel() == 0 or surf_idx.numel() == 0:
                continue
            distance = torch.cdist(ligand_pos[lig_idx], surface_pos[surf_idx])
            assignment = torch.softmax(-distance / self.temperature, dim=-1)
            tokens = self.token_norm(assignment @ self.surface_value(surface_scalar[surf_idx]))
            gate = self.gate(torch.cat([ligand_scalar[lig_idx], tokens], dim=-1))
            updated[lig_idx] = ligand_scalar[lig_idx] + gate * self.ligand_update(tokens)
            # Autocast may evaluate the head in BF16 while contact_logits
            # inherits the FP32 ligand dtype. Index assignment requires an
            # exact dtype match, so cast without breaking the gradient path.
            contact_logits[lig_idx] = self.contact_head(tokens).squeeze(-1).to(
                dtype=contact_logits.dtype
            )
        return updated, contact_logits


class NuSurfAnchorFusion(nn.Module):
    """Bidirectional fusion between surface scalars and base/sugar/phosphate anchors."""

    def __init__(self, scalar_dim: int, nucleic_feat_dim: int = 12,
                 distance_scale: float = 6.0):
        super().__init__()
        self.distance_scale = distance_scale
        self.role_embedding = nn.Embedding(3, scalar_dim)  # base, sugar, phosphate
        self.nucleic_projection = nn.Linear(nucleic_feat_dim, scalar_dim, bias=False)
        self.surface_query = nn.Linear(scalar_dim, scalar_dim, bias=False)
        self.anchor_key = nn.Linear(scalar_dim, scalar_dim, bias=False)
        self.anchor_value = nn.Linear(scalar_dim, scalar_dim, bias=False)
        self.surface_value = nn.Linear(scalar_dim, scalar_dim, bias=False)
        self.surface_gate = nn.Sequential(nn.Linear(2 * scalar_dim, scalar_dim), nn.Sigmoid())
        self.anchor_gate = nn.Sequential(nn.Linear(2 * scalar_dim, scalar_dim), nn.Sigmoid())

    def forward(self, surface_scalar, surface_pos, surface_batch,
                anchor_pos, anchor_type, anchor_parent_index, receptor_batch,
                receptor_nucleic_feat):
        if anchor_pos.numel() == 0:
            return surface_scalar, anchor_pos.new_zeros((0, surface_scalar.shape[-1]))
        anchor_scalar = self.role_embedding(anchor_type.long())
        if receptor_nucleic_feat is not None and receptor_nucleic_feat.numel() > 0:
            anchor_scalar = anchor_scalar + self.nucleic_projection(
                receptor_nucleic_feat[anchor_parent_index].float()
            )
        surface_out = surface_scalar.clone()
        anchor_out = anchor_scalar.clone()
        anchor_batch = receptor_batch[anchor_parent_index]
        scale = math.sqrt(surface_scalar.shape[-1])
        for graph_id in torch.unique(surface_batch):
            surf_idx = torch.where(surface_batch == graph_id)[0]
            anchor_idx = torch.where(anchor_batch == graph_id)[0]
            if surf_idx.numel() == 0 or anchor_idx.numel() == 0:
                continue
            distance = torch.cdist(surface_pos[surf_idx], anchor_pos[anchor_idx])
            distance_bias = -(distance / self.distance_scale).square()
            affinity = (self.surface_query(surface_scalar[surf_idx]) @
                        self.anchor_key(anchor_scalar[anchor_idx]).transpose(0, 1) / scale +
                        distance_bias)
            surface_attention = torch.softmax(affinity, dim=-1)
            surface_message = surface_attention @ self.anchor_value(anchor_scalar[anchor_idx])
            surface_gate = self.surface_gate(torch.cat([surface_scalar[surf_idx], surface_message], dim=-1))
            surface_out[surf_idx] = surface_scalar[surf_idx] + surface_gate * surface_message

            anchor_attention = torch.softmax(affinity.transpose(0, 1), dim=-1)
            anchor_message = anchor_attention @ self.surface_value(surface_scalar[surf_idx])
            anchor_gate = self.anchor_gate(torch.cat([anchor_scalar[anchor_idx], anchor_message], dim=-1))
            anchor_out[anchor_idx] = anchor_scalar[anchor_idx] + anchor_gate * anchor_message
        return surface_out, anchor_out
