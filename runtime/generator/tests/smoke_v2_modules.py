import torch

from models.surfna_v2_modules import LigandAnchoredPatchTokenizer, NuSurfAnchorFusion


def rotation_matrix():
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q


def main():
    torch.manual_seed(7)
    dim = 16
    surface = torch.randn(9, dim, requires_grad=True)
    surface_pos = torch.randn(9, 3)
    surface_batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])
    ligand = torch.randn(5, dim, requires_grad=True)
    ligand_pos = torch.randn(5, 3)
    ligand_batch = torch.tensor([0, 0, 0, 1, 1])
    tokenizer = LigandAnchoredPatchTokenizer(dim).eval()
    ligand_out, logits = tokenizer(ligand, ligand_pos, ligand_batch, surface, surface_pos, surface_batch)

    anchor_pos = torch.randn(6, 3)
    anchor_type = torch.tensor([0, 1, 2, 0, 1, 2])
    anchor_parent_index = torch.tensor([0, 0, 1, 2, 3, 3])
    receptor_batch = torch.tensor([0, 0, 1, 1])
    nuc_feat = torch.randn(4, 12)
    fusion = NuSurfAnchorFusion(dim).eval()
    fused_surface, fused_anchor = fusion(surface, surface_pos, surface_batch, anchor_pos, anchor_type,
                                         anchor_parent_index, receptor_batch, nuc_feat)
    loss = ligand_out.square().mean() + logits.square().mean() + fused_surface.square().mean() + fused_anchor.square().mean()
    loss.backward()
    assert torch.isfinite(surface.grad).all() and torch.isfinite(ligand.grad).all()

    rot = rotation_matrix()
    translated = torch.tensor([2.0, -3.0, 1.0])
    ligand_r, logits_r = tokenizer(ligand.detach(), ligand_pos @ rot + translated, ligand_batch,
                                   surface.detach(), surface_pos @ rot + translated, surface_batch)
    surface_r, anchor_r = fusion(surface.detach(), surface_pos @ rot + translated, surface_batch,
                                 anchor_pos @ rot + translated, anchor_type, anchor_parent_index,
                                 receptor_batch, nuc_feat)
    assert torch.allclose(ligand_out.detach(), ligand_r, atol=1e-5)
    assert torch.allclose(logits.detach(), logits_r, atol=1e-5)
    assert torch.allclose(fused_surface.detach(), surface_r, atol=1e-5)
    assert torch.allclose(fused_anchor.detach(), anchor_r, atol=1e-5)
    print('SurfNA V2 module smoke passed: finite gradients and SE(3)-invariant scalar outputs.')


if __name__ == '__main__':
    main()
