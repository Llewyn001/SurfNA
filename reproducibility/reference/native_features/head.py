"""Small invariant evidence aggregator with an exactly-zero initial residual."""
import torch
from torch import nn
from torch.nn import functional as F


class ContactEvidenceResidual(nn.Module):
    def __init__(self, mean, std, score_scale, hidden=32, dropout=.1, cap=2.):
        super().__init__()
        self.register_buffer('feature_mean',mean.detach().float().clone())
        self.register_buffer('feature_std',std.detach().float().clone())
        self.register_buffer('score_scale',torch.as_tensor(score_scale).detach().float().clone())
        self.cap=cap
        self.atom=nn.Sequential(nn.Linear(len(mean),hidden), nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU())
        self.pose=nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Dropout(dropout),nn.Linear(hidden,1))
        nn.init.zeros_(self.pose[-1].weight)
        nn.init.zeros_(self.pose[-1].bias)

    def forward(self, tokens, mask, baseline):
        # tokens [poses, atoms, 32], never RMSD/reference/identity inputs.
        if not mask.any(-1).all():
            raise ValueError('Every pose must contain at least one unmasked atom')
        x=((tokens-self.feature_mean)/self.feature_std).clamp(-8,8)
        x=self.atom(x)
        mean=(x*mask[...,None]).sum(1)/mask.sum(1,keepdim=True)
        maximum=x.masked_fill(~mask[...,None],-torch.inf).max(1).values
        residual=self.cap*torch.tanh(self.pose(torch.cat([mean,maximum],-1)).squeeze(-1))
        return baseline/self.score_scale+residual, residual


def pair_loss(scores, rmsd, gap=.25):
    """Complex-balanced logistic pairs; lower RMSD means higher score."""
    delta=rmsd[:,None,:]-rmsd[:,:,None]  # [G,i,j]: j is worse than i.
    mask=delta>gap
    crosses=(rmsd[:,:,None]<2)&(rmsd[:,None,:]>=2)
    weight=delta.clamp(0,2)/2*(1+crosses.to(delta.dtype))*mask
    denom=weight.sum((1,2))
    active=denom>0
    loss=F.softplus(-(scores[:,:,None]-scores[:,None,:]))
    per_group=(loss*weight).sum((1,2))/denom.clamp_min(1e-12)
    if not active.any():
        return scores.sum()*0, active
    return per_group[active].mean(),active


def pack(groups, device='cuda'):
    count=len(groups)*8
    atoms=max(g['tokens'].shape[1] for g in groups)
    width=groups[0]['tokens'].shape[-1]
    x=torch.zeros((count,atoms,width),dtype=torch.float32,device=device)
    mask=torch.zeros((count,atoms),dtype=torch.bool,device=device)
    for i,g in enumerate(groups):
        n=g['tokens'].shape[1]
        x[i*8:(i+1)*8,:n]=g['tokens'].to(device)
        mask[i*8:(i+1)*8,:n]=True
    baseline=torch.stack([g['baseline'] for g in groups]).to(device).reshape(-1)
    rmsd=torch.tensor([g['rmsd'] for g in groups],dtype=torch.float32,device=device)
    return x,mask,baseline,rmsd
