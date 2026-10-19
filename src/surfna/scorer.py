"""Frozen SurfNA W0 ranking architecture; 711 atom features and K=40."""
import torch
from torch import nn
from torch.nn import functional as F
from . import scorer_common as BASE

class TopObjectiveScorer(nn.Module):
    """Encode each pose independently and emit an unbounded ranking score."""
    def __init__(self, mean, std, score_scale, *, auxiliary: bool):
        super().__init__()
        mean, std, scale = (torch.as_tensor(x).detach().float().clone()
                            for x in (mean, std, score_scale))
        BASE.finite(mean, std, scale)
        if mean.shape != (711,) or std.shape != mean.shape or not bool((std > 0).all()):
            raise ValueError('Invalid frozen statistics')
        self.register_buffer('feature_mean', mean)
        self.register_buffer('feature_std', std)
        self.register_buffer('score_scale', scale)
        self.auxiliary = bool(auxiliary)
        width = 64
        self.atom = nn.Sequential(nn.Linear(711, width), nn.SiLU(),
                                  nn.Linear(width, width), nn.SiLU())
        self.pose = nn.Sequential(nn.Linear(width * 2 + 3, 96), nn.SiLU(), nn.LayerNorm(96))
        self.success = nn.Linear(96, 1)
        self.log_rmsd = nn.Linear(96, 1)
        for head in (self.success, self.log_rmsd):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)

    def forward(self, tokens, mask, baseline, *, groups: int, k: int = 40):
        if (tokens.ndim != 3 or tokens.shape[-1] != 711 or mask.shape != tokens.shape[:2]
                or mask.dtype != torch.bool or baseline.shape != (tokens.shape[0],)
                or groups <= 0 or k != 40 or tokens.shape[0] != groups * k
                or not bool(mask.any(-1).all())):
            raise ValueError('Invalid candidate-set tensors')
        BASE.finite(tokens, baseline)
        x = self.atom(((tokens - self.feature_mean) / self.feature_std).clamp(-8, 8))
        mean = (x * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        maximum = x.masked_fill(~mask[..., None], -torch.inf).max(1).values
        b = (baseline / self.score_scale).reshape(groups, k)
        centered = b - b.mean(-1, keepdim=True)
        scale = b.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        # Pairwise empirical rank is invariant to input order, including exact
        # baseline-score ties.  It is a candidate-set statistic, not an ordinal.
        delta = b[..., None] - b[:, None, :]
        rank = ((delta > 0).to(b.dtype) + .5 * (delta == 0).to(b.dtype)).mean(-1)
        scalars = torch.stack((b, centered / scale, rank * 2 - 1), -1)
        z = self.pose(torch.cat((mean, maximum, scalars.reshape(groups * k, 3)), -1))
        z = z.reshape(groups, k, -1)
        success = self.success(z).squeeze(-1)
        log_rmsd = self.log_rmsd(z).squeeze(-1)
        score = success - .25 * log_rmsd if self.auxiliary else success
        BASE.finite(score, success, log_rmsd)
        return score, success, log_rmsd

def load_scorer(checkpoint, device="cpu"):
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    state=payload["model"]
    model=TopObjectiveScorer(state["feature_mean"],state["feature_std"],state["score_scale"],auxiliary=True)
    model.load_state_dict(state,strict=True)
    if sum(p.numel() for p in model.parameters()) != 62786:
        raise ValueError("Unexpected Scorer architecture")
    model.requires_grad_(False).eval()
    return model.to(device)

