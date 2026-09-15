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

ARMS=("W0",)
LAMBDA_ARMS=()
SHRINKAGE_ARMS=()

def lambda_top3_loss(score, rmsd):
    """Permutation-invariant binary LambdaRank-style surrogate for Top3.

    Soft ranks avoid any pose-ordinal tie break. Lambda weights are detached,
    as in LambdaRank, while the pairwise logistic term supplies gradients.
    """
    good = rmsd < 2
    with torch.no_grad():
        detached = score.detach()
        other_minus_self = detached[:, None, :] - detached[:, :, None]
        rank = .5 + torch.sigmoid(other_minus_self / .25).sum(-1)
        membership = torch.sigmoid((3.5 - rank) / .5)
        discount = membership / torch.log2(rank + 1.)
        weights = membership[:, :, None] * membership[:, None, :]
        weights = weights * (1. + (discount[:, :, None] - discount[:, None, :]).abs())
        weights = weights * (good[:, :, None] & (~good)[:, None, :])
        denominator = weights.sum((-1, -2))
    logits = score[:, :, None] - score[:, None, :]
    numerator = (weights * F.softplus(-logits)).sum((-1, -2))
    loss = torch.where(denominator > 0, numerator / denominator.clamp_min(1e-30),
                       score.sum(-1) * 0)
    BASE.finite(loss, weights)
    return loss, denominator > 0

def objective(arm, score, success, predicted_log_rmsd, rmsd):
    """U1/V0 objective with optional LambdaRank@3 and score shrinkage."""
    if arm not in ARMS:
        raise ValueError('Unknown top-objective arm')
    BASE._check_scores_labels(score, rmsd)
    BASE.finite(success, predicted_log_rmsd)
    if success.shape != score.shape or predicted_log_rmsd.shape != score.shape:
        raise ValueError('Auxiliary head shape mismatch')
    good = rmsd < 2
    mixed = good.any(-1) & (~good).any(-1)
    mass = score.sum(-1) * 0
    if bool(mixed.any()):
        idx = mixed.nonzero(as_tuple=False).flatten()
        s, positive = score[idx], good[idx]
        mass = mass.index_copy(0, idx, torch.logsumexp(s, -1)
            - torch.logsumexp(s.masked_fill(~positive, -torch.inf), -1))
    target = torch.softmax(-rmsd / 1.5, dim=-1)
    continuous = -(target * torch.log_softmax(score, dim=-1)).sum(-1)
    positive_loss = -F.logsigmoid(success)
    negative_loss = -F.logsigmoid(-success)
    positive_den = good.sum(-1).clamp_min(1)
    negative_den = (~good).sum(-1).clamp_min(1)
    classification = ((positive_loss * good).sum(-1) / positive_den
        + (negative_loss * (~good)).sum(-1) / negative_den)
    classification = classification / (good.any(-1).to(score.dtype)
        + (~good).any(-1).to(score.dtype)).clamp_min(1)
    regression = F.smooth_l1_loss(predicted_log_rmsd,
        torch.log1p(rmsd.clamp_max(20)), reduction='none').mean(-1)
    pair, active = BASE.pair_terms(score, rmsd)
    lambda_loss, lambda_active = lambda_top3_loss(score, rmsd)
    centered = score - score.mean(-1, keepdim=True)
    score_variance = centered.square().mean(-1)
    per_group = (mass + .5 * continuous + .25 * classification
                 + .1 * regression + .1 * pair)
    if arm in LAMBDA_ARMS:
        per_group = per_group + .5 * lambda_loss
    if arm in SHRINKAGE_ARMS:
        per_group = per_group + .01 * score_variance
    loss = per_group.mean()
    BASE.finite(loss, mass, continuous, classification, regression, pair,
                lambda_loss, score_variance)
    return dict(loss=loss, mass_mean=mass.mean(), continuous_mean=continuous.mean(),
        lambda_top3_mean=lambda_loss.mean(), score_variance_mean=score_variance.mean(),
        classification_mean=classification.mean(), regression_mean=regression.mean(),
        pairwise_mean=pair.mean(), groups=score.shape[0],
        zero_positive_groups=int((~good.any(-1)).sum()),
        all_positive_groups=int(good.all(-1).sum()), mixed_groups=int(mixed.sum()),
        comparable_pair_groups=int(active.sum()), lambda_active_groups=int(lambda_active.sum()))

def load_scorer(checkpoint, device="cpu"):
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    state=payload["model"]
    model=TopObjectiveScorer(state["feature_mean"],state["feature_std"],state["score_scale"],auxiliary=True)
    model.load_state_dict(state,strict=True)
    if sum(p.numel() for p in model.parameters()) != 62786:
        raise ValueError("Unexpected Scorer architecture")
    model.requires_grad_(False).eval()
    return model.to(device)

