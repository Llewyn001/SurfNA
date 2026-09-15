"""Feature packing and pairwise loss used by SurfNA W0."""
from collections.abc import Mapping
import torch
from torch.nn import functional as F

def finite(*values):
    for value in values:
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite or non-tensor numerical input; no masking/retry")

def validate_feature_group(group, *, k=None, width=None):
    """Strict four-key transport schema; labels/extra metadata are forbidden."""
    if not isinstance(group, Mapping) or set(group) != {"name", "split", "tokens", "baseline"}:
        raise ValueError("Feature group must contain only name/split/tokens/baseline")
    if (not isinstance(group["name"], str) or not group["name"]
            or group["split"] not in ("train", "val", "test")):
        raise ValueError("Invalid feature identity metadata")
    x, baseline = group["tokens"], group["baseline"]
    finite(x, baseline)
    if (x.dtype != torch.float32 or baseline.dtype != torch.float32
            or x.device.type != "cpu" or baseline.device.type != "cpu"
            or x.ndim != 3 or min(x.shape) <= 0 or baseline.shape != (x.shape[0],)
            or (k is not None and x.shape[0] != k)
            or (width is not None and x.shape[-1] != width)):
        raise ValueError("Feature shape/dtype/device mismatch; no coercion or skipping")
    return tuple(x.shape)

def pack_features(groups, device="cpu"):
    """Pack one equal-K batch; no labels or manifest are accepted or returned."""
    if not groups:
        raise ValueError("Empty feature batch")
    k, _n, width = validate_feature_group(groups[0])
    shapes = [validate_feature_group(group, k=k, width=width) for group in groups]
    if len({group["name"] for group in groups}) != len(groups):
        raise ValueError("Repeated group in batch")
    atoms = max(shape[1] for shape in shapes)
    x = torch.zeros((len(groups)*k, atoms, width), dtype=torch.float32, device=device)
    mask = torch.zeros((len(groups)*k, atoms), dtype=torch.bool, device=device)
    for i, (group, shape) in enumerate(zip(groups, shapes)):
        x[i*k:(i+1)*k, :shape[1]] = group["tokens"].to(device)
        mask[i*k:(i+1)*k, :shape[1]] = True
    baseline = torch.stack([group["baseline"] for group in groups]).to(device).reshape(-1)
    return x, mask, baseline

def _check_scores_labels(scores, rmsd):
    finite(scores, rmsd)
    if scores.ndim != 2 or scores.shape != rmsd.shape or min(scores.shape) <= 0 or bool((rmsd < 0).any()):
        raise ValueError("Invalid score/independent-label matrix")

def pair_terms(scores, rmsd):
    """Original per-group weighted logistic pair formula (before reduction)."""
    _check_scores_labels(scores, rmsd)
    delta = rmsd[:, None, :]-rmsd[:, :, None]
    mask = delta > 0.25
    crosses = (rmsd[:, :, None] < 2) & (rmsd[:, None, :] >= 2)
    weight = delta.clamp(0, 2)/2*(1+crosses.to(delta.dtype))*mask
    denom = weight.sum((1, 2))
    loss = F.softplus(-(scores[:, :, None]-scores[:, None, :]))
    per_group = (loss*weight).sum((1, 2))/denom.clamp_min(1e-12)
    finite(per_group)
    return per_group, denom > 0
