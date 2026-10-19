"""Feature packing and pairwise loss used by SurfNA W0."""
from collections.abc import Mapping
import torch
from torch.nn import functional as F

def finite(*values):
    for value in values:
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite or non-tensor numerical input; no masking/retry")
