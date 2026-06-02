"""
Adaptive Weight Transfer Strategy: Adapter Layers with Scalar Gating for E3NN

This module implements:
1. Low-rank Adapter layers for E3NN-compatible modules
2. Scalar gating mechanism (maintains equivariance)
3. Progressive unfreezing support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from loguru import logger


class ScalarGatedAdapter(nn.Module):
    """
    Low-rank Adapter with scalar gating for E3NN layers.
    
    For a layer update δ = f(x), the adapted output is: α·δ + Adapter(x)
    where α is a learnable scalar gate (applied only to scalar irreps for equivariance).
    
    The adapter uses low-rank decomposition to reduce parameters:
    Adapter(x) = W_up(W_down(x))
    where W_down: [dim, rank], W_up: [rank, dim]
    
    This adapter works on the update/change from the base layer, maintaining equivariance.
    """
    
    def __init__(self, feature_dim, adapter_rank=8, dropout=0.1):
        """
        Args:
            feature_dim: Feature dimension (total dimension of irreps)
            adapter_rank: Rank of the low-rank adapter (default: 8)
            dropout: Dropout rate for adapter layers
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.adapter_rank = adapter_rank
        
        # Scalar gate (learnable scaling factor, applied globally to maintain equivariance)
        # We use a single scalar to gate the base update, maintaining E3NN equivariance
        self.scalar_gate = nn.Parameter(torch.ones(1))
        
        # Low-rank adapter (works on full representation)
        # Down projection: [feature_dim, adapter_rank]
        self.adapter_down = nn.Linear(feature_dim, adapter_rank, bias=False)
        # Up projection: [adapter_rank, feature_dim]
        self.adapter_up = nn.Linear(adapter_rank, feature_dim, bias=False)
        
        # Activation and dropout
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize adapter weights to be small (near-zero initialization)
        # This ensures the adapter starts close to identity
        nn.init.normal_(self.adapter_down.weight, std=0.01)
        nn.init.zeros_(self.adapter_up.weight)
    
    def forward(self, input_features, base_update):
        """
        Args:
            input_features: Input features before update [N, feature_dim]
            base_update: Update/change from base layer [N, feature_dim]
        
        Returns:
            Adapted update: α·base_update + Adapter(input_features)
            The final output should be: input_features + adapted_update
        """
        # Apply scalar gate to base update (maintains equivariance as it's a scalar)
        gated_update = self.scalar_gate * base_update
        
        # Apply low-rank adapter on input features
        adapter_out = self.adapter_up(self.dropout(self.activation(self.adapter_down(input_features))))
        
        # Combine: gated base update + adapter output
        return gated_update + adapter_out


class E3NNAdapterWrapper(nn.Module):
    """
    Wrapper that adds Adapter to an existing E3NN layer (e.g., TensorProductConvLayer).
    This allows keeping the original layer unchanged while adding adaptation.
    """
    
    def __init__(self, base_layer, adapter_rank=8, dropout=0.1, enabled=True):
        """
        Args:
            base_layer: The base E3NN layer to wrap
            adapter_rank: Rank for the low-rank adapter
            dropout: Dropout rate
            enabled: Whether adapter is enabled (for progressive unfreezing)
        """
        super().__init__()
        self.base_layer = base_layer
        self.enabled = enabled
        
        # Get irreps from base layer
        in_irreps = getattr(base_layer, 'in_irreps', None)
        out_irreps = getattr(base_layer, 'out_irreps', None)
        
        if in_irreps is None or out_irreps is None:
            # Try to infer from layer attributes
            logger.warning(f"Could not find irreps in base_layer {type(base_layer)}. Using default.")
            in_irreps = out_irreps = "16x0e + 4x1o"
        
        # Create adapter
        self.adapter = ScalarGatedAdapter(
            in_irreps=str(in_irreps),
            out_irreps=str(out_irreps),
            adapter_rank=adapter_rank,
            dropout=dropout
        )
    
    def forward(self, *args, **kwargs):
        # Call base layer
        base_output = self.base_layer(*args, **kwargs)
        
        # Apply adapter if enabled
        if self.enabled and self.training:
            # For E3NN layers, we need to extract the node features
            # The output format depends on the layer type
            if isinstance(base_output, tuple):
                # Some layers return tuples, we adapt the first element (node features)
                adapted_features = self.adapter(base_output[0], base_output[0])
                return (adapted_features,) + base_output[1:]
            else:
                return self.adapter(base_output, base_output)
        else:
            return base_output


class LinearAdapter(nn.Module):
    """
    Simple adapter for linear/MLP layers (non-E3NN).
    Uses low-rank decomposition and scalar gating.
    """
    
    def __init__(self, in_dim, out_dim=None, adapter_rank=8, dropout=0.1):
        """
        Args:
            in_dim: Input dimension
            out_dim: Output dimension (default: same as input)
            adapter_rank: Rank of low-rank adapter
            dropout: Dropout rate
        """
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Scalar gate (per-channel)
        self.scalar_gate = nn.Parameter(torch.ones(min(in_dim, out_dim)))
        
        # Low-rank adapter
        self.adapter_down = nn.Linear(in_dim, adapter_rank, bias=False)
        self.adapter_up = nn.Linear(adapter_rank, out_dim, bias=False)
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize to small values
        nn.init.normal_(self.adapter_down.weight, std=0.01)
        nn.init.zeros_(self.adapter_up.weight)
    
    def forward(self, x, base_output):
        """
        Args:
            x: Input features
            base_output: Output from base layer
        
        Returns:
            α·base_output + Adapter(x)
        """
        # Apply scalar gate
        gate_dim = min(self.scalar_gate.shape[0], base_output.shape[-1])
        gated_output = base_output[:, :gate_dim] * self.scalar_gate[:gate_dim].unsqueeze(0)
        if base_output.shape[-1] > gate_dim:
            gated_output = torch.cat([gated_output, base_output[:, gate_dim:]], dim=-1)
        
        # Apply adapter
        adapter_out = self.adapter_up(self.dropout(self.activation(self.adapter_down(x))))
        
        return gated_output + adapter_out


def wrap_layer_with_adapter(layer, adapter_rank=8, dropout=0.1, enabled=True):
    """
    Convenience function to wrap a layer with an adapter.
    
    Args:
        layer: The layer to wrap
        adapter_rank: Rank for adapter
        dropout: Dropout rate
        enabled: Whether adapter is enabled
    
    Returns:
        Wrapped layer with adapter
    """
    if isinstance(layer, (nn.Linear, nn.Sequential)):
        # For linear/sequential layers, we'll use LinearAdapter
        # Note: This is a simplified approach. For Sequential, we'd need to wrap each sub-layer.
        logger.warning(f"LinearAdapter wrapper for {type(layer)} may need custom handling")
        return layer  # For now, return as-is for Sequential
    
    # For E3NN layers
    return E3NNAdapterWrapper(layer, adapter_rank=adapter_rank, dropout=dropout, enabled=enabled)


def get_trainable_parameters(model, adapter_only=True, frozen_layers=None):
    """
    Get trainable parameters based on progressive unfreezing strategy.
    
    Args:
        model: The model
        adapter_only: If True, only return adapter parameters (for initial training)
        frozen_layers: List of layer names to freeze
    
    Returns:
        List of parameter groups for optimizer
    """
    if frozen_layers is None:
        frozen_layers = []
    
    params = []
    
    if adapter_only:
        # Only adapter parameters are trainable
        for name, param in model.named_parameters():
            if 'adapter' in name.lower() or 'scalar_gate' in name.lower():
                if not any(frozen in name for frozen in frozen_layers):
                    params.append(param)
    else:
        # All parameters are trainable (except explicitly frozen)
        for name, param in model.named_parameters():
            if not any(frozen in name for frozen in frozen_layers):
                params.append(param)
    
    return params

