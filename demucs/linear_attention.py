"""
Linear Attention implementation for efficient long sequence modeling.
Based on "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


def elu_feature_map(x):
    """ELU-based feature map for linear attention."""
    return F.elu(x) + 1


class LinearAttention(nn.Module):
    """
    Linear Attention mechanism with O(N) complexity.
    
    Args:
        dim: Input dimension
        heads: Number of attention heads
        dim_head: Dimension per head
        feature_map: Feature map function (default: elu_feature_map)
        eps: Small epsilon for numerical stability
    """
    
    def __init__(self, dim, heads=8, dim_head=64, feature_map=None, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.eps = eps
        
        inner_dim = heads * dim_head
        self.feature_map = feature_map or elu_feature_map
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor [B, N, D]
            mask: Optional attention mask [B, N] or [B, N, N]
        Returns:
            Output tensor [B, N, D]
        """
        B, N, D = x.shape
        H = self.heads
        
        # Generate Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=H), qkv)
        
        # Apply feature map
        q = self.feature_map(q)
        k = self.feature_map(k)
        
        # Handle masking if provided
        if mask is not None:
            if mask.dim() == 2:  # [B, N] -> [B, H, N, 1]
                mask = mask.unsqueeze(1).unsqueeze(-1)
            elif mask.dim() == 3:  # [B, N, N] -> use only diagonal for linear attention
                mask = mask.diagonal(dim1=-2, dim2=-1).unsqueeze(1).unsqueeze(-1)
            
            k = k * mask
            v = v * mask
        
        # Linear attention computation: O(N * D^2) instead of O(N^2 * D)
        # Standard: softmax(QK^T)V
        # Linear: φ(Q)(φ(K)^T V) / (φ(Q)(φ(K)^T 1))
        
        # Compute K^T V: [B, H, D, D]
        kv = torch.einsum('bhnd,bhnf->bhdf', k, v)
        
        # Compute K^T 1 (normalization): [B, H, D]
        k_sum = k.sum(dim=-2)
        
        # Compute Q(K^T V): [B, H, N, D]
        qkv_out = torch.einsum('bhnd,bhdf->bhnf', q, kv)
        
        # Compute Q(K^T 1): [B, H, N]
        q_k_sum = torch.einsum('bhnd,bhd->bhn', q, k_sum)
        
        # Normalize: [B, H, N, D]
        out = qkv_out / (q_k_sum.unsqueeze(-1) + self.eps)
        
        # Reshape and project: [B, N, H*D] -> [B, N, D]
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class LinearTransformerBlock(nn.Module):
    """
    Transformer block with Linear Attention.
    """
    
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        ff_mult=4,
        dropout=0.0,
        feature_map=None,
        norm_first=True
    ):
        super().__init__()
        self.norm_first = norm_first
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            feature_map=feature_map
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        if self.norm_first:
            # Pre-norm
            x = x + self.attn(self.norm1(x), mask=mask)
            x = x + self.ff(self.norm2(x))
        else:
            # Post-norm
            x = self.norm1(x + self.attn(x, mask=mask))
            x = self.norm2(x + self.ff(x))
        return x


class LinearTransformerEncoder(nn.Module):
    """
    Stack of Linear Transformer blocks for efficient long sequence modeling.
    """
    
    def __init__(
        self,
        dim,
        depth=6,
        heads=8,
        dim_head=64,
        ff_mult=4,
        dropout=0.0,
        feature_map=None,
        norm_first=True
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            LinearTransformerBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
                feature_map=feature_map,
                norm_first=norm_first
            )
            for _ in range(depth)
        ])
        
        if norm_first:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.Identity()
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)


# Alternative feature maps for experimentation
def softplus_feature_map(x):
    """Softplus-based feature map."""
    return F.softplus(x)


def relu_feature_map(x):
    """ReLU-based feature map with small epsilon."""
    return F.relu(x) + 1e-6


def exp_feature_map(x):
    """Exponential feature map (closer to standard attention)."""
    return torch.exp(x - x.max(dim=-1, keepdim=True)[0])