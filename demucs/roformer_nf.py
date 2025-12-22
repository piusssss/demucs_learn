# Copyright (c) 2019-present, Meta, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# First author is Simon Rouard.
# Modified to use RoPE (Rotary Position Embedding)

import random
import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange

try:
    from rotary_embedding_torch import RotaryEmbedding
    ROPE_AVAILABLE = True
except ImportError:
    print("Warning: rotary-embedding-torch not installed. Install with: pip install rotary-embedding-torch")
    ROPE_AVAILABLE = False


def create_sin_embedding(
    length: int, dim: int, shift: int = 0, device="cpu", max_period=10000
):
    # We aim for TBC format
    assert dim % 2 == 0
    pos = shift + torch.arange(length, device=device).view(-1, 1, 1)
    half_dim = dim // 2
    adim = torch.arange(dim // 2, device=device).view(1, 1, -1)
    phase = pos / (max_period ** (adim / (half_dim - 1)))
    return torch.cat(
        [
            torch.cos(phase),
            torch.sin(phase),
        ],
        dim=-1,
    )


def create_2d_sin_embedding(d_model, height, width, device="cpu", max_period=10000):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dimension (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(
        torch.arange(0.0, d_model, 2) * -(math.log(max_period) / d_model)
    )
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = (
        torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[1:d_model:2, :, :] = (
        torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[d_model::2, :, :] = (
        torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    pe[d_model + 1:: 2, :, :] = (
        torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )

    return pe[None, :].to(device)


def create_sin_embedding_cape(
    length: int,
    dim: int,
    batch_size: int,
    mean_normalize: bool,
    augment: bool,  # True during training
    max_global_shift: float = 0.0,  # delta max
    max_local_shift: float = 0.0,  # epsilon max
    max_scale: float = 1.0,
    device: str = "cpu",
    max_period: float = 10000.0,
):
    # We aim for TBC format
    assert dim % 2 == 0
    pos = 1.0 * torch.arange(length).view(-1, 1, 1)  # (length, 1, 1)
    pos = pos.repeat(1, batch_size, 1)  # (length, batch_size, 1)
    if mean_normalize:
        pos -= torch.nanmean(pos, dim=0, keepdim=True)

    if augment:
        delta = np.random.uniform(
            -max_global_shift, +max_global_shift, size=[1, batch_size, 1]
        )
        delta_local = np.random.uniform(
            -max_local_shift, +max_local_shift, size=[length, batch_size, 1]
        )
        log_lambdas = np.random.uniform(
            -np.log(max_scale), +np.log(max_scale), size=[1, batch_size, 1]
        )
        pos = (pos + delta + delta_local) * np.exp(log_lambdas)

    pos = pos.to(device)

    half_dim = dim // 2
    adim = torch.arange(dim // 2, device=device).view(1, 1, -1)
    phase = pos / (max_period ** (adim / (half_dim - 1)))
    return torch.cat(
        [
            torch.cos(phase),
            torch.sin(phase),
        ],
        dim=-1,
    ).float()


def get_causal_mask(length):
    pos = torch.arange(length)
    return pos > pos[:, None]


def get_elementary_mask(
    T1,
    T2,
    mask_type,
    sparse_attn_window,
    global_window,
    mask_random_seed,
    sparsity,
    device,
):
    """
    When the input of the Decoder has length T1 and the output T2
    The mask matrix has shape (T2, T1)
    """
    assert mask_type in ["diag", "jmask", "random", "global"]

    if mask_type == "global":
        mask = torch.zeros(T2, T1, dtype=torch.bool)
        mask[:, :global_window] = True
        line_window = int(global_window * T2 / T1)
        mask[:line_window, :] = True

    if mask_type == "diag":

        mask = torch.zeros(T2, T1, dtype=torch.bool)
        rows = torch.arange(T2)[:, None]
        cols = (
            (T1 / T2 * rows + torch.arange(-sparse_attn_window, sparse_attn_window + 1))
            .long()
            .clamp(0, T1 - 1)
        )
        mask.scatter_(1, cols, torch.ones(1, dtype=torch.bool).expand_as(cols))

    elif mask_type == "jmask":
        mask = torch.zeros(T2 + 2, T1 + 2, dtype=torch.bool)
        rows = torch.arange(T2 + 2)[:, None]
        t = torch.arange(0, int((2 * T1) ** 0.5 + 1))
        t = (t * (t + 1) / 2).int()
        t = torch.cat([-t.flip(0)[:-1], t])
        cols = (T1 / T2 * rows + t).long().clamp(0, T1 + 1)
        mask.scatter_(1, cols, torch.ones(1, dtype=torch.bool).expand_as(cols))
        mask = mask[1:-1, 1:-1]

    elif mask_type == "random":
        gene = torch.Generator(device=device)
        gene.manual_seed(mask_random_seed)
        mask = (
            torch.rand(T1 * T2, generator=gene, device=device).reshape(T2, T1)
            > sparsity
        )

    mask = mask.to(device)
    return mask


def get_mask(
    T1,
    T2,
    mask_type,
    sparse_attn_window,
    global_window,
    mask_random_seed,
    sparsity,
    device,
):
    """
    Return a SparseCSRTensor mask that is a combination of elementary masks
    mask_type can be a combination of multiple masks: for instance "diag_jmask_random"
    """
    from xformers.sparse import SparseCSRTensor
    # create a list
    mask_types = mask_type.split("_")

    all_masks = [
        get_elementary_mask(
            T1,
            T2,
            mask,
            sparse_attn_window,
            global_window,
            mask_random_seed,
            sparsity,
            device,
        )
        for mask in mask_types
    ]

    final_mask = torch.stack(all_masks).sum(axis=0) > 0

    return SparseCSRTensor.from_dense(final_mask[None])


class ScaledEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        scale: float = 1.0,
        boost: float = 3.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data *= scale / boost
        self.boost = boost

    @property
    def weight(self):
        return self.embedding.weight * self.boost

    def forward(self, x):
        return self.embedding(x) * self.boost


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float = 0, channel_last=False):
        """
        channel_last = False corresponds to (B, C, T) tensors
        channel_last = True corresponds to (T, B, C) tensors
        """
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x


class MyGroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: (B, T, C)
        if num_groups=1: Normalisation on all T and C together for each B
        """
        x = x.transpose(1, 2)
        return super().forward(x).transpose(1, 2)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (from BSRoformer)"""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, dim=-1, p=2, eps=self.eps) * self.scale * self.gamma


class AttentionWithRoPE(nn.Module):
    """Multi-head attention with RoPE (Rotary Position Embedding)"""
    def __init__(
        self,
        dim,
        num_heads=8,
        dim_head=64,
        dropout=0.,
        rotary_embed=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        dim_inner = num_heads * dim_head

        self.rotary_embed = rotary_embed
        
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, N, C]
        x = self.norm(x)
        
        # Generate Q, K, V
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.num_heads)

        # Apply RoPE to Q and K
        if self.rotary_embed is not None:
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = attn @ v
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerLayerWithRoPE(nn.Module):
    """Transformer layer with RoPE support"""
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.gelu,
        rotary_embed=None,
    ):
        super().__init__()
        dim_head = d_model // nhead
        
        self.attn = AttentionWithRoPE(
            dim=d_model,
            num_heads=nhead,
            dim_head=dim_head,
            dropout=dropout,
            rotary_embed=rotary_embed,
        )
        
        # Feedforward
        self.norm_ff = RMSNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU() if activation == F.gelu else activation
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: [B, N, C]
        # Self-attention
        x = x + self.attn(x)
        
        # Feedforward
        ff_out = self.norm_ff(x)
        ff_out = self.linear1(ff_out)
        ff_out = self.activation(ff_out)
        ff_out = self.dropout1(ff_out)
        ff_out = self.linear2(ff_out)
        ff_out = self.dropout2(ff_out)
        x = x + ff_out
        
        return x


class MyTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        group_norm=0,
        norm_first=False,
        norm_out=False,
        layer_norm_eps=1e-5,
        layer_scale=False,
        init_values=1e-4,
        device=None,
        dtype=None,
        sparse=False,
        mask_type="diag",
        mask_random_seed=42,
        sparse_attn_window=500,
        global_window=50,
        auto_sparsity=False,
        sparsity=0.95,
        batch_first=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )
        self.sparse = sparse
        self.auto_sparsity = auto_sparsity
        if sparse:
            if not auto_sparsity:
                self.mask_type = mask_type
                self.sparse_attn_window = sparse_attn_window
                self.global_window = global_window
            self.sparsity = sparsity
        if group_norm:
            self.norm1 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)

        self.norm_out = None
        if self.norm_first & norm_out:
            self.norm_out = MyGroupNorm(num_groups=int(norm_out), num_channels=d_model)
        self.gamma_1 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )
        self.gamma_2 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )

        if sparse:
            self.self_attn = MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=batch_first,
                auto_sparsity=sparsity if auto_sparsity else 0,
            )
            self.__setattr__("src_mask", torch.zeros(1, 1))
            self.mask_random_seed = mask_random_seed

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        if batch_first = False, src shape is (T, B, C)
        the case where batch_first=True is not covered
        """
        device = src.device
        x = src
        T, B, C = x.shape
        if self.sparse and not self.auto_sparsity:
            assert src_mask is None
            src_mask = self.src_mask
            if src_mask.shape[-1] != T:
                src_mask = get_mask(
                    T,
                    T,
                    self.mask_type,
                    self.sparse_attn_window,
                    self.global_window,
                    self.mask_random_seed,
                    self.sparsity,
                    device,
                )
                self.__setattr__("src_mask", src_mask)

        if self.norm_first:
            x = x + self.gamma_1(
                self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            )
            x = x + self.gamma_2(self._ff_block(self.norm2(x)))

            if self.norm_out:
                x = self.norm_out(x)
        else:
            x = self.norm1(
                x + self.gamma_1(self._sa_block(x, src_mask, src_key_padding_mask))
            )
            x = self.norm2(x + self.gamma_2(self._ff_block(x)))

        return x


class CrossTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation=F.relu,
        layer_norm_eps: float = 1e-5,
        layer_scale: bool = False,
        init_values: float = 1e-4,
        norm_first: bool = False,
        group_norm: bool = False,
        norm_out: bool = False,
        sparse=False,
        mask_type="diag",
        mask_random_seed=42,
        sparse_attn_window=500,
        global_window=50,
        sparsity=0.95,
        auto_sparsity=None,
        device=None,
        dtype=None,
        batch_first=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.sparse = sparse
        self.auto_sparsity = auto_sparsity
        if sparse:
            if not auto_sparsity:
                self.mask_type = mask_type
                self.sparse_attn_window = sparse_attn_window
                self.global_window = global_window
            self.sparsity = sparsity

        self.cross_attn: nn.Module
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1: nn.Module
        self.norm2: nn.Module
        self.norm3: nn.Module
        if group_norm:
            self.norm1 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm3 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.norm_out = None
        if self.norm_first & norm_out:
            self.norm_out = MyGroupNorm(num_groups=int(norm_out), num_channels=d_model)

        self.gamma_1 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )
        self.gamma_2 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = self._get_activation_fn(activation)
        else:
            self.activation = activation

        if sparse:
            self.cross_attn = MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=batch_first,
                auto_sparsity=sparsity if auto_sparsity else 0)
            if not auto_sparsity:
                self.__setattr__("mask", torch.zeros(1, 1))
                self.mask_random_seed = mask_random_seed

    def forward(self, q, k, mask=None):
        """
        Args:
            q: tensor of shape (T, B, C)
            k: tensor of shape (S, B, C)
            mask: tensor of shape (T, S)

        """
        device = q.device
        T, B, C = q.shape
        S, B, C = k.shape
        if self.sparse and not self.auto_sparsity:
            assert mask is None
            mask = self.mask
            if mask.shape[-1] != S or mask.shape[-2] != T:
                mask = get_mask(
                    S,
                    T,
                    self.mask_type,
                    self.sparse_attn_window,
                    self.global_window,
                    self.mask_random_seed,
                    self.sparsity,
                    device,
                )
                self.__setattr__("mask", mask)

        if self.norm_first:
            x = q + self.gamma_1(self._ca_block(self.norm1(q), self.norm2(k), mask))
            x = x + self.gamma_2(self._ff_block(self.norm3(x)))
            if self.norm_out:
                x = self.norm_out(x)
        else:
            x = self.norm1(q + self.gamma_1(self._ca_block(q, k, mask)))
            x = self.norm2(x + self.gamma_2(self._ff_block(x)))

        return x

    # self-attention block    cross?
    def _ca_block(self, q, k, attn_mask=None):
        x = self.cross_attn(q, k, k, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


# ----------------- MULTI-BLOCKS MODELS: -----------------------


class CrossTransformerEncoder(nn.Module):
    """
    Multi-resolution Transformer with RoPE (Rotary Position Embedding)
    Pure RoPE implementation - no traditional sin/cos positional encoding
    """
    def __init__(
        self,
        dim: int,
        emb: str = "sin",
        hidden_scale: float = 4.0,
        num_heads: int = 8,
        num_layers: int = 6,
        cross_first: bool = False,
        dropout: float = 0.0,
        max_positions: int = 1000,
        norm_in: bool = True,
        norm_in_group: bool = False,
        group_norm: int = False,
        norm_first: bool = False,
        norm_out: bool = False,
        max_period: float = 10000.0,
        weight_decay: float = 0.0,
        lr: tp.Optional[float] = None,
        layer_scale: bool = False,
        gelu: bool = True,
        sin_random_shift: int = 0,
        weight_pos_embed: float = 1.0,
        cape_mean_normalize: bool = True,
        cape_augment: bool = True,
        cape_glob_loc_scale: list = [5000.0, 1.0, 1.4],
        sparse_self_attn: bool = False,
        sparse_cross_attn: bool = False,
        mask_type: str = "diag",
        mask_random_seed: int = 42,
        sparse_attn_window: int = 500,
        global_window: int = 50,
        auto_sparsity: bool = False,
        sparsity: float = 0.95,
        num_resolutions: int = 4,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert ROPE_AVAILABLE, "rotary-embedding-torch not installed. Run: pip install rotary-embedding-torch"

        hidden_dim = int(dim * hidden_scale)
        dim_head = dim // num_heads

        self.num_layers = num_layers
        self.weight_decay = weight_decay
        self.num_resolutions = num_resolutions
        self.lr = lr

        # RMSNorm for input
        self.norm_in = RMSNorm(dim)

        # Create RoPE embeddings for token axis
        self.token_rope = RotaryEmbedding(dim=dim_head)


        # Multi-resolution self-attention layers with RoPE
        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            self.layers.append(
                TransformerLayerWithRoPE(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    dropout=dropout,
                    activation=F.gelu,
                    rotary_embed=self.token_rope,
                )
            )
        
        # Multi-resolution embedding (static, not positional)
        self.resolution_embedding = nn.Embedding(self.num_resolutions, dim)

    def forward(self, x_list):
        # x_list: list of [B, C, Fr_i, T_i]
        # x_list[0]: [B, C, 1024, 432]
        # x_list[1]: [B, C, 2048, 216]
        # x_list[2]: [B, C, 4096, 108]
        # x_list[3]: [B, C, 8192, 54]
        #print(f"[DEBUG] Input shapes: {[x.shape for x in x_list]}")
        shapes = [(x.shape[2], x.shape[3]) for x in x_list]
        B, C = x_list[0].shape[0], x_list[0].shape[1]
        device = x_list[0].device
        
        # Find minimum time steps for grouping
        time_steps = [x.shape[3] for x in x_list]  # e.g., [431, 215, 107, 53]
        min_time_steps = min(time_steps)  # e.g., 53
        
        # Align all resolutions by padding to make them divisible
        aligned_list = []
        for x in x_list:
            T = x.shape[3]
            # Pad to the smallest multiple of min_time_steps that's >= T
            num_groups = (T + min_time_steps - 1) // min_time_steps  # Ceiling division
            target_T = num_groups * min_time_steps
            
            if target_T > T:
                # Pad time dimension: [B, C, F, T] -> [B, C, F, target_T]
                pad_amount = target_T - T
                x = F.pad(x, (0, pad_amount), mode='constant', value=0)
            
            aligned_list.append(x)
        
        x_list = aligned_list
        #print(f"[DEBUG] After crop: {[x.shape for x in x_list]}, min_time_steps={min_time_steps}")
        
        # First: Norm and group all resolutions into unified square structure
        # Then: Add resolution embeddings in the square space
        # RoPE handles all positional information dynamically during attention
        grouped_list = []
        for i, x in enumerate(x_list):
            Fr, T = x.shape[2], x.shape[3]
            
            # Rearrange to [B, Fr*T, C] for norm only
            x = rearrange(x, 'b c f t -> b (f t) c')
            
            # Normalize with RMSNorm
            x = self.norm_in(x)
            
            # Rearrange back and group: [B, Fr*T, C] -> [B, C, min_time_steps, Fr*group_size]
            x = rearrange(x, 'b (f t) c -> b c f t', f=Fr, t=T)
            group_size = T // min_time_steps  # [8, 4, 2, 1]
            x_grouped = rearrange(x, 'b c f (t g) -> b c t (f g)', g=group_size)
            # x_grouped: [B, C, 54, 32]
            
            grouped_list.append(x_grouped)
        
        # Stack all resolutions: [B, C, num_res, min_time_steps, tokens_per_group]
        all_res = torch.stack(grouped_list, dim=2)
        
        # Rearrange to [B, C, min_time_steps, num_res, tokens_per_group]
        all_res = rearrange(all_res, 'b c r t d -> b c t r d')
        
        # Get actual dimensions
        tokens_per_group = all_res.shape[-1]  # 32: depends on Fr and group_size
        
        # Add resolution embedding in square space (static ID, not positional)
        for i in range(self.num_resolutions):
            res_emb = self.resolution_embedding.weight[i:i+1]  # [1, C]
            res_emb = res_emb.view(1, C, 1, 1, 1).expand(B, C, min_time_steps, 1, tokens_per_group)
            all_res[:, :, :, i:i+1, :] = all_res[:, :, :, i:i+1, :] + res_emb
        
        # Add time group embedding to distinguish which time group (0-53) we're in
        # This is necessary because RoPE only encodes relative positions within each square
        time_group_emb = create_sin_embedding(
            min_time_steps, C, device=device, max_period=10000
        )  # [54, 1, C]
        time_group_emb = rearrange(time_group_emb, 't b c -> b c t 1 1')
        time_group_emb = time_group_emb.expand(B, C, min_time_steps, self.num_resolutions, tokens_per_group)
        all_res = all_res + time_group_emb
        
        # Flatten to sequence: [B*min_time_steps, num_res*tokens_per_group, C]
        # Transformer expects: [batch, sequence, feature]
        x = rearrange(all_res, 'b c t r d -> (b t) (r d) c')
        # x: [B*54, 128, 256] where 128 = 4 resolutions × 32 tokens

        # Apply self-attention layers
        for idx in range(self.num_layers):
            x = self.layers[idx](x)

        # Reshape output back to grouped form: [B, C, 54, 4, 32]
        out = rearrange(x, '(b t) (r d) c -> b c t r d', 
                       b=B, t=min_time_steps, r=self.num_resolutions, d=tokens_per_group)
        
        # Split back to individual resolutions and restore original shapes
        x_list_out = []
        for i, (Fr, T_orig) in enumerate(shapes):
            # Extract this resolution: [B, C, 54, tokens_per_group]
            out_res = out[:, :, :, i, :]
            
            # Calculate actual group_size used (after padding)
            T_padded = x_list[i].shape[3]  # Use the padded T
            group_size = T_padded // min_time_steps
            
            # Ungroup back to padded time steps
            # [B, C, 54, tokens_per_group] -> [B, C, Fr, T_padded]
            out_res = rearrange(out_res, 'b c t (f g) -> b c f (t g)', 
                               f=Fr, g=group_size)
            
            # Crop back to original time steps
            out_res = out_res[:, :, :, :T_orig]
            
            x_list_out.append(out_res)
        
        return x_list_out



    def make_optim_group(self):
        group = {"params": list(self.parameters()), "weight_decay": self.weight_decay}
        if self.lr is not None:
            group["lr"] = self.lr
        return group


# Attention Modules


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        auto_sparsity=None,
    ):
        super().__init__()
        assert auto_sparsity is not None, "sanity check"
        self.num_heads = num_heads
        self.q = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_drop = torch.nn.Dropout(dropout)
        self.proj = torch.nn.Linear(embed_dim, embed_dim, bias)
        self.proj_drop = torch.nn.Dropout(dropout)
        self.batch_first = batch_first
        self.auto_sparsity = auto_sparsity

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
    ):

        if not self.batch_first:  # N, B, C
            query = query.permute(1, 0, 2)  # B, N_q, C
            key = key.permute(1, 0, 2)  # B, N_k, C
            value = value.permute(1, 0, 2)  # B, N_k, C
        B, N_q, C = query.shape
        B, N_k, C = key.shape

        q = (
            self.q(query)
            .reshape(B, N_q, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        q = q.flatten(0, 1)
        k = (
            self.k(key)
            .reshape(B, N_k, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = k.flatten(0, 1)
        v = (
            self.v(value)
            .reshape(B, N_k, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = v.flatten(0, 1)

        if self.auto_sparsity:
            assert attn_mask is None
            x = dynamic_sparse_attention(q, k, v, sparsity=self.auto_sparsity)
        else:
            x = scaled_dot_product_attention(q, k, v, attn_mask, dropout=self.attn_drop)
        x = x.reshape(B, self.num_heads, N_q, C // self.num_heads)

        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if not self.batch_first:
            x = x.permute(1, 0, 2)
        return x, None


def scaled_query_key_softmax(q, k, att_mask):
    from xformers.ops import masked_matmul
    q = q / (k.size(-1)) ** 0.5
    att = masked_matmul(q, k.transpose(-2, -1), att_mask)
    att = torch.nn.functional.softmax(att, -1)
    return att


def scaled_dot_product_attention(q, k, v, att_mask, dropout):
    att = scaled_query_key_softmax(q, k, att_mask=att_mask)
    att = dropout(att)
    y = att @ v
    return y


def _compute_buckets(x, R):
    qq = torch.einsum('btf,bfhi->bhti', x, R)
    qq = torch.cat([qq, -qq], dim=-1)
    buckets = qq.argmax(dim=-1)

    return buckets.permute(0, 2, 1).byte().contiguous()


def dynamic_sparse_attention(query, key, value, sparsity, infer_sparsity=True, attn_bias=None):
    # assert False, "The code for the custom sparse kernel is not ready for release yet."
    from xformers.ops import find_locations, sparse_memory_efficient_attention
    n_hashes = 32
    proj_size = 4
    query, key, value = [x.contiguous() for x in [query, key, value]]
    with torch.no_grad():
        R = torch.randn(1, query.shape[-1], n_hashes, proj_size // 2, device=query.device)
        bucket_query = _compute_buckets(query, R)
        bucket_key = _compute_buckets(key, R)
        row_offsets, column_indices = find_locations(
            bucket_query, bucket_key, sparsity, infer_sparsity)
    return sparse_memory_efficient_attention(
        query, key, value, row_offsets, column_indices, attn_bias)
