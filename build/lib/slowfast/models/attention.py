#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


from os import O_ASYNC
import numpy
import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat

from slowfast.models.common import DropPath, Mlp

from torch.distributions import Categorical

# Multi Scale Attention

def attention_pool(tensor, pool, thw_shape, has_cls_embed=True, norm=None):
    if pool is None:
        return tensor, thw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    tensor = (
        tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
    )

    tensor = pool(tensor)

    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)
    # Assert tensor_dim in [3, 4]
    if tensor_dim == 4:
        pass
    else:  #  tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, thw_shape

class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        # Options include `conv`, `avg`, and `max`.
        mode="conv",
        # If True, perform pool before projection.
        pool_first=False,
        ignore_111_kv_kernel = False,
    ):
        super().__init__()
        self.pool_first = pool_first
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.has_cls_embed = has_cls_embed
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]
        self.pool_kv = len(kernel_kv) > 0
        if ignore_111_kv_kernel:
            self.pool_kv = self.pool_kv and (tuple(stride_kv) != (1, 1, 1))
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1:
            kernel_q = ()
        if numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
            kernel_kv = ()

        if mode in ("avg", "max"):
            pool_op = nn.MaxPool3d if mode == "max" else nn.AvgPool3d
            self.pool_q = (
                pool_op(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "conv":
            self.pool_q = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_q) > 0
                else None
            )
            self.norm_q = norm_layer(head_dim) if len(kernel_q) > 0 else None
            self.pool_k = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                if self.pool_kv
                else None
            )
            self.norm_k = norm_layer(head_dim) if self.pool_kv else None
            self.pool_v = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                if self.pool_kv
                else None
            )
            self.norm_v = norm_layer(head_dim) if self.pool_kv else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

    def forward(self, x, thw_shape,val_output_type=None):
        B, N, C = x.shape
        if self.pool_first:
            x = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(
                0, 2, 1, 3
            )
            q = k = v = x
        else:
            q = k = v = x
            q = (
                self.q(q)
                .reshape(B, N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )
            k = (
                self.k(k)
                .reshape(B, N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )
            v = (
                self.v(v)
                .reshape(B, N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )

        q, q_shape = attention_pool(
            q,
            self.pool_q,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )
        k, k_shape = attention_pool(
            k,
            self.pool_k,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        v, v_shape = attention_pool(
            v,
            self.pool_v,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )

        if self.pool_first:
            q_N = (
                numpy.prod(q_shape) + 1
                if self.has_cls_embed
                else numpy.prod(q_shape)
            )
            k_N = (
                numpy.prod(k_shape) + 1
                if self.has_cls_embed
                else numpy.prod(k_shape)
            )
            v_N = (
                numpy.prod(v_shape) + 1
                if self.has_cls_embed
                else numpy.prod(v_shape)
            )

            q = q.permute(0, 2, 1, 3).reshape(B, q_N, C)
            q = (
                self.q(q)
                .reshape(B, q_N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )

            v = v.permute(0, 2, 1, 3).reshape(B, v_N, C)
            v = (
                self.v(v)
                .reshape(B, v_N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )

            k = k.permute(0, 2, 1, 3).reshape(B, k_N, C)
            k = (
                self.k(k)
                .reshape(B, k_N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        N = q.shape[2]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)

        if val_output_type is not None:
            assert val_output_type in ('t','q','v','k')
            x_out, _shape = {'q':(q, q_shape), 'v':(v, v_shape), 'k':(k, k_shape), 't': (x, q_shape)}[val_output_type]
            if len(x_out.shape) == 4:
                x_out = x_out.transpose(1,2).reshape(
                    B, -1, C
                )
            return ((x, q_shape), (attn, x_out, _shape))

        return ((x, q_shape), attn)

class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        up_rate=None,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        mode="conv",
        has_cls_embed=True,
        pool_first=False,
        ignore_111_kv_kernel=False,
        cfg=None,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        self.attn = MultiScaleAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=nn.LayerNorm,
            has_cls_embed=has_cls_embed,
            mode=mode,
            pool_first=pool_first,
            ignore_111_kv_kernel=ignore_111_kv_kernel,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        # TODO: check the use case for up_rate, and merge the following lines
        if up_rate is not None and up_rate > 1:
            mlp_dim_out = dim * up_rate
        else:
            mlp_dim_out = dim_out
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
            act_layer=act_layer,
            drop_rate=drop_rate,
        )
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        self.pool_skip = (
            nn.MaxPool3d(
                kernel_skip, stride_skip, padding_skip, ceil_mode=False
            )
            if len(kernel_skip) > 0
            else None
        )

    def forward(self, x, metadata, thw_shape, val_output_type=None):
        x_block, thw_shape_new = self.attn(self.norm1(x), thw_shape, val_output_type=val_output_type)[0]
        x_res, _ = attention_pool(
            x, self.pool_skip, thw_shape, has_cls_embed=self.has_cls_embed,
        )
        x = x_res + self.drop_path(x_block)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        if self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)
        return x, thw_shape_new


# Self Attention

class SelfAttention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, proj_drop=0.
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, thw):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, 
            self.num_heads, 
            C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Joint space-time attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, thw


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_drop=drop_rate,
        )

        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop_rate,
        )


    def forward(self, x, metadata, thw_shape):
        x_block, thw_shape_new = self.attn(self.norm1(x), thw_shape)
        x = x + self.drop_path(x_block)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        x = x + self.drop_path(x_mlp)
        return x, thw_shape_new

# Trajectory Attention

def qkv_attn(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out


class TrajectoryAttentionBlock(nn.Module):

    def __init__(
            self, cfg=None, dim=768, num_heads=12,
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code = True,
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TrajectoryAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias, 
            attn_drop=attn_drop,
            proj_drop=drop,
            use_original_code=use_original_code,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.cfg = cfg
        
    def forward(self, x, metadata, thw, with_cls_token=True):
        x  = x + self.drop_path(
            self.attn(
                self.norm1(x), 
                thw,
                with_cls_token=with_cls_token,
            )[0]
        )
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, thw


class TrajectoryAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, thw_prev, with_cls_token=True):
        B, N, C = x.shape
        P = thw_prev[1] * thw_prev[2]
        F = thw_prev[0]
        h = self.num_heads

        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        if with_cls_token:
            # remove CLS token from q, k, v
            (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
                lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

            # let CLS token attend to key / values of all patches across time and space
            cls_out = qkv_attn(cls_q * self.scale, k, v)
            cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h)
        else:
            q_, k_, v_ = q, k, v

        # Using full attention
        q_dot_k = q_ @ k_.transpose(-2, -1)
        q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)
        space_attn = (self.scale * q_dot_k).softmax(dim=-1)
        attn = self.attn_drop(space_attn)
        v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)
        x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_)

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B)
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F)
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2)
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F)
        q2 = self.proj_q(x_diag)
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1)
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x)
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')

        if with_cls_token:
            # concat back the cls token
            x = torch.cat((cls_out, x), dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, thw_prev

# attention to modify human tokens w.r.t. object tokens and vice versa

class CrossAttention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, proj_drop=0.
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pos_emb = torch.nn.Parameter(torch.randn(6, dim))

    def forward(self, x, y, mask=None):
        B, N1, C = x.shape
        B, N2, C = y.shape
        q = self.q(x).reshape(
            B, N1, 1, 
            self.num_heads, 
            C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        
        kv = self.kv(y).reshape(
            B, N2, 2, 
            self.num_heads, 
            C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Joint space-time attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        if mask is not None:
            if B > 1:
                x = mask*x
            else:
                x = mask.unsqueeze(1).float()*x            
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_drop=drop_rate,
        )

        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop_rate,
        )


    def forward(self, x, y, mask=None):
        x_block = self.attn(
                    self.norm1(x), 
                    self.norm2(y), 
                    mask=mask
                )
        x = x + self.drop_path(x_block)
        x_norm = self.norm3(x)
        x_mlp = self.mlp(x_norm)
        x = x + self.drop_path(x_mlp)
        return x


class EntropyPrunedSelfAttention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, proj_drop=0., entropy_threshold=None, decay_rate=5
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # Compute uniform distribution entropy as upper threshold
        if entropy_threshold is None:
            prob_tensor = torch.ones(dim) / dim
            self.entropy_threshold = Categorical(probs = prob_tensor).entropy().item()
        else:
            self.entropy_threshold = entropy_threshold
        self.decay_rate = decay_rate
    def forward(self, x, cur_epoch):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, 
            self.num_heads, 
            C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Joint space-time attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
       
        # prune based on entropy 
        p_tensors = []
        for i in range(N):
            p_tensors.append((attn[:,:,:,i] / self.num_heads).reshape(1,-1))
        p_tensors = torch.stack(p_tensors)
        entropies = Categorical(probs = p_tensors).entropy().squeeze()
        
        factor = 0
        for i in range(cur_epoch):
            factor += self.decay_rate*numpy.exp(-(i+1))
        entropy_threshold = self.entropy_threshold - factor
        print("entropy_threshold: ", entropy_threshold)
        
        attn = attn[:,:,:,entropies <= entropy_threshold]
        v = v[:,:,entropies <= entropy_threshold,:]

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TrajectoryCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, q, thw_prev, with_cls_token=True):
        B, N, C = x.shape
        P = thw_prev[1] * thw_prev[2]
        F = thw_prev[0]
        h = self.num_heads

        # project x to q, k, v vaalues
        k, v = self.kv(x).chunk(2, dim=-1)

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        if with_cls_token:
            # remove CLS token from q, k, v
            (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
                lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

            # let CLS token attend to key / values of all patches across time and space
            cls_out = qkv_attn(cls_q * self.scale, k, v)
            cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h)
        else:
            q_, k_, v_ = q, k, v

        # Using full attention
        q_dot_k = q_ @ k_.transpose(-2, -1)
        q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)
        space_attn = (self.scale * q_dot_k).softmax(dim=-1)
        attn = self.attn_drop(space_attn)
        v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F)
        x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_)

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B)
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F)
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2)
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F)
        q2 = self.proj_q(x_diag)
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1)
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x)
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')

        if with_cls_token:
            # concat back the cls token
            x = torch.cat((cls_out, x), dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, thw_prev