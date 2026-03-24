"""Bird-MAE model — standalone PyTorch version.

ViT-B/16 encoder pretrained via masked autoencoder on BirdSet.
Adapted from https://huggingface.co/DBD-research-group/Bird-MAE-Base
to remove HuggingFace transformers dependency.
"""

import collections
from itertools import repeat
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BirdMAEConfig


# ── Positional encoding ──────────────────────────────────────────────────────

def _get_1d_sincos_pos_embed(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size[0], grid_size[1])
    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return emb


# ── Building blocks ──────────────────────────────────────────────────────────

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
                 proj_drop=0.0, attn_drop=0.0, drop_path=0.0,
                 norm_layer_eps=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_layer_eps)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=norm_layer_eps)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=proj_drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(512, 128), patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        img_size = _ntuple(2)(img_size)
        patch_size = _ntuple(2)(patch_size)
        self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        self.num_patches = self.patch_hw[0] * self.patch_hw[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)       # (B, embed_dim, H', W')
        x = x.flatten(2)       # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


# ── Main model ───────────────────────────────────────────────────────────────

class BirdMAEModel(nn.Module):
    """Bird-MAE-Base encoder (ViT-B/16).

    Input: mel spectrogram of shape (B, 1, 512, 128)
    Output: embedding of shape (B, 768) when global_pool="mean"
    """

    def __init__(self, config: BirdMAEConfig = None):
        super().__init__()
        if config is None:
            config = BirdMAEConfig()
        self.config = config

        self.patch_embed = PatchEmbed(
            img_size=(config.img_size_x, config.img_size_y),
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.embed_dim),
            requires_grad=config.pos_trainable,
        )

        # Initialize sincos positional embedding
        pos_embed_np = _get_2d_sincos_pos_embed(
            config.embed_dim, self.patch_embed.patch_hw, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed_np).float().unsqueeze(0))

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                proj_drop=config.proj_drop_rate,
                attn_drop=config.attn_drop_rate,
                drop_path=dpr[i],
                norm_layer_eps=config.norm_layer_eps,
            )
            for i in range(config.depth)
        ])

        self.pos_drop = nn.Dropout(p=config.pos_drop_rate)
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.norm_layer_eps)
        self.fc_norm = nn.LayerNorm(config.embed_dim, eps=config.norm_layer_eps)
        self.global_pool = config.global_pool

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        """
        Args:
            x: mel spectrogram tensor of shape (B, 1, 512, 128)

        Returns:
            embedding tensor of shape (B, 768)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat([cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.pos_drop(x)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Global pooling
        if self.global_pool == "mean":
            x = x[:, 1:, :].mean(dim=1)
            x = self.fc_norm(x)
        elif self.global_pool == "cls":
            x = self.norm(x)
            x = x[:, 0]
        else:
            raise ValueError(f"Invalid global pool: {self.global_pool}")

        return x  # (B, embed_dim)

    @classmethod
    def from_pretrained(cls, checkpoint_path, config=None):
        """Load pretrained Bird-MAE weights.

        Handles safetensors, HuggingFace format (with 'model.' prefix),
        and raw PyTorch state dicts.
        """
        model = cls(config)

        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if "model" in state_dict:
                state_dict = state_dict["model"]

        # Strip 'model.' prefix if present (HuggingFace wraps in BirdMAEModel.model)
        cleaned = {}
        for k, v in state_dict.items():
            k = k.replace("model.", "", 1) if k.startswith("model.") else k
            cleaned[k] = v

        # Filter out classification head keys if present
        cleaned = {k: v for k, v in cleaned.items()
                   if not k.startswith("head.")}

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"Bird-MAE: missing keys: {missing}")
        if unexpected:
            print(f"Bird-MAE: unexpected keys: {unexpected}")

        return model
