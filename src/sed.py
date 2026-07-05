"""
SED (Sound Event Detection) head with bidirectional GRU.

Replaces the single-CLS-query temporal attention pool with a frame-level
sequence model. Backbone-agnostic: takes any (B, C, H, W) feature map and
produces both per-frame and clip-level scores.

Architecture:
  (B, C, H, W) -> mean over H -> (B, C, T)
                  -> permute   -> (B, T, C)
                  -> biGRU     -> (B, T, 2 * gru_hidden)
                  -> per-frame Linear -> (B, T, num_classes)   # frame-level scores
                  -> attention pool   -> (B, num_classes)      # clip-level (used for loss)

Frame-level scores are returned alongside clip-level for downstream SED-style
inference / analysis. Clip-level loss is the current default (matches existing
training loop); frame-level loss can be added later when the dataloader emits
per-frame targets from soundscape timestamps.

The PANN-style attention pool: a separate Linear(C -> num_classes) learns
attention weights over time, softmaxed over T, then weighted-summed against
per-frame logits. Standard pattern from PANNs (Kong et al. 2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool1d(nn.Module):
    """PANN-style attention pooling over the time axis.

    Given per-frame features (B, T, C), produces:
      - clip_logits: (B, num_classes)
      - frame_logits: (B, T, num_classes)
      - attn_weights: (B, T, num_classes)  (per-class attention over time)
    """

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.attn = nn.Linear(in_dim, num_classes)
        self.cla = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
        # Cast attn computation to fp32 for numerical stability under bf16/fp16.
        with torch.amp.autocast("cuda", enabled=False):
            x_f = x.float()
            attn_logits = self.attn(x_f)              # (B, T, num_classes)
            frame_logits = self.cla(x_f)              # (B, T, num_classes)
            attn = F.softmax(attn_logits, dim=1)      # softmax over T per class
            clip_logits = (attn * frame_logits).sum(dim=1)  # (B, num_classes)
        return clip_logits, frame_logits, attn


class SEDHead(nn.Module):
    """Bidirectional GRU + per-frame classifier + attention pool.

    Input:  (B, C, H, W) backbone feature map
    Output: dict with
        - clip_logits:  (B, num_classes)         used for clip-level loss
        - frame_logits: (B, T, num_classes)      per-frame scores (for SED / analysis)
        - attn_weights: (B, T, num_classes)      per-class attention over time

    Args:
        in_channels: backbone output channels (C dim)
        num_classes: number of output classes (head outputs num_train_classes)
        gru_hidden: biGRU hidden size per direction (output is 2 * gru_hidden)
        gru_layers: number of GRU layers (1 is usually enough for 5s clips)
        gru_dropout: dropout between GRU layers (only matters if gru_layers > 1)
        post_gru_dropout: dropout applied after the GRU before the linear heads
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        gru_hidden: int = 256,
        gru_layers: int = 1,
        gru_dropout: float = 0.0,
        post_gru_dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers

        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(post_gru_dropout)
        self.attn_pool = AttentionPool1d(2 * gru_hidden, num_classes)

    def forward(self, feat_map: torch.Tensor):
        """
        Args:
            feat_map: (B, C, H, W) — H is freq (collapsed by mean), W is time
        Returns:
            dict with clip_logits, frame_logits, attn_weights, latent
        """
        # Collapse freq -> (B, C, T)
        x = feat_map.mean(dim=2)
        # GRU expects (B, T, C)
        x = x.permute(0, 2, 1).contiguous()
        # biGRU; flatten_parameters keeps it cuDNN-fast
        self.gru.flatten_parameters()
        x, _ = self.gru(x)            # (B, T, 2*gru_hidden)
        x = self.dropout(x)
        clip_logits, frame_logits, attn = self.attn_pool(x)

        # latent_output for compat: mean over time of GRU output
        latent = x.mean(dim=1)        # (B, 2*gru_hidden)

        return {
            "clip_logits": clip_logits,
            "frame_logits": frame_logits,
            "attn_weights": attn,
            "latent": latent,
        }
