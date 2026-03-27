"""
EfficientNet-based model for BirdCLEF 2026 multi-label bird call classification.

Architecture:
  - EfficientNet-B0 backbone (ImageNet-pretrained via timm) operating on
    log-mel spectrograms. Fully convolutional — accepts any time dimension.
  - NMF branch: frozen spectral dictionary projected to class logits,
    added to backbone logits before final sigmoid.
  - AdaptiveAvgPool2d(1) before the classifier head so variable-length
    spectrograms are supported natively.

Input: raw waveform tensor of shape (B, num_samples) — variable length
       within a batch is handled by the collate function padding to the
       longest clip and a length mask.
Output dict: {"clipwise_output": (B, num_classes), "latent_output": (B, D)}
"""

import os
import math

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import timm


def build_mel_extractor(sample_rate=32000, n_fft=1024, hop_length=320,
                        n_mels=128, f_min=50.0, f_max=14000.0):
    """Log-mel spectrogram extractor for the CNN backbone."""
    return nn.Sequential(
        T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            window_fn=torch.hann_window,
            power=2.0,
            center=True,
            pad_mode="reflect",
        ),
        T.AmplitudeToDB(stype="power", top_db=80),
    )


def build_nmf_mel_extractor(sample_rate=32000, n_fft=1024, hop_length=320,
                             n_mels=64, f_min=50.0, f_max=14000.0):
    """Power mel spectrogram for NMF (must match the dictionary's mel config)."""
    return T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        window_fn=torch.hann_window,
        power=2.0,
        center=True,
        pad_mode="reflect",
    )


class BirdCLEFModel(nn.Module):
    """EfficientNet + NMF for multi-label bird species classification.

    Args:
        num_classes: number of bird species.
        backbone_name: timm model name (default: tf_efficientnet_b0_ns).
        sample_rate: audio sample rate.
        n_mels: mel bins for backbone spectrogram (128 for richer freq resolution).
        nmf_n_mels: mel bins for NMF branch (64, must match dictionary).
        W_path: path to NMF dictionary .npy file, shape (nmf_n_mels, K).
        pretrained: load ImageNet weights for backbone.
    """

    def __init__(self, num_classes, backbone_name="tf_efficientnet_b0_ns",
                 sample_rate=32000, n_fft=1024, hop_length=320,
                 n_mels=128, f_min=50.0, f_max=14000.0,
                 nmf_n_mels=64, W_path=None, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        # --- Mel spectrogram for backbone ---
        self.mel_extractor = build_mel_extractor(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=f_min, f_max=f_max,
        )

        # --- EfficientNet backbone ---
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, in_chans=1,
            num_classes=0, global_pool="",  # no head, no pool
        )
        backbone_dim = self.backbone.num_features  # 1280 for efficientnet_b0

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head_drop = nn.Dropout(0.3)
        self.head = nn.Linear(backbone_dim, num_classes)

        # --- Spec augmentation (training only) ---
        self.freq_mask = T.FrequencyMasking(freq_mask_param=16)
        self.time_mask = T.TimeMasking(time_mask_param=64)

        # --- Batch norm on mel ---
        self.bn0 = nn.BatchNorm2d(n_mels)

        # --- NMF branch ---
        self.nmf_mel = build_nmf_mel_extractor(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=nmf_n_mels, f_min=f_min, f_max=f_max,
        )

        if W_path is None:
            W_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "nmf_analysis", "output", "W_k56.npy",
            )
        if os.path.isfile(W_path):
            _W = torch.from_numpy(np.load(W_path)).float()
            self.register_buffer("W_nmf", _W)
            self.nmf_k = _W.shape[1]
        else:
            self.register_buffer("W_nmf", None)
            self.nmf_k = 0

        nmf_feat_dim = 2 * self.nmf_k  # mean + max
        if self.nmf_k > 0:
            self.nmf_proj = nn.Linear(nmf_feat_dim, num_classes)
        else:
            self.nmf_proj = None

        # latent dim for augmented embedding
        self.latent_dim = backbone_dim + nmf_feat_dim

    def _compute_mel(self, x):
        """Waveform (B, T) -> log-mel (B, 1, n_mels, time_frames)."""
        mel = self.mel_extractor(x)  # (B, n_mels, time)
        mel = mel.unsqueeze(1)       # (B, 1, n_mels, time)
        return mel

    def _apply_spec_aug(self, mel):
        """Apply SpecAugment during training."""
        mel = self.freq_mask(mel)
        mel = self.time_mask(mel)
        return mel

    @staticmethod
    def _solve_nmf_h(V, W, num_iters=50, eps=1e-8):
        """Solve V ~= W @ H for H using multiplicative updates."""
        B, F, T_time = V.shape
        K = W.shape[1]
        device, dtype = V.device, V.dtype
        W = W.to(device=device, dtype=dtype)

        H = torch.ones((B, K, T_time), device=device, dtype=dtype)
        WT = W.T                                         # (K, F)
        WTW = WT @ W                                     # (K, K)
        WTV = torch.matmul(WT.unsqueeze(0), V)           # (B, K, T)

        for _ in range(num_iters):
            denom = torch.matmul(WTW.unsqueeze(0), H) + eps
            H = H * (WTV / denom)
            H = torch.clamp(H, min=eps)
        return H

    @staticmethod
    def _summarize_nmf_h(H):
        """H (B, K, T) -> (B, 2K) via mean + max pooling."""
        return torch.cat([H.mean(dim=2), H.amax(dim=2)], dim=1)

    def _nmf_features(self, x):
        """Waveform -> NMF summary features (B, 2K)."""
        if self.W_nmf is None or self.nmf_k == 0:
            return None
        with torch.no_grad():
            mel_pow = self.nmf_mel(x)                    # (B, 64, T)
            mel_pow = torch.clamp(mel_pow, min=1e-10)
            H = self._solve_nmf_h(mel_pow, self.W_nmf)
            return self._summarize_nmf_h(H)

    def forward(self, x, mixup_lambda=None):
        """
        Args:
            x: waveform tensor, (B, T) or (B, 1, T).
        Returns:
            dict with 'clipwise_output' (B, num_classes) and 'latent_output'.
        """
        if x.dim() == 3 and x.shape[1] == 1:
            x = x[:, 0, :]  # (B, T)

        # NMF branch (detached, no grad)
        nmf_feat = self._nmf_features(x)

        # Backbone branch
        mel = self._compute_mel(x)  # (B, 1, n_mels, time)

        # Batch norm over freq axis: BN2d expects (B, C, H, W) with C=n_mels
        # mel is (B, 1, n_mels, time) -> squeeze channel, treat n_mels as C
        mel = mel.squeeze(1)                    # (B, n_mels, time)
        mel = mel.unsqueeze(-1)                 # (B, n_mels, time, 1)
        mel = self.bn0(mel)
        mel = mel.squeeze(-1).unsqueeze(1)      # (B, 1, n_mels, time)

        if self.training:
            mel = self._apply_spec_aug(mel)

        # EfficientNet expects (B, C, H, W) — our (B, 1, n_mels, time) is fine
        features = self.backbone(mel)          # (B, backbone_dim, H', W')
        pooled = self.global_pool(features)    # (B, backbone_dim, 1, 1)
        pooled = pooled.flatten(1)             # (B, backbone_dim)

        logits = self.head(self.head_drop(pooled))  # (B, num_classes)

        # Add NMF contribution
        if nmf_feat is not None and self.nmf_proj is not None:
            nmf_logits = self.nmf_proj(nmf_feat)
            logits = logits + nmf_logits
            latent = torch.cat([pooled, nmf_feat], dim=1)
        else:
            latent = pooled

        output_dict = {
            "clipwise_output": torch.sigmoid(logits),
            "latent_output": latent,
        }
        return output_dict
