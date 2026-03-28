"""
BirdSet-EfficientNet-based model for BirdCLEF multi-label bird call classification.

Architecture:
  - BirdSet pretrained EfficientNet-B1 backbone from Hugging Face
    (DBD-research-group/EfficientNet-B1-BirdSet-XCL).
  - BirdSet-compatible log-mel frontend (32kHz, n_fft=2048, hop=256, n_mels=256,
    power->dB, mean/std normalization).
  - Temporal attention pooling (SED-style): collapses the encoder's spatial
    feature map to a single vector by attending over time frames with a
    learnable CLS query, so the model can focus on frames that contain calls
    rather than averaging over silence.
  - MLP head on the attended 1280-dim features for BirdCLEF classification.
  - BirdSet classifier logits still available for teacher-student distillation.

Input: raw waveform tensor (B, T) or (B, 1, T) with variable T.
Output dict: {
    "clipwise_output": (B, num_classes),
    "logits": (B, num_classes),
    "latent_output": (B, hidden_dim),
    "student_birdset_logits": (B, birdset_num_classes),
    "attn_weights": (B, T_frames)   -- which time frames were attended to
}
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T

try:
    from transformers import EfficientNetForImageClassification
except Exception as exc:
    EfficientNetForImageClassification = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


class TemporalAttentionPool(nn.Module):
    """SED-style attention pooling over the time axis of a 2D feature map.

    The EfficientNet encoder produces (B, C, H, W) where H=8 (freq) and W=T
    (time, scales with audio length). Since the frequency dimension has been
    fully collapsed by the encoder's striding to just 8 bins with no remaining
    spectral structure, we first mean-pool over H to get a clean (B, C, T)
    time sequence, then pool T -> 1 using cross-attention with a single
    learnable CLS query.

    This is a standard off-the-shelf pattern:
      - nn.MultiheadAttention is the PyTorch built-in.
      - The CLS query is a learnable parameter (one vector, same idea as
        BERT's [CLS] token or the ViT class token).
      - The attention weights (B, T) tell you which time frames the model
        found most discriminative -- essentially free frame-level SED.

    Works with any input length at both train and eval time because MHA has
    no constraint on sequence length.

    Args:
        hidden_dim: channel dimension of the feature map (1280 for EfficientNet-B1).
        num_heads: number of attention heads (hidden_dim must be divisible by this).
        dropout: dropout on the attention weights.
    """

    def __init__(self, hidden_dim: int = 1280, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Learnable CLS query: the single "question" the model asks of the
        # time sequence. Initialized from N(0, hidden_dim^-0.5) like BERT.
        self.cls_query = nn.Parameter(
            torch.empty(1, 1, hidden_dim).normal_(std=hidden_dim ** -0.5)
        )

        # Standard PyTorch MHA. query=cls_token, key=value=time_sequence.
        # batch_first=False keeps (T, B, C) layout which avoids an extra
        # transpose and matches MHA's default (slightly faster).
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )

        # LayerNorm on the attended output for training stability.
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, H, W) encoder feature map. H is freq (always 8),
               W is time (scales with audio length).
        Returns:
            pooled: (B, C) attended feature vector.
            attn_weights: (B, W) attention weight per time frame.
        """
        B, C, H, W = x.shape

        # 1. Collapse freq (H=8) by mean-pooling -- no structure left there.
        #    (B, C, H, W) -> (B, C, W)
        x = x.mean(dim=2)

        # 2. Reshape to (W, B, C) for MHA's (seq, batch, embed) layout.
        x = x.permute(2, 0, 1)  # (W, B, C)

        # 3. Expand the CLS query to the current batch size.
        q = self.cls_query.expand(1, B, C)  # (1, B, C)

        # 4. Cross-attend: query is CLS token, keys/values are time frames.
        #    attn_weights: (B, 1, W) -- one query attending over W time steps.
        attended, attn_weights = self.mha(q, x, x)  # attended: (1, B, C)

        # 5. Squeeze query dim and apply LayerNorm.
        pooled = self.norm(attended.squeeze(0))  # (B, C)

        return pooled, attn_weights.squeeze(1)  # (B, C), (B, W)


class BirdCLEFModel(nn.Module):
    """BirdSet EfficientNet-B1 + temporal attention pooling + MLP head.

    Args:
        num_classes: Number of BirdCLEF classes.
        sample_rate: Audio sample rate.
        birdset_model_name: HF model id for BirdSet EfficientNet.
        pretrained: If False, initializes from config only (for checkpoint loading).
        attn_heads: Number of attention heads in temporal pooling.
        attn_dropout: Dropout on attention weights.
    """

    def __init__(
        self,
        num_classes,
        sample_rate=32000,
        birdset_model_name="DBD-research-group/EfficientNet-B1-BirdSet-XCL",
        max_time_frames=768,
        chunk_hop_frames=512,
        pretrained=True,
        attn_heads=8,
        attn_dropout=0.0,
    ):
        super().__init__()
        if EfficientNetForImageClassification is None:
            raise ImportError(
                "transformers is required for BirdSet backbone. "
                "Install with: pip install transformers safetensors"
            ) from _TRANSFORMERS_IMPORT_ERROR

        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.max_time_frames = int(max_time_frames)
        self.chunk_hop_frames = int(chunk_hop_frames)

        # BirdSet-compatible frontend settings
        self.n_fft = 2048
        self.hop_length = 256
        self.n_mels = 256
        self.spec_norm_mean = -4.268
        self.spec_norm_std = 4.569

        self.spectrogram = T.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
            center=True,
            pad_mode="reflect",
        )
        self.mel_scale = T.MelScale(
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            n_stft=(self.n_fft // 2) + 1,
        )
        self.amp_to_db = T.AmplitudeToDB(stype="power", top_db=80)

        # Training-time spec augment
        self.freq_mask = T.FrequencyMasking(freq_mask_param=24)
        self.time_mask = T.TimeMasking(time_mask_param=48)

        self.register_buffer(
            "_spec_mean",
            torch.tensor(self.spec_norm_mean, dtype=torch.float32).view(1, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_spec_std",
            torch.tensor(self.spec_norm_std, dtype=torch.float32).view(1, 1, 1, 1),
            persistent=False,
        )

        if pretrained:
            self.birdset_model = EfficientNetForImageClassification.from_pretrained(
                birdset_model_name,
                num_channels=1,
                ignore_mismatched_sizes=True,
            )
        else:
            from transformers import EfficientNetConfig
            config = EfficientNetConfig.from_pretrained(
                birdset_model_name,
                local_files_only=True,
            )
            config.num_channels = 1
            self.birdset_model = EfficientNetForImageClassification(config)

        birdset_dim = int(self.birdset_model.config.num_labels)
        hidden_dim = int(self.birdset_model.config.hidden_dim)  # 1280 for B1
        self.latent_dim = hidden_dim

        # Temporal attention pooling -- replaces AvgPool2d from the backbone.
        self.attn_pool = TemporalAttentionPool(
            hidden_dim=hidden_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
        )

        # MLP head on attended 1280-dim features.
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def _compute_birdset_mel(self, x):
        """Waveform (B, T) -> normalized BirdSet mel image (B, 1, 256, time)."""
        spec = self.spectrogram(x)
        mel = self.mel_scale(spec)
        mel = self.amp_to_db(mel)
        mel = mel.unsqueeze(1)
        mel = (mel - self._spec_mean) / (self._spec_std + 1e-8)
        return mel

    def _apply_spec_aug(self, mel):
        mel = self.freq_mask(mel)
        mel = self.time_mask(mel)
        return mel

    def preprocess_waveform(self, x, apply_spec_aug=False):
        """Public preprocessing utility for teacher-student distillation."""
        if x.dim() == 3 and x.shape[1] == 1:
            x = x[:, 0, :]
        mel = self._compute_birdset_mel(x)
        if apply_spec_aug:
            mel = self._apply_spec_aug(mel)
        return mel

    def _backbone_forward(self, mel):
        """Run encoder and return attended features, attn weights, and BirdSet logits.

        Returns:
            pooled:           (B, 1280) attended feature vector
            attn_weights:     (B, T_frames) attention weight per time frame
            birdset_logits:   (B, birdset_dim) for distillation
        """
        # Run encoder -- bypasses the backbone's own AvgPool2d
        enc = self.birdset_model.efficientnet.encoder(
            self.birdset_model.efficientnet.embeddings(mel),
            return_dict=True,
        )
        feat_map = enc.last_hidden_state  # (B, 1280, 8, T_frames)

        # Temporal attention pooling
        pooled, attn_weights = self.attn_pool(feat_map)  # (B, 1280), (B, T_frames)

        # BirdSet logits (needed for distillation) -- uses backbone's own pooler
        birdset_pooled = self.birdset_model.efficientnet.pooler(feat_map)
        birdset_pooled = birdset_pooled.reshape(birdset_pooled.shape[:2])
        birdset_logits = self.birdset_model.classifier(
            self.birdset_model.dropout(birdset_pooled)
        )

        return pooled, attn_weights, birdset_logits

    def forward_features(self, x, apply_spec_aug=False, return_mel=False):
        """Extract attended features with optional waveform chunking.

        Training: single random crop to bound GPU memory.
        Eval (short audio): single pass.
        Eval (long audio): sliding window, average pooled features across chunks.
        """
        if x.dim() == 3 and x.shape[1] == 1:
            x = x[:, 0, :]

        max_wav_samples = self.max_time_frames * self.hop_length + self.n_fft

        if self.training:
            if x.shape[-1] > max_wav_samples:
                max_start = x.shape[-1] - max_wav_samples
                start = int(torch.randint(0, max_start + 1, (1,)).item())
                x = x[..., start:start + max_wav_samples]

            mel_clean = self._compute_birdset_mel(x)
            mel = self._apply_spec_aug(mel_clean) if apply_spec_aug else mel_clean
            pooled, attn_weights, birdset_logits = self._backbone_forward(mel)
            if return_mel:
                return pooled, attn_weights, birdset_logits, mel_clean
            return pooled, attn_weights, birdset_logits

        if x.shape[-1] <= max_wav_samples:
            mel = self._compute_birdset_mel(x)
            pooled, attn_weights, birdset_logits = self._backbone_forward(mel)
            if return_mel:
                return pooled, attn_weights, birdset_logits, mel
            return pooled, attn_weights, birdset_logits

        # Sliding window -- average pooled features across chunks
        pooled_sum = None
        logits_sum = None
        n_chunks = 0
        wav_len = x.shape[-1]
        wav_hop = self.chunk_hop_frames * self.hop_length

        for start in range(0, wav_len, wav_hop):
            end = min(start + max_wav_samples, wav_len)
            mel_chunk = self._compute_birdset_mel(x[..., start:end])
            c_pooled, _, c_logits = self._backbone_forward(mel_chunk)
            pooled_sum = c_pooled if pooled_sum is None else pooled_sum + c_pooled
            logits_sum = c_logits if logits_sum is None else logits_sum + c_logits
            n_chunks += 1
            if end >= wav_len:
                break

        pooled = pooled_sum / max(n_chunks, 1)
        birdset_logits = logits_sum / max(n_chunks, 1)
        # attn_weights not meaningful when averaging across chunks
        attn_weights = torch.zeros(x.shape[0], 1, device=x.device)
        if return_mel:
            return pooled, attn_weights, birdset_logits, None
        return pooled, attn_weights, birdset_logits

    def forward(self, x, mixup_lambda=None, apply_spec_aug=None, return_mel=False):
        """
        Args:
            x: waveform tensor, (B, T) or (B, 1, T).
            return_mel: return clean mel in output dict (for distillation).
        Returns:
            dict with 'clipwise_output', 'logits', 'latent_output',
            'student_birdset_logits', 'attn_weights', and optionally 'mel_for_teacher'.
        """
        if apply_spec_aug is None:
            apply_spec_aug = self.training

        fwd = self.forward_features(
            x, apply_spec_aug=apply_spec_aug, return_mel=return_mel,
        )
        if return_mel:
            pooled, attn_weights, student_birdset_logits, mel_clean = fwd
        else:
            pooled, attn_weights, student_birdset_logits = fwd
            mel_clean = None

        logits = self.head(pooled)

        output = {
            "clipwise_output": torch.sigmoid(logits),
            "logits": logits,
            "latent_output": pooled,
            "student_birdset_logits": student_birdset_logits,
            "attn_weights": attn_weights,
        }
        if mel_clean is not None:
            output["mel_for_teacher"] = mel_clean
        return output
