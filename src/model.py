"""
BirdSet-EfficientNet-based model for BirdCLEF multi-label bird call classification.

Architecture:
  - BirdSet pretrained EfficientNet-B1 backbone from Hugging Face
    (DBD-research-group/EfficientNet-B1-BirdSet-XCL).
  - BirdSet-compatible log-mel frontend (32kHz, n_fft=2048, hop=256, n_mels=256,
    power->dB, mean/std normalization).
  - Task head projects BirdSet logits to BirdCLEF target classes.

Input: raw waveform tensor (B, T) or (B, 1, T) with variable T. The collate
       function pads within-batch to max length; model remains fully convolutional
       over time and supports variable-length inputs.
Output dict: {
    "clipwise_output": (B, num_classes),
    "latent_output": (B, birdset_num_classes),
    "student_birdset_logits": (B, birdset_num_classes),
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


class BirdCLEFModel(nn.Module):
    """BirdSet EfficientNet-B1 + BirdCLEF projection head.

    Args:
        num_classes: Number of BirdCLEF classes.
        sample_rate: Audio sample rate.
        birdset_model_name: HF model id for BirdSet EfficientNet.
        pretrained: If False, initializes from config defaults (not recommended).
    """

    def __init__(
        self,
        num_classes,
        sample_rate=32000,
        birdset_model_name="DBD-research-group/EfficientNet-B1-BirdSet-XCL",
        max_time_frames=768,
        chunk_hop_frames=512,
        pretrained=True,
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

        # Training-time spec augment (kept moderate)
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
            # Initialize from config only (no pretrained weights) —
            # used when loading weights from a checkpoint instead.
            from transformers import EfficientNetConfig
            try:
                config = EfficientNetConfig.from_pretrained(birdset_model_name)
            except Exception:
                config = EfficientNetConfig.from_pretrained(
                    birdset_model_name, local_files_only=True,
                )
            config.num_channels = 1
            self.birdset_model = EfficientNetForImageClassification(config)

        birdset_dim = int(self.birdset_model.config.num_labels)
        self.latent_dim = birdset_dim

        self.head_drop = nn.Dropout(0.2)
        self.head = nn.Linear(birdset_dim, num_classes)

    def _compute_birdset_mel(self, x):
        """Waveform (B, T) -> normalized BirdSet mel image (B, 1, 256, time)."""
        spec = self.spectrogram(x)          # (B, F, T)
        mel = self.mel_scale(spec)          # (B, 256, T)
        mel = self.amp_to_db(mel)           # (B, 256, T)
        mel = mel.unsqueeze(1)              # (B, 1, 256, T)
        mel = (mel - self._spec_mean) / (self._spec_std + 1e-8)
        return mel

    def _apply_spec_aug(self, mel):
        """Apply SpecAugment during training."""
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

    def forward_birdset_logits(self, x, apply_spec_aug=False, return_mel=False):
        """Return BirdSet logits before BirdCLEF projection.

        Crops waveform to a bounded length BEFORE computing the STFT so that
        GPU memory is always bounded regardless of input audio length.

        During training, a single random waveform chunk is used.
        During eval, a sliding window over the waveform processes each chunk
        through STFT+backbone individually and averages logits.

        Args:
            x: waveform (B, T) or (B, 1, T)
            apply_spec_aug: apply SpecAugment to mel
            return_mel: if True, also return the clean mel (for distillation)
        Returns:
            logits (B, birdset_dim)  or  (logits, mel_clean) if return_mel
        """
        if x.dim() == 3 and x.shape[1] == 1:
            x = x[:, 0, :]

        # Max waveform samples that produce ~max_time_frames mel frames
        max_wav_samples = self.max_time_frames * self.hop_length + self.n_fft

        if self.training:
            # ── Random crop waveform BEFORE STFT (bounded memory) ──
            if x.shape[-1] > max_wav_samples:
                max_start = x.shape[-1] - max_wav_samples
                start = int(torch.randint(0, max_start + 1, (1,)).item())
                x = x[..., start:start + max_wav_samples]

            mel_clean = self._compute_birdset_mel(x)
            mel = self._apply_spec_aug(mel_clean) if apply_spec_aug else mel_clean
            logits = self.birdset_model(pixel_values=mel).logits
            if return_mel:
                return logits, mel_clean
            return logits

        # ── Eval: short audio fits in one pass ──
        if x.shape[-1] <= max_wav_samples:
            mel = self._compute_birdset_mel(x)
            logits = self.birdset_model(pixel_values=mel).logits
            if return_mel:
                return logits, mel
            return logits

        # ── Eval: sliding window over waveform chunks ──
        logits_sum = None
        n_chunks = 0
        wav_len = x.shape[-1]
        wav_hop = self.chunk_hop_frames * self.hop_length

        for start in range(0, wav_len, wav_hop):
            end = min(start + max_wav_samples, wav_len)
            x_chunk = x[..., start:end]
            mel_chunk = self._compute_birdset_mel(x_chunk)
            chunk_logits = self.birdset_model(pixel_values=mel_chunk).logits
            if logits_sum is None:
                logits_sum = chunk_logits
            else:
                logits_sum = logits_sum + chunk_logits
            n_chunks += 1
            if end >= wav_len:
                break

        result = logits_sum / max(n_chunks, 1)
        if return_mel:
            return result, None  # no single mel in multi-chunk eval
        return result

    def forward(self, x, mixup_lambda=None, apply_spec_aug=None, return_mel=False):
        """
        Args:
            x: waveform tensor, (B, T) or (B, 1, T).
            return_mel: return clean mel in output dict (for distillation).
        Returns:
            dict with 'clipwise_output', 'logits', 'latent_output',
            'student_birdset_logits', and optionally 'mel_for_teacher'.
        """
        if apply_spec_aug is None:
            apply_spec_aug = self.training

        fwd = self.forward_birdset_logits(
            x, apply_spec_aug=apply_spec_aug, return_mel=return_mel,
        )
        if return_mel:
            student_birdset_logits, mel_clean = fwd
        else:
            student_birdset_logits = fwd
            mel_clean = None

        logits = self.head(self.head_drop(student_birdset_logits))

        output = {
            "clipwise_output": torch.sigmoid(logits),
            "logits": logits,
            "latent_output": student_birdset_logits,
            "student_birdset_logits": student_birdset_logits,
        }
        if mel_clean is not None:
            output["mel_for_teacher"] = mel_clean
        return output
