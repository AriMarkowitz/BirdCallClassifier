"""Bird-MAE feature extractor — standalone version.

Converts raw waveforms to normalized fbank mel spectrograms matching
the Bird-MAE-Base preprocessing pipeline.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torchaudio.compliance.kaldi import fbank


class BirdMAEFeatureExtractor:
    """Extracts fbank features from raw waveforms for Bird-MAE.

    Input: waveform tensor of shape (batch, samples) at 32kHz
    Output: mel spectrogram tensor of shape (batch, 1, 512, 128)
    """

    def __init__(self, sampling_rate=32000, num_mel_bins=128,
                 target_length=512, mean=-7.2, std=4.43):
        self.sampling_rate = sampling_rate
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.mean = mean
        self.std = std

    def __call__(self, waveforms):
        """Convert waveforms to normalized fbank features.

        Args:
            waveforms: tensor of shape (batch, samples) or (samples,)

        Returns:
            tensor of shape (batch, 1, target_length, num_mel_bins)
        """
        if not torch.is_tensor(waveforms):
            waveforms = torch.from_numpy(np.array(waveforms))

        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0)

        # Pad/truncate to exactly 5 seconds
        clip_samples = self.sampling_rate * 5
        if waveforms.shape[1] < clip_samples:
            pad = clip_samples - waveforms.shape[1]
            waveforms = F.pad(waveforms, (0, pad))
        elif waveforms.shape[1] > clip_samples:
            waveforms = waveforms[:, :clip_samples]

        # Compute fbank features per sample
        features = []
        for waveform in waveforms:
            feat = fbank(
                waveform.unsqueeze(0),
                htk_compat=True,
                sample_frequency=self.sampling_rate,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=self.num_mel_bins,
                dither=0.0,
                frame_shift=10,
            )
            features.append(feat)
        features = torch.stack(features)  # (batch, time, mel_bins)

        # Pad/truncate to target_length
        if features.shape[1] > self.target_length:
            features = features[:, :self.target_length, :]
        elif features.shape[1] < self.target_length:
            diff = self.target_length - features.shape[1]
            min_val = features.min()
            features = F.pad(features, (0, 0, 0, diff), value=min_val.item())

        # Normalize
        features = (features - self.mean) / (self.std * 2)

        # Add channel dim: (batch, 1, 512, 128)
        return features.unsqueeze(1)
