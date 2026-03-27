"""
BirdCLEF 2026 — Kaggle Inference Notebook
EfficientNet fine-tuned on BirdCLEF 2026 training data.
Supports multi-model ensemble (averages predictions across all checkpoints).
Includes temporal MLP for time-of-day and seasonality awareness.

Constraints: CPU-only, no internet, 90-minute limit.
"""

import sys
import os
import types
import glob
import time
import math
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import librosa
import timm

# ── CPU optimization ──────────────────────────────────────────────────────────
torch.set_num_threads(4)  # Kaggle has 4 CPU cores
torch.set_grad_enabled(False)

# ── Paths ──────────────────────────────────────────────────────────────────────
_INPUT = "/kaggle/input"

# Competition data
COMPETITION_DATA = None
for p in [f"{_INPUT}/birdclef-2026", f"{_INPUT}/competitions/birdclef-2026"]:
    if os.path.isfile(os.path.join(p, "sample_submission.csv")):
        COMPETITION_DATA = p
        break

# Model dataset
MODEL_DATASET = None
for p in [f"{_INPUT}/birdclef-2026-model",
          f"{_INPUT}/datasets/arimarkowitz/birdclef-2026-model"]:
    if os.path.isdir(p):
        MODEL_DATASET = p
        break

assert COMPETITION_DATA, f"Competition data not found. /kaggle/input contains: {os.listdir(_INPUT)}"
assert MODEL_DATASET, f"Model dataset not found. /kaggle/input contains: {os.listdir(_INPUT)}"

# Find all checkpoints for ensemble
CHECKPOINT_PATHS = sorted(glob.glob(os.path.join(MODEL_DATASET, "birdclef-enet-*.ckpt")))

NMF_W_PATH = os.path.join(MODEL_DATASET, "W_k56.npy")
TAXONOMY_PATH = os.path.join(COMPETITION_DATA, "taxonomy.csv")

# Add model code to path
MODEL_CODE_DIR = os.path.join(MODEL_DATASET, "src")
if os.path.isdir(MODEL_CODE_DIR):
    sys.path.insert(0, MODEL_CODE_DIR)

print(f"Competition data: {COMPETITION_DATA}")
print(f"Model dataset: {MODEL_DATASET}")
print(f"Checkpoints found: {len(CHECKPOINT_PATHS)}")
for cp in CHECKPOINT_PATHS:
    print(f"  - {os.path.basename(cp)}")
print(f"NMF W matrix: {NMF_W_PATH}")
print(f"Taxonomy: {TAXONOMY_PATH}")

assert len(CHECKPOINT_PATHS) > 0, "No checkpoints found"

TEST_SOUNDSCAPES = os.path.join(COMPETITION_DATA, "test_soundscapes")
SAMPLE_SUB_PATH = os.path.join(COMPETITION_DATA, "sample_submission.csv")
OUTPUT_PATH = "/kaggle/working/submission.csv"


# ── Audio config (must match training) ─────────────────────────────────────────
SAMPLE_RATE = 32000
SEGMENT_SECONDS = 5
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS
BATCH_SIZE = 8
TEMPORAL_DIM = 4


# ── Import model ──────────────────────────────────────────────────────────────
# Try importing from model dataset, fall back to inline definition
try:
    from model import BirdCLEFModel
except ImportError:
    # Inline model definition for Kaggle (self-contained)
    def build_mel_extractor(sample_rate=32000, n_fft=1024, hop_length=320,
                            n_mels=128, f_min=50.0, f_max=14000.0):
        return nn.Sequential(
            T.MelSpectrogram(
                sample_rate=sample_rate, n_fft=n_fft, win_length=n_fft,
                hop_length=hop_length, f_min=f_min, f_max=f_max,
                n_mels=n_mels, window_fn=torch.hann_window,
                power=2.0, center=True, pad_mode="reflect",
            ),
            T.AmplitudeToDB(stype="power", top_db=80),
        )

    def build_nmf_mel_extractor(sample_rate=32000, n_fft=1024, hop_length=320,
                                 n_mels=64, f_min=50.0, f_max=14000.0):
        return T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=n_fft,
            hop_length=hop_length, f_min=f_min, f_max=f_max,
            n_mels=n_mels, window_fn=torch.hann_window,
            power=2.0, center=True, pad_mode="reflect",
        )

    class BirdCLEFModel(nn.Module):
        def __init__(self, num_classes, backbone_name="tf_efficientnet_b0_ns",
                     sample_rate=32000, n_fft=1024, hop_length=320,
                     n_mels=128, f_min=50.0, f_max=14000.0,
                     nmf_n_mels=64, W_path=None, pretrained=False):
            super().__init__()
            self.num_classes = num_classes
            self.sample_rate = sample_rate
            self.hop_length = hop_length

            self.mel_extractor = build_mel_extractor(
                sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
                n_mels=n_mels, f_min=f_min, f_max=f_max,
            )

            self.backbone = timm.create_model(
                backbone_name, pretrained=pretrained, in_chans=1,
                num_classes=0, global_pool="",
            )
            backbone_dim = self.backbone.num_features

            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.head_drop = nn.Dropout(0.3)
            self.head = nn.Linear(backbone_dim, num_classes)

            self.freq_mask = T.FrequencyMasking(freq_mask_param=16)
            self.time_mask = T.TimeMasking(time_mask_param=64)
            self.bn0 = nn.BatchNorm2d(n_mels)

            self.nmf_mel = build_nmf_mel_extractor(
                sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
                n_mels=nmf_n_mels, f_min=f_min, f_max=f_max,
            )

            if W_path and os.path.isfile(W_path):
                _W = torch.from_numpy(np.load(W_path)).float()
                self.register_buffer("W_nmf", _W)
                self.nmf_k = _W.shape[1]
            else:
                self.register_buffer("W_nmf", None)
                self.nmf_k = 0

            nmf_feat_dim = 2 * self.nmf_k
            if self.nmf_k > 0:
                self.nmf_proj = nn.Linear(nmf_feat_dim, num_classes)
            else:
                self.nmf_proj = None

            self.latent_dim = backbone_dim + nmf_feat_dim

        @staticmethod
        def _solve_nmf_h(V, W, num_iters=50, eps=1e-8):
            B, F, T_time = V.shape
            K = W.shape[1]
            device, dtype = V.device, V.dtype
            W = W.to(device=device, dtype=dtype)
            H = torch.ones((B, K, T_time), device=device, dtype=dtype)
            WT = W.T
            WTW = WT @ W
            WTV = torch.matmul(WT.unsqueeze(0), V)
            for _ in range(num_iters):
                denom = torch.matmul(WTW.unsqueeze(0), H) + eps
                H = H * (WTV / denom)
                H = torch.clamp(H, min=eps)
            return H

        @staticmethod
        def _summarize_nmf_h(H):
            return torch.cat([H.mean(dim=2), H.amax(dim=2)], dim=1)

        def _nmf_features(self, x):
            if self.W_nmf is None or self.nmf_k == 0:
                return None
            with torch.no_grad():
                mel_pow = self.nmf_mel(x)
                mel_pow = torch.clamp(mel_pow, min=1e-10)
                H = self._solve_nmf_h(mel_pow, self.W_nmf)
                return self._summarize_nmf_h(H)

        def forward(self, x, mixup_lambda=None):
            if x.dim() == 3 and x.shape[1] == 1:
                x = x[:, 0, :]

            nmf_feat = self._nmf_features(x)

            mel = self.mel_extractor(x).unsqueeze(1)
            mel = mel.squeeze(1).permute(0, 2, 1).unsqueeze(-1)
            mel = self.bn0(mel)
            mel = mel.squeeze(-1).unsqueeze(1).permute(0, 1, 3, 2)
            mel = mel.permute(0, 1, 3, 2)

            features = self.backbone(mel)
            pooled = self.global_pool(features).flatten(1)

            logits = self.head(self.head_drop(pooled))

            if nmf_feat is not None and self.nmf_proj is not None:
                nmf_logits = self.nmf_proj(nmf_feat)
                logits = logits + nmf_logits
                latent = torch.cat([pooled, nmf_feat], dim=1)
            else:
                latent = pooled

            return {
                "clipwise_output": torch.sigmoid(logits),
                "latent_output": latent,
            }


# ── Temporal features ─────────────────────────────────────────────────────────
def parse_temporal_features(filename):
    """Extract cyclical temporal features from soundscape filename."""
    m = re.search(r'_(\d{8})_(\d{6})', filename)
    if m is None:
        return np.zeros(TEMPORAL_DIM, dtype=np.float32)

    date_str, time_str = m.group(1), m.group(2)
    hour = int(time_str[:2]) + int(time_str[2:4]) / 60.0

    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    from datetime import date
    doy = date(year, month, day).timetuple().tm_yday

    hour_rad = 2.0 * math.pi * hour / 24.0
    doy_rad = 2.0 * math.pi * doy / 365.0

    return np.array([
        math.sin(hour_rad), math.cos(hour_rad),
        math.sin(doy_rad), math.cos(doy_rad),
    ], dtype=np.float32)


class TemporalMLP(nn.Module):
    """Small MLP that maps cyclical temporal features to a per-class logit bias."""

    def __init__(self, temporal_dim, num_classes, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, temporal_features):
        return self.mlp(temporal_features)


# ── Label map ──────────────────────────────────────────────────────────────────
def build_label_map(taxonomy_path):
    df = pd.read_csv(taxonomy_path)
    labels = sorted(df["primary_label"].astype(str).unique())
    return {lbl: i for i, lbl in enumerate(labels)}, labels


label_map, sorted_labels = build_label_map(TAXONOMY_PATH)
NUM_CLASSES = len(label_map)
print(f"Classes: {NUM_CLASSES}")

sample_sub = pd.read_csv(SAMPLE_SUB_PATH, nrows=0)
sub_columns = [c for c in sample_sub.columns if c != "row_id"]
assert len(sub_columns) == NUM_CLASSES, (
    f"Column count mismatch: {len(sub_columns)} vs {NUM_CLASSES}"
)
col_to_model_idx = {lbl: label_map[lbl] for lbl in sub_columns}


# ── Load model ─────────────────────────────────────────────────────────────────
def load_model(checkpoint_path):
    """Load EfficientNet model and temporal MLP from checkpoint."""
    model = BirdCLEFModel(
        num_classes=NUM_CLASSES,
        backbone_name="tf_efficientnet_b0_ns",
        sample_rate=SAMPLE_RATE,
        W_path=NMF_W_PATH,
        pretrained=False,
    )

    temporal_mlp = TemporalMLP(TEMPORAL_DIM, NUM_CLASSES)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    # Load model weights (strip model. prefix from Lightning wrapper)
    model_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            model_dict[k.replace("model.", "", 1)] = v
    model.load_state_dict(model_dict, strict=False)
    model.eval()

    # Load temporal MLP weights if present
    temporal_dict = {k.replace("temporal_mlp.", ""): v
                     for k, v in state_dict.items()
                     if k.startswith("temporal_mlp.")}
    if temporal_dict:
        temporal_mlp.load_state_dict(temporal_dict, strict=True)
        print(f"  Loaded temporal MLP from checkpoint")
    else:
        print(f"  No temporal MLP in checkpoint (using zero-init)")
    temporal_mlp.eval()

    return model, temporal_mlp


# Load all models for ensemble
t_load = time.time()
models = []
temporal_mlps = []
for cp in CHECKPOINT_PATHS:
    print(f"Loading {os.path.basename(cp)} ...")
    m, t = load_model(cp)
    models.append(m)
    temporal_mlps.append(t)
print(f"Loaded {len(models)} models for ensemble in {time.time() - t_load:.1f}s")

col_indices = np.array([col_to_model_idx[c] for c in sub_columns])


# ── Audio helpers ──────────────────────────────────────────────────────────────
def load_audio(path):
    """Load audio file, resample to SAMPLE_RATE, convert to mono."""
    try:
        audio, sr = sf.read(path, dtype="float32")
    except Exception:
        audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        return audio

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32)


def segment_audio(audio):
    """Split audio into 5s segments (no tiling — model handles variable length)."""
    n_segments = len(audio) // SEGMENT_SAMPLES
    if n_segments == 0:
        return None
    segments = np.empty((n_segments, SEGMENT_SAMPLES), dtype=np.float32)
    for i in range(n_segments):
        segments[i] = audio[i * SEGMENT_SAMPLES : (i + 1) * SEGMENT_SAMPLES]
    return segments


# ── Inference ──────────────────────────────────────────────────────────────────
def predict_batch_ensemble(models, temporal_mlps, waveforms, temporal_features):
    """Run inference with all models and average predictions."""
    x = torch.from_numpy(waveforms)
    t = torch.from_numpy(temporal_features)
    batch_size = len(waveforms)
    probs_sum = np.zeros((batch_size, NUM_CLASSES), dtype=np.float32)
    eps = 1e-6

    for model, temp_mlp in zip(models, temporal_mlps):
        output_dict = model(x)
        clipwise = output_dict["clipwise_output"]

        logits = torch.log(clipwise.clamp(eps, 1 - eps) / (1 - clipwise.clamp(eps, 1 - eps)))
        temporal_bias = temp_mlp(t)
        probs = torch.sigmoid(logits + temporal_bias)

        probs_sum += probs.numpy()
    probs_sum /= len(models)
    return probs_sum


# ── Main loop ──────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    test_files = sorted(
        glob.glob(os.path.join(TEST_SOUNDSCAPES, "*.ogg"))
        + glob.glob(os.path.join(TEST_SOUNDSCAPES, "*.wav"))
        + glob.glob(os.path.join(TEST_SOUNDSCAPES, "*.mp3"))
        + glob.glob(os.path.join(TEST_SOUNDSCAPES, "*.flac"))
    )
    print(f"Found {len(test_files)} test soundscapes")

    all_row_ids = []
    all_probs = []

    for file_idx, filepath in enumerate(test_files):
        fname = os.path.basename(filepath)
        fname_stem = os.path.splitext(fname)[0]

        temporal = parse_temporal_features(fname)

        try:
            audio = load_audio(filepath)
        except Exception as e:
            print(f"WARNING: Failed to load {filepath}: {e}")
            continue

        segments = segment_audio(audio)
        if segments is None:
            continue

        n_segments = len(segments)
        row_ids = [f"{fname_stem}_{(i + 1) * SEGMENT_SECONDS}" for i in range(n_segments)]

        for batch_start in range(0, n_segments, BATCH_SIZE):
            batch = segments[batch_start : batch_start + BATCH_SIZE]
            batch_temporal = np.tile(temporal, (len(batch), 1))
            probs = predict_batch_ensemble(models, temporal_mlps, batch, batch_temporal)
            all_probs.append(probs[:, col_indices])
            all_row_ids.extend(row_ids[batch_start : batch_start + len(batch)])

        if (file_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {file_idx + 1}/{len(test_files)} files ({elapsed:.0f}s)")

    # ── Build submission ───────────────────────────────────────────────────────
    print(f"Building submission with {len(all_row_ids)} rows...")

    if len(all_row_ids) == 0:
        print("WARNING: No predictions generated. Using sample submission.")
        sub = pd.read_csv(SAMPLE_SUB_PATH)
    else:
        probs_matrix = np.concatenate(all_probs, axis=0)
        sub = pd.DataFrame(probs_matrix, columns=sub_columns)
        sub.insert(0, "row_id", all_row_ids)

    sub.to_csv(OUTPUT_PATH, index=False)
    elapsed = time.time() - t0
    print(f"Submission saved to {OUTPUT_PATH} ({len(sub)} rows, {elapsed:.0f}s)")


if __name__ == "__main__":
    main()
