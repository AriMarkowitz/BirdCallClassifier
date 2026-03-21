"""
BirdCLEF 2026 — Kaggle Inference Notebook
HTSAT-tiny fine-tuned on BirdCLEF 2026 training data.
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
import soundfile as sf
import librosa

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
CHECKPOINT_PATHS = sorted(glob.glob(os.path.join(MODEL_DATASET, "birdclef-htsat-*.ckpt")))

HTSAT_CODE_DIR = os.path.join(MODEL_DATASET, "htsat")
TAXONOMY_PATH = os.path.join(COMPETITION_DATA, "taxonomy.csv")

print(f"Competition data: {COMPETITION_DATA}")
print(f"Model dataset: {MODEL_DATASET}")
print(f"Checkpoints found: {len(CHECKPOINT_PATHS)}")
for cp in CHECKPOINT_PATHS:
    print(f"  - {os.path.basename(cp)}")
print(f"HTSAT code: {HTSAT_CODE_DIR}")
print(f"Taxonomy: {TAXONOMY_PATH}")

assert len(CHECKPOINT_PATHS) > 0, "No checkpoints found"
assert os.path.isdir(HTSAT_CODE_DIR), f"HTSAT code dir not found: {HTSAT_CODE_DIR}"
assert os.path.isfile(TAXONOMY_PATH), f"Taxonomy not found: {TAXONOMY_PATH}"

TEST_SOUNDSCAPES = os.path.join(COMPETITION_DATA, "test_soundscapes")
SAMPLE_SUB_PATH = os.path.join(COMPETITION_DATA, "sample_submission.csv")
OUTPUT_PATH = "/kaggle/working/submission.csv"

print(f"Test soundscapes dir exists: {os.path.isdir(TEST_SOUNDSCAPES)}")
print(f"Sample submission exists: {os.path.isfile(SAMPLE_SUB_PATH)}")


# ── Audio config (must match training) ─────────────────────────────────────────
SAMPLE_RATE = 32000
SEGMENT_SECONDS = 5
CLIP_SECONDS = 10
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS
BATCH_SIZE = 8  # smaller batches for CPU memory efficiency
TEMPORAL_DIM = 4

# ── Stub out problematic imports before loading HTSAT code ─────────────────────
for mod_name in ["h5py", "museval", "museval.metrics", "torchcontrib", "torchcontrib.optim"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)
sys.modules["torchcontrib.optim"].SWA = type("SWA", (), {})

# Fix numpy 2.x: sed_model.py does `from numpy.lib.function_base import average`
try:
    from numpy.lib.function_base import average  # noqa: F401
except (ImportError, ModuleNotFoundError):
    if not hasattr(np.lib, "function_base"):
        np.lib.function_base = types.ModuleType("numpy.lib.function_base")
    np.lib.function_base.average = np.average

# Add HTSAT code to sys.path
sys.path.insert(0, HTSAT_CODE_DIR)
print(f"HTSAT code dir exists: {os.path.isdir(HTSAT_CODE_DIR)}")

import config as htsat_config
from model.htsat import HTSAT_Swin_Transformer

# ── Config overrides (must match training) ─────────────────────────────────────
NUM_CLASSES = 234

htsat_config.classes_num = NUM_CLASSES
htsat_config.loss_type = "clip_bce"
htsat_config.enable_tscam = True
htsat_config.htsat_attn_heatmap = False
htsat_config.enable_repeat_mode = False


# ── Temporal features ─────────────────────────────────────────────────────────
def parse_temporal_features(filename):
    """Extract cyclical temporal features from soundscape filename.

    Filename format: BC2026_Test_0001_S05_20250227_010002.ogg
                                          YYYYMMDD  HHMMSS

    Returns np.float32 array of shape (4,):
        [sin(2π·hour/24), cos(2π·hour/24), sin(2π·doy/365), cos(2π·doy/365)]
    """
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
htsat_config.classes_num = NUM_CLASSES
print(f"Classes: {NUM_CLASSES}")

# Read exact column order from sample submission
sample_sub = pd.read_csv(SAMPLE_SUB_PATH, nrows=0)
sub_columns = [c for c in sample_sub.columns if c != "row_id"]
print(f"Submission columns: {len(sub_columns)}")

assert len(sub_columns) == NUM_CLASSES, (
    f"Column count mismatch: {len(sub_columns)} vs {NUM_CLASSES}"
)
col_to_model_idx = {lbl: label_map[lbl] for lbl in sub_columns}


# ── Load model ─────────────────────────────────────────────────────────────────
def load_model(checkpoint_path):
    """Load HTSAT model and temporal MLP from checkpoint."""
    model = HTSAT_Swin_Transformer(
        spec_size=htsat_config.htsat_spec_size,
        patch_size=htsat_config.htsat_patch_size,
        patch_stride=htsat_config.htsat_stride,
        num_classes=NUM_CLASSES,
        embed_dim=htsat_config.htsat_dim,
        depths=htsat_config.htsat_depth,
        num_heads=htsat_config.htsat_num_head,
        window_size=htsat_config.htsat_window_size,
        config=htsat_config,
    )

    temporal_mlp = TemporalMLP(TEMPORAL_DIM, NUM_CLASSES)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    # Load HTSAT weights (strip sed_model. prefix)
    htsat_dict = {k.replace("sed_model.", ""): v
                  for k, v in state_dict.items()
                  if k.startswith("sed_model.")}
    model.load_state_dict(htsat_dict, strict=True)
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

# Pre-compute column index mapping once
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
    """Split audio into 5s segments, each padded/tiled to 10s."""
    n_segments = len(audio) // SEGMENT_SAMPLES
    if n_segments == 0:
        return None
    segments = np.empty((n_segments, CLIP_SAMPLES), dtype=np.float32)
    for i in range(n_segments):
        seg = audio[i * SEGMENT_SAMPLES : (i + 1) * SEGMENT_SAMPLES]
        # Tile 5s to 10s
        segments[i] = np.tile(seg, 2)[:CLIP_SAMPLES]
    return segments



# ── Inference ──────────────────────────────────────────────────────────────────
def predict_batch_ensemble(models, temporal_mlps, waveforms, temporal_features):
    """Run inference with all models and average predictions.

    Applies temporal MLP bias in logit space before averaging.
    """
    x = torch.from_numpy(waveforms)
    t = torch.from_numpy(temporal_features)  # (batch, 4)
    batch_size = len(waveforms)
    probs_sum = np.zeros((batch_size, NUM_CLASSES), dtype=np.float32)
    eps = 1e-6

    for model, temp_mlp in zip(models, temporal_mlps):
        output_dict = model(x)
        clipwise = output_dict["clipwise_output"]  # sigmoided

        # Apply temporal bias in logit space
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

        # Parse temporal features once per file (same for all segments)
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

        # Batch inference
        for batch_start in range(0, n_segments, BATCH_SIZE):
            batch = segments[batch_start : batch_start + BATCH_SIZE]
            # Broadcast temporal features to batch size
            batch_temporal = np.tile(temporal, (len(batch), 1))
            probs = predict_batch_ensemble(models, temporal_mlps, batch, batch_temporal)
            # Reorder columns to match submission format
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
