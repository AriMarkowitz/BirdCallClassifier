"""
BirdCLEF 2026 — Kaggle Inference Notebook
HTSAT-tiny fine-tuned on BirdCLEF 2026 training data.
Supports multi-model ensemble (averages predictions across all checkpoints).

Constraints: CPU-only, no internet, 90-minute limit.
"""

import sys
import os
import types
import glob
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import soundfile as sf
import librosa

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
BATCH_SIZE = 16

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

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    state_dict = {k.replace("sed_model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# Load all models for ensemble
models = []
for cp in CHECKPOINT_PATHS:
    print(f"Loading {os.path.basename(cp)} ...")
    models.append(load_model(cp))
print(f"Loaded {len(models)} models for ensemble.")


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


def pad_to_clip(segment):
    """Pad a 5s segment to CLIP_SAMPLES (10s) by tiling."""
    if len(segment) < CLIP_SAMPLES:
        repeats = (CLIP_SAMPLES // max(len(segment), 1)) + 1
        segment = np.tile(segment, repeats)[:CLIP_SAMPLES]
    return segment[:CLIP_SAMPLES].astype(np.float32)


# ── Inference ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_batch_ensemble(models, waveforms):
    """Run inference with all models and average predictions."""
    x = torch.from_numpy(waveforms).float()
    probs_sum = None
    for model in models:
        output_dict = model(x)
        probs = output_dict["clipwise_output"].cpu().numpy()
        if probs_sum is None:
            probs_sum = probs
        else:
            probs_sum += probs
    return probs_sum / len(models)


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

    results = []

    for file_idx, filepath in enumerate(test_files):
        fname_stem = os.path.splitext(os.path.basename(filepath))[0]

        try:
            audio = load_audio(filepath)
        except Exception as e:
            print(f"WARNING: Failed to load {filepath}: {e}")
            continue

        duration_samples = len(audio)
        n_segments = duration_samples // SEGMENT_SAMPLES
        if n_segments == 0:
            continue

        segments = []
        row_ids = []
        for i in range(n_segments):
            start = i * SEGMENT_SAMPLES
            end = start + SEGMENT_SAMPLES
            seg = audio[start:end]
            seg = pad_to_clip(seg)
            segments.append(seg)
            end_second = (i + 1) * SEGMENT_SECONDS
            row_ids.append(f"{fname_stem}_{end_second}")

        segments = np.stack(segments, axis=0)
        for batch_start in range(0, len(segments), BATCH_SIZE):
            batch = segments[batch_start : batch_start + BATCH_SIZE]
            probs = predict_batch_ensemble(models, batch)
            for j in range(len(probs)):
                results.append((row_ids[batch_start + j], probs[j]))

        if (file_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {file_idx + 1}/{len(test_files)} files ({elapsed:.0f}s)")

    # ── Build submission ───────────────────────────────────────────────────────
    print(f"Building submission with {len(results)} rows...")

    if len(results) == 0:
        print("WARNING: No predictions generated. Using sample submission.")
        sub = pd.read_csv(SAMPLE_SUB_PATH)
    else:
        row_ids = [r[0] for r in results]
        probs_matrix = np.stack([r[1] for r in results], axis=0)

        col_indices = [col_to_model_idx[c] for c in sub_columns]
        probs_matrix = probs_matrix[:, col_indices]

        sub = pd.DataFrame(probs_matrix, columns=sub_columns)
        sub.insert(0, "row_id", row_ids)

    sub.to_csv(OUTPUT_PATH, index=False)
    elapsed = time.time() - t0
    print(f"Submission saved to {OUTPUT_PATH} ({len(sub)} rows, {elapsed:.0f}s)")


if __name__ == "__main__":
    main()
