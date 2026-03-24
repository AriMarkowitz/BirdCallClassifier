"""
BirdCLEF 2026 — Kaggle Inference Notebook
Bird-MAE-Base fine-tuned on BirdCLEF 2026 training data.
Supports multi-model ensemble (averages predictions across all checkpoints).

Constraints: CPU-only, no internet, 90-minute limit.
"""

import sys
import os
import glob
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import librosa

# ── CPU optimization ──────────────────────────────────────────────────────────
torch.set_num_threads(4)
torch.set_grad_enabled(False)

# ── Paths ──────────────────────────────────────────────────────────────────────
_INPUT = "/kaggle/input"

COMPETITION_DATA = None
for p in [f"{_INPUT}/birdclef-2026", f"{_INPUT}/competitions/birdclef-2026"]:
    if os.path.isfile(os.path.join(p, "sample_submission.csv")):
        COMPETITION_DATA = p
        break

MODEL_DATASET = None
for p in [f"{_INPUT}/birdclef-2026-model",
          f"{_INPUT}/datasets/arimarkowitz/birdclef-2026-model"]:
    if os.path.isdir(p):
        MODEL_DATASET = p
        break

assert COMPETITION_DATA, f"Competition data not found. /kaggle/input contains: {os.listdir(_INPUT)}"
assert MODEL_DATASET, f"Model dataset not found. /kaggle/input contains: {os.listdir(_INPUT)}"

# Robust checkpoint discovery with fallback patterns
checkpoint_patterns = [
    os.path.join(MODEL_DATASET, "birdclef-birdmae-*.ckpt"),
    os.path.join(MODEL_DATASET, "*birdmae*.ckpt"),
    os.path.join(MODEL_DATASET, "*.ckpt"),  # fallback: any .ckpt file
]
CHECKPOINT_PATHS = sorted({cp for pattern in checkpoint_patterns for cp in glob.glob(pattern)})

# If glob found nothing, list the directory for debugging
if not CHECKPOINT_PATHS:
    print(f"DEBUG: No checkpoints found via glob patterns.")
    print(f"Contents of {MODEL_DATASET}:")
    try:
        for item in sorted(os.listdir(MODEL_DATASET)):
            full_path = os.path.join(MODEL_DATASET, item)
            if os.path.isfile(full_path):
                size_mb = os.path.getsize(full_path) / (1024 ** 2)
                print(f"  [FILE] {item} ({size_mb:.1f} MB)")
            elif os.path.isdir(full_path):
                print(f"  [DIR]  {item}/")
    except Exception as e:
        print(f"  Error listing directory: {e}")

BIRD_MAE_CODE = os.path.join(MODEL_DATASET, "bird_mae")
TAXONOMY_PATH = os.path.join(COMPETITION_DATA, "taxonomy.csv")

print(f"Competition data: {COMPETITION_DATA}")
print(f"Model dataset: {MODEL_DATASET}")
print(f"Checkpoints found: {len(CHECKPOINT_PATHS)}")
for cp in CHECKPOINT_PATHS:
    print(f"  - {os.path.basename(cp)}")

assert len(CHECKPOINT_PATHS) > 0, "No checkpoints found"
assert os.path.isdir(BIRD_MAE_CODE), "bird_mae code directory missing in model dataset"

TEST_SOUNDSCAPES = os.path.join(COMPETITION_DATA, "test_soundscapes")
SAMPLE_SUB_PATH = os.path.join(COMPETITION_DATA, "sample_submission.csv")
OUTPUT_PATH = "/kaggle/working/submission.csv"

# ── Audio config ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 32000
SEGMENT_SECONDS = 5
CLIP_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS
BATCH_SIZE = 16

# ── Add Bird-MAE code to path ────────────────────────────────────────────────
sys.path.insert(0, MODEL_DATASET)
from bird_mae.model import BirdMAEModel
from bird_mae.config import BirdMAEConfig
from bird_mae.feature_extractor import BirdMAEFeatureExtractor

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
assert len(sub_columns) == NUM_CLASSES
col_to_model_idx = {lbl: label_map[lbl] for lbl in sub_columns}
col_indices = np.array([col_to_model_idx[c] for c in sub_columns])


# ── Load model ─────────────────────────────────────────────────────────────────
def load_model(checkpoint_path):
    """Load Bird-MAE model with classification head from checkpoint."""
    config = BirdMAEConfig()
    backbone = BirdMAEModel(config)
    head = nn.Linear(config.embed_dim, NUM_CLASSES)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    # Load backbone weights (strip 'backbone.' prefix)
    backbone_dict = {k.replace("backbone.", ""): v
                     for k, v in state_dict.items()
                     if k.startswith("backbone.")}
    backbone.load_state_dict(backbone_dict, strict=True)
    backbone.eval()

    # Load head weights
    head_dict = {k.replace("head.", ""): v
                 for k, v in state_dict.items()
                 if k.startswith("head.")}
    head.load_state_dict(head_dict, strict=True)
    head.eval()

    return backbone, head


feature_extractor = BirdMAEFeatureExtractor()

t_load = time.time()
models = []
for cp in CHECKPOINT_PATHS:
    print(f"Loading {os.path.basename(cp)} ...")
    models.append(load_model(cp))
print(f"Loaded {len(models)} models for ensemble in {time.time() - t_load:.1f}s")


# ── Audio helpers ──────────────────────────────────────────────────────────────
def load_audio(path):
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
    """Split audio into 5s segments."""
    n_segments = len(audio) // CLIP_SAMPLES
    if n_segments == 0:
        return None
    segments = np.empty((n_segments, CLIP_SAMPLES), dtype=np.float32)
    for i in range(n_segments):
        segments[i] = audio[i * CLIP_SAMPLES : (i + 1) * CLIP_SAMPLES]
    return segments


# ── Inference ──────────────────────────────────────────────────────────────────
def predict_batch_ensemble(models, waveforms):
    """Run inference with all models and average predictions."""
    mel = feature_extractor(waveforms)  # (B, 1, 512, 128)
    batch_size = len(waveforms)
    logits_sum = np.zeros((batch_size, NUM_CLASSES), dtype=np.float32)

    with torch.no_grad():
        for backbone, head in models:
            embeddings = backbone(mel)
            logits = head(embeddings)
            logits_sum += logits.detach().cpu().numpy()

    logits_avg = logits_sum / len(models)
    probs = torch.sigmoid(torch.from_numpy(logits_avg)).numpy()
    return probs


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
        fname_stem = os.path.splitext(os.path.basename(filepath))[0]

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
            probs = predict_batch_ensemble(models, batch)
            all_probs.append(probs[:, col_indices])
            all_row_ids.extend(row_ids[batch_start : batch_start + len(batch)])

        if (file_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {file_idx + 1}/{len(test_files)} files ({elapsed:.0f}s)")

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
