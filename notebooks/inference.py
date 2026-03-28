"""
BirdCLEF 2026 — Kaggle Inference Notebook
BirdSet EfficientNet-B1 fine-tuned on BirdCLEF 2026 training data.
Supports multi-model ensemble (averages predictions across all checkpoints).

Constraints: CPU-only, no internet, 90-minute limit.
"""

import sys
import os
import glob
import time
import math
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import soundfile as sf
import librosa

# ── CPU optimization ──────────────────────────────────────────────────────────
torch.set_num_threads(4)
torch.set_grad_enabled(False)

# ── Paths ─────────────────────────────────────────────────────────────────────
_INPUT = "/kaggle/input"

COMPETITION_DATA = None
for p in [f"{_INPUT}/birdclef-2026", f"{_INPUT}/competitions/birdclef-2026"]:
    if os.path.isfile(os.path.join(p, "sample_submission.csv")):
        COMPETITION_DATA = p
        break
assert COMPETITION_DATA, f"Competition data not found under {_INPUT}"

MODEL_DATASET = None
for p in [f"{_INPUT}/birdclef-2026-model",
          f"{_INPUT}/datasets/arimarkowitz/birdclef-2026-model"]:
    if os.path.isdir(p):
        MODEL_DATASET = p
        break
assert MODEL_DATASET, f"Model dataset not found under {_INPUT}"

# src/model.py must be present — no fallback
MODEL_CODE_DIR = os.path.join(MODEL_DATASET, "src")
assert os.path.isdir(MODEL_CODE_DIR), (
    f"src/ not found in model dataset at {MODEL_CODE_DIR}. "
    "Re-run submit.sh to sync model code."
)
sys.path.insert(0, MODEL_CODE_DIR)
from model import BirdCLEFModel

# BirdSet config must be present — no HF download at inference time
BIRDSET_MODEL_DIR = os.path.join(MODEL_DATASET, "DBD-research-group", "EfficientNet-B1-BirdSet-XCL")
assert os.path.isfile(os.path.join(BIRDSET_MODEL_DIR, "config.json")), (
    f"BirdSet config not found at {BIRDSET_MODEL_DIR}. "
    "Add DBD-research-group/EfficientNet-B1-BirdSet-XCL/ to the Kaggle dataset."
)

CHECKPOINT_PATHS = sorted(glob.glob(os.path.join(MODEL_DATASET, "birdclef-birdset-*.ckpt")))
assert len(CHECKPOINT_PATHS) > 0, (
    f"No birdclef-birdset-*.ckpt checkpoints found in {MODEL_DATASET}"
)

TAXONOMY_PATH = os.path.join(COMPETITION_DATA, "taxonomy.csv")
TEST_SOUNDSCAPES = os.path.join(COMPETITION_DATA, "test_soundscapes")
SAMPLE_SUB_PATH = os.path.join(COMPETITION_DATA, "sample_submission.csv")
OUTPUT_PATH = "/kaggle/working/submission.csv"

print(f"Competition data: {COMPETITION_DATA}")
print(f"Model dataset:    {MODEL_DATASET}")
print(f"Checkpoints ({len(CHECKPOINT_PATHS)}):")
for cp in CHECKPOINT_PATHS:
    print(f"  {os.path.basename(cp)}")


# ── Audio config (must match training) ────────────────────────────────────────
SAMPLE_RATE = 32000
SEGMENT_SECONDS = 5
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS
BATCH_SIZE = 8


# ── Label map ─────────────────────────────────────────────────────────────────
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
    f"Column mismatch: submission has {len(sub_columns)}, taxonomy has {NUM_CLASSES}"
)
col_indices = np.array([label_map[c] for c in sub_columns])


# ── Load model ────────────────────────────────────────────────────────────────
def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    model = BirdCLEFModel(
        num_classes=NUM_CLASSES,
        sample_rate=SAMPLE_RATE,
        birdset_model_name=BIRDSET_MODEL_DIR,
        pretrained=False,
    )
    model_dict = {k.replace("model.", "", 1): v
                  for k, v in state_dict.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(model_dict, strict=True)
    assert not missing, f"Missing keys loading {os.path.basename(checkpoint_path)}: {missing}"
    model.eval()
    print(f"  Loaded {os.path.basename(checkpoint_path)}")
    return model


t_load = time.time()
models = [load_model(cp) for cp in CHECKPOINT_PATHS]
print(f"Loaded {len(models)} model(s) in {time.time() - t_load:.1f}s")


# ── Audio helpers ─────────────────────────────────────────────────────────────
def load_audio(path):
    try:
        audio, sr = sf.read(path, dtype="float32")
    except Exception:
        audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        return audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32)


def segment_audio(audio):
    n_segments = len(audio) // SEGMENT_SAMPLES
    if n_segments == 0:
        return None
    segments = np.empty((n_segments, SEGMENT_SAMPLES), dtype=np.float32)
    for i in range(n_segments):
        segments[i] = audio[i * SEGMENT_SAMPLES:(i + 1) * SEGMENT_SAMPLES]
    return segments


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_batch_ensemble(waveforms):
    x = torch.from_numpy(waveforms)
    probs_sum = np.zeros((len(waveforms), NUM_CLASSES), dtype=np.float32)
    for model in models:
        logits = model(x)["logits"]
        probs_sum += torch.sigmoid(logits).numpy()
    return probs_sum / len(models)


# ── Main loop ─────────────────────────────────────────────────────────────────
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
            batch = segments[batch_start:batch_start + BATCH_SIZE]
            probs = predict_batch_ensemble(batch)
            all_probs.append(probs[:, col_indices])
            all_row_ids.extend(row_ids[batch_start:batch_start + len(batch)])

        if (file_idx + 1) % 50 == 0:
            print(f"  Processed {file_idx + 1}/{len(test_files)} ({time.time() - t0:.0f}s)")

    print(f"Building submission with {len(all_row_ids)} rows...")
    if not all_probs:
        # No test audio found — write empty submission matching sample format
        sub = pd.read_csv(SAMPLE_SUB_PATH)
        sub.to_csv(OUTPUT_PATH, index=False)
        print(f"No test audio found — wrote empty submission to {OUTPUT_PATH}")
        return
    probs_matrix = np.concatenate(all_probs, axis=0)
    sub = pd.DataFrame(probs_matrix, columns=sub_columns)
    sub.insert(0, "row_id", all_row_ids)
    sub.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH} ({len(sub)} rows, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
