"""
Detect active vocal regions in train_audio files.

For each file, computes smoothed energy in the 1-10 kHz band and marks frames
above the file's median energy as "active." Contiguous inactive regions longer
than --min-dead-sec are removed. Short active fragments (< 2s) surrounded by
dead zones are also dropped.

This uses a global (per-file) median threshold, which works well because:
- Vocalizations produce clear energy spikes above ambient noise
- The median naturally adapts to each recording's noise floor
- Human noise (talking, footsteps) is mostly below 1 kHz and doesn't inflate the threshold

Output: JSON mapping filename -> list of [start_sec, end_sec] valid regions.

Usage:
    python scripts/preprocess_activity.py --data-dir data --output data/valid_regions.json
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


SAMPLE_RATE = 32000
N_FFT = 2048
HOP_LENGTH = 512  # ~16ms per frame at 32kHz
FREQ_LOW = 1000   # Hz — lower bound of animal vocalization band
FREQ_HIGH = 10000 # Hz — upper bound
FRAME_DURATION = HOP_LENGTH / SAMPLE_RATE
SMOOTH_SEC = 0.5  # smoothing window for energy


def compute_band_energy(audio, sr=SAMPLE_RATE):
    """Compute smoothed per-frame energy in the 1-10 kHz band."""
    S = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    band_mask = (freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH)
    band_energy = S[band_mask, :].sum(axis=0)

    # Smooth with half-second window to reduce frame-level noise
    smooth_frames = max(int(SMOOTH_SEC * sr / HOP_LENGTH), 1)
    kernel = np.ones(smooth_frames) / smooth_frames
    band_energy = np.convolve(band_energy, kernel, mode="same")

    return band_energy


def find_active_regions(band_energy, min_dead_sec=5.0, min_active_sec=2.0):
    """Find active regions using per-file median threshold.

    Args:
        band_energy: smoothed per-frame energy array
        min_dead_sec: inactive stretches shorter than this are bridged
        min_active_sec: active regions shorter than this are dropped

    Returns:
        List of [start_sec, end_sec] for active regions.
    """
    if len(band_energy) == 0:
        return []

    threshold = np.median(band_energy)
    threshold = max(threshold, 1e-10)

    active = band_energy > threshold
    min_dead_frames = int(min_dead_sec / FRAME_DURATION)
    min_active_frames = int(min_active_sec / FRAME_DURATION)

    # Step 1: find raw active runs
    raw_regions = []
    in_active = False
    start = 0
    for i in range(len(active)):
        if active[i] and not in_active:
            start = i
            in_active = True
        elif not active[i] and in_active:
            raw_regions.append((start, i))
            in_active = False
    if in_active:
        raw_regions.append((start, len(active)))

    if not raw_regions:
        return []

    # Step 2: bridge short gaps (< min_dead_frames) between active regions
    merged = [list(raw_regions[0])]
    for start, end in raw_regions[1:]:
        gap = start - merged[-1][1]
        if gap < min_dead_frames:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    # Step 3: drop short active fragments (< min_active_frames)
    filtered = []
    for start, end in merged:
        if (end - start) >= min_active_frames:
            filtered.append([start, end])

    # Convert to seconds
    result = []
    for start, end in filtered:
        result.append([round(start * FRAME_DURATION, 3),
                       round(end * FRAME_DURATION, 3)])

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Detect active vocal regions in train_audio files")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="data/valid_regions.json")
    parser.add_argument("--min-dead-sec", type=float, default=5.0,
                        help="Minimum inactive duration (sec) to mark as dead zone")
    parser.add_argument("--min-active-sec", type=float, default=2.0,
                        help="Minimum active region duration (sec) to keep")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_csv = pd.read_csv(data_dir / "train.csv")
    audio_dir = data_dir / "train_audio"

    filenames = train_csv["filename"].values
    print(f"Processing {len(filenames)} files...")

    valid_regions = {}
    n_fully_active = 0
    n_has_dead = 0
    n_failed = 0
    total_duration = 0.0
    total_active = 0.0

    for i, fn in enumerate(tqdm(filenames, desc="Detecting activity", mininterval=30)):
        path = str(audio_dir / fn)
        try:
            audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            if i < 5:
                print(f"  Skipping {fn}: {e}")
            n_failed += 1
            continue

        file_duration = len(audio) / SAMPLE_RATE
        total_duration += file_duration

        band_energy = compute_band_energy(audio)
        regions = find_active_regions(
            band_energy,
            min_dead_sec=args.min_dead_sec,
            min_active_sec=args.min_active_sec,
        )

        if not regions:
            # Fallback: use entire file if no active regions found
            regions = [[0.0, round(file_duration, 3)]]
            n_fully_active += 1
        else:
            active_dur = sum(e - s for s, e in regions)
            total_active += active_dur
            if active_dur >= file_duration - 0.5:
                n_fully_active += 1
            else:
                n_has_dead += 1

        valid_regions[fn] = regions

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(filenames)}, "
                  f"{n_has_dead} files with dead zones so far")

    # Summary stats
    pct_active = (total_active / total_duration * 100) if total_duration > 0 else 0
    print(f"\nResults:")
    print(f"  Total files: {len(filenames)}")
    print(f"  Failed to load: {n_failed}")
    print(f"  Fully active (no dead zones): {n_fully_active}")
    print(f"  Has dead zones removed: {n_has_dead}")
    print(f"  Total duration: {total_duration / 3600:.1f} hours")
    print(f"  Active duration: {total_active / 3600:.1f} hours ({pct_active:.1f}%)")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(valid_regions, f)
    print(f"\nSaved valid regions to {output_path} ({len(valid_regions)} files)")


if __name__ == "__main__":
    main()
