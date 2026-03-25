"""
Step 1: Build the concatenated spectrogram matrix V for NMFk.

Stratified sampling ensures every species appears at least `min_clips_per_species`
times. Also includes soundscape clips for background/polyphonic coverage.

Output: saves V (f x T), metadata (which clip contributed which columns), and
preprocessing params to nmf_analysis/output/.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Config — match HTSAT's spectrogram pipeline
# --------------------------------------------------------------------------- #
SAMPLE_RATE = 32000
N_FFT = 1024
HOP_LENGTH = 320  # 10ms at 32kHz (matches HTSAT hop_size)
N_MELS = 64       # matches HTSAT mel_bins
FMIN = 50         # matches HTSAT fmin
FMAX = 14000      # matches HTSAT fmax
CLIP_DURATION = 5.0  # seconds
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)


def compute_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Compute non-negative mel spectrogram (power) for a single clip.

    Returns array of shape (n_mels, n_frames).
    """
    S = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    # Floor to small positive value to keep nonneg
    S = np.maximum(S, 1e-10)
    return S


def load_clip(path: str, offset: float = 0.0, duration: float = CLIP_DURATION) -> np.ndarray:
    """Load a single audio clip, zero-pad if shorter than duration."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, offset=offset, duration=duration, mono=True)
    if len(audio) < CLIP_SAMPLES:
        audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
    return audio[:CLIP_SAMPLES]


def sample_train_audio(train_csv: pd.DataFrame, data_dir: Path,
                        min_per_species: int, max_per_species: int,
                        rng: np.random.Generator) -> list[dict]:
    """Stratified sample from train_audio: at least min, at most max per species."""
    clips = []
    for label, group in train_csv.groupby("primary_label"):
        n = max(min_per_species, min(max_per_species, len(group)))
        sampled = group.sample(n=min(n, len(group)), random_state=int(rng.integers(1e9)))
        for _, row in sampled.iterrows():
            clips.append({
                "path": str(data_dir / "train_audio" / row["filename"]),
                "source": "train_audio",
                "label": str(label),
                "offset": 0.0,
            })
    return clips


def sample_soundscapes(data_dir: Path, n_files: int,
                        rng: np.random.Generator) -> list[dict]:
    """Sample clips from soundscape files (both labeled and unlabeled)."""
    soundscape_dir = data_dir / "train_soundscapes"
    all_files = sorted(soundscape_dir.glob("*.ogg"))

    selected = rng.choice(all_files, size=min(n_files, len(all_files)), replace=False)
    clips = []
    for fpath in selected:
        # Get file duration and sample a few random 5s windows
        try:
            dur = librosa.get_duration(path=str(fpath))
        except Exception:
            continue
        n_windows = max(1, min(5, int(dur // CLIP_DURATION)))
        max_offset = max(0, dur - CLIP_DURATION)
        offsets = rng.uniform(0, max_offset, size=n_windows) if max_offset > 0 else [0.0]
        for off in offsets:
            clips.append({
                "path": str(fpath),
                "source": "soundscape",
                "label": "soundscape",
                "offset": float(off),
            })
    return clips


def main():
    parser = argparse.ArgumentParser(description="Build spectrogram matrix for NMFk")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Path to data/ directory")
    parser.add_argument("--output-dir", type=str, default="nmf_analysis/output",
                        help="Where to save output files")
    parser.add_argument("--min-per-species", type=int, default=3,
                        help="Min clips per species from train_audio")
    parser.add_argument("--max-per-species", type=int, default=10,
                        help="Max clips per species from train_audio")
    parser.add_argument("--n-soundscape-files", type=int, default=50,
                        help="Number of soundscape files to sample from")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Load metadata
    train_csv = pd.read_csv(data_dir / "train.csv")
    n_species = train_csv["primary_label"].nunique()
    print(f"Found {n_species} species in train.csv ({len(train_csv)} total clips)")

    # Stratified sampling
    train_clips = sample_train_audio(train_csv, data_dir,
                                      args.min_per_species, args.max_per_species, rng)
    soundscape_clips = sample_soundscapes(data_dir, args.n_soundscape_files, rng)
    all_clips = train_clips + soundscape_clips

    print(f"Sampled {len(train_clips)} train_audio clips + "
          f"{len(soundscape_clips)} soundscape clips = {len(all_clips)} total")

    # Build concatenated spectrogram matrix
    spectrograms = []
    metadata = []
    col_offset = 0

    for clip_info in tqdm(all_clips, desc="Computing spectrograms"):
        try:
            audio = load_clip(clip_info["path"], offset=clip_info["offset"])
            S = compute_mel_spectrogram(audio)
        except Exception as e:
            print(f"  Skipping {clip_info['path']}: {e}")
            continue

        n_frames = S.shape[1]
        metadata.append({
            **clip_info,
            "col_start": col_offset,
            "col_end": col_offset + n_frames,
            "n_frames": n_frames,
        })
        spectrograms.append(S)
        col_offset += n_frames

    V = np.concatenate(spectrograms, axis=1)
    print(f"Final matrix V shape: {V.shape}  ({V.shape[0]} mel bins x {V.shape[1]} time frames)")
    print(f"Memory: {V.nbytes / 1e9:.2f} GB (float64), {V.nbytes / 2e9:.2f} GB as float32")

    # Save
    np.save(output_dir / "V_matrix.npy", V.astype(np.float32))
    pd.DataFrame(metadata).to_csv(output_dir / "clip_metadata.csv", index=False)

    params = {
        "sample_rate": SAMPLE_RATE,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "clip_duration": CLIP_DURATION,
        "n_clips": len(metadata),
        "n_species_represented": len(set(m["label"] for m in metadata if m["label"] != "soundscape")),
        "matrix_shape": list(V.shape),
    }
    with open(output_dir / "preprocessing_params.json", "w") as f:
        json.dump(params, f, indent=2)

    print(f"\nSaved to {output_dir}/:")
    print(f"  V_matrix.npy          ({V.shape})")
    print(f"  clip_metadata.csv     ({len(metadata)} clips)")
    print(f"  preprocessing_params.json")
    print(f"\nSpecies coverage: {params['n_species_represented']}/{n_species}")


if __name__ == "__main__":
    main()
