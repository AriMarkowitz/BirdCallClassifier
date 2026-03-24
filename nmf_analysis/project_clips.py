"""
Step 3: Project all training clips into the fixed NMF basis.

For each clip, compute mel spectrogram V_clip and solve:
    V_clip ≈ W @ H_clip   (W fixed, solve for H_clip >= 0)

Uses GPU-accelerated batch NNLS for speed.

Then extract summary features from H_clip:
    - mean, max, std of each component over time
    - sparsity (fraction of frames above threshold per component)

Output: per-clip latent feature vectors saved as a single .npy array + metadata.
"""

import argparse
import json
import logging
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from build_spectrogram_matrix import (
    CLIP_DURATION, CLIP_SAMPLES, HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE,
    compute_mel_spectrogram, load_clip,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def gpu_nnls(W: torch.Tensor, V_clip: torch.Tensor,
             max_iter: int = 200, tol: float = 1e-6) -> torch.Tensor:
    """Solve V_clip ≈ W @ H for H >= 0 via multiplicative updates on GPU.

    W: (f, k), V_clip: (f, T) -> H: (k, T)
    Much faster than scipy.nnls column-by-column on CPU.
    """
    eps = 1e-10
    k = W.shape[1]
    T = V_clip.shape[1]

    # Initialize H
    H = torch.rand(k, T, device=W.device) * 0.01 + eps

    # Precompute W^T W (k, k) — constant across iterations
    WtW = W.T @ W  # (k, k)
    WtV = W.T @ V_clip  # (k, T)

    for _ in range(max_iter):
        # Multiplicative update: H <- H * (W^T V) / (W^T W H + eps)
        denom = WtW @ H + eps
        H = H * (WtV / denom)

    return H


def gpu_nnls_batch(W: torch.Tensor, V_batch: torch.Tensor,
                   max_iter: int = 200) -> torch.Tensor:
    """Solve NNLS for a batch of clips simultaneously.

    W: (f, k)
    V_batch: (batch, f, T)
    Returns H_batch: (batch, k, T)
    """
    eps = 1e-10
    B, f, T = V_batch.shape
    k = W.shape[1]

    H = torch.rand(B, k, T, device=W.device) * 0.01 + eps

    WtW = W.T @ W  # (k, k)
    WtV = torch.bmm(W.T.unsqueeze(0).expand(B, -1, -1), V_batch)  # (B, k, T)

    for _ in range(max_iter):
        denom = torch.bmm(WtW.unsqueeze(0).expand(B, -1, -1), H) + eps  # (B, k, T)
        H = H * (WtV / denom)

    return H


def extract_features(H: np.ndarray, sparsity_threshold: float = 0.01) -> np.ndarray:
    """Extract summary features from activation matrix H (k, T).

    Returns 1D feature vector of length 4*k:
        [mean_per_component, max_per_component, std_per_component, sparsity_per_component]
    """
    h_mean = H.mean(axis=1)
    h_max = H.max(axis=1)
    h_std = H.std(axis=1)
    h_sparsity = (H > sparsity_threshold).mean(axis=1)

    return np.concatenate([h_mean, h_max, h_std, h_sparsity]).astype(np.float32)


def extract_features_batch(H_batch: torch.Tensor,
                           sparsity_threshold: float = 0.01) -> np.ndarray:
    """Extract features from a batch of H matrices on GPU.

    H_batch: (B, k, T)
    Returns: (B, 4*k) numpy array
    """
    h_mean = H_batch.mean(dim=2)   # (B, k)
    h_max = H_batch.amax(dim=2)    # (B, k)
    h_std = H_batch.std(dim=2)     # (B, k)
    h_sparsity = (H_batch > sparsity_threshold).float().mean(dim=2)  # (B, k)

    features = torch.cat([h_mean, h_max, h_std, h_sparsity], dim=1)  # (B, 4k)
    return features.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Project clips into NMF basis")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--nmf-dir", type=str, default="nmf_analysis/output")
    parser.add_argument("--output-dir", type=str, default="nmf_analysis/output")
    parser.add_argument("--source", type=str, default="all",
                        choices=["train_audio", "soundscapes", "all"],
                        help="Which clips to project")
    parser.add_argument("--max-clips", type=int, default=0,
                        help="Max clips to process (0 = all)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for GPU NNLS projection")
    parser.add_argument("--nnls-iters", type=int, default=200,
                        help="Max iterations for NNLS solver")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    nmf_dir = Path(args.nmf_dir)
    output_dir = Path(args.output_dir)

    W_np = np.load(nmf_dir / "W_global.npy")
    k = W_np.shape[1]
    W = torch.from_numpy(W_np).float().to(device)
    log.info(f"Loaded W_global: {W_np.shape} (k={k})")

    # Build clip list
    clips = []

    if args.source in ("train_audio", "all"):
        train_csv = pd.read_csv(data_dir / "train.csv")
        for _, row in train_csv.iterrows():
            clips.append({
                "path": str(data_dir / "train_audio" / row["filename"]),
                "source": "train_audio",
                "label": str(row["primary_label"]),
                "offset": 0.0,
            })
        log.info(f"train_audio: {len(clips)} clips")

    n_train = len(clips)

    if args.source in ("soundscapes", "all"):
        soundscape_dir = data_dir / "train_soundscapes"
        for fpath in sorted(soundscape_dir.glob("*.ogg")):
            try:
                dur = librosa.get_duration(path=str(fpath))
            except Exception:
                continue
            # Process every 5s window
            for offset in np.arange(0, dur - CLIP_DURATION + 0.1, CLIP_DURATION):
                clips.append({
                    "path": str(fpath),
                    "source": "soundscape",
                    "label": "soundscape",
                    "offset": float(offset),
                })
        log.info(f"soundscapes: {len(clips) - n_train} clips")

    if args.max_clips > 0:
        clips = clips[:args.max_clips]

    feature_dim = 4 * k  # mean, max, std, sparsity
    log.info(f"Projecting {len(clips)} clips into {k}-dim NMF basis "
             f"(feature dim = {feature_dim}, batch_size = {args.batch_size})")

    features = np.zeros((len(clips), feature_dim), dtype=np.float32)
    valid_mask = np.ones(len(clips), dtype=bool)

    # Process in batches for GPU efficiency
    batch_size = args.batch_size
    n_batches = (len(clips) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(n_batches), desc="Projecting"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(clips))
        batch_clips = clips[start:end]

        # Load and compute spectrograms for this batch
        specs = []
        batch_valid = []
        for i, clip_info in enumerate(batch_clips):
            try:
                audio = load_clip(clip_info["path"], offset=clip_info["offset"])
                S = compute_mel_spectrogram(audio)
                specs.append(S)
                batch_valid.append(True)
            except Exception as e:
                if batch_idx == 0:  # only log first batch errors
                    log.warning(f"Skipping {clip_info['path']}: {e}")
                batch_valid.append(False)
                valid_mask[start + i] = False

        if not specs:
            continue

        # Stack into batch tensor (B, f, T) — pad to same T if needed
        max_T = max(s.shape[1] for s in specs)
        V_batch = np.zeros((len(specs), N_MELS, max_T), dtype=np.float32)
        for j, s in enumerate(specs):
            V_batch[j, :, :s.shape[1]] = s

        V_batch_t = torch.from_numpy(V_batch).to(device)

        # Batch NNLS solve
        H_batch = gpu_nnls_batch(W, V_batch_t, max_iter=args.nnls_iters)

        # Extract features
        batch_features = extract_features_batch(H_batch)

        # Write back to features array
        valid_idx = 0
        for i, v in enumerate(batch_valid):
            if v:
                features[start + i] = batch_features[valid_idx]
                valid_idx += 1

    # Filter out failed clips
    features = features[valid_mask]
    valid_clips = [c for c, v in zip(clips, valid_mask) if v]

    np.save(output_dir / "nmf_latent_features.npy", features)
    pd.DataFrame(valid_clips).to_csv(output_dir / "projected_clip_metadata.csv", index=False)

    log.info(f"Saved {len(valid_clips)} clip features of dim {feature_dim}")
    log.info(f"  nmf_latent_features.npy: {features.shape}")
    log.info(f"  projected_clip_metadata.csv")

    # Summary stats
    log.info(f"Feature stats — mean: {features.mean():.4f}, "
             f"std: {features.std():.4f}, "
             f"min: {features.min():.4f}, "
             f"max: {features.max():.4f}")


if __name__ == "__main__":
    main()
