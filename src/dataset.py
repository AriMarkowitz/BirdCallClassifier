"""
Memory-efficient dataloaders for BirdCLEF 2026.

Two dataset classes:
  - TrainAudioDataset:      individual species clips from train_audio/
  - SoundscapeDataset:      5-second segments from train_soundscapes/

Both return {"waveform": np.float32 (clip_samples,), "target": np.float32 (num_classes,)}
which matches the HTSAT SEDWrapper training interface.
"""

import os
import re
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from torch.utils.data import Dataset, ConcatDataset, DataLoader, WeightedRandomSampler
import torch
import logging

logger = logging.getLogger(__name__)


def build_label_map(taxonomy_path):
    """Build mapping from primary_label (str) -> class index using taxonomy.csv."""
    df = pd.read_csv(taxonomy_path)
    labels = sorted(df["primary_label"].astype(str).unique())
    return {lbl: i for i, lbl in enumerate(labels)}


def _parse_site_id(filename):
    """Extract site ID (e.g. 'S08') from soundscape filename like BC2026_Train_0001_S08_..."""
    m = re.search(r'_S(\d+)_', filename)
    return f"S{m.group(1)}" if m else None


class TrainAudioDataset(Dataset):
    """
    Loads individual species recordings from train_audio/.
    Each audio file maps to one primary_label (single-label, stored as multi-hot).
    Audio is loaded on-the-fly from .ogg for memory efficiency.
    """

    def __init__(self, csv_path, audio_dir, label_map, sample_rate=32000,
                 clip_duration=10.0, label_smoothing=0.0):
        self.audio_dir = audio_dir
        self.label_map = label_map
        self.sample_rate = sample_rate
        self.clip_samples = int(sample_rate * clip_duration)
        self.num_classes = len(label_map)
        self.label_smoothing = label_smoothing

        df = pd.read_csv(csv_path)
        df["primary_label"] = df["primary_label"].astype(str)
        df = df[df["primary_label"].isin(label_map)].reset_index(drop=True)
        self.filenames = df["filename"].values
        self.labels = df["primary_label"].values

        # Parse secondary labels
        import ast
        self.secondary_labels = []
        for val in df["secondary_labels"].values:
            sec = []
            try:
                parsed = ast.literal_eval(str(val))
                for lbl in parsed:
                    lbl = str(lbl).strip()
                    if lbl in label_map:
                        sec.append(label_map[lbl])
            except (ValueError, SyntaxError):
                pass
            self.secondary_labels.append(sec)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filepath = os.path.join(self.audio_dir, self.filenames[index])
        waveform = self._load_audio(filepath)
        eps = self.label_smoothing
        target = np.full(self.num_classes, eps / self.num_classes, dtype=np.float32)
        target[self.label_map[self.labels[index]]] = 1.0 - eps
        for idx in self.secondary_labels[index]:
            target[idx] = 1.0 - eps
        return {"waveform": waveform, "target": target}

    def _load_audio(self, path):
        """Load, resample to mono, pad or randomly crop to clip_samples."""
        try:
            audio, sr = sf.read(path, dtype="float32")
        except Exception:
            audio, sr = librosa.load(path, sr=self.sample_rate, mono=True)
            return self._fit_length(audio)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        return self._fit_length(audio)

    def _fit_length(self, audio):
        """Pad with zeros or randomly crop to exactly clip_samples."""
        if len(audio) < self.clip_samples:
            pad = np.zeros(self.clip_samples - len(audio), dtype=np.float32)
            audio = np.concatenate([audio, pad])
        elif len(audio) > self.clip_samples:
            start = np.random.randint(0, len(audio) - self.clip_samples)
            audio = audio[start:start + self.clip_samples]
        return audio.astype(np.float32)


def _time_to_seconds(t):
    """Convert HH:MM:SS string to seconds."""
    parts = t.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


class SoundscapeDataset(Dataset):
    """
    Loads 5-second segments from train_soundscapes/.
    Labels are multi-hot (multiple species per segment).
    Audio is loaded on-the-fly; only the needed segment is read.
    """

    def __init__(self, labels_csv, soundscape_dir, label_map, sample_rate=32000,
                 clip_duration=10.0, label_smoothing=0.0):
        self.soundscape_dir = soundscape_dir
        self.label_map = label_map
        self.sample_rate = sample_rate
        self.clip_samples = int(sample_rate * clip_duration)
        self.num_classes = len(label_map)
        self.label_smoothing = label_smoothing

        df = pd.read_csv(labels_csv)
        self.entries = []
        for _, row in df.iterrows():
            filename = row["filename"]
            start_sec = _time_to_seconds(row["start"])
            end_sec = _time_to_seconds(row["end"])
            label_strs = str(row["primary_label"]).split(";")
            label_indices = []
            for lbl in label_strs:
                lbl = lbl.strip()
                if lbl in label_map:
                    label_indices.append(label_map[lbl])
            if len(label_indices) == 0:
                continue
            self.entries.append({
                "filename": filename,
                "start_sample": start_sec * sample_rate,
                "end_sample": end_sec * sample_rate,
                "label_indices": label_indices,
                "site_id": _parse_site_id(filename),
            })

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        filepath = os.path.join(self.soundscape_dir, entry["filename"])

        start = entry["start_sample"]
        n_frames = entry["end_sample"] - start
        try:
            audio, sr = sf.read(filepath, start=start, frames=n_frames, dtype="float32")
        except Exception:
            audio, sr = librosa.load(filepath, sr=self.sample_rate, mono=True)
            s = int(start * self.sample_rate / sr) if sr != self.sample_rate else start
            audio = audio[s:s + int(n_frames * self.sample_rate / sr)]

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        audio = audio.astype(np.float32)
        if len(audio) < self.clip_samples:
            repeats = (self.clip_samples // max(len(audio), 1)) + 1
            audio = np.tile(audio, repeats)[:self.clip_samples]

        eps = self.label_smoothing
        target = np.full(self.num_classes, eps / self.num_classes, dtype=np.float32)
        for idx in entry["label_indices"]:
            target[idx] = 1.0 - eps

        return {"waveform": audio, "target": target}


def get_datasets(data_dir, sample_rate=32000, clip_duration=10.0, val_frac=0.25,
                 seed=42, label_smoothing=0.0, n_folds=5, fold=0):
    """
    Build train/val datasets using k-fold cross-validation on soundscape segments.

    Train = all train_audio + (k-1)/k of soundscape segments.
    Val   = 1/k of soundscape segments (fold-based split).

    Each fold holds out a different 20% of soundscapes, so every segment is
    validated exactly once across all folds.

    Args:
        n_folds: number of CV folds (default 5)
        fold: which fold to hold out for validation (0 to n_folds-1)

    Returns:
        train_dataset, val_dataset, label_map, num_classes, n_train_audio, n_soundscape_train
    """
    taxonomy_path = os.path.join(data_dir, "taxonomy.csv")
    train_csv = os.path.join(data_dir, "train.csv")
    soundscape_csv = os.path.join(data_dir, "train_soundscapes_labels.csv")
    audio_dir = os.path.join(data_dir, "train_audio")
    soundscape_dir = os.path.join(data_dir, "train_soundscapes")

    label_map = build_label_map(taxonomy_path)
    num_classes = len(label_map)
    logger.info(f"Number of classes: {num_classes}")

    audio_ds = TrainAudioDataset(
        train_csv, audio_dir, label_map,
        sample_rate=sample_rate, clip_duration=clip_duration,
        label_smoothing=label_smoothing,
    )
    # Train soundscapes use label smoothing; val soundscapes use hard labels
    soundscape_ds_train = SoundscapeDataset(
        soundscape_csv, soundscape_dir, label_map,
        sample_rate=sample_rate, clip_duration=clip_duration,
        label_smoothing=label_smoothing,
    )
    soundscape_ds_val = SoundscapeDataset(
        soundscape_csv, soundscape_dir, label_map,
        sample_rate=sample_rate, clip_duration=clip_duration,
        label_smoothing=0.0,  # hard labels for accurate AUC
    )

    # K-fold split of soundscape segments
    n_sc = len(soundscape_ds_train)
    indices = np.arange(n_sc)
    # Deterministic shuffle so folds are consistent across runs
    rng = np.random.RandomState(42)  # always use 42 for fold assignment
    rng.shuffle(indices)

    fold_size = n_sc // n_folds
    val_start = fold * fold_size
    val_end = val_start + fold_size if fold < n_folds - 1 else n_sc
    val_indices = indices[val_start:val_end]
    train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

    sc_train = torch.utils.data.Subset(soundscape_ds_train, train_indices.tolist())
    sc_val = torch.utils.data.Subset(soundscape_ds_val, val_indices.tolist())

    # Train = all train_audio + train soundscape segments
    train_ds = ConcatDataset([audio_ds, sc_train])
    val_ds = sc_val

    logger.info(f"Fold {fold}/{n_folds}: Train = {len(audio_ds)} audio + {len(sc_train)} soundscape, "
                f"Val = {len(sc_val)} soundscape segments")

    return train_ds, val_ds, label_map, num_classes, len(audio_ds), len(sc_train)


def get_dataloaders(data_dir, batch_size=32, num_workers=4, sample_rate=32000,
                    clip_duration=10.0, val_frac=0.25, seed=42,
                    soundscape_weight=3.0, label_smoothing=0.0,
                    n_folds=5, fold=0):
    """
    Convenience function returning DataLoaders ready for training.

    soundscape_weight: how much to upweight soundscape samples relative to train_audio.
        E.g. 3.0 means each soundscape sample is ~3x more likely to be drawn per epoch.
    """
    train_ds, val_ds, label_map, num_classes, n_audio, n_soundscape = get_datasets(
        data_dir, sample_rate=sample_rate, clip_duration=clip_duration,
        val_frac=val_frac, seed=seed, label_smoothing=label_smoothing,
        n_folds=n_folds, fold=fold,
    )

    # Build WeightedRandomSampler to upweight soundscape data in training
    if soundscape_weight > 1.0 and isinstance(train_ds, ConcatDataset):
        # ConcatDataset: first n_audio are TrainAudioDataset, rest are soundscape
        n_total = len(train_ds)
        weights = np.ones(n_total, dtype=np.float64)
        weights[n_audio:] = soundscape_weight
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=n_total, replacement=True
        )
        logger.info(f"WeightedRandomSampler: {n_audio} audio (w=1.0) + "
                    f"{n_total - n_audio} soundscape (w={soundscape_weight})")
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return train_loader, val_loader, label_map, num_classes, n_audio
