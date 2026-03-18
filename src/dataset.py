"""
Memory-efficient dataloaders for BirdCLEF 2026.

Two dataset classes:
  - TrainAudioDataset:      individual species clips from train_audio/
  - SoundscapeDataset:      5-second segments from train_soundscapes/

Both return {"waveform": np.float32 (clip_samples,), "target": np.float32 (num_classes,)}
which matches the HTSAT SEDWrapper training interface.
"""

import os
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


class TrainAudioDataset(Dataset):
    """
    Loads individual species recordings from train_audio/.
    Each audio file maps to one primary_label (single-label, stored as multi-hot).
    Audio is loaded on-the-fly from .ogg for memory efficiency.
    """

    def __init__(self, csv_path, audio_dir, label_map, sample_rate=32000,
                 clip_duration=10.0):
        """
        Args:
            csv_path:      path to train.csv
            audio_dir:     path to data/train_audio/
            label_map:     dict mapping primary_label (str) -> class index
            sample_rate:   target sample rate (must match HTSAT config)
            clip_duration: clip length in seconds
        """
        self.audio_dir = audio_dir
        self.label_map = label_map
        self.sample_rate = sample_rate
        self.clip_samples = int(sample_rate * clip_duration)
        self.num_classes = len(label_map)

        df = pd.read_csv(csv_path)
        # Keep only rows whose label is in the taxonomy
        df["primary_label"] = df["primary_label"].astype(str)
        df = df[df["primary_label"].isin(label_map)].reset_index(drop=True)
        self.filenames = df["filename"].values
        self.labels = df["primary_label"].values

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filepath = os.path.join(self.audio_dir, self.filenames[index])
        waveform = self._load_audio(filepath)
        target = np.zeros(self.num_classes, dtype=np.float32)
        target[self.label_map[self.labels[index]]] = 1.0
        return {"waveform": waveform, "target": target}

    def _load_audio(self, path):
        """Load, resample to mono, pad or randomly crop to clip_samples."""
        try:
            audio, sr = sf.read(path, dtype="float32")
        except Exception:
            # Fallback to librosa for problematic files
            audio, sr = librosa.load(path, sr=self.sample_rate, mono=True)
            return self._fit_length(audio)

        # Convert to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
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
                 clip_duration=10.0):
        """
        Args:
            labels_csv:     path to train_soundscapes_labels.csv
            soundscape_dir: path to data/train_soundscapes/
            label_map:      dict mapping primary_label (str) -> class index
            sample_rate:    target sample rate
            clip_duration:  duration to pad the 5-sec segment to (HTSAT expects 10s)
        """
        self.soundscape_dir = soundscape_dir
        self.label_map = label_map
        self.sample_rate = sample_rate
        self.clip_samples = int(sample_rate * clip_duration)
        self.num_classes = len(label_map)

        df = pd.read_csv(labels_csv)
        self.entries = []
        for _, row in df.iterrows():
            filename = row["filename"]
            start_sec = _time_to_seconds(row["start"])
            end_sec = _time_to_seconds(row["end"])
            label_strs = str(row["primary_label"]).split(";")
            # Map labels, skip unknown ones
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
            })

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        filepath = os.path.join(self.soundscape_dir, entry["filename"])

        # Read only the needed segment for memory efficiency
        start = entry["start_sample"]
        n_frames = entry["end_sample"] - start
        try:
            audio, sr = sf.read(filepath, start=start, frames=n_frames, dtype="float32")
        except Exception:
            # Fallback: load full and slice
            audio, sr = librosa.load(filepath, sr=self.sample_rate, mono=True)
            s = int(start * self.sample_rate / sr) if sr != self.sample_rate else start
            audio = audio[s:s + int(n_frames * self.sample_rate / sr)]

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        # Pad 5s segment to clip_duration (10s) by repeating
        audio = audio.astype(np.float32)
        if len(audio) < self.clip_samples:
            repeats = (self.clip_samples // max(len(audio), 1)) + 1
            audio = np.tile(audio, repeats)[:self.clip_samples]

        target = np.zeros(self.num_classes, dtype=np.float32)
        for idx in entry["label_indices"]:
            target[idx] = 1.0

        return {"waveform": audio, "target": target}


def get_datasets(data_dir, sample_rate=32000, clip_duration=10.0, val_frac=0.1,
                 seed=42):
    """
    Build train/val datasets combining TrainAudioDataset + SoundscapeDataset.

    Returns:
        train_dataset, val_dataset, label_map, num_classes
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
        sample_rate=sample_rate, clip_duration=clip_duration
    )
    soundscape_ds = SoundscapeDataset(
        soundscape_csv, soundscape_dir, label_map,
        sample_rate=sample_rate, clip_duration=clip_duration
    )

    combined = ConcatDataset([audio_ds, soundscape_ds])
    total = len(combined)
    val_size = int(total * val_frac)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(
        combined, [train_size, val_size], generator=generator
    )

    logger.info(f"Train: {train_size}, Val: {val_size} "
                f"(audio: {len(audio_ds)}, soundscape: {len(soundscape_ds)})")

    return train_ds, val_ds, label_map, num_classes


def get_dataloaders(data_dir, batch_size=32, num_workers=4, sample_rate=32000,
                    clip_duration=10.0, val_frac=0.1, seed=42):
    """
    Convenience function returning DataLoaders ready for training.
    """
    train_ds, val_ds, label_map, num_classes = get_datasets(
        data_dir, sample_rate=sample_rate, clip_duration=clip_duration,
        val_frac=val_frac, seed=seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, label_map, num_classes
