"""
Memory-efficient dataloaders for BirdCLEF 2026.

Dataset classes:
  - TrainAudioDataset:       individual species clips from train_audio/
  - SoundscapeDataset:       5-second segments from train_soundscapes/
  - MultiSpeciesMixDataset:  wraps TrainAudioDataset, overlays multiple clips
                             to simulate polyphonic soundscapes

Both return {"waveform": np.float32 (clip_samples,), "target": np.float32 (num_classes,)}
"""

import os
import re
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset, WeightedRandomSampler
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

    When preload=True, all waveforms are loaded into RAM at init time
    (random-cropped to clip_samples). This avoids repeated disk I/O and
    is essential for MultiSpeciesMixDataset which loads 2-5 clips per sample.
    Memory cost: ~N * clip_samples * 4 bytes (e.g. 28k * 320k * 4 ≈ 34 GB).
    """

    def __init__(self, csv_path, audio_dir, label_map, sample_rate=32000,
                 clip_duration=10.0, label_smoothing=0.0, preload=False):
        self.audio_dir = audio_dir
        self.label_map = label_map
        self.sample_rate = sample_rate
        self.clip_samples = int(sample_rate * clip_duration)
        self.num_classes = len(label_map)
        self.label_smoothing = label_smoothing
        self.preload = preload

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

        # Pre-compute all target vectors (tiny: N * num_classes * 4 bytes)
        self._targets = np.full((len(self.filenames), self.num_classes),
                                label_smoothing / self.num_classes, dtype=np.float32)
        for i in range(len(self.filenames)):
            self._targets[i, self.label_map[self.labels[i]]] = 1.0 - label_smoothing
            for idx in self.secondary_labels[i]:
                self._targets[i, idx] = 1.0 - label_smoothing

        # Optionally preload all waveforms into RAM
        self._waveforms = None
        if preload:
            self._preload_all()

    def _preload_all(self):
        """Load all audio files into RAM, cropped to clip_samples.

        Clips longer than clip_samples get a random crop (fixed per epoch,
        but multi-species mixing + SuMix provide ample diversity).
        This keeps RAM predictable: N * clip_samples * 4 bytes.
        """
        logger.info(f"Preloading {len(self.filenames)} audio files into RAM "
                     f"(cropped to {self.clip_samples} samples = "
                     f"{self.clip_samples / self.sample_rate:.1f}s)...")
        waveforms = []
        for i, fn in enumerate(self.filenames):
            path = os.path.join(self.audio_dir, fn)
            audio = self._load_audio_raw(path)
            audio = self._fit_length(audio)  # crop/pad to clip_samples
            waveforms.append(audio)
            if (i + 1) % 5000 == 0:
                logger.info(f"  Preloaded {i + 1}/{len(self.filenames)}")
        self._waveforms = waveforms
        total_gb = sum(w.nbytes for w in waveforms) / 1e9
        logger.info(f"Preload complete: {len(waveforms)} clips, {total_gb:.1f} GB")

    def _load_audio_raw(self, path):
        """Load and resample audio to mono, but do NOT crop to clip_samples."""
        try:
            audio, sr = sf.read(path, dtype="float32")
        except Exception:
            audio, sr = librosa.load(path, sr=self.sample_rate, mono=True)
            return audio.astype(np.float32)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        return audio.astype(np.float32)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        if self._waveforms is not None:
            waveform = self._waveforms[index].copy()
        else:
            filepath = os.path.join(self.audio_dir, self.filenames[index])
            waveform = self._fit_length(self._load_audio_raw(filepath))
        return {
            "waveform": waveform,
            "target": self._targets[index].copy(),
        }

    def _fit_length(self, audio):
        """Pad with zeros or randomly crop to exactly clip_samples."""
        if len(audio) < self.clip_samples:
            pad = np.zeros(self.clip_samples - len(audio), dtype=np.float32)
            audio = np.concatenate([audio, pad])
        elif len(audio) > self.clip_samples:
            start = np.random.randint(0, len(audio) - self.clip_samples)
            audio = audio[start:start + self.clip_samples]
        else:
            audio = audio.copy()
        return audio.astype(np.float32)


class MultiSpeciesMixDataset(Dataset):
    """
    Wraps a TrainAudioDataset to create synthetic polyphonic mixtures.

    Each sample overlays the anchor clip with 1-4 additional randomly chosen
    clips at random gains (0.1-0.7), simulating the polyphonic conditions
    found in real soundscapes. The target is the union of all constituent
    labels (multi-hot OR).

    This is the competition-specific multi-species mixing strategy used by
    past BirdCLEF winners to bridge the domain gap between clean focal
    recordings and noisy multi-species test soundscapes.
    """

    def __init__(self, base_dataset, min_mix=1, max_mix=4, mix_prob=0.7,
                 gain_range=(0.1, 0.7)):
        self.base = base_dataset
        self.min_mix = min_mix
        self.max_mix = max_mix
        self.mix_prob = mix_prob
        self.gain_range = gain_range

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        waveform = sample["waveform"].copy()
        target = sample["target"].copy()

        # With mix_prob, overlay additional clips
        if np.random.rand() < self.mix_prob:
            n_extra = np.random.randint(self.min_mix, self.max_mix + 1)
            extra_indices = np.random.randint(0, len(self.base), size=n_extra)
            for extra_idx in extra_indices:
                extra = self.base[extra_idx]
                gain = np.random.uniform(*self.gain_range)
                waveform = waveform + gain * extra["waveform"]
                # Scale extra labels by gain so target reflects audibility:
                # a species mixed at 0.2 gain gets a 0.2 target, not 1.0
                target = np.maximum(target, gain * extra["target"])

        return {
            "waveform": waveform,
            "target": target,
        }


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

        return {
            "waveform": audio,
            "target": target,
        }


def get_datasets(data_dir, sample_rate=32000, clip_duration=10.0, val_frac=0.25,
                 seed=42, label_smoothing=0.0, n_folds=5, fold=0,
                 multi_mix=True, mix_prob=0.7, preload=False):
    """
    Build train/val datasets with k-fold CV.

    Validation includes BOTH a holdout of train_audio AND a fold of soundscape
    segments, so val AUC reflects performance across all species (not just the
    few present in soundscapes).

    Train audio clips are wrapped in MultiSpeciesMixDataset for polyphonic
    augmentation when multi_mix=True.

    Args:
        n_folds: number of CV folds (default 5)
        fold: which fold to hold out for validation (0 to n_folds-1)
        multi_mix: enable multi-species mixing augmentation on train_audio
        mix_prob: probability of mixing per sample (0-1)

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

    # --- Train audio: split into train/val by primary_label (stratified) ---
    # Preload waveforms into RAM to avoid repeated disk I/O during mixing
    audio_ds_full = TrainAudioDataset(
        train_csv, audio_dir, label_map,
        sample_rate=sample_rate, clip_duration=clip_duration,
        label_smoothing=label_smoothing, preload=preload,
    )
    # Val dataset: hard labels, shares preloaded waveforms to avoid double RAM
    audio_ds_val_hard = TrainAudioDataset(
        train_csv, audio_dir, label_map,
        sample_rate=sample_rate, clip_duration=clip_duration,
        label_smoothing=0.0, preload=False,
    )
    if preload and audio_ds_full._waveforms is not None:
        audio_ds_val_hard._waveforms = audio_ds_full._waveforms

    # Stratified split: hold out 1/n_folds of each label's clips for val
    n_audio = len(audio_ds_full)
    rng_audio = np.random.RandomState(42)
    label_to_indices = {}
    for i, lbl in enumerate(audio_ds_full.labels):
        label_to_indices.setdefault(lbl, []).append(i)

    audio_train_idx = []
    audio_val_idx = []
    for lbl, idxs in label_to_indices.items():
        idxs = np.array(idxs)
        rng_audio.shuffle(idxs)
        fold_sz = max(len(idxs) // n_folds, 1)
        v_start = fold * fold_sz
        v_end = v_start + fold_sz if fold < n_folds - 1 else len(idxs)
        audio_val_idx.extend(idxs[v_start:v_end].tolist())
        audio_train_idx.extend(np.concatenate([idxs[:v_start], idxs[v_end:]]).tolist())

    audio_train_sub = Subset(audio_ds_full, audio_train_idx)
    audio_val_sub = Subset(audio_ds_val_hard, audio_val_idx)

    # Wrap train audio in multi-species mixing
    if multi_mix:
        audio_train_mixed = MultiSpeciesMixDataset(
            audio_train_sub, min_mix=1, max_mix=4, mix_prob=mix_prob,
        )
        logger.info(f"Multi-species mixing enabled: prob={mix_prob}, 1-4 extra clips per sample")
    else:
        audio_train_mixed = audio_train_sub

    # --- Soundscape segments: k-fold split ---
    soundscape_ds_train = SoundscapeDataset(
        soundscape_csv, soundscape_dir, label_map,
        sample_rate=sample_rate, clip_duration=clip_duration,
        label_smoothing=label_smoothing,
    )
    soundscape_ds_val = SoundscapeDataset(
        soundscape_csv, soundscape_dir, label_map,
        sample_rate=sample_rate, clip_duration=clip_duration,
        label_smoothing=0.0,
    )

    n_sc = len(soundscape_ds_train)
    sc_indices = np.arange(n_sc)
    rng_sc = np.random.RandomState(42)
    rng_sc.shuffle(sc_indices)

    fold_size = n_sc // n_folds
    val_start = fold * fold_size
    val_end = val_start + fold_size if fold < n_folds - 1 else n_sc
    sc_val_indices = sc_indices[val_start:val_end]
    sc_train_indices = np.concatenate([sc_indices[:val_start], sc_indices[val_end:]])

    sc_train = Subset(soundscape_ds_train, sc_train_indices.tolist())
    sc_val = Subset(soundscape_ds_val, sc_val_indices.tolist())

    # --- Combine ---
    # Train = mixed train_audio (train fold) + soundscape (train fold)
    n_train_audio = len(audio_train_mixed)
    train_ds = ConcatDataset([audio_train_mixed, sc_train])

    # Val = train_audio (val fold, hard labels) + soundscape (val fold, hard labels)
    val_ds = ConcatDataset([audio_val_sub, sc_val])

    logger.info(f"Fold {fold}/{n_folds}: "
                f"Train = {n_train_audio} audio (mixed) + {len(sc_train)} soundscape, "
                f"Val = {len(audio_val_sub)} audio + {len(sc_val)} soundscape")

    # Return extra info needed for class-balanced sampling
    split_info = {
        "audio_ds": audio_ds_full,
        "audio_train_idx": audio_train_idx,
        "soundscape_ds": soundscape_ds_train,
        "sc_train_indices": sc_train_indices.tolist(),
    }

    return train_ds, val_ds, label_map, num_classes, n_train_audio, len(sc_train), split_info


def _build_class_balanced_sampler(audio_ds, audio_train_idx,
                                  soundscape_ds, sc_train_indices,
                                  num_classes, n_train_audio):
    """Build a WeightedRandomSampler that upweights rare classes.

    Each sample's weight = 1 / freq(rarest positive label in that sample).
    This ensures every class gets roughly equal representation per epoch.
    """
    # Count class frequencies across all training samples
    class_counts = np.zeros(num_classes, dtype=np.float64)

    # Train audio: use primary label (+ secondary labels)
    for idx in audio_train_idx:
        label_idx = audio_ds.label_map[audio_ds.labels[idx]]
        class_counts[label_idx] += 1
        for sec_idx in audio_ds.secondary_labels[idx]:
            class_counts[sec_idx] += 1

    # Soundscape segments: use label_indices
    for sc_idx in sc_train_indices:
        entry = soundscape_ds.entries[sc_idx]
        for label_idx in entry["label_indices"]:
            class_counts[label_idx] += 1

    # Avoid division by zero for classes with no training samples
    class_counts = np.maximum(class_counts, 1.0)
    class_weights = 1.0 / class_counts  # inverse frequency

    # Per-sample weight = weight of rarest positive label
    n_total = n_train_audio + len(sc_train_indices)
    sample_weights = np.ones(n_total, dtype=np.float64)

    # Train audio samples (first n_train_audio in ConcatDataset)
    for i, idx in enumerate(audio_train_idx):
        label_idx = audio_ds.label_map[audio_ds.labels[idx]]
        w = class_weights[label_idx]
        for sec_idx in audio_ds.secondary_labels[idx]:
            w = max(w, class_weights[sec_idx])
        sample_weights[i] = w

    # Soundscape samples (after train audio in ConcatDataset)
    for i, sc_idx in enumerate(sc_train_indices):
        entry = soundscape_ds.entries[sc_idx]
        w = max(class_weights[li] for li in entry["label_indices"])
        sample_weights[n_train_audio + i] = w

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=n_total, replacement=True
    )

    # Log class balance stats
    min_c, max_c = class_counts.min(), class_counts.max()
    logger.info(f"Class-balanced sampler: class counts range [{min_c:.0f}, {max_c:.0f}], "
                f"weight range [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")

    return sampler


def get_dataloaders(data_dir, batch_size=32, num_workers=4, sample_rate=32000,
                    clip_duration=10.0, val_frac=0.25, seed=42,
                    label_smoothing=0.0,
                    n_folds=5, fold=0, multi_mix=True, mix_prob=0.7,
                    preload=False):
    """
    Convenience function returning DataLoaders ready for training.

    Uses class-balanced sampling so every species is well represented per epoch.
    """
    train_ds, val_ds, label_map, num_classes, n_audio, n_soundscape, split_info = get_datasets(
        data_dir, sample_rate=sample_rate, clip_duration=clip_duration,
        val_frac=val_frac, seed=seed, label_smoothing=label_smoothing,
        n_folds=n_folds, fold=fold, multi_mix=multi_mix, mix_prob=mix_prob,
        preload=preload,
    )

    sampler = _build_class_balanced_sampler(
        split_info["audio_ds"], split_info["audio_train_idx"],
        split_info["soundscape_ds"], split_info["sc_train_indices"],
        num_classes, n_audio,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=8 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=8 if num_workers > 0 else None,
    )

    return train_loader, val_loader, label_map, num_classes, n_audio
