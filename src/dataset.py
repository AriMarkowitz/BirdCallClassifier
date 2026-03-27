"""
Memory-efficient dataloaders for BirdCLEF 2026.

Dataset classes:
  - TrainAudioDataset:       individual species clips from train_audio/
  - SoundscapeDataset:       5-second segments from train_soundscapes/
  - MultiSpeciesMixDataset:  wraps TrainAudioDataset, overlays multiple clips
                             to simulate polyphonic soundscapes

Variable-length support:
  - Datasets return waveforms of varying lengths (capped at max_duration).
  - A custom collate function pads each batch to the longest waveform.
  - The EfficientNet backbone handles arbitrary time dimensions natively
    via AdaptiveAvgPool2d before the classifier head.

Both return {"waveform": np.float32 (T,), "target": np.float32 (num_classes,)}
where T varies per sample.
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

# Default duration bounds (seconds)
DEFAULT_MIN_DURATION = 3.0
DEFAULT_MAX_DURATION = 30.0


def build_label_map(taxonomy_path):
    """Build mapping from primary_label (str) -> class index using taxonomy.csv."""
    df = pd.read_csv(taxonomy_path)
    labels = sorted(df["primary_label"].astype(str).unique())
    return {lbl: i for i, lbl in enumerate(labels)}


def _parse_site_id(filename):
    """Extract site ID (e.g. 'S08') from soundscape filename like BC2026_Train_0001_S08_..."""
    m = re.search(r'_S(\d+)_', filename)
    return f"S{m.group(1)}" if m else None


def variable_length_collate_fn(batch):
    """Collate variable-length waveforms by padding to max length in the batch.

    Returns:
        dict with:
          "waveform": (B, T_max) float32 tensor, zero-padded
          "target":   (B, num_classes) float32 tensor
          "lengths":  (B,) int64 tensor of original waveform lengths
    """
    waveforms = [sample["waveform"] for sample in batch]
    targets = [sample["target"] for sample in batch]

    lengths = [len(w) for w in waveforms]
    max_len = max(lengths)

    padded = np.zeros((len(batch), max_len), dtype=np.float32)
    for i, w in enumerate(waveforms):
        padded[i, :len(w)] = w

    return {
        "waveform": torch.from_numpy(padded),
        "target": torch.from_numpy(np.stack(targets)),
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }


class TrainAudioDataset(Dataset):
    """
    Loads individual species recordings from train_audio/.
    Each audio file maps to one primary_label (single-label, stored as multi-hot).

    Returns variable-length waveforms between min_duration and max_duration.
    When preload=True, all waveforms are loaded into RAM (cropped to max_duration).
    """

    def __init__(self, csv_path, audio_dir, label_map, sample_rate=32000,
                 min_duration=DEFAULT_MIN_DURATION,
                 max_duration=DEFAULT_MAX_DURATION,
                 label_smoothing=0.0, preload=False,
                 valid_regions=None):
        self.audio_dir = audio_dir
        self.label_map = label_map
        self.sample_rate = sample_rate
        self.min_samples = int(sample_rate * min_duration)
        self.max_samples = int(sample_rate * max_duration)
        self.num_classes = len(label_map)
        self.label_smoothing = label_smoothing
        self.preload = preload

        df = pd.read_csv(csv_path)
        df["primary_label"] = df["primary_label"].astype(str)
        df = df[df["primary_label"].isin(label_map)].reset_index(drop=True)
        self.filenames = df["filename"].values
        self.labels = df["primary_label"].values

        # Valid vocal regions per file (from preprocess_activity.py)
        # Maps filename -> list of (start_sample, end_sample) tuples
        self._valid_regions = {}
        if valid_regions is not None:
            n_with_regions = 0
            for fn in self.filenames:
                if fn in valid_regions:
                    # Convert seconds to samples
                    regions = []
                    for start_sec, end_sec in valid_regions[fn]:
                        regions.append((
                            int(start_sec * sample_rate),
                            int(end_sec * sample_rate),
                        ))
                    self._valid_regions[fn] = regions
                    n_with_regions += 1
            logger.info(f"Valid regions loaded for {n_with_regions}/{len(self.filenames)} files")

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
        """Load all audio files into RAM, cropped to max_samples.

        Uses valid_regions if available to crop from active vocal regions.
        """
        logger.info(f"Preloading {len(self.filenames)} audio files into RAM "
                     f"(capped at {self.max_samples} samples = "
                     f"{self.max_samples / self.sample_rate:.1f}s)...")
        waveforms = []
        for i, fn in enumerate(self.filenames):
            path = os.path.join(self.audio_dir, fn)
            audio = self._load_audio_raw(path)
            regions = self._valid_regions.get(fn)
            audio = self._crop_from_regions(audio, regions)
            waveforms.append(audio)
            if (i + 1) % 5000 == 0:
                logger.info(f"  Preloaded {i + 1}/{len(self.filenames)}")
        self._waveforms = waveforms
        total_gb = sum(w.nbytes for w in waveforms) / 1e9
        logger.info(f"Preload complete: {len(waveforms)} clips, {total_gb:.1f} GB")

    def _load_audio_raw(self, path):
        """Load and resample audio to mono, but do NOT crop."""
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
            audio = self._load_audio_raw(filepath)
            regions = self._valid_regions.get(self.filenames[index])
            waveform = self._crop_from_regions(audio, regions)
        return {
            "waveform": waveform,
            "target": self._targets[index].copy(),
        }

    def _crop_from_regions(self, audio, regions):
        """Crop a variable-length segment from a valid vocal region.

        Returns a waveform between min_samples and max_samples in length.
        If valid_regions are available, picks a random region weighted by
        duration. Falls back to random crop if no regions are available.
        """
        if not regions:
            return self._fit_length(audio)

        # Weight regions by duration so longer active stretches are preferred
        durations = np.array([max(end - start, 1) for start, end in regions],
                             dtype=np.float64)
        probs = durations / durations.sum()
        region_idx = np.random.choice(len(regions), p=probs)
        reg_start, reg_end = regions[region_idx]

        # Clamp to actual audio length
        reg_start = max(0, min(reg_start, len(audio)))
        reg_end = max(reg_start, min(reg_end, len(audio)))
        reg_len = reg_end - reg_start

        if reg_len >= self.max_samples:
            # Random crop to max_samples within the region
            start = reg_start + np.random.randint(0, reg_len - self.max_samples + 1)
            return audio[start:start + self.max_samples].astype(np.float32)
        elif reg_len >= self.min_samples:
            # Region is within [min, max] — use as-is (variable length)
            return audio[reg_start:reg_end].astype(np.float32)
        else:
            # Region shorter than min_samples — pad with silence to min_samples
            clip = audio[reg_start:reg_end].astype(np.float32)
            if len(clip) == 0:
                return np.zeros(self.min_samples, dtype=np.float32)
            pad = np.zeros(self.min_samples - len(clip), dtype=np.float32)
            return np.concatenate([clip, pad])

    def _fit_length(self, audio):
        """Return variable-length waveform capped at max_samples, padded to min_samples."""
        if len(audio) > self.max_samples:
            start = np.random.randint(0, len(audio) - self.max_samples)
            return audio[start:start + self.max_samples].astype(np.float32)
        elif len(audio) < self.min_samples:
            pad = np.zeros(self.min_samples - len(audio), dtype=np.float32)
            return np.concatenate([audio, pad]).astype(np.float32)
        else:
            # Already in [min_samples, max_samples] — return as-is
            return audio.copy().astype(np.float32)


class MultiSpeciesMixDataset(Dataset):
    """
    Wraps a TrainAudioDataset to create synthetic polyphonic mixtures.

    Each sample overlays the anchor clip with 1-4 additional randomly chosen
    clips at random gains (0.1-0.7), simulating the polyphonic conditions
    found in real soundscapes. The target is the union of all constituent
    labels (multi-hot OR).

    When mixing variable-length clips, extras are truncated or zero-padded
    to match the anchor's length.
    """

    def __init__(self, base_dataset, min_mix=1, max_mix=2, mix_prob=0.3,
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
            anchor_len = len(waveform)
            for extra_idx in extra_indices:
                extra = self.base[extra_idx]
                extra_wav = extra["waveform"]
                # Align lengths: truncate or pad extra to anchor length
                if len(extra_wav) > anchor_len:
                    extra_wav = extra_wav[:anchor_len]
                elif len(extra_wav) < anchor_len:
                    pad = np.zeros(anchor_len - len(extra_wav), dtype=np.float32)
                    extra_wav = np.concatenate([extra_wav, pad])
                gain = np.random.uniform(*self.gain_range)
                waveform = waveform + gain * extra_wav
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

    Returns natural segment length (no forced padding/tiling).
    Short segments are padded to min_samples.
    """

    def __init__(self, labels_csv, soundscape_dir, label_map, sample_rate=32000,
                 min_duration=DEFAULT_MIN_DURATION, label_smoothing=0.0):
        self.soundscape_dir = soundscape_dir
        self.label_map = label_map
        self.sample_rate = sample_rate
        self.min_samples = int(sample_rate * min_duration)
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

        # Pad very short segments to minimum length
        if len(audio) < self.min_samples:
            if len(audio) == 0:
                audio = np.zeros(self.min_samples, dtype=np.float32)
            else:
                repeats = (self.min_samples // len(audio)) + 1
                audio = np.tile(audio, repeats)[:self.min_samples]

        eps = self.label_smoothing
        target = np.full(self.num_classes, eps / self.num_classes, dtype=np.float32)
        for idx in entry["label_indices"]:
            target[idx] = 1.0 - eps

        return {
            "waveform": audio,
            "target": target,
        }


def get_datasets(data_dir, sample_rate=32000,
                 min_duration=DEFAULT_MIN_DURATION,
                 max_duration=DEFAULT_MAX_DURATION,
                 val_frac=0.25,
                 seed=42, label_smoothing=0.0, n_folds=5, fold=0,
                 multi_mix=True, mix_prob=0.7, preload=False,
                 valid_regions_path=None, pseudo_labels_csv=None):
    """
    Build train/val datasets with k-fold CV.

    Validation includes BOTH a holdout of train_audio AND a fold of soundscape
    segments, so val AUC reflects performance across all species (not just the
    few present in soundscapes).

    Train audio clips are wrapped in MultiSpeciesMixDataset for polyphonic
    augmentation when multi_mix=True.

    Args:
        min_duration: minimum clip duration in seconds (shorter clips padded)
        max_duration: maximum clip duration in seconds (longer clips cropped)
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

    # Load valid vocal regions if available
    import json
    valid_regions = None
    if valid_regions_path and os.path.isfile(valid_regions_path):
        with open(valid_regions_path) as f:
            valid_regions = json.load(f)
        logger.info(f"Loaded valid regions from {valid_regions_path} "
                     f"({len(valid_regions)} files)")
    elif valid_regions_path:
        logger.warning(f"Valid regions file not found: {valid_regions_path}, "
                        f"falling back to random cropping")

    # --- Train audio: split into train/val by primary_label (stratified) ---
    audio_ds_full = TrainAudioDataset(
        train_csv, audio_dir, label_map,
        sample_rate=sample_rate,
        min_duration=min_duration,
        max_duration=max_duration,
        label_smoothing=label_smoothing, preload=preload,
        valid_regions=valid_regions,
    )
    # Val dataset: hard labels, shares preloaded waveforms to avoid double RAM
    audio_ds_val_hard = TrainAudioDataset(
        train_csv, audio_dir, label_map,
        sample_rate=sample_rate,
        min_duration=min_duration,
        max_duration=max_duration,
        label_smoothing=0.0, preload=False,
        valid_regions=valid_regions,
    )
    if preload and audio_ds_full._waveforms is not None:
        audio_ds_val_hard._waveforms = audio_ds_full._waveforms

    # Stratified split: hold out 1/n_folds of each label's clips for val.
    # Guarantee every species has at least 1 sample in val so all classes
    # are evaluable for per-class AUC.
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
        n = len(idxs)

        if n == 1:
            audio_val_idx.extend(idxs.tolist())
            audio_train_idx.extend(idxs.tolist())
            continue

        fold_sz = max(n // n_folds, 1)
        v_start = fold * fold_sz
        v_end = v_start + fold_sz if fold < n_folds - 1 else n

        if v_start >= n:
            audio_val_idx.append(idxs[-1])
            audio_train_idx.extend(idxs.tolist())
            continue

        v_end = min(v_end, n)
        audio_val_idx.extend(idxs[v_start:v_end].tolist())
        audio_train_idx.extend(np.concatenate([idxs[:v_start], idxs[v_end:]]).tolist())

    audio_train_sub = Subset(audio_ds_full, audio_train_idx)
    audio_val_sub = Subset(audio_ds_val_hard, audio_val_idx)

    # Wrap train audio in multi-species mixing
    if multi_mix:
        audio_train_mixed = MultiSpeciesMixDataset(
            audio_train_sub, min_mix=1, max_mix=2, mix_prob=mix_prob,
        )
        logger.info(f"Multi-species mixing enabled: prob={mix_prob}, 1-2 extra clips per sample")
    else:
        audio_train_mixed = audio_train_sub

    # --- Soundscape segments: stratified k-fold split ---
    soundscape_ds_gt = SoundscapeDataset(
        soundscape_csv, soundscape_dir, label_map,
        sample_rate=sample_rate, min_duration=min_duration,
        label_smoothing=label_smoothing,
    )
    soundscape_ds_val = SoundscapeDataset(
        soundscape_csv, soundscape_dir, label_map,
        sample_rate=sample_rate, min_duration=min_duration,
        label_smoothing=0.0,
    )

    # Stratified split by species so val contains all soundscape species
    n_sc_gt = len(soundscape_ds_gt)
    rng_sc = np.random.RandomState(42)

    sc_label_to_indices = {}
    for i, entry in enumerate(soundscape_ds_gt.entries):
        key = entry["label_indices"][0]
        sc_label_to_indices.setdefault(key, []).append(i)

    sc_train_indices = []
    sc_val_indices = []
    for key, idxs in sc_label_to_indices.items():
        idxs = np.array(idxs)
        rng_sc.shuffle(idxs)
        n = len(idxs)

        if n == 1:
            sc_val_indices.extend(idxs.tolist())
            sc_train_indices.extend(idxs.tolist())
            continue

        fold_sz = max(n // n_folds, 1)
        v_start = fold * fold_sz
        v_end = v_start + fold_sz if fold < n_folds - 1 else n

        if v_start >= n:
            sc_val_indices.append(idxs[-1])
            sc_train_indices.extend(idxs.tolist())
            continue

        v_end = min(v_end, n)
        sc_val_indices.extend(idxs[v_start:v_end].tolist())
        sc_train_indices.extend(np.concatenate([idxs[:v_start], idxs[v_end:]]).tolist())

    sc_train_gt = Subset(soundscape_ds_gt, sc_train_indices)
    sc_val = Subset(soundscape_ds_val, sc_val_indices)

    # Add pseudo-labeled segments to training only.
    pseudo_ds = None
    if pseudo_labels_csv and os.path.isfile(pseudo_labels_csv):
        pseudo_df = pd.read_csv(pseudo_labels_csv)
        sc_mask = ~pseudo_df["filename"].str.contains("/", na=False)
        audio_mask = pseudo_df["filename"].str.contains("/", na=False)

        pseudo_parts = []
        if sc_mask.any():
            sc_pseudo_csv = pseudo_labels_csv.replace(".csv", "_sc.csv") \
                if isinstance(pseudo_labels_csv, str) \
                else str(pseudo_labels_csv).replace(".csv", "_sc.csv")
            pseudo_df[sc_mask].to_csv(sc_pseudo_csv, index=False)
            sc_pseudo_ds = SoundscapeDataset(
                sc_pseudo_csv, soundscape_dir, label_map,
                sample_rate=sample_rate, min_duration=min_duration,
                label_smoothing=label_smoothing,
            )
            pseudo_parts.append(sc_pseudo_ds)
            logger.info(f"Pseudo-labels (soundscape): {len(sc_pseudo_ds)} segments")

        if audio_mask.any():
            audio_pseudo_csv = pseudo_labels_csv.replace(".csv", "_audio.csv") \
                if isinstance(pseudo_labels_csv, str) \
                else str(pseudo_labels_csv).replace(".csv", "_audio.csv")
            pseudo_df[audio_mask].to_csv(audio_pseudo_csv, index=False)
            audio_pseudo_ds = SoundscapeDataset(
                audio_pseudo_csv, audio_dir, label_map,
                sample_rate=sample_rate, min_duration=min_duration,
                label_smoothing=label_smoothing,
            )
            pseudo_parts.append(audio_pseudo_ds)
            logger.info(f"Pseudo-labels (train_audio secondary): {len(audio_pseudo_ds)} segments")

        if pseudo_parts:
            pseudo_ds = ConcatDataset(pseudo_parts) if len(pseudo_parts) > 1 else pseudo_parts[0]
            sc_train = ConcatDataset([sc_train_gt, pseudo_ds])
            logger.info(f"Pseudo-labels total: {len(pseudo_ds)} segments added to training")
        else:
            sc_train = sc_train_gt
    else:
        sc_train = sc_train_gt

    # --- Combine ---
    n_train_audio = len(audio_train_mixed)
    train_ds = ConcatDataset([audio_train_mixed, sc_train])
    val_ds = ConcatDataset([audio_val_sub, sc_val])

    logger.info(f"Fold {fold}/{n_folds}: "
                f"Train = {n_train_audio} audio (mixed) + {len(sc_train)} soundscape, "
                f"Val = {len(audio_val_sub)} audio + {len(sc_val)} soundscape")

    split_info = {
        "audio_ds": audio_ds_full,
        "audio_train_idx": audio_train_idx,
        "soundscape_ds_gt": soundscape_ds_gt,
        "sc_train_indices": sc_train_indices,
        "pseudo_ds": pseudo_ds,
    }

    return train_ds, val_ds, label_map, num_classes, n_train_audio, len(sc_train), split_info


def _get_all_entries(ds):
    """Get all entries from a SoundscapeDataset or ConcatDataset of SoundscapeDatasets."""
    if ds is None:
        return []
    if hasattr(ds, "entries"):
        return ds.entries
    if hasattr(ds, "datasets"):
        entries = []
        for sub in ds.datasets:
            entries.extend(_get_all_entries(sub))
        return entries
    return []


def _build_class_balanced_sampler(split_info, num_classes, n_train_audio, n_sc_train,
                                  balance_alpha=0.5):
    """Build a WeightedRandomSampler that upweights rare classes.

    balance_alpha controls how aggressively rare classes are upweighted:
      0.0 = uniform sampling (natural frequency, no rebalancing)
      0.5 = sqrt of inverse frequency (moderate boost for rare classes)
      1.0 = full inverse frequency (every class sampled equally)

    The ConcatDataset layout is:
      [0 .. n_train_audio-1]  = train audio (mixed)
      [n_train_audio .. n_train_audio + n_sc_gt_train - 1] = ground-truth soundscapes
      [n_train_audio + n_sc_gt_train .. end] = pseudo-labeled soundscapes (if any)
    """
    audio_ds = split_info["audio_ds"]
    audio_train_idx = split_info["audio_train_idx"]
    soundscape_ds_gt = split_info["soundscape_ds_gt"]
    sc_train_indices = split_info["sc_train_indices"]
    pseudo_ds = split_info.get("pseudo_ds")

    # Count class frequencies across all training samples
    class_counts = np.zeros(num_classes, dtype=np.float64)

    for idx in audio_train_idx:
        label_idx = audio_ds.label_map[audio_ds.labels[idx]]
        class_counts[label_idx] += 1
        for sec_idx in audio_ds.secondary_labels[idx]:
            class_counts[sec_idx] += 1

    for sc_idx in sc_train_indices:
        entry = soundscape_ds_gt.entries[sc_idx]
        for label_idx in entry["label_indices"]:
            class_counts[label_idx] += 1

    pseudo_entries = _get_all_entries(pseudo_ds)
    n_pseudo = len(pseudo_entries)
    for entry in pseudo_entries:
        for label_idx in entry["label_indices"]:
            class_counts[label_idx] += 1

    class_counts = np.maximum(class_counts, 1.0)
    class_weights = (1.0 / class_counts) ** balance_alpha

    n_total = n_train_audio + n_sc_train
    sample_weights = np.ones(n_total, dtype=np.float64)

    for i, idx in enumerate(audio_train_idx):
        label_idx = audio_ds.label_map[audio_ds.labels[idx]]
        w = class_weights[label_idx]
        for sec_idx in audio_ds.secondary_labels[idx]:
            w = max(w, class_weights[sec_idx])
        sample_weights[i] = w

    n_sc_gt = len(sc_train_indices)
    for i, sc_idx in enumerate(sc_train_indices):
        entry = soundscape_ds_gt.entries[sc_idx]
        w = max(class_weights[li] for li in entry["label_indices"])
        sample_weights[n_train_audio + i] = w

    for i, entry in enumerate(pseudo_entries):
        w = max(class_weights[li] for li in entry["label_indices"])
        sample_weights[n_train_audio + n_sc_gt + i] = w

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=n_total, replacement=True
    )

    min_c, max_c = class_counts.min(), class_counts.max()
    logger.info(f"Class-balanced sampler (alpha={balance_alpha}): "
                f"class counts range [{min_c:.0f}, {max_c:.0f}], "
                f"weight range [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
    if n_pseudo > 0:
        logger.info(f"  (includes {n_pseudo} pseudo-labeled segments)")

    return sampler


def get_dataloaders(data_dir, batch_size=32, num_workers=4, sample_rate=32000,
                    min_duration=DEFAULT_MIN_DURATION,
                    max_duration=DEFAULT_MAX_DURATION,
                    val_frac=0.25, seed=42,
                    label_smoothing=0.0,
                    n_folds=5, fold=0, multi_mix=True, mix_prob=0.7,
                    preload=False, valid_regions_path=None,
                    pseudo_labels_csv=None, balance_alpha=0.5):
    """
    Convenience function returning DataLoaders ready for training.

    Uses class-balanced sampling so every species is well represented per epoch.
    Uses variable_length_collate_fn to pad batches to the longest waveform.
    """
    train_ds, val_ds, label_map, num_classes, n_audio, n_soundscape, split_info = get_datasets(
        data_dir, sample_rate=sample_rate,
        min_duration=min_duration, max_duration=max_duration,
        val_frac=val_frac, seed=seed, label_smoothing=label_smoothing,
        n_folds=n_folds, fold=fold, multi_mix=multi_mix, mix_prob=mix_prob,
        preload=preload, valid_regions_path=valid_regions_path,
        pseudo_labels_csv=pseudo_labels_csv,
    )

    sampler = _build_class_balanced_sampler(
        split_info, num_classes, n_audio, n_soundscape,
        balance_alpha=balance_alpha,
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
        collate_fn=variable_length_collate_fn,
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
        collate_fn=variable_length_collate_fn,
    )

    return train_loader, val_loader, label_map, num_classes, n_audio
