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

    result = {
        "waveform": torch.from_numpy(padded),
        "target": torch.from_numpy(np.stack(targets)),
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }

    # Pass through teacher logits if any sample has them (pseudo-label distillation).
    # Samples without logits get zeros; a mask indicates which samples have real logits.
    if any("teacher_logits" in s for s in batch):
        # Find the actual teacher logits size (may differ from target size due to padding)
        tl_size = next(s["teacher_logits"].shape[0] for s in batch if "teacher_logits" in s)
        logits = np.zeros((len(batch), tl_size), dtype=np.float32)
        mask = np.zeros(len(batch), dtype=np.float32)
        for i, s in enumerate(batch):
            if "teacher_logits" in s:
                logits[i] = s["teacher_logits"]
                mask[i] = 1.0
        result["teacher_logits"] = torch.from_numpy(logits)
        result["teacher_logits_mask"] = torch.from_numpy(mask)

    return result


class TrainAudioDataset(Dataset):
    """
    Loads individual species recordings from train_audio/.
    Each audio file maps to one primary_label (single-label, stored as multi-hot).

    Returns variable-length waveforms between min_duration and max_duration,
    unless full_files=True (then full recordings are returned, padded only to
    min_duration for very short clips).
    When preload=True, all waveforms are loaded into RAM (cropped to max_duration).
    """

    def __init__(self, csv_path, audio_dir, label_map, sample_rate=32000,
                 min_duration=DEFAULT_MIN_DURATION,
                 max_duration=DEFAULT_MAX_DURATION,
                 label_smoothing=0.0, preload=False,
                 valid_regions=None, full_files=False):
        self.audio_dir = audio_dir
        self.label_map = label_map
        self.sample_rate = sample_rate
        self.min_samples = int(sample_rate * min_duration)
        self.max_samples = int(sample_rate * max_duration)
        self.full_files = full_files
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
        if valid_regions is not None and not self.full_files:
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
        if self.full_files:
            logger.info(f"Preloading {len(self.filenames)} full audio files into RAM...")
        else:
            logger.info(f"Preloading {len(self.filenames)} audio files into RAM "
                        f"(capped at {self.max_samples} samples = "
                        f"{self.max_samples / self.sample_rate:.1f}s)...")
        waveforms = []
        for i, fn in enumerate(self.filenames):
            path = os.path.join(self.audio_dir, fn)
            audio = self._load_audio_raw(path)
            if self.full_files:
                audio = self._fit_min_only(audio)
            else:
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
            if self.full_files:
                waveform = self._fit_min_only(audio)
            else:
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

    def _fit_min_only(self, audio):
        """Return full audio, capped at max_samples, padded if shorter than min_samples."""
        if len(audio) > self.max_samples:
            start = np.random.randint(0, len(audio) - self.max_samples)
            audio = audio[start:start + self.max_samples]
        if len(audio) < self.min_samples:
            pad = np.zeros(self.min_samples - len(audio), dtype=np.float32)
            return np.concatenate([audio, pad]).astype(np.float32)
        return audio.copy().astype(np.float32)


class SoundscapeBackgroundMix(Dataset):
    """Wraps any dataset to mix in random crops from unlabeled soundscape files.

    This bridges the domain gap between clean focal recordings (train_audio)
    and noisy test soundscapes by adding real background noise at random SNR.
    Labels are unchanged because the background is unlabeled.

    Args:
        base_dataset: any dataset returning {"waveform": ..., "target": ...}
        soundscape_dir: path to train_soundscapes/ directory
        sample_rate: target sample rate
        mix_prob: probability of mixing background per sample (0-1)
        snr_range: (min_snr_db, max_snr_db) — signal-to-noise ratio range.
            Lower SNR = more background noise. Typical range: (3, 15) dB.
    """

    def __init__(self, base_dataset, soundscape_dir, sample_rate=32000,
                 mix_prob=0.5, snr_range=(3.0, 15.0)):
        self.base = base_dataset
        self.sample_rate = sample_rate
        self.mix_prob = mix_prob
        self.snr_min, self.snr_max = snr_range

        # Collect all soundscape file paths
        self.bg_files = []
        if os.path.isdir(soundscape_dir):
            for f in os.listdir(soundscape_dir):
                if f.endswith(('.ogg', '.wav', '.flac', '.mp3')):
                    self.bg_files.append(os.path.join(soundscape_dir, f))
        logger.info(f"SoundscapeBackgroundMix: {len(self.bg_files)} background files, "
                     f"mix_prob={mix_prob}, SNR={snr_range}")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]

        if not self.bg_files or np.random.rand() >= self.mix_prob:
            return sample

        waveform = sample["waveform"].copy()
        target = sample["target"]
        clip_len = len(waveform)

        # Load a random crop from a random soundscape file
        bg_path = self.bg_files[np.random.randint(len(self.bg_files))]
        try:
            info = sf.info(bg_path)
            bg_total = int(info.frames)
            sr = info.samplerate

            # Pick a random start position
            need_frames = int(clip_len * sr / self.sample_rate)
            if bg_total > need_frames:
                start = np.random.randint(0, bg_total - need_frames)
            else:
                start = 0
            bg_audio, bg_sr = sf.read(bg_path, start=start,
                                       frames=min(need_frames, bg_total),
                                       dtype="float32")
            if bg_audio.ndim > 1:
                bg_audio = bg_audio.mean(axis=1)
            if bg_sr != self.sample_rate:
                bg_audio = librosa.resample(bg_audio, orig_sr=bg_sr,
                                            target_sr=self.sample_rate)
        except Exception:
            return sample

        # Pad or truncate background to match clip length
        if len(bg_audio) > clip_len:
            bg_audio = bg_audio[:clip_len]
        elif len(bg_audio) < clip_len:
            pad = np.zeros(clip_len - len(bg_audio), dtype=np.float32)
            bg_audio = np.concatenate([bg_audio, pad])

        # Mix at random SNR
        sig_power = max(np.mean(waveform ** 2), 1e-10)
        bg_power = max(np.mean(bg_audio ** 2), 1e-10)
        snr_db = np.random.uniform(self.snr_min, self.snr_max)
        # Scale background so that 10*log10(sig/bg_scaled) = snr_db
        target_bg_power = sig_power / (10.0 ** (snr_db / 10.0))
        gain = np.sqrt(target_bg_power / bg_power)
        waveform = waveform + gain * bg_audio

        return {
            "waveform": waveform.astype(np.float32),
            "target": target,
        }


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


class PseudoLabelDataset(Dataset):
    """Loads pseudo-labeled soundscape segments with teacher logits.

    Unlike SoundscapeDataset which uses hard multi-hot targets, this dataset
    loads the teacher model's raw logits from a companion .npz file. During
    training, the student is trained to match these logits via KL divergence,
    preserving the full distribution of teacher knowledge rather than just
    the thresholded binary decisions.

    The CSV and .npz must be aligned row-by-row (same order, same count).

    Returns:
        waveform: np.float32 (T,)
        target: np.float32 (num_classes,) — soft targets (sigmoid of teacher logits)
        teacher_logits: np.float32 (num_classes,) — raw pre-sigmoid logits
    """

    def __init__(self, labels_csv, logits_npz, soundscape_dir, audio_dir,
                 label_map, sample_rate=32000, min_duration=3.0):
        self.soundscape_dir = soundscape_dir
        self.audio_dir = audio_dir
        self.label_map = label_map
        self.sample_rate = sample_rate
        self.min_samples = int(sample_rate * min_duration)
        self.num_classes = len(label_map)

        df = pd.read_csv(labels_csv)

        # Load teacher logits — aligned row-by-row with the CSV
        npz = np.load(logits_npz)
        all_logits = npz["logits"].astype(np.float32)
        assert len(df) == len(all_logits), (
            f"CSV has {len(df)} rows but logits has {len(all_logits)} — "
            f"regenerate pseudo-labels")

        self.entries = []
        self.teacher_logits = []
        for row_idx, (_, row) in enumerate(df.iterrows()):
            filename = row["filename"]
            start_sec = _time_to_seconds(row["start"])
            end_sec = _time_to_seconds(row["end"])

            # Determine if this is a soundscape or train_audio file
            is_train_audio = "/" in str(filename)

            # Parse label_indices for class-balanced sampler compatibility
            label_strs = str(row["primary_label"]).split(";")
            label_indices = [label_map[l.strip()] for l in label_strs
                             if l.strip() in label_map]

            self.entries.append({
                "filename": filename,
                "start_sample": int(start_sec * sample_rate),
                "end_sample": int(end_sec * sample_rate),
                "is_train_audio": is_train_audio,
                "label_indices": label_indices,
            })
            self.teacher_logits.append(all_logits[row_idx])

        self.teacher_logits = np.stack(self.teacher_logits) if self.teacher_logits else np.zeros((0, self.num_classes), dtype=np.float32)
        logger.info(f"PseudoLabelDataset: {len(self.entries)} segments with teacher logits")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        if entry["is_train_audio"]:
            filepath = os.path.join(self.audio_dir, entry["filename"])
        else:
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

        if len(audio) < self.min_samples:
            if len(audio) == 0:
                audio = np.zeros(self.min_samples, dtype=np.float32)
            else:
                repeats = (self.min_samples // len(audio)) + 1
                audio = np.tile(audio, repeats)[:self.min_samples]

        # Soft target = sigmoid of teacher logits (for supervised loss compatibility)
        logits = self.teacher_logits[index]
        soft_target = 1.0 / (1.0 + np.exp(-logits))

        return {
            "waveform": audio,
            "target": soft_target,
            "teacher_logits": logits,
        }


class DistillAudioDataset(Dataset):
    """Loads supplemental audio from distill_audio/ using distill_manifest.csv.

    Maps species to class indices:
      - Species matching the BirdCLEF taxonomy get their standard index (0..num_target-1).
      - Extra species get indices num_target..num_target+N (hard negatives).
      - If hard_negatives=False, only matched species are loaded.

    Args:
        manifest_path: path to distill_manifest.csv
        distill_dir: root directory containing distill_audio/ files
        target_label_map: {primary_label: idx} for the 234 BirdCLEF classes
        taxonomy_path: path to taxonomy.csv (to map scientific_name -> primary_label)
        sample_rate: target sample rate
        min_duration: minimum clip duration in seconds
        max_duration: maximum clip duration in seconds
        label_smoothing: label smoothing epsilon
        hard_negatives: if True, include non-target species as extra classes
    """

    def __init__(self, manifest_path, distill_dir, target_label_map, taxonomy_path,
                 sample_rate=32000,
                 min_duration=DEFAULT_MIN_DURATION,
                 max_duration=DEFAULT_MAX_DURATION,
                 label_smoothing=0.0,
                 hard_negatives=True):
        self.distill_dir = distill_dir
        self.sample_rate = sample_rate
        self.min_samples = int(sample_rate * min_duration)
        self.max_samples = int(sample_rate * max_duration)
        self.label_smoothing = label_smoothing

        # Build scientific_name -> primary_label mapping from taxonomy
        tax_df = pd.read_csv(taxonomy_path)
        sci_to_primary = {}
        for _, row in tax_df.iterrows():
            sci_to_primary[str(row["scientific_name"]).strip().lower()] = str(row["primary_label"])

        # Read manifest and assign class indices
        manifest_df = pd.read_csv(manifest_path)

        # Map species_name (scientific name) to target label index or extra index
        extra_species = {}  # species_name -> extra_class_idx
        next_extra_idx = len(target_label_map)

        self.entries = []
        self.num_target_classes = len(target_label_map)

        for _, row in manifest_df.iterrows():
            species_name = str(row.get("species_name", "")).strip()
            rel_path = str(row.get("relative_path", "")).strip()
            if not species_name or not rel_path:
                continue

            # Try to match to BirdCLEF taxonomy
            primary_label = sci_to_primary.get(species_name.lower())
            if primary_label and primary_label in target_label_map:
                class_idx = target_label_map[primary_label]
            elif hard_negatives:
                # Assign a hard-negative class index
                if species_name not in extra_species:
                    extra_species[species_name] = next_extra_idx
                    next_extra_idx += 1
                class_idx = extra_species[species_name]
            else:
                continue  # skip non-target species

            audio_path = os.path.join(distill_dir, rel_path)
            if not os.path.isfile(audio_path):
                continue

            self.entries.append({
                "audio_path": audio_path,
                "class_idx": class_idx,
                "species_name": species_name,
            })

        self.num_extra_classes = len(extra_species)
        self.num_classes = self.num_target_classes + self.num_extra_classes
        self.extra_species_map = extra_species  # species_name -> idx

        logger.info(
            f"DistillAudioDataset: {len(self.entries)} files, "
            f"{self.num_target_classes} target classes matched, "
            f"{self.num_extra_classes} extra hard-negative classes"
        )

        # Pre-compute targets
        eps = label_smoothing
        self._targets = np.full(
            (len(self.entries), self.num_classes),
            eps / self.num_classes, dtype=np.float32
        )
        for i, entry in enumerate(self.entries):
            self._targets[i, entry["class_idx"]] = 1.0 - eps

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        audio = self._load_audio(entry["audio_path"])
        audio = self._fit_length(audio)
        return {
            "waveform": audio,
            "target": self._targets[index].copy(),
        }

    def _load_audio(self, path):
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

    def _fit_length(self, audio):
        if len(audio) > self.max_samples:
            start = np.random.randint(0, len(audio) - self.max_samples)
            return audio[start:start + self.max_samples].astype(np.float32)
        elif len(audio) < self.min_samples:
            pad = np.zeros(self.min_samples - len(audio), dtype=np.float32)
            return np.concatenate([audio, pad]).astype(np.float32)
        return audio.copy().astype(np.float32)


def get_datasets(data_dir, sample_rate=32000,
                 min_duration=DEFAULT_MIN_DURATION,
                 max_duration=DEFAULT_MAX_DURATION,
                 val_frac=0.25,
                 seed=42, label_smoothing=0.0, n_folds=5, fold=0,
                 multi_mix=True, mix_prob=0.7, preload=False,
                 valid_regions_path=None, pseudo_labels_csv=None,
                 full_files=True,
                 distill_manifest=None, hard_negatives=True,
                 bg_mix_prob=0.0, bg_snr_range=(3.0, 15.0)):
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
        full_files: if True, use full train_audio files (no max-duration crop,
                no valid-region sub-cropping)
        n_folds: number of CV folds (default 5)
        fold: which fold to hold out for validation (0 to n_folds-1)
        multi_mix: enable multi-species mixing augmentation on train_audio
        mix_prob: probability of mixing per sample (0-1)

    Returns:
        train_dataset, val_dataset, label_map, num_classes, n_train_audio, n_soundscape_train,
        split_info (dict with num_train_classes — may be > num_classes when hard negatives enabled)
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
    if full_files and valid_regions_path:
        logger.info("full_files=True: ignoring valid_regions cropping for train_audio")
    elif valid_regions_path and os.path.isfile(valid_regions_path):
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
        full_files=full_files,
    )
    # Val dataset: hard labels, shares preloaded waveforms to avoid double RAM
    audio_ds_val_hard = TrainAudioDataset(
        train_csv, audio_dir, label_map,
        sample_rate=sample_rate,
        min_duration=min_duration,
        max_duration=max_duration,
        label_smoothing=0.0, preload=False,
        valid_regions=valid_regions,
        full_files=full_files,
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

    # Wrap train audio with soundscape background noise injection
    if bg_mix_prob > 0:
        audio_train_mixed = SoundscapeBackgroundMix(
            audio_train_mixed, soundscape_dir,
            sample_rate=sample_rate, mix_prob=bg_mix_prob,
            snr_range=bg_snr_range,
        )

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
    # Prefer logit-based loading if .npz exists alongside the CSV.
    pseudo_ds = None
    if pseudo_labels_csv and os.path.isfile(pseudo_labels_csv):
        logits_npz = pseudo_labels_csv.replace(".csv", ".npz") \
            if isinstance(pseudo_labels_csv, str) \
            else str(pseudo_labels_csv).replace(".csv", ".npz")

        if os.path.isfile(logits_npz):
            # Logit-based pseudo-labels (preferred): preserves full teacher distribution
            pseudo_ds = PseudoLabelDataset(
                pseudo_labels_csv, logits_npz,
                soundscape_dir, audio_dir, label_map,
                sample_rate=sample_rate, min_duration=min_duration,
            )
            logger.info(f"Pseudo-labels (logit-based): {len(pseudo_ds)} segments with teacher logits")
        else:
            # Fallback: hard-label pseudo-labels (legacy .csv-only format)
            logger.warning("No .npz logits found — falling back to hard pseudo-labels")
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

            if pseudo_parts:
                pseudo_ds = ConcatDataset(pseudo_parts) if len(pseudo_parts) > 1 else pseudo_parts[0]
                logger.info(f"Pseudo-labels (hard): {len(pseudo_ds)} segments")

        if pseudo_ds is not None:
            sc_train = ConcatDataset([sc_train_gt, pseudo_ds])
        else:
            sc_train = sc_train_gt
    else:
        sc_train = sc_train_gt

    # --- Distill data (supplemental audio) ---
    distill_ds = None
    num_train_classes = num_classes  # may grow if hard negatives enabled
    if distill_manifest and os.path.isfile(distill_manifest):
        distill_dir = os.path.join(data_dir, "distill_audio")
        distill_ds = DistillAudioDataset(
            manifest_path=distill_manifest,
            distill_dir=distill_dir,
            target_label_map=label_map,
            taxonomy_path=taxonomy_path,
            sample_rate=sample_rate,
            min_duration=min_duration,
            max_duration=max_duration,
            label_smoothing=label_smoothing,
            hard_negatives=hard_negatives,
        )
        num_train_classes = distill_ds.num_classes
        logger.info(f"Distill data: {len(distill_ds)} files, "
                     f"num_train_classes={num_train_classes} "
                     f"(+{distill_ds.num_extra_classes} hard negatives)")
    elif distill_manifest:
        logger.warning(f"Distill manifest not found: {distill_manifest}")

    # --- Pad targets to num_train_classes if distill added extra classes ---
    # All existing datasets have targets of size num_classes (234).
    # If distill adds hard negatives, we need a wrapper to zero-pad targets.
    class _PadTargetDataset(Dataset):
        """Wraps a dataset to zero-pad targets from num_classes to num_train_classes."""
        def __init__(self, ds, target_size):
            self.ds = ds
            self.target_size = target_size
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            sample = self.ds[idx]
            t = sample["target"]
            if len(t) < self.target_size:
                padded = np.zeros(self.target_size, dtype=np.float32)
                padded[:len(t)] = t
                sample = {**sample, "target": padded}
            return sample

    # --- Combine ---
    n_train_audio = len(audio_train_mixed)
    train_parts = [audio_train_mixed, sc_train]

    if distill_ds is not None and len(distill_ds) > 0:
        if bg_mix_prob > 0:
            distill_ds_aug = SoundscapeBackgroundMix(
                distill_ds, soundscape_dir,
                sample_rate=sample_rate, mix_prob=bg_mix_prob,
                snr_range=bg_snr_range,
            )
            train_parts.append(distill_ds_aug)
        else:
            train_parts.append(distill_ds)

    if num_train_classes > num_classes:
        # Pad targets for datasets that output num_classes-sized targets.
        # Distill datasets (and their wrappers) already output num_train_classes.
        def _needs_pad(ds):
            """Check if dataset outputs targets smaller than num_train_classes."""
            try:
                t = ds[0]["target"]
                return len(t) < num_train_classes
            except Exception:
                return True
        train_parts = [_PadTargetDataset(ds, num_train_classes)
                       if _needs_pad(ds) else ds
                       for ds in train_parts]

    train_ds = ConcatDataset(train_parts)

    # Val stays at num_classes — we only evaluate on target classes
    val_ds = ConcatDataset([audio_val_sub, sc_val])

    logger.info(f"Fold {fold}/{n_folds}: "
                f"Train = {n_train_audio} audio (mixed) + {len(sc_train)} soundscape"
                f"{f' + {len(distill_ds)} distill' if distill_ds else ''}, "
                f"Val = {len(audio_val_sub)} audio + {len(sc_val)} soundscape")

    split_info = {
        "audio_ds": audio_ds_full,
        "audio_train_idx": audio_train_idx,
        "soundscape_ds_gt": soundscape_ds_gt,
        "sc_train_indices": sc_train_indices,
        "pseudo_ds": pseudo_ds,
        "distill_ds": distill_ds,
        "num_train_classes": num_train_classes,
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
      [n_train_audio .. n_train_audio + n_sc_train - 1] = soundscapes (gt + pseudo)
      [n_train_audio + n_sc_train .. end] = distill audio (if any)
    """
    audio_ds = split_info["audio_ds"]
    audio_train_idx = split_info["audio_train_idx"]
    soundscape_ds_gt = split_info["soundscape_ds_gt"]
    sc_train_indices = split_info["sc_train_indices"]
    pseudo_ds = split_info.get("pseudo_ds")
    distill_ds = split_info.get("distill_ds")
    num_train_classes = split_info.get("num_train_classes", num_classes)

    # Count class frequencies across all training samples (over all train classes)
    class_counts = np.zeros(num_train_classes, dtype=np.float64)

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

    n_distill = len(distill_ds) if distill_ds else 0
    if distill_ds:
        for entry in distill_ds.entries:
            class_counts[entry["class_idx"]] += 1

    class_counts = np.maximum(class_counts, 1.0)
    class_weights = (1.0 / class_counts) ** balance_alpha

    n_total = n_train_audio + n_sc_train + n_distill
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

    if distill_ds:
        offset = n_train_audio + n_sc_train
        for i, entry in enumerate(distill_ds.entries):
            sample_weights[offset + i] = class_weights[entry["class_idx"]]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=n_total, replacement=True
    )

    min_c, max_c = class_counts[:num_classes].min(), class_counts[:num_classes].max()
    logger.info(f"Class-balanced sampler (alpha={balance_alpha}): "
                f"target class counts range [{min_c:.0f}, {max_c:.0f}], "
                f"weight range [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
    if n_pseudo > 0:
        logger.info(f"  (includes {n_pseudo} pseudo-labeled segments)")
    if n_distill > 0:
        logger.info(f"  (includes {n_distill} distill audio files)")

    return sampler


def get_dataloaders(data_dir, batch_size=32, num_workers=4, sample_rate=32000,
                    min_duration=DEFAULT_MIN_DURATION,
                    max_duration=DEFAULT_MAX_DURATION,
                    val_frac=0.25, seed=42,
                    label_smoothing=0.0,
                    n_folds=5, fold=0, multi_mix=True, mix_prob=0.7,
                    preload=False, valid_regions_path=None,
                    pseudo_labels_csv=None, balance_alpha=0.5,
                    full_files=True,
                    distill_manifest=None, hard_negatives=True,
                    bg_mix_prob=0.0, bg_snr_range=(3.0, 15.0)):
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
        full_files=full_files,
        distill_manifest=distill_manifest, hard_negatives=hard_negatives,
        bg_mix_prob=bg_mix_prob, bg_snr_range=bg_snr_range,
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

    num_train_classes = split_info.get("num_train_classes", num_classes)
    return train_loader, val_loader, label_map, num_classes, n_audio, num_train_classes
