"""
Generate pseudo-labels for unlabeled soundscape files.

Runs a trained BirdCLEF EfficientNet model on all unlabeled soundscape files
in 5-second sliding windows, filters predictions by confidence threshold,
and outputs a CSV in the same format as train_soundscapes_labels.csv so
SoundscapeDataset can load it directly.

Usage:
    python scripts/pseudo_label.py \
        --checkpoint checkpoints/best.ckpt \
        --data-dir data \
        --output data/pseudo_labels.csv \
        --threshold 0.8
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_ROOT, "src"))

from model import BirdCLEFModel
from dataset import build_label_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 32000
SEGMENT_DURATION = 5.0  # seconds — matches soundscape label format
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)


def load_model(checkpoint_path, num_classes, device):
    """Load a trained BirdCLEFWrapper from a Lightning checkpoint."""
    model = BirdCLEFModel(
        num_classes=num_classes,
        sample_rate=SAMPLE_RATE,
        pretrained=False,  # weights come from checkpoint
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Lightning wraps in "model." prefix
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("model.model.", "model.")
        k = k.replace("model.", "", 1) if k.startswith("model.") else k
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    logger.info(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        logger.debug(f"  Missing keys: {missing[:10]}")

    model = model.to(device)
    model.eval()
    return model


def load_soundscape_audio(path, sample_rate=SAMPLE_RATE):
    """Load a full soundscape file."""
    try:
        audio, sr = sf.read(path, dtype="float32")
    except Exception:
        audio, sr = librosa.load(path, sr=sample_rate, mono=True)
        return audio

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    return audio.astype(np.float32)


def seconds_to_timestr(sec):
    """Convert seconds to HH:MM:SS format."""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def predict_file(model, audio, device, batch_size=32):
    """Run inference on a soundscape file in 5s sliding windows.

    Each 5s segment is passed directly to the model (variable-length capable).

    Returns:
        segments: list of (start_sec, end_sec)
        probs: np.array of shape (n_segments, num_classes)
    """
    n_samples = len(audio)
    segments = []
    waveforms = []

    for start in range(0, n_samples, SEGMENT_SAMPLES):
        end = start + SEGMENT_SAMPLES
        if end > n_samples:
            break

        segment = audio[start:end]
        waveforms.append(segment)
        segments.append((start / SAMPLE_RATE, end / SAMPLE_RATE))

    if not waveforms:
        return [], np.array([])

    # Batch inference — all segments are same length (5s) so no padding needed
    all_probs = []
    for i in range(0, len(waveforms), batch_size):
        batch = np.stack(waveforms[i:i + batch_size])
        batch_tensor = torch.from_numpy(batch).to(device)

        with torch.no_grad():
            output = model(batch_tensor)
            clipwise = output["clipwise_output"]
            all_probs.append(clipwise.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    return segments, probs


def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels for unlabeled soundscapes")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="data/pseudo_labels.csv")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="birdclef-2026")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    label_map = build_label_map(str(data_dir / "taxonomy.csv"))
    num_classes = len(label_map)
    idx_to_label = {v: k for k, v in label_map.items()}
    logger.info(f"Classes: {num_classes}")

    model = load_model(args.checkpoint, num_classes, device)

    # --- Collect files to pseudo-label ---
    soundscape_dir = data_dir / "train_soundscapes"
    labels_csv = data_dir / "train_soundscapes_labels.csv"
    labeled_df = pd.read_csv(labels_csv)
    labeled_files = set(labeled_df["filename"].unique())

    all_sc_files = sorted(f.name for f in soundscape_dir.glob("*.ogg"))
    unlabeled_sc = [f for f in all_sc_files if f not in labeled_files]
    logger.info(f"Soundscapes: {len(all_sc_files)} total, "
                f"{len(labeled_files)} labeled, {len(unlabeled_sc)} unlabeled")

    # Train audio files (discover secondary species)
    audio_dir = data_dir / "train_audio"
    train_df = pd.read_csv(data_dir / "train.csv")
    known_labels_per_file = {}
    for _, row in train_df.iterrows():
        fn = row["filename"]
        labels = {str(row["primary_label"])}
        try:
            import ast
            sec = ast.literal_eval(str(row.get("secondary_labels", "[]")))
            labels.update(str(s).strip() for s in sec)
        except (ValueError, SyntaxError):
            pass
        known_labels_per_file[fn] = labels

    train_audio_files = sorted(train_df["filename"].values)
    logger.info(f"Train audio: {len(train_audio_files)} files to scan for secondary species")

    # --- Run inference ---
    rows_soundscape = []
    rows_train_audio = []
    n_segments_total = 0
    n_segments_kept = 0
    all_max_confs = []
    species_confs = {}

    for fn in tqdm(unlabeled_sc, desc="Pseudo-label soundscapes", mininterval=30):
        path = str(soundscape_dir / fn)
        try:
            audio = load_soundscape_audio(path)
        except Exception as e:
            logger.warning(f"Skipping {fn}: {e}")
            continue

        segments, probs = predict_file(model, audio, device, batch_size=args.batch_size)

        for (start_sec, end_sec), prob in zip(segments, probs):
            n_segments_total += 1
            max_conf = float(prob.max())
            all_max_confs.append(max_conf)

            above = np.where(prob >= args.threshold)[0]
            if len(above) == 0:
                continue

            n_segments_kept += 1
            species_labels = [idx_to_label[i] for i in above]
            label_str = ";".join(species_labels)

            for i in above:
                sp = idx_to_label[i]
                species_confs.setdefault(sp, []).append(float(prob[i]))

            rows_soundscape.append({
                "filename": fn,
                "start": seconds_to_timestr(start_sec),
                "end": seconds_to_timestr(end_sec),
                "primary_label": label_str,
            })

    logger.info(f"Soundscape pseudo-labeling: {n_segments_kept}/{n_segments_total} segments kept")

    # Process train_audio for secondary species discovery
    n_train_segments = 0
    n_train_discoveries = 0
    for fn in tqdm(train_audio_files, desc="Discover secondary species", mininterval=30):
        path = str(audio_dir / fn)
        try:
            audio = load_soundscape_audio(path)
        except Exception as e:
            logger.warning(f"Skipping {fn}: {e}")
            continue

        segments, probs = predict_file(model, audio, device, batch_size=args.batch_size)

        known = known_labels_per_file.get(fn, set())
        for (start_sec, end_sec), prob in zip(segments, probs):
            n_train_segments += 1

            above = np.where(prob >= args.threshold)[0]
            if len(above) == 0:
                continue

            novel = [i for i in above if idx_to_label[i] not in known]
            if not novel:
                continue

            n_train_discoveries += 1
            novel_labels = [idx_to_label[i] for i in novel]
            label_str = ";".join(novel_labels)

            for i in novel:
                sp = idx_to_label[i]
                species_confs.setdefault(sp, []).append(float(prob[i]))
                all_max_confs.append(float(prob[i]))

            rows_train_audio.append({
                "filename": fn,
                "start": seconds_to_timestr(start_sec),
                "end": seconds_to_timestr(end_sec),
                "primary_label": label_str,
            })

    logger.info(f"Train audio secondary species: {n_train_discoveries}/{n_train_segments} "
                f"segments with novel species")

    rows = rows_soundscape + rows_train_audio

    # Save CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        out_df = pd.DataFrame(rows)
        out_df.to_csv(output_path, index=False)
    else:
        pd.DataFrame(columns=["filename", "start", "end", "primary_label"]
                      ).to_csv(output_path, index=False)

    # --- Compute metrics ---
    all_max_confs = np.array(all_max_confs) if all_max_confs else np.array([])
    yield_rate = n_segments_kept / max(n_segments_total, 1)
    unique_species = sorted(species_confs.keys())

    per_species_stats = {}
    for sp in unique_species:
        confs = species_confs[sp]
        per_species_stats[sp] = {
            "count": len(confs),
            "mean_conf": float(np.mean(confs)),
            "min_conf": float(np.min(confs)),
        }

    conf_hist_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    conf_hist_counts = np.histogram(all_max_confs, bins=conf_hist_bins)[0].tolist() if len(all_max_confs) > 0 else []

    species_by_count = sorted(per_species_stats.items(), key=lambda x: -x[1]["count"])
    top10 = species_by_count[:10]
    bottom10 = species_by_count[-10:] if len(species_by_count) > 10 else []

    total_kept = len(rows_soundscape) + len(rows_train_audio)
    logger.info(f"\nPseudo-labeling complete:")
    logger.info(f"  Soundscape: {len(unlabeled_sc)} files, "
                f"{n_segments_kept}/{n_segments_total} segments kept")
    logger.info(f"  Train audio: {len(train_audio_files)} files, "
                f"{n_train_discoveries}/{n_train_segments} segments with novel species")
    logger.info(f"  Total pseudo-labeled rows: {total_kept}")
    logger.info(f"  Soundscape yield rate: {yield_rate:.1%}")
    logger.info(f"  Unique species: {len(unique_species)}/{num_classes}")
    if len(all_max_confs) > 0:
        logger.info(f"  Max-confidence distribution (all segments):")
        logger.info(f"    mean={all_max_confs.mean():.3f}, "
                     f"median={np.median(all_max_confs):.3f}, "
                     f"p90={np.percentile(all_max_confs, 90):.3f}")
    logger.info(f"  Top 10 species by pseudo-label count:")
    for sp, stats in top10:
        logger.info(f"    {sp}: {stats['count']} segments, "
                     f"mean_conf={stats['mean_conf']:.3f}")
    if bottom10:
        logger.info(f"  Bottom 10 species by pseudo-label count:")
        for sp, stats in bottom10:
            logger.info(f"    {sp}: {stats['count']} segments, "
                         f"mean_conf={stats['mean_conf']:.3f}")
    logger.info(f"  Saved to: {output_path}")

    # --- Save JSON summary ---
    summary = {
        "checkpoint": args.checkpoint,
        "threshold": args.threshold,
        "soundscape_files": len(unlabeled_sc),
        "soundscape_segments_kept": len(rows_soundscape),
        "train_audio_files": len(train_audio_files),
        "train_audio_discoveries": len(rows_train_audio),
        "total_segments": n_segments_total + n_train_segments,
        "segments_kept": total_kept,
        "yield_rate": round(yield_rate, 4),
        "unique_species": len(unique_species),
        "total_classes": num_classes,
        "confidence_distribution": {
            "mean": round(float(all_max_confs.mean()), 4) if len(all_max_confs) > 0 else None,
            "median": round(float(np.median(all_max_confs)), 4) if len(all_max_confs) > 0 else None,
            "p10": round(float(np.percentile(all_max_confs, 10)), 4) if len(all_max_confs) > 0 else None,
            "p90": round(float(np.percentile(all_max_confs, 90)), 4) if len(all_max_confs) > 0 else None,
            "histogram_bins": conf_hist_bins,
            "histogram_counts": conf_hist_counts,
        },
        "per_species": per_species_stats,
    }

    summary_path = output_path.with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Summary saved to: {summary_path}")

    # --- W&B logging ---
    if args.use_wandb:
        try:
            import wandb
            run_name = args.run_name or f"pseudo-label-t{args.threshold}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                job_type="pseudo-label",
                config={
                    "checkpoint": args.checkpoint,
                    "threshold": args.threshold,
                    "batch_size": args.batch_size,
                },
            )

            wandb.log({
                "pseudo/total_segments": n_segments_total,
                "pseudo/segments_kept": n_segments_kept,
                "pseudo/yield_rate": yield_rate,
                "pseudo/unique_species": len(unique_species),
                "pseudo/species_coverage": len(unique_species) / max(num_classes, 1),
            })

            if len(all_max_confs) > 0:
                wandb.log({
                    "pseudo/confidence_hist": wandb.Histogram(all_max_confs.tolist()),
                    "pseudo/mean_max_conf": float(all_max_confs.mean()),
                    "pseudo/median_max_conf": float(np.median(all_max_confs)),
                })

            species_table = wandb.Table(
                columns=["species", "count", "mean_conf", "min_conf"])
            for sp, stats in species_by_count:
                species_table.add_data(sp, stats["count"],
                                       round(stats["mean_conf"], 4),
                                       round(stats["min_conf"], 4))
            wandb.log({"pseudo/species_stats": species_table})

            wandb.finish()
            logger.info("  W&B logging complete.")
        except Exception as e:
            logger.warning(f"  W&B logging failed: {e}")


if __name__ == "__main__":
    main()
