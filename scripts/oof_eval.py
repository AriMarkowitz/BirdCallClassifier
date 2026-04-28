"""
Out-of-fold (OOF) ensemble validation.

Each ensemble member trains on a different CV fold, so per-model val_macro_auc
values are computed on different data and aren't directly comparable. This
script:

  1. For each model i in the ensemble manifest:
     - Loads checkpoint
     - Recomputes the dataloader for fold i (the held-out validation fold for that model)
     - Predicts on its own val fold (which it never trained on — no leakage)
  2. Concatenates per-fold predictions into one prediction-per-sample matrix.
  3. Reports unified OOF macro-AUC + per-class AUC.

This is a leak-free ensemble-quality estimate that doesn't require a Kaggle
submission.

Usage:
  python scripts/oof_eval.py --manifest checkpoints/ensemble_manifest_<JOB_ID>.txt
  python scripts/oof_eval.py --manifest <path> --output oof_predictions.npz

The manifest is the file produced by ensemble_pipeline.sh — tab-separated
backbone\tfold\tcheckpoint per non-comment line.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJ_ROOT, "scripts"))

from model import BirdCLEFModel  # noqa: E402
from dataset import build_label_map, get_datasets, variable_length_collate_fn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from pseudo_label import _detect_backbone_from_state_dict, _remap_legacy_keys  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_manifest(path):
    """Returns list of (backbone, fold, ckpt_path)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            backbone, fold, ckpt = parts
            rows.append((backbone.strip(), int(fold.strip()), ckpt.strip()))
    return rows


def load_model(ckpt_path, backbone, num_classes, device):
    """Load BirdCLEFModel from a Lightning checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("model.model.", "model.")
        k = k.replace("model.", "", 1) if k.startswith("model.") else k
        cleaned[k] = v
    cleaned = _remap_legacy_keys(cleaned)

    head_w = "head.5.weight"
    num_train_classes = num_classes
    if head_w in cleaned:
        ckpt_classes = cleaned[head_w].shape[0]
        if ckpt_classes > num_classes:
            num_train_classes = ckpt_classes

    model = BirdCLEFModel(
        num_classes=num_classes,
        num_train_classes=num_train_classes,
        backbone_name=backbone,
        pretrained=False,
    )
    missing, unexpected = model.load_state_dict(cleaned, strict=True)
    assert not missing, f"Missing keys for {ckpt_path}: {missing}"
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_val_fold(model, val_loader, device, num_classes):
    """Returns (preds: (N, num_classes), targets: (N, num_classes))."""
    preds, targets = [], []
    for batch in val_loader:
        wav = batch["waveform"].to(device)
        target = batch["target"]
        out = model(wav)
        preds.append(out["clipwise_output"].detach().float().cpu().numpy())
        targets.append(target.detach().float().cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def main():
    ap = argparse.ArgumentParser(description="Out-of-fold ensemble validation")
    ap.add_argument("--manifest", required=True, help="Path to ensemble_manifest_*.txt")
    ap.add_argument("--data-dir", default=os.path.join(PROJ_ROOT, "data"))
    ap.add_argument("--output", default=None,
                    help="Optional .npz to save (preds, targets, fold_per_sample)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    logger.info(f"Device: {device}")

    members = parse_manifest(args.manifest)
    logger.info(f"Manifest: {len(members)} ensemble members")
    for backbone, fold, ckpt in members:
        logger.info(f"  fold={fold}  backbone={backbone}  ckpt={Path(ckpt).name}")

    label_map = build_label_map(os.path.join(args.data_dir, "taxonomy.csv"))
    num_classes = len(label_map)
    idx_to_label = {v: k for k, v in label_map.items()}

    # Stitch predictions per-fold (each member predicts on the data it never trained on).
    all_preds = []
    all_targets = []
    fold_ids = []
    per_member_aucs = {}

    for backbone, fold, ckpt in members:
        logger.info(f"\n── Member fold={fold}, backbone={backbone} ──")
        # Recreate the SAME dataloader as training (so val-fold composition matches).
        _, val_ds, _, _, _, _, _ = get_datasets(
            args.data_dir,
            min_duration=5.0, max_duration=5.0,
            n_folds=args.n_folds, fold=fold,
            multi_mix=False, mix_prob=0.0,  # off — we're evaluating, not training
            valid_regions_path=os.path.join(args.data_dir, "valid_regions.json"),
            full_files=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=variable_length_collate_fn,
            pin_memory=True,
        )

        model = load_model(ckpt, backbone, num_classes, device)
        preds, targets = predict_val_fold(model, val_loader, device, num_classes)
        del model
        torch.cuda.empty_cache()

        # Per-member AUC on its own fold (not directly comparable across members
        # since folds differ — but logged for sanity).
        col_mask = targets.sum(axis=0) > 0
        try:
            auc = roc_auc_score(targets[:, col_mask], preds[:, col_mask], average="macro")
        except Exception:
            auc = float("nan")
        per_member_aucs[(backbone, fold)] = (auc, int(col_mask.sum()), len(targets))
        logger.info(f"  Per-member val AUC = {auc:.4f}  ({int(col_mask.sum())} classes, {len(targets)} samples)")

        all_preds.append(preds)
        all_targets.append(targets)
        fold_ids.append(np.full(len(targets), fold, dtype=np.int32))

    # Stitch
    preds_oof = np.concatenate(all_preds, axis=0)
    targets_oof = np.concatenate(all_targets, axis=0)
    folds_oof = np.concatenate(fold_ids, axis=0)

    logger.info(f"\n── OOF aggregate: {len(targets_oof)} samples across {len(members)} folds ──")

    col_mask = targets_oof.sum(axis=0) > 0
    n_evaluable = int(col_mask.sum())
    try:
        oof_auc = roc_auc_score(
            targets_oof[:, col_mask], preds_oof[:, col_mask], average="macro"
        )
    except Exception as e:
        oof_auc = float("nan")
        logger.warning(f"OOF AUC failed: {e}")

    logger.info(f"OOF macro-AUC = {oof_auc:.4f}  ({n_evaluable} evaluable classes)")

    # Worst/best class report
    per_class = {}
    for i in range(num_classes):
        if not col_mask[i]:
            continue
        try:
            per_class[idx_to_label.get(i, f"class_{i}")] = roc_auc_score(
                targets_oof[:, i], preds_oof[:, i]
            )
        except Exception:
            pass
    if per_class:
        sorted_aucs = sorted(per_class.items(), key=lambda x: x[1])
        logger.info("Worst 10 OOF AUC: " + ", ".join(f"{n}={a:.3f}" for n, a in sorted_aucs[:10]))
        logger.info("Best 10 OOF AUC:  " + ", ".join(f"{n}={a:.3f}" for n, a in sorted_aucs[-10:]))

    if args.output:
        np.savez_compressed(
            args.output,
            preds=preds_oof.astype(np.float32),
            targets=targets_oof.astype(np.float32),
            folds=folds_oof,
            class_indices=np.arange(num_classes),
        )
        logger.info(f"Saved OOF predictions: {args.output}")


if __name__ == "__main__":
    main()
