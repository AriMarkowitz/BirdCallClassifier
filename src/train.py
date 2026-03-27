"""
Fine-tune EfficientNet on BirdCLEF 2026 for multi-label classification.

Usage:
    python src/train.py [--backbone tf_efficientnet_b0_ns] [--max_duration 30]

Key features:
  - EfficientNet backbone (ImageNet-pretrained via timm)
  - Variable-length audio input (3-30s default, padded per batch)
  - NMF spectral branch (frozen dictionary → logit addition)
  - SuMix augmentation, focal loss, class-balanced sampling
"""

import os
import sys
import argparse
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_ROOT, "src"))

from model import BirdCLEFModel
from torch.utils.data import DataLoader
from dataset import get_dataloaders, get_datasets, build_label_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Binary focal loss for multi-label classification.

    Focuses training on hard examples by down-weighting easy negatives.
    With gamma=0 this is equivalent to standard BCE.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.where(target >= 0.5, pred, 1.0 - pred)
        alpha_t = torch.where(target >= 0.5, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * (1.0 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class BirdCLEFWrapper(pl.LightningModule):
    """Lightning wrapper for EfficientNet fine-tuning on BirdCLEF."""

    def __init__(self, model, num_classes, learning_rate=1e-4,
                 warmup_epochs=1, max_epochs=30,
                 loss_type="bce", focal_alpha=0.25, focal_gamma=2.0,
                 mixup_alpha=0.4, idx_to_label=None):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.mixup_alpha = mixup_alpha
        self.idx_to_label = idx_to_label or {}
        if loss_type == "focal":
            self.loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            logger.info(f"Using FocalLoss (alpha={focal_alpha}, gamma={focal_gamma})")
        else:
            self.loss_fn = nn.BCELoss()
            logger.info("Using BCELoss")
        if mixup_alpha > 0:
            logger.info(f"SuMix enabled: alpha={mixup_alpha}")

    def forward(self, x):
        output_dict = self.model(x)
        return output_dict["clipwise_output"]

    def _safe_loss(self, pred, target):
        """Clamp predictions to [eps, 1-eps] and compute loss in fp32."""
        with torch.amp.autocast("cuda", enabled=False):
            pred = pred.float().clamp(1e-6, 1.0 - 1e-6)
            target = target.float()
            return self.loss_fn(pred, target)

    def _sumix(self, waveform, target):
        """SuMix: shuffle batch and mix waveforms additively with soft labels.

        Waveform: anchor + (1-lam) * shuffled (additive, not convex).
        Target: element-wise max of (anchor_target, (1-lam) * shuffled_target).
        """
        if self.mixup_alpha <= 0:
            return waveform, target

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        lam = max(lam, 1.0 - lam)  # ensure anchor dominates

        batch_size = waveform.size(0)
        perm = torch.randperm(batch_size, device=waveform.device)

        waveform_mixed = waveform + (1.0 - lam) * waveform[perm]
        target_mixed = torch.maximum(target, (1.0 - lam) * target[perm])

        return waveform_mixed, target_mixed

    def training_step(self, batch, batch_idx):
        waveform = batch["waveform"]
        target = batch["target"]

        if self.training and self.mixup_alpha > 0:
            waveform, target = self._sumix(waveform, target)

        pred = self(waveform)
        loss = self._safe_loss(pred, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if batch_idx % 50 == 0:
            logger.info(f"Step {self.global_step} (epoch {self.current_epoch}, "
                        f"batch {batch_idx}): train_loss={loss.item():.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch["waveform"])
        loss = self._safe_loss(pred, batch["target"])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        if not hasattr(self, "_val_preds"):
            self._val_preds = []
            self._val_targets = []
        self._val_preds.append(pred.detach().cpu())
        self._val_targets.append(batch["target"].detach().cpu())

    def on_validation_epoch_end(self):
        import numpy as np
        from sklearn.metrics import roc_auc_score

        if not hasattr(self, "_val_preds") or len(self._val_preds) == 0:
            return

        preds = torch.cat(self._val_preds).numpy()
        targets = torch.cat(self._val_targets).numpy()
        self._val_preds.clear()
        self._val_targets.clear()

        per_class_auc = {}
        col_mask = targets.sum(axis=0) > 0
        for i in range(self.num_classes):
            if not col_mask[i]:
                continue
            try:
                auc_i = roc_auc_score(targets[:, i], preds[:, i])
                sp_name = self.idx_to_label.get(i, f"class_{i}")
                per_class_auc[sp_name] = auc_i
            except Exception:
                pass

        try:
            if col_mask.any():
                macro_auc = roc_auc_score(
                    targets[:, col_mask], preds[:, col_mask], average="macro"
                )
            else:
                macro_auc = 0.0
        except Exception:
            macro_auc = 0.0

        self.log("val_macro_auc", macro_auc, prog_bar=True)
        self.log("val_n_evaluable_classes", float(len(per_class_auc)))
        logger.info(f"Epoch {self.current_epoch}: val_macro_auc={macro_auc:.4f} "
                     f"({len(per_class_auc)} evaluable classes)")

        if per_class_auc:
            sorted_aucs = sorted(per_class_auc.items(), key=lambda x: x[1])
            worst10 = sorted_aucs[:10]
            best10 = sorted_aucs[-10:]

            logger.info(f"  Worst 10 AUC: " +
                         ", ".join(f"{sp}={auc:.3f}" for sp, auc in worst10))
            logger.info(f"  Best 10 AUC: " +
                         ", ".join(f"{sp}={auc:.3f}" for sp, auc in best10))

            if self.logger and hasattr(self.logger, "experiment"):
                try:
                    import wandb
                    table = wandb.Table(columns=["species", "auc", "rank"])
                    for rank, (sp, auc) in enumerate(sorted_aucs):
                        table.add_data(sp, round(auc, 4), rank + 1)
                    self.logger.experiment.log({
                        "val/per_class_auc_table": table,
                        "val/worst_class_auc": worst10[0][1],
                        "val/best_class_auc": best10[-1][1],
                        "val/median_class_auc": float(np.median(
                            [v for v in per_class_auc.values()])),
                    })
                except Exception:
                    pass

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())

        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.05,
        )

        return optimizer


def _load_pseudo_label_summary(pseudo_labels_csv):
    """Load pseudo-label summary JSON (saved alongside CSV) for W&B config."""
    if not pseudo_labels_csv:
        return {}
    import json
    from pathlib import Path
    summary_path = Path(pseudo_labels_csv).with_suffix(".json")
    if not summary_path.is_file():
        return {}
    try:
        with open(summary_path) as f:
            s = json.load(f)
        return {
            "pseudo_threshold": s.get("threshold"),
            "pseudo_segments_kept": s.get("segments_kept"),
            "pseudo_yield_rate": s.get("yield_rate"),
            "pseudo_unique_species": s.get("unique_species"),
            "pseudo_mean_conf": s.get("confidence_distribution", {}).get("mean"),
        }
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet on BirdCLEF 2026")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(PROJ_ROOT, "data"))
    parser.add_argument("--backbone", type=str, default="tf_efficientnet_b0_ns",
                        help="timm backbone name (default: tf_efficientnet_b0_ns)")
    parser.add_argument("--n_mels", type=int, default=128,
                        help="Number of mel bins for backbone spectrogram")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--val_frac", type=float, default=0.25)
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(PROJ_ROOT, "checkpoints"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to Lightning checkpoint to resume training from")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "focal"])
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="birdclef-2026")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--multi_mix", action="store_true", default=True)
    parser.add_argument("--no_multi_mix", action="store_false", dest="multi_mix")
    parser.add_argument("--mix_prob", type=float, default=0.7)
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--valid_regions", type=str, default=None)
    parser.add_argument("--pseudo_labels", type=str, default=None)
    parser.add_argument("--balance_alpha", type=float, default=0.5)
    parser.add_argument("--min_duration", type=float, default=3.0,
                        help="Minimum clip duration in seconds")
    parser.add_argument("--max_duration", type=float, default=30.0,
                        help="Maximum clip duration in seconds")
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    label_map = build_label_map(os.path.join(args.data_dir, "taxonomy.csv"))
    num_classes = len(label_map)

    logger.info(f"=== EfficientNet training (backbone={args.backbone}) ===")
    logger.info(f"Variable-length input: {args.min_duration}s - {args.max_duration}s")

    # Build dataloaders
    train_loader, val_loader, label_map, num_classes, n_train_audio = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=32000,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        val_frac=args.val_frac,
        seed=args.seed,
        label_smoothing=args.label_smoothing,
        n_folds=args.n_folds,
        fold=args.fold,
        multi_mix=args.multi_mix,
        mix_prob=args.mix_prob,
        preload=args.preload,
        valid_regions_path=args.valid_regions,
        pseudo_labels_csv=args.pseudo_labels,
        balance_alpha=args.balance_alpha,
    )
    logger.info(f"Classes: {num_classes}, Train batches: {len(train_loader)}, "
                f"Val batches: {len(val_loader)}")

    # Build model
    model = BirdCLEFModel(
        num_classes=num_classes,
        backbone_name=args.backbone,
        sample_rate=32000,
        n_mels=args.n_mels,
        pretrained=True,
    )
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    idx_to_label = {v: k for k, v in label_map.items()}
    wrapper = BirdCLEFWrapper(
        model, num_classes,
        learning_rate=args.lr,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        loss_type=args.loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        mixup_alpha=args.mixup_alpha,
        idx_to_label=idx_to_label,
    )

    # Save checkpoints in a run-specific subdirectory
    run_id = args.run_id or f"seed{args.seed}"
    run_save_dir = os.path.join(args.save_dir, run_id)
    os.makedirs(run_save_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {run_save_dir}")

    ckpt_callback = ModelCheckpoint(
        dirpath=run_save_dir,
        filename="birdclef-enet-{epoch:02d}-{val_macro_auc:.4f}",
        monitor="val_macro_auc",
        mode="max",
        save_top_k=5,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    loggers = []
    if args.use_wandb:
        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.run_name,
            config={
                "lr": args.lr,
                "batch_size": args.batch_size,
                "max_epochs": args.max_epochs,
                "seed": args.seed,
                "val_frac": args.val_frac,
                "warmup_epochs": args.warmup_epochs,
                "label_smoothing": args.label_smoothing,
                "fold": args.fold,
                "n_folds": args.n_folds,
                "model": args.backbone,
                "n_mels": args.n_mels,
                "loss": args.loss,
                "focal_alpha": args.focal_alpha,
                "focal_gamma": args.focal_gamma,
                "multi_mix": args.multi_mix,
                "mix_prob": args.mix_prob,
                "mixup_alpha": args.mixup_alpha,
                "pseudo_labels": args.pseudo_labels,
                "balance_alpha": args.balance_alpha,
                "min_duration": args.min_duration,
                "max_duration": args.max_duration,
                "n_train_audio": n_train_audio,
                **_load_pseudo_label_summary(args.pseudo_labels),
            },
        )
        loggers.append(wandb_logger)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[ckpt_callback, lr_monitor],
        logger=loggers if loggers else True,
        default_root_dir=args.save_dir,
        log_every_n_steps=50,
        precision=32,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(wrapper, train_loader, val_loader, ckpt_path=args.resume_from)


if __name__ == "__main__":
    main()
