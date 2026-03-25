"""
Fine-tune HTSAT on BirdCLEF 2026 for multi-label classification.

Usage:
    python src/train.py --checkpoint path/to/HTSAT_AudioSet_Saved.ckpt

Key adjustments from AudioSet defaults:
  - classes_num: 234 (from taxonomy.csv)
  - Warmup reduced: 1 epoch at 0.1x LR, then cosine decay
  - Lower LR: 3e-5 default (fine-tuning, not training from scratch)
  - Batch size: 32 (single GPU default)
  - Site-based validation split for realistic eval
  - WeightedRandomSampler to upweight soundscape data
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

# Add HTSAT source to path
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_ROOT, "external", "htsat"))
sys.path.insert(0, os.path.join(PROJ_ROOT, "src"))

import config as htsat_config
from model.htsat import HTSAT_Swin_Transformer
from sed_model import SEDWrapper
from torch.utils.data import DataLoader
from dataset import get_dataloaders, get_datasets, build_label_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Binary focal loss for multi-label classification.

    Focuses training on hard examples by down-weighting easy negatives.
    With gamma=0 this is equivalent to standard BCE.

    Args:
        alpha: Weighting factor for positives (1-alpha for negatives).
        gamma: Focusing parameter — higher values down-weight easy examples more.
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


def load_pretrained_htsat(checkpoint_path, config, num_classes):
    """
    Load an AudioSet-pretrained HTSAT checkpoint and replace the
    classification head for `num_classes`.
    """
    # Build model with original 527 classes to load weights
    model_527 = HTSAT_Swin_Transformer(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        patch_stride=config.htsat_stride,
        num_classes=527,
        embed_dim=config.htsat_dim,
        depths=config.htsat_depth,
        num_heads=config.htsat_num_head,
        window_size=config.htsat_window_size,
        config=config,
    )

    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            state_dict = {
                k.replace("sed_model.", ""): v
                for k, v in state_dict.items()
            }
        else:
            state_dict = ckpt

        filtered = {
            k: v for k, v in state_dict.items()
            if "head" not in k and "tscam_conv" not in k and "nmf_proj" not in k
        }
        missing, unexpected = model_527.load_state_dict(filtered, strict=False)
        logger.info(f"Loaded pretrained weights. Missing: {len(missing)}, "
                    f"Unexpected: {len(unexpected)}")
    else:
        logger.warning("No checkpoint provided — training from scratch.")

    # Rebuild with correct num_classes
    model = HTSAT_Swin_Transformer(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        patch_stride=config.htsat_stride,
        num_classes=num_classes,
        embed_dim=config.htsat_dim,
        depths=config.htsat_depth,
        num_heads=config.htsat_num_head,
        window_size=config.htsat_window_size,
        config=config,
    )

    # Copy backbone weights
    src_dict = model_527.state_dict()
    tgt_dict = model.state_dict()
    for k in tgt_dict:
        if "head" not in k and "tscam_conv" not in k and "nmf_proj" not in k and k in src_dict:
            tgt_dict[k] = src_dict[k]
    model.load_state_dict(tgt_dict)
    logger.info(f"Head replaced: 527 -> {num_classes} classes")

    return model


class BirdCLEFWrapper(pl.LightningModule):
    """Lightning wrapper for HTSAT fine-tuning on BirdCLEF."""

    def __init__(self, sed_model, config, num_classes, learning_rate=1e-4,
                 warmup_epochs=1, max_epochs=30,
                 loss_type="bce", focal_alpha=0.25, focal_gamma=2.0,
                 mixup_alpha=0.4):
        super().__init__()
        self.sed_model = sed_model
        self.config = config
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.mixup_alpha = mixup_alpha
        if loss_type == "focal":
            self.loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            logger.info(f"Using FocalLoss (alpha={focal_alpha}, gamma={focal_gamma})")
        else:
            self.loss_fn = nn.BCELoss()
            logger.info("Using BCELoss")
        if mixup_alpha > 0:
            logger.info(f"SuMix enabled: alpha={mixup_alpha}")

    def forward(self, x, mix_lambda=None):
        output_dict = self.sed_model(x, mix_lambda)
        clipwise = output_dict["clipwise_output"]  # already sigmoided
        return clipwise, output_dict["framewise_output"]

    def _safe_loss(self, pred, target):
        """Clamp predictions to [eps, 1-eps] and compute BCE in fp32."""
        with torch.amp.autocast("cuda", enabled=False):
            pred = pred.float().clamp(1e-6, 1.0 - 1e-6)
            target = target.float()
            return self.loss_fn(pred, target)

    def _sumix(self, waveform, target):
        """SuMix: shuffle batch and mix waveforms additively with soft labels.

        Waveform: anchor + (1-lam) * shuffled (additive, not convex).
        Target: element-wise max of (anchor_target, (1-lam) * shuffled_target).
        This ensures the anchor labels stay at full strength while the secondary
        labels are scaled by their actual gain — a species mixed at 10% gain
        gets a 0.1 target, not 1.0.
        """
        if self.mixup_alpha <= 0:
            return waveform, target

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        lam = max(lam, 1.0 - lam)  # ensure anchor dominates

        batch_size = waveform.size(0)
        perm = torch.randperm(batch_size, device=waveform.device)

        # Additive waveform mix: anchor stays at full gain, secondary scaled
        waveform_mixed = waveform + (1.0 - lam) * waveform[perm]
        # Anchor labels stay at full strength; secondary labels scaled by
        # their mixing gain so the target reflects actual audibility
        target_mixed = torch.maximum(target, (1.0 - lam) * target[perm])

        return waveform_mixed, target_mixed

    def training_step(self, batch, batch_idx):
        waveform = batch["waveform"]
        target = batch["target"]

        # Apply SuMix batch-level augmentation
        if self.training and self.mixup_alpha > 0:
            waveform, target = self._sumix(waveform, target)

        pred, _ = self(waveform)
        loss = self._safe_loss(pred, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if batch_idx % 50 == 0:
            logger.info(f"Step {self.global_step} (epoch {self.current_epoch}, "
                        f"batch {batch_idx}): train_loss={loss.item():.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        pred, _ = self(batch["waveform"])
        loss = self._safe_loss(pred, batch["target"])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        # Accumulate for epoch-end AUC computation
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

        try:
            col_mask = targets.sum(axis=0) > 0
            if col_mask.any():
                macro_auc = roc_auc_score(
                    targets[:, col_mask], preds[:, col_mask], average="macro"
                )
            else:
                macro_auc = 0.0
        except Exception:
            macro_auc = 0.0

        self.log("val_macro_auc", macro_auc, prog_bar=True)
        logger.info(f"Epoch {self.current_epoch}: val_macro_auc={macro_auc:.4f}")

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



def main():
    parser = argparse.ArgumentParser(description="Fine-tune HTSAT on BirdCLEF 2026")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(PROJ_ROOT, "data"))
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to AudioSet pretrained HTSAT checkpoint")
    parser.add_argument("--batch_size", type=int, default=128)
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
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor (0=hard labels, 0.1=recommended)")
    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "focal"],
                        help="Loss function: 'bce' or 'focal'")
    parser.add_argument("--focal_alpha", type=float, default=0.25,
                        help="Focal loss alpha (positive class weight)")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss gamma (focusing parameter)")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="birdclef-2026",
                        help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run name (defaults to auto-generated)")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Run ID for checkpoint subdir (defaults to seed<N>)")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--fold", type=int, default=0,
                        help="Which fold to hold out for validation (0 to n_folds-1)")
    parser.add_argument("--multi_mix", action="store_true", default=True,
                        help="Enable multi-species mixing augmentation on train_audio")
    parser.add_argument("--no_multi_mix", action="store_false", dest="multi_mix",
                        help="Disable multi-species mixing augmentation")
    parser.add_argument("--mix_prob", type=float, default=0.7,
                        help="Probability of multi-species mixing per sample")
    parser.add_argument("--mixup_alpha", type=float, default=0.4,
                        help="Beta distribution alpha for SuMix (0=disabled)")
    parser.add_argument("--preload", action="store_true",
                        help="Preload all train_audio waveforms into RAM (~34GB) for faster mixing")
    parser.add_argument("--valid_regions", type=str, default=None,
                        help="Path to valid_regions.json from preprocess_activity.py")
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # Use tensor cores for fp32 matmuls (L40S has them)
    torch.set_float32_matmul_precision("medium")

    # Patch config for our dataset
    htsat_config.classes_num = len(
        build_label_map(os.path.join(args.data_dir, "taxonomy.csv"))
    )
    htsat_config.loss_type = "clip_bce"
    htsat_config.enable_tscam = True

    logger.info(f"=== Full model training ===")

    # Build dataloaders
    train_loader, val_loader, label_map, num_classes, n_train_audio = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=htsat_config.sample_rate,
        clip_duration=htsat_config.clip_samples / htsat_config.sample_rate,
        val_frac=args.val_frac,
        seed=args.seed,
        label_smoothing=args.label_smoothing,
        n_folds=args.n_folds,
        fold=args.fold,
        multi_mix=args.multi_mix,
        mix_prob=args.mix_prob,
        preload=args.preload,
        valid_regions_path=args.valid_regions,
    )
    logger.info(f"Classes: {num_classes}, Train batches: {len(train_loader)}, "
                f"Val batches: {len(val_loader)}")

    # Build model
    htsat_config.classes_num = num_classes
    model = load_pretrained_htsat(args.checkpoint, htsat_config, num_classes)
    wrapper = BirdCLEFWrapper(
        model, htsat_config, num_classes,
        learning_rate=args.lr,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        loss_type=args.loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        mixup_alpha=args.mixup_alpha,
    )

    # Save checkpoints in a run-specific subdirectory
    run_id = args.run_id or f"seed{args.seed}"
    run_save_dir = os.path.join(args.save_dir, run_id)
    os.makedirs(run_save_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {run_save_dir}")

    # Callbacks
    ckpt_callback = ModelCheckpoint(
        dirpath=run_save_dir,
        filename="birdclef-htsat-{epoch:02d}-{val_macro_auc:.4f}",
        monitor="val_macro_auc",
        mode="max",
        save_top_k=5,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Logger
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
                "model": "htsat-tiny",
                "loss": args.loss,
                "focal_alpha": args.focal_alpha,
                "focal_gamma": args.focal_gamma,
                "multi_mix": args.multi_mix,
                "mix_prob": args.mix_prob,
                "mixup_alpha": args.mixup_alpha,
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
    )

    trainer.fit(wrapper, train_loader, val_loader, ckpt_path=args.resume_from)


if __name__ == "__main__":
    main()
