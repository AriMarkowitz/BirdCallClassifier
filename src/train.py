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
from dataset import get_dataloaders, build_label_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        ckpt = torch.load(checkpoint_path, map_location="cpu")
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
            if "head" not in k and "tscam_conv" not in k
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
        if "head" not in k and "tscam_conv" not in k and k in src_dict:
            tgt_dict[k] = src_dict[k]
    model.load_state_dict(tgt_dict)
    logger.info(f"Head replaced: 527 -> {num_classes} classes")

    return model


class BirdCLEFWrapper(pl.LightningModule):
    """
    Lightning wrapper for HTSAT fine-tuning on BirdCLEF.
    """

    def __init__(self, sed_model, config, num_classes, learning_rate=3e-5,
                 warmup_epochs=1, max_epochs=30):
        super().__init__()
        self.sed_model = sed_model
        self.config = config
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.loss_fn = nn.BCELoss()

    def forward(self, x, mix_lambda=None):
        output_dict = self.sed_model(x, mix_lambda)
        return output_dict["clipwise_output"], output_dict["framewise_output"]

    def _safe_loss(self, pred, target):
        """Clamp predictions to [eps, 1-eps] and compute BCE in fp32."""
        with torch.amp.autocast("cuda", enabled=False):
            pred = pred.float().clamp(1e-6, 1.0 - 1e-6)
            target = target.float()
            return self.loss_fn(pred, target)

    def training_step(self, batch, batch_idx):
        pred, _ = self(batch["waveform"])
        loss = self._safe_loss(pred, batch["target"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if batch_idx % 50 == 0:
            logger.info(f"Step {self.global_step} (epoch {self.current_epoch}, "
                        f"batch {batch_idx}): train_loss={loss.item():.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        pred, _ = self(batch["waveform"])
        loss = self._safe_loss(pred, batch["target"])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return {"pred": pred.detach(), "target": batch["target"].detach()}

    def validation_epoch_end(self, outputs):
        import numpy as np
        from sklearn.metrics import roc_auc_score

        preds = torch.cat([o["pred"] for o in outputs]).cpu().numpy()
        targets = torch.cat([o["target"] for o in outputs]).cpu().numpy()

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
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(PROJ_ROOT, "checkpoints"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to Lightning checkpoint to resume training from")
    # New args
    parser.add_argument("--val_sites", type=str, nargs="+", default=["S22", "S23"],
                        help="Site IDs to hold out for validation (e.g. S22 S23)")
    parser.add_argument("--soundscape_weight", type=float, default=3.0,
                        help="Upweight soundscape samples in training sampler")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="birdclef-2026",
                        help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run name (defaults to auto-generated)")
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # Patch config for our dataset
    htsat_config.classes_num = len(
        build_label_map(os.path.join(args.data_dir, "taxonomy.csv"))
    )
    htsat_config.loss_type = "clip_bce"
    htsat_config.enable_tscam = True

    # Build dataloaders
    train_loader, val_loader, label_map, num_classes = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=htsat_config.sample_rate,
        clip_duration=htsat_config.clip_samples / htsat_config.sample_rate,
        val_frac=args.val_frac,
        seed=args.seed,
        val_sites=args.val_sites,
        soundscape_weight=args.soundscape_weight,
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
    )

    # Save each seed's checkpoints in a subdirectory
    seed_save_dir = os.path.join(args.save_dir, f"seed{args.seed}")
    os.makedirs(seed_save_dir, exist_ok=True)

    # Callbacks
    ckpt_callback = ModelCheckpoint(
        dirpath=seed_save_dir,
        filename="birdclef-htsat-{epoch:02d}-{val_macro_auc:.4f}",
        monitor="val_macro_auc",
        mode="max",
        save_top_k=3,
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
                "val_sites": args.val_sites,
                "soundscape_weight": args.soundscape_weight,
                "warmup_epochs": args.warmup_epochs,
                "model": "htsat-tiny",
            },
        )
        loggers.append(wandb_logger)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus,
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
