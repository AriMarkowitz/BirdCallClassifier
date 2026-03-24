"""
Fine-tune Bird-MAE-Base on BirdCLEF 2026 for multi-label classification.

Usage:
    python src/train.py --checkpoint path/to/bird-mae-base.safetensors

Bird-MAE-Base is a ViT-B/16 (85M params) pretrained via masked autoencoder
on BirdSet's 9.7k bird species. We add a linear classification head and
fine-tune on BirdCLEF 2026 training data.
"""

import os
import sys
import argparse
import logging
import math

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_ROOT, "src"))

from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import get_dataloaders, get_datasets, build_label_map, TEMPORAL_DIM
from bird_mae.model import BirdMAEModel
from bird_mae.config import BirdMAEConfig
from bird_mae.feature_extractor import BirdMAEFeatureExtractor
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BirdMAEWrapper(pl.LightningModule):
    """
    Lightning wrapper for Bird-MAE fine-tuning on BirdCLEF.

    The model:
    1. Converts raw waveforms to fbank mel spectrograms (on GPU)
    2. Runs Bird-MAE encoder to get 768-dim embeddings
    3. Applies linear classification head -> num_classes logits
    4. Sigmoid -> multi-label probabilities
    """

    def __init__(self, backbone, num_classes, learning_rate=1e-4,
                 warmup_epochs=1, max_epochs=30, n_train_audio=0):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.n_train_audio = n_train_audio
        self.embed_dim = backbone.config.embed_dim  # 768

        self.head = nn.Linear(self.embed_dim, num_classes)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.feature_extractor = BirdMAEFeatureExtractor()

    def forward(self, waveform):
        """
        Args:
            waveform: (B, samples) raw audio at 32kHz

        Returns:
            probs: (B, num_classes) sigmoid probabilities
        """
        mel = self.feature_extractor(waveform)  # (B, 1, 512, 128)
        mel = mel.to(self.device)
        embeddings = self.backbone(mel)  # (B, 768)
        logits = self.head(embeddings)  # (B, num_classes)
        return logits

    def _safe_loss(self, logits, target):
        return self.loss_fn(logits, target)

    def training_step(self, batch, batch_idx):
        logits = self(batch["waveform"])
        loss = self._safe_loss(logits, batch["target"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if batch_idx % 50 == 0:
            logger.info(f"Step {self.global_step} (epoch {self.current_epoch}, "
                        f"batch {batch_idx}): train_loss={loss.item():.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["waveform"])
        loss = self._safe_loss(logits, batch["target"])
        probs = torch.sigmoid(logits)
        return {"val_loss": loss, "preds": probs.detach().cpu(), "targets": batch["target"].detach().cpu()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        preds = torch.cat([x["preds"] for x in outputs]).numpy()
        targets = torch.cat([x["targets"] for x in outputs]).numpy()

        # Macro AUC (skip classes not present in this fold's val set)
        try:
            per_class_auc = []
            for i in range(self.num_classes):
                if targets[:, i].sum() > 0 and targets[:, i].sum() < len(targets):
                    per_class_auc.append(roc_auc_score(targets[:, i], preds[:, i]))
            macro_auc = np.mean(per_class_auc) if per_class_auc else 0.0
        except Exception:
            macro_auc = 0.0

        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_macro_auc", macro_auc, prog_bar=True)
        logger.info(f"Epoch {self.current_epoch}: val_macro_auc={macro_auc:.4f}")

    def configure_optimizers(self):
        # Different LR for backbone vs head
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.head.parameters())

        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": self.learning_rate},
            {"params": head_params, "lr": self.learning_rate * 10},
        ], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05)

        return optimizer

    def on_train_epoch_start(self):
        """Curriculum learning: ramp soundscape weight over training."""
        sampler = None
        dl = self.trainer.train_dataloader
        if hasattr(dl, 'loaders'):
            dl = dl.loaders
        if isinstance(dl, dict):
            for key, val in dl.items():
                if hasattr(val, 'sampler'):
                    dl = val
                    break
        if hasattr(dl, 'sampler'):
            s = dl.sampler
            if isinstance(s, WeightedRandomSampler):
                sampler = s

        if sampler is None:
            if self.current_epoch == 0:
                logger.warning("Curriculum: could not find WeightedRandomSampler, skipping")
            return

        progress = self.current_epoch / max(self.max_epochs - 1, 1)
        weight = 0.5 + 2.5 * progress

        n_total = len(sampler.weights)
        new_weights = torch.ones(n_total, dtype=torch.float64)
        new_weights[self.n_train_audio:] = weight
        sampler.weights = new_weights

        self.log("soundscape_weight", weight)
        logger.info(f"Epoch {self.current_epoch}: curriculum soundscape_weight={weight:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Bird-MAE on BirdCLEF 2026")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(PROJ_ROOT, "data"))
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to Bird-MAE pretrained checkpoint (.safetensors)")
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
    parser.add_argument("--soundscape_weight", type=float, default=3.0,
                        help="Upweight soundscape samples in training sampler")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="birdclef-2026",
                        help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run name")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Run ID for checkpoint subdir")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--fold", type=int, default=0,
                        help="Which fold to hold out for validation")
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    # Build dataloaders (5s clips for Bird-MAE)
    train_loader, val_loader, label_map, num_classes, n_train_audio = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=32000,
        clip_duration=5.0,  # Bird-MAE uses 5s clips
        val_frac=args.val_frac,
        seed=args.seed,
        soundscape_weight=args.soundscape_weight,
        label_smoothing=args.label_smoothing,
        n_folds=args.n_folds,
        fold=args.fold,
    )
    logger.info(f"Classes: {num_classes}, Train batches: {len(train_loader)}, "
                f"Val batches: {len(val_loader)}")

    # Curriculum schedule summary
    logger.info(f"Curriculum training: soundscape_weight ramps 0.50 -> 3.00 "
                f"linearly over {args.max_epochs} epochs")

    # Build model
    mae_config = BirdMAEConfig()
    if args.checkpoint and os.path.isfile(args.checkpoint):
        logger.info(f"Loading pretrained Bird-MAE from: {args.checkpoint}")
        backbone = BirdMAEModel.from_pretrained(args.checkpoint, config=mae_config)
    else:
        logger.warning("No checkpoint — training Bird-MAE from scratch (not recommended)")
        backbone = BirdMAEModel(mae_config)

    wrapper = BirdMAEWrapper(
        backbone, num_classes,
        learning_rate=args.lr,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        n_train_audio=n_train_audio,
    )

    total = sum(p.numel() for p in wrapper.parameters())
    trainable = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    logger.info(f"Model params: {total:,} total, {trainable:,} trainable")

    # Save checkpoints
    run_id = args.run_id or f"seed{args.seed}"
    run_save_dir = os.path.join(args.save_dir, run_id)
    os.makedirs(run_save_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {run_save_dir}")

    ckpt_callback = ModelCheckpoint(
        dirpath=run_save_dir,
        filename="birdclef-birdmae-{epoch:02d}-{val_macro_auc:.4f}",
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
                "soundscape_weight": args.soundscape_weight,
                "label_smoothing": args.label_smoothing,
                "fold": args.fold,
                "n_folds": args.n_folds,
                "model": "bird-mae-base",
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
