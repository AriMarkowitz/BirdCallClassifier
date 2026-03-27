"""
Fine-tune BirdSet EfficientNet on BirdCLEF 2026 for multi-label classification.

Key features:
  - BirdSet EfficientNet-B1 backbone via Hugging Face
  - Variable-length/full-file audio input (default) padded per batch
  - Teacher-student distillation to reduce overfitting under domain shift
  - Moderate SuMix + focal/BCE loss + class-balanced sampling
"""

import os
import sys
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_ROOT, "src"))

from model import BirdCLEFModel
from dataset import get_dataloaders, build_label_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Binary focal loss on logits for multi-label classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        prob = torch.sigmoid(logits)
        pt = torch.where(target >= 0.5, prob, 1.0 - prob)
        alpha_t = torch.where(target >= 0.5, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * (1.0 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class BirdCLEFWrapper(pl.LightningModule):
    """Lightning wrapper with supervised loss + optional distillation."""

    def __init__(
        self,
        model,
        num_classes,
        learning_rate=1e-4,
        max_epochs=30,
        loss_type="bce",
        focal_alpha=0.25,
        focal_gamma=2.0,
        mixup_alpha=0.0,
        distill_weight=0.0,
        distill_temperature=2.0,
        teacher_model=None,
        idx_to_label=None,
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.mixup_alpha = mixup_alpha
        self.distill_weight = distill_weight
        self.distill_temperature = distill_temperature
        self.teacher_model = teacher_model
        self.idx_to_label = idx_to_label or {}

        if loss_type == "focal":
            self.loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            logger.info(f"Using FocalLoss (alpha={focal_alpha}, gamma={focal_gamma})")
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
            logger.info("Using BCEWithLogitsLoss")

        if mixup_alpha > 0:
            logger.info(f"SuMix enabled: alpha={mixup_alpha}")
        if teacher_model is not None and distill_weight > 0:
            logger.info(
                f"Distillation enabled: weight={distill_weight}, "
                f"temperature={distill_temperature}"
            )

    def forward(self, x):
        return self.model(x)["clipwise_output"]

    def _safe_supervised_loss(self, logits, target):
        with torch.amp.autocast("cuda", enabled=False):
            logits = logits.float()
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
            target = torch.nan_to_num(target.float(), nan=0.0, posinf=1.0, neginf=0.0)
            target = target.clamp(0.0, 1.0)
            return self.loss_fn(logits, target)

    def _current_distill_weight(self):
        """Cosine decay: full distill_weight at epoch 0, tapering to 0 by final epoch."""
        progress = self.current_epoch / max(self.max_epochs - 1, 1)
        return self.distill_weight * 0.5 * (1.0 + np.cos(np.pi * progress))

    def _distill_loss(self, student_logits, teacher_logits):
        """KL-divergence distillation with temperature-scaled softmax.

        Softmax over the BirdSet class dimension is more principled than
        independent sigmoids: it captures relative confidence across all
        ~9.7K BirdSet species rather than treating each as independent.
        """
        t = float(self.distill_temperature)
        with torch.amp.autocast("cuda", enabled=False):
            s = torch.nan_to_num(student_logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
            te = torch.nan_to_num(teacher_logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
            student_log_prob = nn.functional.log_softmax(s / t, dim=-1)
            teacher_prob = nn.functional.softmax(te / t, dim=-1)
            # KL(teacher || student), scaled by T^2 to keep gradient magnitude
            # consistent across temperature values (Hinton et al. 2015)
            return (t * t) * nn.functional.kl_div(
                student_log_prob, teacher_prob, reduction="batchmean",
            )

    def _sumix(self, waveform, target):
        """Additive waveform mixing with soft target max-merge."""
        if self.mixup_alpha <= 0:
            return waveform, target

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        lam = max(lam, 1.0 - lam)

        batch_size = waveform.size(0)
        perm = torch.randperm(batch_size, device=waveform.device)

        waveform_mixed = lam * waveform + (1.0 - lam) * waveform[perm]
        target_mixed = torch.maximum(target, (1.0 - lam) * target[perm])
        return waveform_mixed, target_mixed

    def training_step(self, batch, batch_idx):
        waveform = batch["waveform"]
        target = batch["target"]
        distill_active = self.teacher_model is not None and self.distill_weight > 0

        if self.training and self.mixup_alpha > 0:
            waveform, target = self._sumix(waveform, target)

        output_dict = self.model(
            waveform,
            apply_spec_aug=self.training,
            return_mel=distill_active,
        )
        pred = output_dict["clipwise_output"]
        logits = output_dict["logits"]
        supervised_loss = self._safe_supervised_loss(logits, target)

        total_loss = supervised_loss
        distill_loss = torch.tensor(0.0, device=waveform.device)

        if distill_active:
            # Reuse the same (clean) mel chunk the student used (no extra STFT)
            mel_clean = output_dict["mel_for_teacher"]
            student_logits = output_dict["student_birdset_logits"]
            with torch.no_grad():
                teacher_logits = self.teacher_model(pixel_values=mel_clean).logits
            distill_loss = self._distill_loss(student_logits, teacher_logits)
            dw = self._current_distill_weight()
            if torch.isfinite(distill_loss):
                total_loss = supervised_loss + dw * distill_loss
            else:
                logger.warning(f"Step {self.global_step}: distill_loss is NaN/Inf, using supervised only")
                distill_loss = torch.tensor(0.0, device=waveform.device)
                total_loss = supervised_loss

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_supervised_loss", supervised_loss, on_step=True, on_epoch=True)
        if self.teacher_model is not None and self.distill_weight > 0:
            self.log("train_distill_loss", distill_loss, on_step=True, on_epoch=True)
            self.log("distill_weight_current", self._current_distill_weight(), on_step=False, on_epoch=True)

        if batch_idx % 50 == 0:
            logger.info(
                f"Step {self.global_step} (epoch {self.current_epoch}, "
                f"batch {batch_idx}): train_loss={total_loss.item():.4f}"
            )
        return total_loss

    def validation_step(self, batch, batch_idx):
        output_dict = self.model(batch["waveform"], apply_spec_aug=False)
        pred = output_dict["clipwise_output"]
        logits = output_dict["logits"]
        loss = self._safe_supervised_loss(logits, batch["target"])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        if not hasattr(self, "_val_preds"):
            self._val_preds = []
            self._val_targets = []
        self._val_preds.append(pred.detach().float().cpu())
        self._val_targets.append(batch["target"].detach().float().cpu())

    def on_validation_epoch_end(self):
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
        logger.info(
            f"Epoch {self.current_epoch}: val_macro_auc={macro_auc:.4f} "
            f"({len(per_class_auc)} evaluable classes)"
        )

        if per_class_auc:
            sorted_aucs = sorted(per_class_auc.items(), key=lambda x: x[1])
            worst10 = sorted_aucs[:10]
            best10 = sorted_aucs[-10:]
            logger.info("  Worst 10 AUC: " + ", ".join(f"{sp}={auc:.3f}" for sp, auc in worst10))
            logger.info("  Best 10 AUC: " + ", ".join(f"{sp}={auc:.3f}" for sp, auc in best10))

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.05,
        )
        # Cosine annealing with linear warmup (5% of training)
        warmup_epochs = max(1, self.max_epochs // 20)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs - warmup_epochs, eta_min=1e-7,
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs,
        )
        combined = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, scheduler], milestones=[warmup_epochs],
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": combined, "interval": "epoch"}}


def _build_teacher(model_name):
    """Create frozen BirdSet teacher model for distillation."""
    from transformers import EfficientNetForImageClassification

    teacher = EfficientNetForImageClassification.from_pretrained(
        model_name,
        num_channels=1,
        ignore_mismatched_sizes=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BirdSet EfficientNet on BirdCLEF 2026")
    parser.add_argument("--data_dir", type=str, default=os.path.join(PROJ_ROOT, "data"))
    parser.add_argument(
        "--birdset_model_name",
        type=str,
        default="DBD-research-group/EfficientNet-B1-BirdSet-XCL",
        help="HF model id for BirdSet EfficientNet backbone",
    )
    parser.add_argument("--max_time_frames", type=int, default=768)
    parser.add_argument("--chunk_hop_frames", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--precision", type=str, default="16",
                        choices=["16", "bf16", "32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save_dir", type=str, default=os.path.join(PROJ_ROOT, "checkpoints"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default=None)

    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--loss", type=str, default="focal", choices=["bce", "focal"])
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
    parser.add_argument("--mix_prob", type=float, default=0.0)
    parser.add_argument("--mixup_alpha", type=float, default=0.0)

    parser.add_argument("--distill", dest="distill", action="store_true")
    parser.add_argument("--no_distill", dest="distill", action="store_false")
    parser.set_defaults(distill=False)
    parser.add_argument("--distill_weight", type=float, default=0.15)
    parser.add_argument("--distill_temperature", type=float, default=2.0)

    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--valid_regions", type=str, default=None)
    parser.add_argument("--pseudo_labels", type=str, default=None)
    parser.add_argument("--balance_alpha", type=float, default=0.5)

    parser.add_argument("--min_duration", type=float, default=3.0)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--full_files", dest="full_files", action="store_true")
    parser.add_argument("--no_full_files", dest="full_files", action="store_false")
    parser.set_defaults(full_files=True)

    args = parser.parse_args()

    if args.precision == "16-mixed":
        args.precision = "16"
    elif args.precision == "bf16-mixed":
        args.precision = "bf16"

    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    label_map = build_label_map(os.path.join(args.data_dir, "taxonomy.csv"))
    num_classes = len(label_map)

    logger.info("=== BirdSet EfficientNet training ===")
    logger.info(f"BirdSet backbone: {args.birdset_model_name}")
    logger.info(f"Variable-length input: {args.min_duration}s - {args.max_duration}s")
    if args.full_files:
        logger.info("Full-file mode enabled for train_audio (no cropping)")

    train_loader, val_loader, label_map, num_classes, n_train_audio = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=32000,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
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
        full_files=args.full_files,
    )
    logger.info(f"Classes: {num_classes}, Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = BirdCLEFModel(
        num_classes=num_classes,
        sample_rate=32000,
        birdset_model_name=args.birdset_model_name,
        max_time_frames=args.max_time_frames,
        chunk_hop_frames=args.chunk_hop_frames,
        pretrained=True,
    )
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    teacher = _build_teacher(args.birdset_model_name) if args.distill else None

    idx_to_label = {v: k for k, v in label_map.items()}
    wrapper = BirdCLEFWrapper(
        model=model,
        num_classes=num_classes,
        learning_rate=args.lr,
        max_epochs=args.max_epochs,
        loss_type=args.loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        mixup_alpha=args.mixup_alpha,
        distill_weight=args.distill_weight,
        distill_temperature=args.distill_temperature,
        teacher_model=teacher,
        idx_to_label=idx_to_label,
    )

    run_id = args.run_id or f"seed{args.seed}"
    run_save_dir = os.path.join(args.save_dir, run_id)
    os.makedirs(run_save_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {run_save_dir}")

    ckpt_callback = ModelCheckpoint(
        dirpath=run_save_dir,
        filename="birdclef-birdset-{epoch:02d}-{val_macro_auc:.4f}",
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
                "fold": args.fold,
                "n_folds": args.n_folds,
                "birdset_model_name": args.birdset_model_name,
                "max_time_frames": args.max_time_frames,
                "chunk_hop_frames": args.chunk_hop_frames,
                "loss": args.loss,
                "focal_alpha": args.focal_alpha,
                "focal_gamma": args.focal_gamma,
                "multi_mix": args.multi_mix,
                "mix_prob": args.mix_prob,
                "mixup_alpha": args.mixup_alpha,
                "distill": args.distill,
                "distill_weight": args.distill_weight,
                "distill_temperature": args.distill_temperature,
                "pseudo_labels": args.pseudo_labels,
                "balance_alpha": args.balance_alpha,
                "min_duration": args.min_duration,
                "max_duration": args.max_duration,
                "full_files": args.full_files,
                "n_train_audio": n_train_audio,
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
        precision=args.precision,
        gradient_clip_val=1.0,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(wrapper, train_loader, val_loader, ckpt_path=args.resume_from)


if __name__ == "__main__":
    main()
