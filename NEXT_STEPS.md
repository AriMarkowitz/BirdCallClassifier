# BirdCLEF 2026 — Next Steps

## Done (v2 training)
- [x] Site-based validation split (S22, S23 held out) — no data leakage
- [x] WeightedRandomSampler upweighting soundscape data 3x
- [x] Lower LR (3e-5) with CosineAnnealingWarmRestarts
- [x] Wandb logging (train_loss, val_loss, val_macro_auc, LR)
- [x] 5-model ensemble (seeds 42, 123, 456, 789, 2026)

## High priority
- [ ] **Location metadata conditioning** — train.csv has lat/lon per recording. Could add a small MLP that encodes (lat, lon) and fuses with HTSAT latent_output before the classification head. Requires modifying HTSAT forward pass to accept auxiliary features.
- [ ] **Augmentation** — mixup, time-shift, Gaussian noise, SpecAugment (frequency/time masking). Soundscapes are noisy; augmenting clean train_audio to be more soundscape-like should reduce domain gap.
- [ ] **Test-time augmentation (TTA)** — average predictions over original + time-shifted versions of each 5s segment. Free accuracy at the cost of inference time (~2-3x).

## Medium priority
- [ ] **Focal loss** instead of BCELoss — down-weights easy negatives, may help with class imbalance (234 classes, most absent per segment).
- [ ] **Label smoothing** — reduce overconfidence, especially for single-label train_audio samples.
- [ ] **Larger model** — HTSAT-base (htsat_dim=128, depths=[2,2,12,2]) if training budget allows.
- [ ] **Pseudo-labeling** — use current model to label unlabeled soundscape segments, add confident predictions to training set.
- [ ] **Cross-validation ensemble** — instead of random seeds, train on different site-based folds for more diverse ensemble members.

## Low priority / experimental
- [ ] **Knowledge distillation** from a larger teacher model
- [ ] **Multi-scale inference** — process segments at multiple window sizes (3s, 5s, 10s)
- [ ] **Threshold tuning** — optimize per-class thresholds on validation set instead of using raw probabilities
