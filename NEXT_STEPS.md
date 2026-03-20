# BirdCLEF 2026 — Next Steps

## Done (current training)
- [x] Site-based validation split — no data leakage
- [x] WeightedRandomSampler upweighting soundscape data
- [x] Wandb logging (train_loss, val_loss, val_macro_auc)
- [x] 5-model ensemble with k-fold cross-validation
- [x] Curriculum training (soundscape weight ramps 1.0 → 3.0)
- [x] Secondary labels in train_audio (multi-hot targets)
- [x] Label smoothing (ε=0.1)
- [x] All taxonomy classes trained (birds, insects, amphibians, mammals, reptiles)

## High priority
- [ ] **Audio augmentation** — mixup, Gaussian noise, SpecAugment (frequency/time masking). Soundscapes are noisy; augmenting clean train_audio to be more soundscape-like should reduce domain gap. This is likely the biggest remaining gap between val AUC (0.999) and Kaggle score (0.848).
- [ ] **Pseudo-labeling unlabeled soundscapes** — only a subset of train_soundscapes are labeled. Use current model to pseudo-label the rest, add high-confidence predictions to training set. This would massively increase training data for soundscape-only species (insect sonotypes, etc.).
- [ ] **Focal loss** instead of BCELoss — down-weights easy negatives (230+ absent species per segment), focuses gradients on hard cases.
- [ ] **Test-time augmentation (TTA)** — average predictions over original + time-shifted versions of each 5s segment. Free accuracy at the cost of inference time.

## Contextual awareness (metadata features)
These features could help the model make more informed predictions but require careful implementation to avoid spurious correlations. All of these are available in soundscape filenames (`BC2026_Test_0001_S05_20250227_010002.ogg`):

- [ ] **Spatial awareness (location)** — train.csv has lat/lon per recording but soundscapes do not have per-recording coordinates. Attempted: spatial embedding injected into HTSAT patch embeddings (location encoder MLP maps [lat, lon] → embed_dim, added as bias). Reverted because soundscapes lack coordinates, causing the model to learn a dataset-type signal rather than geography. **Alternative approach**: post-processing geographic prior — build species×location co-occurrence from train.csv, downweight predictions for species never observed near the Pantanal. No retraining needed.
- [ ] **Time of day awareness** — nocturnal vs diurnal species have very different calling patterns. Time is encoded in soundscape filenames (UTC). Could be injected as a cyclical embedding (sin/cos encoding of hour) similar to the location approach.
- [ ] **Seasonality (day of year)** — some species are migratory or have seasonal calling patterns. Date is in soundscape filenames. Could use cyclical encoding (sin/cos of day-of-year / 365).
- [ ] **Site-specific priors** — different recording sites (S03, S05, S08, etc.) have different species communities. Could learn site embeddings or build per-site species priors from labeled soundscapes.

## Medium priority
- [ ] **Larger model** — HTSAT-base (htsat_dim=128, depths=[2,2,12,2]) if training budget allows.
- [ ] **Cross-validation ensemble with hyperparameter diversity** — vary label smoothing, augmentation strength, or learning rate across folds for more diverse ensemble members.
- [ ] **Threshold tuning** — optimize per-class thresholds on validation set instead of using raw probabilities.

## Low priority / experimental
- [ ] **Knowledge distillation** from a larger teacher model
- [ ] **Multi-scale inference** — process segments at multiple window sizes (3s, 5s, 10s)
