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

## Ranked ideas (best → worst)

### Tier 1 — High impact, low risk
1. **Audio augmentation** — mixup, Gaussian noise, SpecAugment (frequency/time masking). Soundscapes are noisy; augmenting clean train_audio to be more soundscape-like should reduce domain gap. This is likely the biggest remaining gap between val AUC (0.999) and Kaggle score (0.848).
2. **Two-stage temporal fusion** — see detailed plan below. Injects time-of-day and seasonality metadata via a two-stage training approach. Well-designed to avoid the data-asymmetry problem that killed spatial embeddings.
3. **Focal loss** instead of BCELoss — down-weights easy negatives (230+ absent species per segment), focuses gradients on hard cases. Easy to implement, drop-in replacement.

### Tier 2 — High potential, more effort
4. **Pseudo-labeling unlabeled soundscapes** — only a subset of train_soundscapes are labeled. Use current model to pseudo-label the rest, add high-confidence predictions to training set. Would massively increase training data for soundscape-only species (insect sonotypes, etc.).
5. **Bird-MAE-Base as backbone replacement** — self-supervised ViT-B/16 (85M params, 768-dim embeddings) pretrained via masked autoencoder on BirdSet ([model](https://huggingface.co/DBD-research-group/Bird-MAE-Base)). Bird-specific representations should outperform AudioSet-general HTSAT features. Larger refactor — new spectrogram pipeline, model loading, inference notebook. ~2.5x larger than HTSAT-tiny so need to verify Kaggle inference timing. Must bundle custom model code in dataset upload (`trust_remote_code=True`). **Suggested first step**: freeze Bird-MAE, train linear probe on soundscape val set, compare AUC to current HTSAT. Prototype on a separate git branch.
6. **BirdSet XCM as extra pretraining data** — 89k focal recordings, 409 species, 89GB. Fine-tune HTSAT on XCM first (stage 0), then continue with BirdCLEF fine-tuning (stage 1). Stays within current architecture. Need to check species overlap with our 234 BirdCLEF classes.

### Tier 3 — Quick wins / incremental
7. **Threshold tuning** — optimize per-class thresholds on validation set instead of using raw probabilities. Quick to implement, no retraining.
8. **Spatial/site post-processing priors** — build species×location co-occurrence from train.csv, downweight predictions for species never observed near the Pantanal. No retraining needed. Site-specific priors from labeled soundscapes could also help.
9. **Test-time augmentation (TTA)** — average predictions over original + time-shifted versions of each 5s segment. Free accuracy but tight on Kaggle's 90-min CPU inference budget.
10. **Cross-validation ensemble with hyperparameter diversity** — vary label smoothing, augmentation strength, or learning rate across folds for more diverse ensemble members. Diminishing returns over current uniform ensemble.

### Tier 4 — Blocked or speculative
11. **Larger HTSAT model** — HTSAT-base (htsat_dim=128, depths=[2,2,12,2]) if training budget allows. More compute for marginal gain.
12. **Google Perch v2 (EfficientNet-B3, 12M params)** — Google Research bioacoustics model, ~15k species, 1536-dim embeddings ([model](https://huggingface.co/cgeorgiaw/Perch)). State-of-the-art on benchmarks but **blocked**: TensorFlow-only, GPU-only (no CPU variant yet), no Kaggle offline install. Worth revisiting if a PyTorch port or ONNX export becomes available.
13. **AudioProtoPNet (ConvNeXt-base, 0.3B params)** — BirdSet XCL trained ([model](https://huggingface.co/DBD-research-group/AudioProtoPNet-20-BirdSet-XCL)). Too large for Kaggle CPU inference. Lower priority unless distilled.
14. **Knowledge distillation** from a larger teacher model — speculative, no clear teacher available yet.
15. **Multi-scale inference** — process segments at multiple window sizes (3s, 5s, 10s). Experimental, adds inference cost.

---

## Two-stage temporal fusion (detailed plan)
Two-stage training to inject temporal metadata (time-of-day, seasonality) into predictions. The architecture includes temporal embedding inputs from the start, but trains them in two phases to handle the data asymmetry (train_audio lacks date/time, soundscapes have it).

- [ ] **Stage 1: Full model with zeroed temporal inputs** — fine-tune HTSAT backbone + classification head + temporal MLP together on all data (train_audio + soundscapes). Temporal features are set to zeros for all samples. The model learns audio classification while the temporal pathway learns to be a no-op. This means the head is jointly trained with the backbone from the start, avoiding distribution shift when temporal features are introduced later.
- [ ] **Stage 2: Train temporal MLP on real features** — freeze HTSAT backbone, unfreeze temporal MLP only, train on soundscape data only (where real temporal metadata is available). The MLP learns to modulate predictions based on time-of-day and seasonality. Since the head already works well with zero temporal input, stage 2 only needs to learn the temporal delta.
  - **Temporal features** (4-dim): `[sin(2π·hour/24), cos(2π·hour/24), sin(2π·doy/365), cos(2π·doy/365)]` parsed from soundscape filenames (e.g. `_20211231_201500` → Dec 31, 20:15 UTC).
  - **Architecture**: temporal MLP maps 4-dim → embed_dim, added to HTSAT audio embeddings before the classification head. Only the temporal MLP is trained in stage 2; backbone + head weights stay frozen.
  - **Training data**: soundscape segments only (these have temporal metadata). Same k-fold split as stage 1 for consistent validation.
  - **Inference**: parse date/time from test soundscape filenames, run HTSAT + temporal MLP. Adds negligible inference cost — should not hit the 90-minute Kaggle time limit even with ensemble.
  - **Considerations**: keep the MLP small to avoid overfitting on the limited soundscape training data. Regularize (dropout, weight decay). Monitor whether temporal features actually improve val AUC vs stage 1 alone.

Any backbone replacement (Bird-MAE, Perch, etc.) should be prototyped on a separate git branch to avoid disrupting the current HTSAT pipeline.
