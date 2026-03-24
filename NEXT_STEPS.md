# BirdCLEF 2026 — Next Steps

## Done (current training)
- [x] Site-based validation split — no data leakage
- [x] WeightedRandomSampler upweighting soundscape data
- [x] Wandb logging (train_loss, val_loss, val_macro_auc)
- [x] 4-model ensemble with k-fold cross-validation
- [x] Curriculum training (soundscape weight ramps 1.0 → 3.0)
- [x] Secondary labels in train_audio (multi-hot targets)
- [x] Label smoothing (ε=0.1)
- [x] All taxonomy classes trained (birds, insects, amphibians, mammals, reptiles)
- [x] Two-stage temporal fusion (implemented but made Kaggle score worse — temporal MLP overfits on 1183 soundscape samples, deprioritized)

## Key problem: domain shift
Val AUC is 0.999 but Kaggle score is 0.848. The model memorizes clean focal recordings from train_audio but cannot generalize to noisy, polyphonic test soundscapes. Past BirdCLEF winners confirm this is THE problem to solve. Everything below is ordered by expected impact on closing this gap.

---

## Tier 1 — Proven winning strategies (from past BirdCLEF solutions)

### 1. Noise-robust training with strong augmentation
The model trains on clean single-species focal recordings but tests on noisy multi-species soundscapes. Augmentation bridges this domain gap.

- [ ] **Multi-species mixing** — randomly overlay 3-5 train_audio clips at varying volumes per sample. Union their multi-hot labels. Teaches the model to detect individual species in polyphonic audio. This is the competition-specific version of MixUp that past winners used.
- [ ] **Soundscape background injection** — mix a random crop from the 10,593 unlabeled soundscape files as background noise behind clean train_audio clips. This is the most realistic noise source possible — same equipment, same sites, same ambient conditions as test data. ~5GB of free noise available in `data/train_soundscapes/`.
- [ ] **SpecAugment (aggressive)** — increase frequency/time masking beyond current defaults. Force model to be robust to partial information.
- [ ] **Gaussian noise / gain variation** — random SNR noise injection and volume scaling for additional robustness.

### 2. Iterative pseudo-labeling of unlabeled soundscapes
Only 66 of 10,658 soundscape files have labels (1,478 segments). The other 10,593 files are unlabeled — this is a massive untapped resource. Past winners treated this as the #1 lever.

- [ ] **Round 1: Generate pseudo-labels** — run current best model on all unlabeled soundscapes, segment into 5s windows, predict species probabilities. Filter by confidence threshold (e.g. keep predictions > 0.8).
- [ ] **Round 2: Retrain with pseudo-labels** — add high-confidence pseudo-labeled segments to training set. Use soft labels (raw probabilities) instead of hard labels to reduce noise propagation.
- [ ] **Round 3+: Iterate** — retrain → relabel → retrain. Each round improves the model's soundscape predictions, producing better pseudo-labels for the next round. Typically 2-3 rounds before diminishing returns.
- [ ] **Pseudo-label filtering** — track prediction confidence across rounds. Remove samples where the model flip-flops (unstable predictions). Weight pseudo-labeled samples lower than ground-truth labeled samples in the loss.

### 3. Focal loss
- [ ] **Replace BCELoss with focal loss** — down-weights easy negatives (230+ absent species per segment), focuses gradients on hard positives. Complements pseudo-labeling since pseudo-labels are noisier than ground truth. Drop-in replacement, α=0.25, γ=2.0 as starting point.

### 4. Bird-MAE-Base backbone replacement
- [ ] **Replace HTSAT with Bird-MAE-Base** — self-supervised ViT-B/16 (85M params, 768-dim embeddings) pretrained via masked autoencoder on BirdSet's 9.7k species ([model](https://huggingface.co/DBD-research-group/Bird-MAE-Base)). Bird-specific representations should outperform AudioSet-general HTSAT for fine-grained species discrimination. Requires new spectrogram pipeline (HuggingFace feature extractor), new model wrapper, new inference notebook. Must bundle custom model code in Kaggle dataset upload (`trust_remote_code=True`). **Prototype on separate branch.** ~2.5x larger than HTSAT-tiny — need to verify it fits in Kaggle's 90-min CPU inference window.

---

## Tier 2 — High potential, more effort

### 5. Longer-context SED-style training
- [ ] **Train on longer windows** — instead of 10s clips, train on 30-60s windows with frame-level (SED) predictions. Past winners found that longer context helps the model learn temporal patterns of species calls within a soundscape. Requires adjusting the HTSAT input size or using a sliding window with aggregation.

### 6. Transfer learning from larger bird-audio sources
- [ ] **BirdSet XCM pretraining (stage 0)** — 89k focal recordings, 409 species, 89GB. Fine-tune backbone on XCM first, then on BirdCLEF data. Stays within current architecture. Need to check species overlap with our 234 classes.
- [ ] **BirdSet XCL (full dataset)** — 528k recordings, 9.7k species, 484GB. Maximum pretraining data but significant storage/compute requirements.

### 7. Post-processing
- [ ] **Threshold tuning** — optimize per-class thresholds on validation set instead of using 0.5 cutoff. Quick win, no retraining.
- [ ] **Spatial/site priors** — build species×location co-occurrence from train.csv, downweight predictions for species never observed near the Pantanal. No retraining needed.
- [ ] **Test-time augmentation (TTA)** — average predictions over original + time-shifted segments. Free accuracy but tight on inference budget.

---

## Tier 3 — Deprioritized / blocked

- [ ] **Two-stage temporal fusion** — implemented, made score worse. Temporal MLP overfits on small soundscape training set. May revisit after pseudo-labeling massively increases soundscape training data.
- [ ] **Cross-validation ensemble diversity** — vary hyperparameters across folds. Diminishing returns.
- [ ] **Google Perch v2** — blocked by TensorFlow-only, GPU-only. Worth revisiting if PyTorch port appears.
- [ ] **AudioProtoPNet** — 0.3B params, too large for Kaggle CPU inference.
- [ ] **Knowledge distillation** — no clear teacher model yet.
- [ ] **Multi-scale inference** — experimental, adds inference cost.
