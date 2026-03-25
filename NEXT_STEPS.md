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
- [ ] **MixUp / Sumix** — test label-aware waveform or spectrogram mixing beyond simple overlay. Compare classic MixUp against Sumix-style stronger sample mixing to improve robustness on overlapping species and noisy backgrounds.
- [ ] **Soundscape background injection** — mix a random crop from the 10,593 unlabeled soundscape files as background noise behind clean train_audio clips. This is the most realistic noise source possible — same equipment, same sites, same ambient conditions as test data. ~5GB of free noise available in `data/train_soundscapes/`.
- [ ] **SpecAugment (aggressive)** — increase frequency/time masking beyond current defaults. Force model to be robust to partial information.
- [ ] **FilterAugment** — apply random band-wise spectral shaping / filtering to mimic microphone, habitat, and distance variation. Useful for domain robustness without changing labels.
- [ ] **Time shift** — randomly roll or offset clips within the 5s window so detections are less position-dependent. Cheap augmentation and also useful to mirror test-time offset variability.
- [ ] **Pitch shift (light)** — small semitone perturbations only (e.g. ±0.5 to ±1.0 semitones) to improve robustness while avoiding unrealistic bird vocal transformations.
- [ ] **Gaussian noise / gain variation** — random SNR noise injection and volume scaling for additional robustness.

### 2. Iterative pseudo-labeling of unlabeled soundscapes
Only 66 of 10,658 soundscape files have labels (1,478 segments). The other 10,593 files are unlabeled — this is a massive untapped resource. Past winners treated this as the #1 lever.

- [ ] **Round 1: Generate pseudo-labels** — run current best model on all unlabeled soundscapes, segment into 5s windows, predict species probabilities. Filter by confidence threshold (e.g. keep predictions > 0.8).
- [ ] **Round 2: Retrain with pseudo-labels** — add high-confidence pseudo-labeled segments to training set. Use soft labels (raw probabilities) instead of hard labels to reduce noise propagation.
- [ ] **Round 3+: Iterate** — retrain → relabel → retrain. Each round improves the model's soundscape predictions, producing better pseudo-labels for the next round. Typically 2-3 rounds before diminishing returns.
- [ ] **Pseudo-label filtering** — track prediction confidence across rounds. Remove samples where the model flip-flops (unstable predictions). Weight pseudo-labeled samples lower than ground-truth labeled samples in the loss.

### 3. Focal loss
- [ ] **Replace BCELoss with focal loss** — down-weights easy negatives (230+ absent species per segment), focuses gradients on hard positives. Complements pseudo-labeling since pseudo-labels are noisier than ground truth. Drop-in replacement, α=0.25, γ=2.0 as starting point.
- [ ] **Focal loss ablation** — compare BCE vs focal loss under the current Bird-MAE setup, especially once stronger augmentation is enabled. Check whether focal improves rare/quiet species recall without destabilizing calibration.

### 4. Per-label error analysis + class imbalance audit
- [ ] **Best / worst label analysis** — compute per-class metrics (AUC, AP, recall at fixed precision, calibration) and rank labels from easiest to hardest. Break this out separately for birds, insects, amphibians, etc.
- [ ] **Performance vs training frequency** — join per-label metrics with number of training examples, number of positive soundscape segments, and site coverage. Check whether weak labels are mostly just low-resource labels or whether some abundant classes are still failing due to confusion/domain shift.
- [ ] **Confusion and co-occurrence review** — inspect which labels are commonly over-predicted, under-predicted, or confused with acoustically similar species / taxa. Use this to guide augmentation, thresholding, and targeted data collection.
- [ ] **Rare-label validation slice** — maintain a reporting view focused on low-resource classes so leaderboard improvements are not driven only by already-common species.

### 5. Bird-MAE-Base backbone replacement
- [ ] **Replace HTSAT with Bird-MAE-Base** — self-supervised ViT-B/16 (85M params, 768-dim embeddings) pretrained via masked autoencoder on BirdSet's 9.7k species ([model](https://huggingface.co/DBD-research-group/Bird-MAE-Base)). Bird-specific representations should outperform AudioSet-general HTSAT for fine-grained species discrimination. Requires new spectrogram pipeline (HuggingFace feature extractor), new model wrapper, new inference notebook. Must bundle custom model code in Kaggle dataset upload (`trust_remote_code=True`). **Prototype on separate branch.** ~2.5x larger than HTSAT-tiny — need to verify it fits in Kaggle's 90-min CPU inference window.

---

## Tier 2 — High potential, more effort

### 6. Longer-context SED-style training
- [ ] **Train on longer windows** — instead of 10s clips, train on 30-60s windows with frame-level (SED) predictions. Past winners found that longer context helps the model learn temporal patterns of species calls within a soundscape. Requires adjusting the HTSAT input size or using a sliding window with aggregation.

### 7. Transfer learning from larger bird-audio sources
- [ ] **BirdSet XCM pretraining (stage 0)** — 89k focal recordings, 409 species, 89GB. Fine-tune backbone on XCM first, then on BirdCLEF data. Stays within current architecture. Need to check species overlap with our 234 classes.
- [ ] **BirdSet XCL (full dataset)** — 528k recordings, 9.7k species, 484GB. Maximum pretraining data but significant storage/compute requirements.
- [ ] **Add geographically relevant non-bird data** — incorporate insects / amphibians (and possibly other taxa present in the competition taxonomy) from similar South American / Pantanal-adjacent soundscapes. Goal: improve negative class modeling, reduce false bird positives, and better match the real acoustic background distribution at test time.

### 8. Class-balanced sampling / reweighting
- [ ] **Balanced label sampling** — test a sampler that increases exposure of underrepresented labels so all classes are seen more uniformly during training. Important for multi-label data: balance by positive-label coverage, not just by clip count.
- [ ] **Per-class loss weighting** — compare sampler-based balancing against per-class positive weights in the loss. Some labels may benefit more from loss reweighting than aggressive resampling.
- [ ] **Hybrid sampling scheme** — keep some natural-frequency batches for calibration while reserving part of each epoch for rare-label upsampling. Goal: reduce imbalance without making the train distribution too unrealistic.
- [ ] **Taxon-aware sampling** — ensure insects / amphibians / mammals / reptiles are not drowned out by abundant bird labels when constructing batches.

### 9. Global NMF latent basis + learned projection
Build a fixed global latent dictionary from representative spectrogram data, then train a small CNN to project any clip into that latent space at inference time.

**Why this architecture:** Running NMFk per clip is both expensive and semantically wrong — without a shared dictionary, each clip's latent dimensions represent different spectral patterns, so component 3 in one clip (e.g. insect drone) has no relation to component 3 in another (e.g. wind). A global W gives every component a fixed meaning across all clips ("component 12 always captures this frequency pattern"), making the activations H comparable and useful as features. However, we can't run NMFk on the entire concatenated training audio — it's too large. So we learn W from a representative subsample, generate per-clip H targets with W fixed (cheap NNLS solve, not full NMFk), then train a CNN to predict those targets directly from spectrograms. At inference time, only the CNN runs — no NMF at all.

- [ ] **Step 1: Learn global dictionary W** — take a large representative sample of training soundscapes, convert to nonneg representation (mel or PCEN spectrograms), concatenate enough time for NMFk to see broad backgrounds + bird calls. Run NMFk once on that pooled matrix to choose k and learn W ∈ R+^{f×k}. Output: fixed preprocessing params (sr, FFT, hop, mel bins, normalization) and fixed global dictionary W.
- [ ] **Step 2: Project clips into fixed basis** — for each 5s clip, compute the same spectrogram and solve only for activation matrix H in V ≈ WH with W fixed (no per-clip NMFk). Extract fixed-size latent features from H: mean, max, std over time per component; sparsity; top temporal peaks per component. One latent feature vector per clip.
- [ ] **Step 3: Train small CNN encoder to approximate latent features** — train a small CNN (not MLP — input has time-frequency structure) to predict the NMF latent summary vector from Step 2 directly from the clip spectrogram. Target is the NMF latent vector, not the class label. This replaces the per-clip NNLS solve with a fast learned forward pass at inference time.
- [ ] **Step 4: Fuse latent branch with main classifier** — two options to evaluate:
  - **Option A: Feature concatenation** — concatenate (a) learned deep features from main backbone and (b) CNN-predicted NMF latent features before the classification head. NMF branch stays fully independent of Bird-MAE — easy to train, debug, and ablate separately. Adds minor inference cost (MobileNetV3-Small side branch).
  - **Option B: Auxiliary loss (no inference cost)** — add a second head to Bird-MAE that predicts the NMF latent vector as a regularization signal during training, then discard that head at inference. Pushes Bird-MAE's internal representations to capture spectral structure NMF finds, without adding any inference cost. More exotic — try Option A first.
- [ ] **Step 5: Ablation ladder** — test in order: (1) baseline, (2) baseline + exact NMF features, (3) baseline + CNN-predicted latent features, (4) optionally both. Judge by val performance and robustness on noisy soundscapes. Stop early if no improvement.

### 10. Post-processing
- [ ] **Threshold tuning** — optimize per-class thresholds on validation set instead of using 0.5 cutoff. Quick win, no retraining.
- [ ] **Spatial/site priors** — build species×location co-occurrence from train.csv, downweight predictions for species never observed near the Pantanal. No retraining needed.
- [ ] **Test-time augmentation (TTA)** — average predictions over original + time-shifted segments. Free accuracy but tight on inference budget.

---

## Tier 3 — Deprioritized / blocked

- [x] **Two-stage temporal fusion** — implemented and removed. Temporal MLP overfits on small soundscape training set, did not improve Kaggle score. Code removed from train.py and dataset.py.
- [ ] **Cross-validation ensemble diversity** — vary hyperparameters across folds. Diminishing returns.
- [ ] **Google Perch v2** — blocked by TensorFlow-only, GPU-only. Worth revisiting if PyTorch port appears.
- [ ] **AudioProtoPNet** — 0.3B params, too large for Kaggle CPU inference.
- [ ] **Knowledge distillation** — no clear teacher model yet.
- [ ] **Multi-scale inference** — experimental, adds inference cost.
