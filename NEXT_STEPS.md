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
- [x] NMF latent feature integration into HTSAT forward pass (global dictionary W_k56, nmf_proj head)
- [x] Two-stage temporal fusion (implemented but made Kaggle score worse — temporal MLP overfits on 1183 soundscape samples, deprioritized)

## Key problem: domain shift
Val AUC was 0.999 but Kaggle score is 0.848. The inflated val AUC was because validation only used soundscape segments (most species absent → trivially high AUC). **Fixed:** validation now includes a stratified holdout of train_audio + soundscape fold, so val AUC reflects all species. The model memorizes clean focal recordings but cannot generalize to noisy, polyphonic test soundscapes. Past BirdCLEF winners confirm this is THE problem to solve. Everything below is ordered by expected impact on closing this gap.

---

## Tier 1 — Proven winning strategies (from past BirdCLEF solutions)

### 1. Noise-robust training with strong augmentation
The model trains on clean single-species focal recordings but tests on noisy multi-species soundscapes. Augmentation bridges this domain gap.

- [x] **Multi-species mixing** — implemented as `MultiSpeciesMixDataset` in dataset.py. Overlays 1-4 random train_audio clips at gains 0.1-0.7 with probability 0.7 per sample. Union of multi-hot labels. Enabled by default (`--multi_mix`, `--mix_prob 0.7`).
- [x] **MixUp / SuMix** — implemented as batch-level SuMix in train.py `_sumix()`. Shuffles batch and additively mixes waveforms with Beta-distributed lambda, soft-union labels. Enabled by default (`--mixup_alpha 0.4`).
- [ ] **Soundscape background injection** — mix a random crop from the 10,593 unlabeled soundscape files as background noise behind clean train_audio clips. This is the most realistic noise source possible — same equipment, same sites, same ambient conditions as test data. ~5GB of free noise available in `data/train_soundscapes/`.
- [x] **SpecAugment** — HTSAT already applies SpecAugmentation (time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2) during training. Built into the backbone.
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
- [x] **Replace BCELoss with focal loss** — implemented as `--loss focal` flag in train.py. Down-weights easy negatives (230+ absent species per segment), focuses gradients on hard positives. Default α=0.25, γ=2.0.
- [ ] **Focal loss ablation** — compare BCE vs focal loss under the current HTSAT+NMF setup, especially once stronger augmentation is enabled. Check whether focal improves rare/quiet species recall without destabilizing calibration.

### 4. Per-label error analysis + class imbalance audit
- [ ] **Best / worst label analysis** — compute per-class metrics (AUC, AP, recall at fixed precision, calibration) and rank labels from easiest to hardest. Break this out separately for birds, insects, amphibians, etc.
- [ ] **Performance vs training frequency** — join per-label metrics with number of training examples, number of positive soundscape segments, and site coverage. Check whether weak labels are mostly just low-resource labels or whether some abundant classes are still failing due to confusion/domain shift.
- [ ] **Confusion and co-occurrence review** — inspect which labels are commonly over-predicted, under-predicted, or confused with acoustically similar species / taxa. Use this to guide augmentation, thresholding, and targeted data collection.
- [ ] **Rare-label validation slice** — maintain a reporting view focused on low-resource classes so leaderboard improvements are not driven only by already-common species.

### 5. Bird-MAE-Base backbone (deprioritized)
- [ ] **Replace HTSAT with Bird-MAE-Base** — self-supervised ViT-B/16 (85M params, 768-dim embeddings) pretrained via masked autoencoder on BirdSet's 9.7k species ([model](https://huggingface.co/DBD-research-group/Bird-MAE-Base)). Potentially better representations but ~2.5x larger than HTSAT-tiny. Previously prototyped on `bird-mae-backbone` branch but deprioritized in favor of HTSAT+NMF approach. Would need to verify it fits in Kaggle's 90-min CPU inference window.

---

## Tier 2 — High potential, more effort

### 6. Longer-context SED-style training
- [ ] **Train on longer windows** — instead of 10s clips, train on 30-60s windows with frame-level (SED) predictions. Past winners found that longer context helps the model learn temporal patterns of species calls within a soundscape. Requires adjusting the HTSAT input size or using a sliding window with aggregation.

### 7. Transfer learning from larger bird-audio sources
- [ ] **BirdSet XCM pretraining (stage 0)** — 89k focal recordings, 409 species, 89GB. Fine-tune backbone on XCM first, then on BirdCLEF data. Stays within current architecture. Need to check species overlap with our 234 classes.
- [ ] **BirdSet XCL (full dataset)** — 528k recordings, 9.7k species, 484GB. Maximum pretraining data but significant storage/compute requirements.
- [ ] **Add geographically relevant non-bird data** — incorporate insects / amphibians (and possibly other taxa present in the competition taxonomy) from similar South American / Pantanal-adjacent soundscapes. Goal: improve negative class modeling, reduce false bird positives, and better match the real acoustic background distribution at test time.
- [ ] **Supplemental audio for undersampled species** — 14 species have ≤5 training samples and zero soundscape labels (mostly frogs, plus a marmoset, a nightjar, a titi monkey, and feral horse). These are effectively unlearnable from our data alone. Sources to pull from:
  - **Xeno-canto** — largest open bird/wildlife sound archive; some frog coverage but spotty for rare Neotropical species
  - **iNaturalist** — has audio observations, especially for herps; search by species + Pantanal/Mato Grosso region
  - **Fonoteca Neotropical (FN)** — specialized Neotropical animal sound archive hosted by Instituto Humboldt; best coverage for South American frogs and mammals
  - **YouTube field recordings** — search species common name + "call" or "vocalization"; extract audio clips and manually verify
  - Priority targets (1 training sample each): Hooded Capuchin (516975), Waxy Monkey Tree Frog (23724), Southern Spectacled Caiman (116570), Central Dwarf Frog (23150)
  - Secondary targets (2-3 samples): Mato Grosso Snouted Tree Frog (24321), Cei's White-lipped Frog (70711), Feral Horse (209233), Cuyaba Dwarf Frog (476521), Muller's Termite Frog (25214), Yungas de la Paz Poison Frog (64898), Black-tailed Marmoset (74580), Usina Tree Frog (555123), Spot-tailed Nightjar (sptnig1), Cope's Swamp Froglet (23176)

### 8. Class-balanced sampling / reweighting
- [ ] **Balanced label sampling** — test a sampler that increases exposure of underrepresented labels so all classes are seen more uniformly during training. Important for multi-label data: balance by positive-label coverage, not just by clip count.
- [ ] **Per-class loss weighting** — compare sampler-based balancing against per-class positive weights in the loss. Some labels may benefit more from loss reweighting than aggressive resampling.
- [ ] **Hybrid sampling scheme** — keep some natural-frequency batches for calibration while reserving part of each epoch for rare-label upsampling. Goal: reduce imbalance without making the train distribution too unrealistic.
- [ ] **Taxon-aware sampling** — ensure insects / amphibians / mammals / reptiles are not drowned out by abundant bird labels when constructing batches.

### 9. NMF latent basis — next steps
Steps 1-2 are done: global dictionary W (k=56) learned via NMFk, and per-clip projection integrated directly into HTSAT forward pass (multiplicative updates solver + nmf_proj linear head). Remaining work:

- [x] **Step 1: Learn global dictionary W** — done. W_k56.npy in `nmf_analysis/output/`, learned from representative soundscape spectrograms via NMFk.
- [x] **Step 2: Project clips into fixed basis** — done. Integrated into HTSAT forward pass: power mel spectrogram → solve H via multiplicative updates (W fixed) → mean+max summary → nmf_proj linear layer → added to classification logits.
- [ ] **Step 3: Train small CNN encoder to approximate latent features** — train a small CNN (not MLP — input has time-frequency structure) to predict the NMF latent summary vector from Step 2 directly from the clip spectrogram. Target is the NMF latent vector, not the class label. This replaces the per-clip multiplicative updates solve with a fast learned forward pass at inference time.
- [ ] **Step 4: Fuse latent branch with main classifier** — two options to evaluate:
  - **Option A: Feature concatenation** — concatenate (a) learned deep features from HTSAT backbone and (b) CNN-predicted NMF latent features before the classification head. NMF branch stays fully independent — easy to train, debug, and ablate separately. Adds minor inference cost (MobileNetV3-Small side branch).
  - **Option B: Auxiliary loss (no inference cost)** — add a second head to HTSAT that predicts the NMF latent vector as a regularization signal during training, then discard that head at inference. Pushes HTSAT's internal representations to capture spectral structure NMF finds, without adding any inference cost. More exotic — try Option A first.
- [ ] **Step 5: Ablation ladder** — test in order: (1) baseline, (2) baseline + exact NMF features (current), (3) baseline + CNN-predicted latent features, (4) optionally both. Judge by val performance and robustness on noisy soundscapes. Stop early if no improvement.

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
